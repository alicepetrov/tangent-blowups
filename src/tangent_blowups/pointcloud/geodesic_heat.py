"""
Lifted Heat Method
------------------
Approximate geodesic distances on a lifted point cloud using the heat method
(Crane et al. 2013), with the graph Laplacian built on the lifted space
and the tangent-plane gradient from ``geometry.kernels``.

Algorithm:

1) Heat diffusion:   (M + t L) u = e_i
2) Gradient flow:    X = -grad(u) / ||grad(u)||
3) Poisson solve:    L phi = -div_edge(X)

The graph Laplacian ``L = D - W`` is used for both the heat and Poisson
steps (symmetric PSD, correct null space).  The tangent-plane gradient
(``lifted_gradient``) gives ambient-space gradient vectors at each vertex.
The Poisson RHS uses an **edge-compatible divergence**: the normalised
gradient is projected onto each k-NN edge and summed with affinity
weights, so that the divergence lies in the range of L.

A lumped mass matrix ``M`` (estimated from k-NN Voronoi volumes) scales
the initial impulse to account for non-uniform point density.
"""
from __future__ import annotations

from time import perf_counter
from typing import Literal, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import connected_components

from ..geometry.iterated_grassmann import BlowUpLevel
from ..geometry.kernels import (
    estimate_product_bandwidths,
    lifted_affinity,
    product_affinity,
    affinity_to_laplacian,
    lifted_gradient,
)


def _resolve_device(device: str | None) -> str:
    """Resolve device string: None -> auto-detect, else validate."""
    if device is None:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False



def _normals_to_tangent_frames(normals: np.ndarray) -> np.ndarray:
    """Build an orthonormal 2-frame from unit normals via Gram-Schmidt.

    Args:
        normals: (N, 3) unit surface normals.

    Returns:
        (N, 3, 2) orthonormal tangent frames.
    """
    N = normals.shape[0]
    # Pick seed axis with smallest |n| component for numerical stability
    ax = np.argmin(np.abs(normals), axis=1)         # (N,)
    seed = np.zeros((N, 3), dtype=float)
    seed[np.arange(N), ax] = 1.0

    t1 = np.cross(normals, seed)                    # (N, 3)
    norms1 = np.linalg.norm(t1, axis=1, keepdims=True)

    # Fallback for degenerate cases (nearly zero cross product)
    degen = (norms1.ravel() < 1e-12)
    if degen.any():
        ax2 = (ax[degen] + 1) % 3
        seed2 = np.zeros((int(degen.sum()), 3), dtype=float)
        seed2[np.arange(len(ax2)), ax2] = 1.0
        t1[degen] = np.cross(normals[degen], seed2)
        norms1[degen] = np.linalg.norm(t1[degen], axis=1, keepdims=True)

    t1 /= norms1
    t2 = np.cross(normals, t1)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)

    frames = np.empty((N, 3, 2), dtype=float)
    frames[:, :, 0] = t1
    frames[:, :, 1] = t2
    return frames


def _resolve_source_indices(
    source_index: int | Sequence[int] | np.ndarray,
    n_points: int,
) -> np.ndarray:
    idx = np.asarray(source_index, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError("source_index must contain at least one index.")

    idx = np.where(idx < 0, idx + n_points, idx)
    if np.any(idx < 0) or np.any(idx >= n_points):
        raise ValueError("source_index out of range for point set.")
    return idx


def _estimate_time_step(
    level: BlowUpLevel,
    W: sparse.spmatrix,
    t_scale: float,
) -> float:
    """Estimate the diffusion time step from mean spatial edge length.

    Returns ``t = t_scale * mean(h)^2`` (Crane et al. 2013), where
    *h* is the mean spatial edge length in the graph.  Computed in the
    **original spatial coordinates** so that geodesic distances remain
    in spatial units.
    """
    coo = W.tocoo()
    mask = coo.row != coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    if rows.size == 0:
        return float(t_scale)

    positions = level.embedded[:, :level.n_orig]
    d = positions[rows] - positions[cols]
    dist = np.linalg.norm(d, axis=1)
    dist = dist[dist > 0.0]
    if dist.size == 0:
        return float(t_scale)
    return float(t_scale) * float(np.mean(dist) ** 2)


def _estimate_mass_matrix(
    level: BlowUpLevel,
    W: sparse.spmatrix,
) -> np.ndarray:
    """Estimate a diagonal lumped mass matrix from the lifted graph.

    Uses the lifted graph ``W`` for neighbour structure (so the mass
    respects the component separation established by the product kernel)
    but measures distances in the **original spatial coordinates**.

    For each vertex *i*, the characteristic radius ``r_k`` is estimated
    as the **median** spatial distance to its lifted-graph neighbours
    (rather than the max), which is robust to cross-sheet outliers near
    non-manifold intersections.

    Returns:
        (N,) diagonal mass entries.
    """
    N = level.N
    d = level.d  # manifold dimension
    positions = level.embedded[:, :level.n_orig]

    coo = W.tocoo()
    mask = coo.row != coo.col
    rows, cols = coo.row[mask], coo.col[mask]

    if rows.size == 0:
        return np.ones(N, dtype=float)

    dx = positions[rows] - positions[cols]
    dist = np.linalg.norm(dx, axis=1)

    # r_k(i) = median spatial distance to lifted-graph neighbours.
    # Using median instead of max avoids inflation from cross-sheet
    # neighbours that are close in the lifted space but far spatially.
    # Group by source vertex using argsort for vectorised median.
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    sorted_dist = dist[order]
    splits = np.searchsorted(sorted_rows, np.arange(N + 1))

    # Vectorised median: find max neighbours-per-vertex, pad into a
    # regular (N, max_k) array, then take np.median along axis=1.
    counts = np.diff(splits)
    max_k = int(counts.max()) if counts.size > 0 else 0
    r_k = np.zeros(N, dtype=float)
    if max_k > 0:
        padded = np.full((N, max_k), np.nan, dtype=float)
        for i in range(N):
            lo, hi = splits[i], splits[i + 1]
            if hi > lo:
                padded[i, :hi - lo] = sorted_dist[lo:hi]
        r_k = np.nanmedian(padded, axis=1)
        r_k = np.nan_to_num(r_k, nan=0.0)

    # Count neighbours per vertex (= k for interior points)
    k_per_vertex = np.bincount(rows, minlength=N).astype(float)
    k_per_vertex = np.clip(k_per_vertex, 1.0, None)

    # Volume of d-dimensional unit ball
    from math import gamma as math_gamma
    V_d = np.pi ** (d / 2.0) / math_gamma(d / 2.0 + 1.0)

    mass = V_d * (r_k ** d) / k_per_vertex

    # Safety: replace zeros with global median
    zero = mass <= 0.0
    if zero.any():
        pos = mass[~zero]
        fallback = float(np.median(pos)) if pos.size > 0 else 1.0
        mass[zero] = fallback

    return mass


def _edge_divergence(
    level: BlowUpLevel,
    X: np.ndarray,
    W: sparse.spmatrix,
) -> np.ndarray:
    """Edge-compatible divergence of a vertex vector field.

    For each edge (i, j) in the k-NN graph, the normalised gradient
    field ``X`` is averaged at the two endpoints and projected onto the
    **spatial** edge direction ``x_j - x_i``.  The weighted sum gives a
    divergence that is compatible with the graph Laplacian ``L = D - W``
    (i.e. the result lies in the range of L).

    Args:
        level:  BlowUpLevel (any level).
        X:      (N, D) normalised gradient field (unit vectors).
        W:      Sparse affinity matrix.

    Returns:
        (N,) divergence values.
    """
    N = level.N
    positions = level.embedded[:, :level.n_orig]

    coo = W.tocoo()
    rows, cols = coo.row, coo.col
    ws = np.asarray(coo.data, dtype=float)

    # Spatial edge vectors
    edge_vec = positions[cols] - positions[rows]  # (E, n_orig)

    # Average gradient at both endpoints, project onto spatial edge
    # Only use the first n_orig components of X (spatial part).
    X_spatial = X[:, :level.n_orig]
    X_avg = 0.5 * (X_spatial[rows] + X_spatial[cols])  # (E, n_orig)
    X_edge = np.einsum("ij,ij->i", X_avg, edge_vec)    # (E,)

    # Graph divergence: div_i = sum_j w_ij X_ij
    div = np.bincount(rows, weights=ws * X_edge, minlength=N)

    return div


def _apply_dirichlet_constraint(
    L: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    anchor_index: int,
    anchor_value: float = 0.0,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    L = L.tolil(copy=True)
    rhs = np.array(rhs, copy=True)

    L.rows[anchor_index] = [anchor_index]
    L.data[anchor_index] = [1.0]
    rhs[anchor_index] = anchor_value
    return L.tocsr(), rhs


def _coerce_to_level(
    input_data: BlowUpLevel | np.ndarray,
    subspace_basis: np.ndarray | None,
    *,
    alpha: float,
    k: int,
    lam: float,
) -> BlowUpLevel:
    """
    Convert any supported input to a BlowUpLevel suitable for the heat method.

    - BlowUpLevel: returned as-is (already lifted if the caller chose to).
    - Raw arrays: converted to level-0 BlowUpLevel, then optionally lifted
      to level 1 when alpha > 0.

    When ``subspace_basis`` is a 2-D array of shape ``(N, 3)`` in 3-D ambient
    space, it is interpreted as **surface normals** and converted to orthonormal
    tangent 2-frames.  In all other cases a ``(N, n)`` array is treated as a
    single tangent vector (d=1, curves).
    """
    if isinstance(input_data, BlowUpLevel):
        return input_data

    if subspace_basis is None:
        raise ValueError(
            "subspace_basis is required when passing raw point arrays."
        )
    spatial = np.asarray(input_data, dtype=float)
    basis = np.asarray(subspace_basis, dtype=float)
    if basis.ndim == 2:
        # (N, 3) in 3-D ambient space -> surface normals -> tangent 2-frames
        if spatial.shape[1] == 3:
            basis = _normals_to_tangent_frames(basis)
        else:
                basis = basis[:, :, np.newaxis]

    level = BlowUpLevel.from_point_tangents(spatial, basis)

    # Lift to level 1 when alpha > 0 so the Laplacian, gradient, and
    # divergence all operate on the sheet-aware Chordal-Sasaki embedding.
    if alpha > 0.0:
        level = level.lift(alpha=alpha, k=k, lam=lam)

    return level


def lifted_heat_method(
    points_or_sample: BlowUpLevel | np.ndarray,
    subspace_basis: np.ndarray | None = None,
    *,
    source_index: int | Sequence[int] | np.ndarray = 0,
    k: int | None = 20,
    kernel: Literal["product", "self-tuning"] = "product",
    uniform_regression: bool = False,
    diffusion_steps: int = 1,
    sigma_x: float | None = None,
    sigma_u: float | list[float] | np.ndarray | None = None,
    h: float | Literal["local"] | None = "local",
    alpha: float = 1.0,
    laplacian_normalized: bool = False,
    symmetrize: bool = True,
    t: float | None = None,
    t_scale: float = 1.0,
    gradient_eps: float = 1e-12,
    gradient_lam: float = 0.0,
    anchor_index: Optional[int] = None,
    return_intermediate: bool = False,
    timings: dict[str, float] | None = None,
    device: str | None = None,
    _precomputed: dict | None = None,
    # Legacy parameters — accepted for backward compatibility
    radius: float | None = None,  # noqa: ARG001
    subspace_metric: str = "chordal",  # noqa: ARG001
    include_self: bool = False,  # noqa: ARG001
    subspace_kind: str = "auto",  # noqa: ARG001
    normalized: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """
    Approximate geodesic distances via the heat method on a lifted point cloud.

    Implements the three-step heat method of Crane et al. (2013) using the
    graph Laplacian on the lifted space for heat diffusion and Poisson solve,
    and the tangent-plane gradient from ``geometry.kernels`` for the
    normalised-gradient step.

    The Poisson RHS is computed via an edge-compatible divergence: the
    normalised gradient is projected onto k-NN spatial edges, ensuring
    consistency with the graph Laplacian.  A lumped mass matrix scales
    the initial impulse to account for non-uniform sampling density.

    Args:
        points_or_sample:
            Input data.  One of:

            - ``BlowUpLevel`` at any level (preferred).  The Laplacian,
              gradient, and divergence are computed directly on this level.
            - ``np.ndarray`` of shape ``(N, n)`` with ``subspace_basis``
              of shape ``(N, n, d)`` or ``(N, n)`` (backward-compatible).
            For raw arrays, a level-0 ``BlowUpLevel``
            is created internally and lifted to level 1 when ``alpha > 0``.

        subspace_basis: Tangent frames, required when ``points_or_sample``
            is a raw array.
        source_index: Index (or indices) of the source point(s).
        k:  k-NN neighbourhood size.
        kernel: Kernel type for the affinity matrix.

            - ``"product"`` (default): Fixed-bandwidth product kernel with
              independent spatial and angular bandwidths::

                  W_ij = exp(-||x_i - x_j||^2 / sigma_x^2)
                       * exp(-||P_i - P_j||_F^2 / sigma_u^2)

              Smaller ``sigma_x`` tightens spatial neighbourhoods; smaller
              ``sigma_u`` increases emphasis on angular (normal) separation.
              If ``sigma_x`` or ``sigma_u`` are None, they are auto-estimated
              from the median k-NN spatial and angular distances respectively.
              PD by the Schur product theorem.

            - ``"self-tuning"``: Zelnik-Manor & Perona adaptive bandwidth
              on the Chordal-Sasaki embedding.  Uses the ``h`` parameter.
              NOT guaranteed PSD.

        uniform_regression: If True, the gradient and divergence use uniform
            weights (w_ij = 1) while the Laplacian keeps the kernel-weighted
            affinity.  Default False (gradient/divergence use the same W as
            the Laplacian).
        diffusion_steps: Number of implicit diffusion steps in Step 1.
            Solves ``(M + tL) u = rhs`` iteratively, feeding each output as
            the next RHS.  Equivalent to ``(M + tL)^n u = delta``, spreading
            heat farther without increasing t.  Default 1.
        sigma_x: Spatial bandwidth for the product kernel (> 0, or None for
            auto-estimation).  Ignored when ``kernel="self-tuning"``.
        sigma_u: Angular bandwidth(s) for the product kernel.  A single float
            applies the same bandwidth to all projector levels; a list/array
            assigns one per level.  None for auto-estimation.
            Ignored when ``kernel="self-tuning"``.
        h:  Bandwidth for the self-tuning kernel (``"local"``, ``None``, or
            float).  Ignored when ``kernel="product"``.
        alpha:  Chordal-Sasaki weight for the lift.  ``alpha=0`` gives the
            standard Euclidean heat method (no lifting).
        laplacian_normalized: Use symmetric-normalised Laplacian if True.
        symmetrize: Symmetrise the affinity matrix.
        t:  Diffusion time (auto-estimated if None).
        t_scale: Multiplier for auto-estimated *t*.
        gradient_eps: Floor for gradient norms (avoids division by zero).
        gradient_lam: Tikhonov regularisation for gradient/divergence.
        anchor_index: Dirichlet anchor for Poisson solve (default: first source).
        return_intermediate: If True, return ``(phi, u, X, div)``.
        timings: Optional dict populated with elapsed times for the
            precompute reuse/build, heat diffusion, gradient, Poisson, and
            total solve stages.
        device: Compute device.  ``"cuda"`` uses GPU-accelerated sparse
            solvers and edge operations; ``"cpu"`` uses the original
            numpy/scipy path.  ``None`` (default) auto-detects CUDA.
        _precomputed: Optional dict of cached intermediate results from a
            previous call.  When provided, the affinity matrix, Laplacian,
            mass matrix, heat operator, and edge cache are reused instead
            of recomputed.  Call :func:`precompute_heat_method` to build
            this dict.

    Returns:
        Geodesic distances ``phi`` (shape ``(N,)``), or
        ``(phi, u, X, div)`` if ``return_intermediate=True``.
    """
    # --- Resolve device ---
    use_device = _resolve_device(device)
    use_gpu = use_device == "cuda" and _gpu_available()
    total_start = perf_counter()

    # --- Resolve backward-compatible parameter aliases ---
    use_normalized = laplacian_normalized
    if normalized is not None:
        use_normalized = normalized

    k_eff = k if k is not None else 30

    # --- Convert input to BlowUpLevel ---
    level = _coerce_to_level(
        points_or_sample, subspace_basis,
        alpha=alpha, k=k_eff, lam=0.0,
    )

    n_points = level.N
    if n_points == 0:
        empty = np.zeros((0,), dtype=float)
        return (empty, empty, empty, empty) if return_intermediate else empty

    src_idx = _resolve_source_indices(source_index, n_points)
    if anchor_index is None:
        anchor_index = int(src_idx[0])

    # --- Build or reuse precomputed data ---
    if _precomputed is not None:
        precompute_elapsed = 0.0
        L = _precomputed["L"]
        W = _precomputed["W"]
        W_reg = _precomputed["W_reg"]
        A = _precomputed["A"]
        edge_cache = _precomputed.get("edge_cache")
        n_comp = _precomputed["n_comp"]
        comp_labels = _precomputed["comp_labels"]
    else:
        precompute_start = perf_counter()
        if kernel == "product":
            sx, su = sigma_x, sigma_u
            if sx is None or su is None:
                auto_sx, auto_su = estimate_product_bandwidths(
                    level, k_eff,
                )
                if sx is None:
                    sx = auto_sx
                if su is None:
                    su = auto_su
            W = product_affinity(
                level, sx, su, k=k_eff, symmetrize=symmetrize,
                device=use_device if use_gpu else "cpu",
            )
        elif kernel == "self-tuning":
            W = lifted_affinity(level, k=k_eff, h=h, symmetrize=symmetrize)
        else:
            raise ValueError(f"Unknown kernel type: {kernel!r}")

        L, W, _ = affinity_to_laplacian(W, normalized=use_normalized)
        W = W.tocsr()

        if uniform_regression:
            W_reg = W.copy()
            W_reg.data[:] = 1.0
        else:
            W_reg = W

        mass = _estimate_mass_matrix(level, W)

        if t is None:
            t = _estimate_time_step(level, W, t_scale)

        M = sparse.diags(mass, format="csr")
        A = M + float(t) * L
        n_comp, comp_labels = connected_components(
            W, directed=False, connection="weak",
        )

        if use_gpu:
            from ..geometry._gpu_ops import precompute_edges_gpu
            edge_cache = precompute_edges_gpu(level, W_reg, gradient_lam)
        else:
            edge_cache = None
        precompute_elapsed = perf_counter() - precompute_start

    # --- Step 1: Heat diffusion  (M + tL)^n u = e_i ---
    # The Dirac delta as a FEM functional gives rhs = e_i (indicator at
    # source), matching geometry-central's implementation exactly.
    # Multiple steps spread heat farther without increasing t.
    u = np.zeros(n_points, dtype=float)
    u[src_idx] = 1.0

    diffusion_start = perf_counter()
    if use_gpu:
        from ..solvers.gpu_sparse import sparse_solve_gpu
        from ..geometry._gpu_ops import (
            lifted_gradient_gpu,
            edge_divergence_gpu,
        )
        for _ in range(max(diffusion_steps, 1)):
            u = sparse_solve_gpu(A, u, x0_numpy=u)
    else:
        for _ in range(max(diffusion_steps, 1)):
            u = spla.spsolve(A, u)
    diffusion_elapsed = perf_counter() - diffusion_start

    # --- Step 2: Normalised gradient  X = -grad(u) / |grad(u)| ---
    # The divergence operator uses only the spatial (first n_orig) components
    # of X, so we must normalise the spatial part specifically — otherwise
    # the unit-length constraint in D dimensions leaves the spatial
    # projection with norm << 1, systematically deflating the divergence.
    gradient_start = perf_counter()
    if use_gpu:
        grad_u = lifted_gradient_gpu(
            level, u, W_reg, lam=gradient_lam, _edge_cache=edge_cache,
        )
    else:
        grad_u = lifted_gradient(level, u, W_reg, lam=gradient_lam)
    n_orig = level.n_orig
    grad_spatial = grad_u[:, :n_orig]
    grad_spatial_norms = np.linalg.norm(grad_spatial, axis=1)
    grad_spatial_norms = np.clip(grad_spatial_norms, gradient_eps, None)
    X = np.zeros_like(grad_u)
    X[:, :n_orig] = -grad_spatial / grad_spatial_norms[:, None]
    gradient_elapsed = perf_counter() - gradient_start

    # --- Step 3: Poisson solve  L phi = -div_edge(X) ---
    # Use edge-compatible divergence so the RHS lies in the range of L.
    poisson_start = perf_counter()
    if use_gpu:
        div_X = edge_divergence_gpu(level, X, W_reg)
    else:
        div_X = _edge_divergence(level, X, W_reg)

    L_poisson, div_rhs = _apply_dirichlet_constraint(
        L,
        -div_X,
        anchor_index=int(anchor_index),
        anchor_value=0.0,
    )

    # When tight kernels create disconnected components, anchor one node
    # per extra component so the Poisson system is non-singular.
    if n_comp > 1:
        src_label = comp_labels[anchor_index]
        L_poisson = L_poisson.tolil()
        for c in range(n_comp):
            if c == src_label:
                continue
            idx = int(np.flatnonzero(comp_labels == c)[0])
            L_poisson.rows[idx] = [idx]
            L_poisson.data[idx] = [1.0]
            div_rhs[idx] = 0.0
        L_poisson = L_poisson.tocsr()

    if use_gpu:
        phi = sparse_solve_gpu(L_poisson, div_rhs)
    else:
        phi = spla.spsolve(L_poisson, div_rhs)
    poisson_elapsed = perf_counter() - poisson_start

    if timings is not None:
        timings["precompute"] = precompute_elapsed
        timings["diffusion"] = diffusion_elapsed
        timings["gradient"] = gradient_elapsed
        timings["poisson"] = poisson_elapsed
        timings["total"] = perf_counter() - total_start

    if return_intermediate:
        return phi, u, X, div_X
    return phi


def precompute_heat_method(
    level: BlowUpLevel,
    *,
    k: int = 20,
    kernel: Literal["product", "self-tuning"] = "product",
    uniform_regression: bool = False,
    sigma_x: float | None = None,
    sigma_u: float | list[float] | np.ndarray | None = None,
    h: float | Literal["local"] | None = "local",
    laplacian_normalized: bool = False,
    symmetrize: bool = True,
    t: float | None = None,
    t_scale: float = 1.0,
    gradient_lam: float = 0.0,
    device: str | None = None,
) -> dict:
    """Precompute the static parts of the heat method for repeated queries.

    Returns a dict that can be passed as ``_precomputed`` to
    :func:`lifted_heat_method` to skip affinity, Laplacian, mass matrix,
    and edge precomputation on each call.

    Args:
        level:  BlowUpLevel (any level).
        k, kernel, sigma_x, sigma_u, h, laplacian_normalized, symmetrize,
        t, t_scale, gradient_lam, device:
            Same meaning as in :func:`lifted_heat_method`.

    Returns:
        Dict of cached arrays (L, W, W_reg, A, edge_cache, etc.).
    """
    use_device = _resolve_device(device)
    use_gpu = use_device == "cuda" and _gpu_available()
    use_normalized = laplacian_normalized

    if kernel == "product":
        sx, su = sigma_x, sigma_u
        if sx is None or su is None:
            auto_sx, auto_su = estimate_product_bandwidths(level, k)
            if sx is None:
                sx = auto_sx
            if su is None:
                su = auto_su
        W = product_affinity(
            level, sx, su, k=k, symmetrize=symmetrize,
            device=use_device if use_gpu else "cpu",
        )
    elif kernel == "self-tuning":
        W = lifted_affinity(level, k=k, h=h, symmetrize=symmetrize)
    else:
        raise ValueError(f"Unknown kernel type: {kernel!r}")

    L, W, _ = affinity_to_laplacian(W, normalized=use_normalized)
    W = W.tocsr()

    if uniform_regression:
        W_reg = W.copy()
        W_reg.data[:] = 1.0
    else:
        W_reg = W

    mass = _estimate_mass_matrix(level, W)
    if t is None:
        t = _estimate_time_step(level, W, t_scale)

    M = sparse.diags(mass, format="csr")
    A = M + float(t) * L

    n_comp, comp_labels = connected_components(
        W, directed=False, connection="weak",
    )

    edge_cache = None
    if use_gpu:
        from ..geometry._gpu_ops import precompute_edges_gpu
        edge_cache = precompute_edges_gpu(level, W_reg, gradient_lam)

    return dict(
        L=L, W=W, W_reg=W_reg, A=A,
        edge_cache=edge_cache,
        n_comp=n_comp, comp_labels=comp_labels,
    )


__all__ = ["lifted_heat_method", "precompute_heat_method"]
