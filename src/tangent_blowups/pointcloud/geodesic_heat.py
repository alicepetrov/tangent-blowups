"""
Lifted Heat Method
------------------
Approximate geodesic distances on a lifted point cloud using the heat method
(Crane et al. 2013), with the graph Laplacian built on the lifted space
and the tangent-plane gradient from ``geometry.kernels``.

Algorithm:

1) Heat diffusion:   (I + t L) u = M^{-1} delta
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

from typing import Literal, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import connected_components

from ..geometry.iterated_grassmann import BlowUpLevel
from ..geometry.kernels import (
    lifted_affinity,
    product_affinity,
    affinity_to_laplacian,
    lifted_gradient,
    lifted_divergence,
    _precompute_edges,
    _knn_edges,
)

# Keep old import available for backward-compatible input coercion
from ..geometry.grassmann import BlownUpSample


def _normals_to_tangent_frames(normals: np.ndarray) -> np.ndarray:
    """Build an orthonormal 2-frame from unit normals via Gram-Schmidt.

    Args:
        normals: (N, 3) unit surface normals.

    Returns:
        (N, 3, 2) orthonormal tangent frames.
    """
    N = normals.shape[0]
    frames = np.zeros((N, 3, 2), dtype=float)
    for i in range(N):
        n = normals[i]
        ax = int(np.argmin(np.abs(n)))
        seed = np.zeros(3)
        seed[ax] = 1.0
        t1 = np.cross(n, seed)
        norm1 = np.linalg.norm(t1)
        if norm1 < 1e-12:
            seed = np.zeros(3)
            seed[(ax + 1) % 3] = 1.0
            t1 = np.cross(n, seed)
            norm1 = np.linalg.norm(t1)
        t1 /= norm1
        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2)
        frames[i, :, 0] = t1
        frames[i, :, 1] = t2
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
    """Estimate the diffusion time step from mean edge distance.

    Uses ``mean(edge_length)^2`` (matching geometry-central) rather than
    median, computed in the **original spatial coordinates** so that
    geodesic distances remain in spatial units.
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
    """Estimate a diagonal lumped mass matrix from k-NN distances.

    For a *d*-dimensional manifold sampled with *N* points, the Voronoi
    volume at point *i* is approximated by

        M_ii = V_d * r_k^d / k

    where *r_k* is the distance to the k-th nearest neighbour (the
    farthest edge in the k-NN graph), *V_d* is the volume of the
    *d*-dimensional unit ball, and *k* is the neighbourhood size.

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

    # r_k(i) = max distance to any neighbour of i in the k-NN graph
    r_k = np.zeros(N, dtype=float)
    np.maximum.at(r_k, rows, dist)

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
    div = np.zeros(N, dtype=float)
    np.add.at(div, rows, ws * X_edge)

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
    input_data: BlowUpLevel | BlownUpSample | np.ndarray,
    subspace_basis: np.ndarray | None,
    *,
    alpha: float,
    k: int,
    lam: float,
) -> BlowUpLevel:
    """
    Convert any supported input to a BlowUpLevel suitable for the heat method.

    - BlowUpLevel: returned as-is (already lifted if the caller chose to).
    - BlownUpSample / raw arrays: converted to level-0 BlowUpLevel, then
      optionally lifted to level 1 when alpha > 0.

    When ``subspace_basis`` is a 2-D array of shape ``(N, 3)`` in 3-D ambient
    space, it is interpreted as **surface normals** and converted to orthonormal
    tangent 2-frames.  In all other cases a ``(N, n)`` array is treated as a
    single tangent vector (d=1, curves).
    """
    if isinstance(input_data, BlowUpLevel):
        return input_data

    # Convert old-style inputs to a level-0 BlowUpLevel
    if isinstance(input_data, BlownUpSample):
        spatial = input_data.spatial
        basis = input_data.basis
    else:
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


def _auto_estimate_product_bandwidths(
    level: BlowUpLevel,
    k: int,
) -> tuple[float, float]:
    """Estimate sigma_x and sigma_u from median k-NN distances.

    Builds a k-NN graph in the Chordal-Sasaki embedding, then computes
    the median spatial and angular neighbour distances separately.

    Returns:
        (sigma_x, sigma_u) — median spatial distance, median angular distance.
    """
    rows, cols, _ = _knn_edges(level.embedded, k)

    # Spatial distances
    positions = level.embedded[:, :level.n_orig]
    dx = positions[rows] - positions[cols]
    dist_x = np.sqrt(np.einsum("ij,ij->i", dx, dx))
    dist_x = dist_x[dist_x > 0.0]
    sigma_x = float(np.median(dist_x)) if dist_x.size > 0 else 1.0

    # Angular distances (chordal: ||P_i - P_j||_F)
    U = level.frame  # (N, D, d)
    M = np.einsum("eka,ekb->eab", U[rows], U[cols])   # (E, d, d)
    inner_sq = np.einsum("eab,eab->e", M, M)
    proj_dist2 = np.clip(2.0 * level.d - 2.0 * inner_sq, 0.0, None)
    dist_u = np.sqrt(proj_dist2)
    dist_u = dist_u[dist_u > 0.0]
    sigma_u = float(np.median(dist_u)) if dist_u.size > 0 else 1.0

    return sigma_x, sigma_u


def lifted_heat_method(
    points_or_sample: BlowUpLevel | np.ndarray | BlownUpSample,
    subspace_basis: np.ndarray | None = None,
    *,
    source_index: int | Sequence[int] | np.ndarray = 0,
    k: int | None = 30,
    kernel: Literal["product", "self-tuning"] = "product",
    sigma_x: float | None = None,
    sigma_u: float | list[float] | np.ndarray | None = None,
    h: float | Literal["local"] | None = "local",
    alpha: float = 1.0,
    laplacian_normalized: bool = False,
    symmetrize: bool = True,
    t: float | None = None,
    t_scale: float = 1.0,
    gradient_eps: float = 1e-12,
    gradient_lam: float = 1e-3,
    anchor_index: Optional[int] = None,
    return_intermediate: bool = False,
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
            - ``BlownUpSample`` (legacy, backward-compatible).

            For raw arrays and ``BlownUpSample``, a level-0 ``BlowUpLevel``
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

    Returns:
        Geodesic distances ``phi`` (shape ``(N,)``), or
        ``(phi, u, X, div)`` if ``return_intermediate=True``.
    """
    # --- Resolve backward-compatible parameter aliases ---
    use_normalized = laplacian_normalized
    if normalized is not None:
        use_normalized = normalized

    k_eff = k if k is not None else 30

    # --- Convert input to BlowUpLevel ---
    level = _coerce_to_level(
        points_or_sample, subspace_basis,
        alpha=alpha, k=k_eff, lam=1e-3,
    )

    n_points = level.N
    if n_points == 0:
        empty = np.zeros((0,), dtype=float)
        return (empty, empty, empty, empty) if return_intermediate else empty

    src_idx = _resolve_source_indices(source_index, n_points)
    if anchor_index is None:
        anchor_index = int(src_idx[0])

    # --- Build affinity matrix ---
    if kernel == "product":
        sx, su = sigma_x, sigma_u
        if sx is None or su is None:
            auto_sx, auto_su = _auto_estimate_product_bandwidths(level, k_eff)
            if sx is None:
                sx = auto_sx
            if su is None:
                su = auto_su
        W = product_affinity(level, sx, su, k=k_eff, symmetrize=symmetrize)
    elif kernel == "self-tuning":
        W = lifted_affinity(level, k=k_eff, h=h, symmetrize=symmetrize)
    else:
        raise ValueError(f"Unknown kernel type: {kernel!r}")

    L, W, _ = affinity_to_laplacian(W, normalized=use_normalized)
    W = W.tocsr()

    # --- Estimate mass matrix and time step ---
    mass = _estimate_mass_matrix(level, W)

    if t is None:
        t = _estimate_time_step(level, W, t_scale)

    # --- Step 1: Heat diffusion  (I + tL) u = M^{-1} delta ---
    # The mass-scaled RHS accounts for non-uniform sampling density.
    rhs = np.zeros(n_points, dtype=float)
    rhs[src_idx] = 1.0 / mass[src_idx]

    A = sparse.eye(n_points, format="csr") + float(t) * L
    u = spla.spsolve(A, rhs)

    # --- Step 2: Normalised gradient  X = -grad(u) / |grad(u)| ---
    grad_u = lifted_gradient(level, u, W, lam=gradient_lam)
    grad_norms = np.linalg.norm(grad_u, axis=1)
    grad_norms = np.clip(grad_norms, gradient_eps, None)
    X = -grad_u / grad_norms[:, None]

    # --- Step 3: Poisson solve  L phi = -div_edge(X) ---
    # Use edge-compatible divergence so the RHS lies in the range of L.
    div_X = _edge_divergence(level, X, W)

    L_poisson, div_rhs = _apply_dirichlet_constraint(
        L,
        -div_X,
        anchor_index=int(anchor_index),
        anchor_value=0.0,
    )

    # When tight kernels create disconnected components, anchor one node
    # per extra component so the Poisson system is non-singular.
    n_comp, comp_labels = connected_components(W, directed=False, connection="weak")
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

    phi = spla.spsolve(L_poisson, div_rhs)

    if return_intermediate:
        return phi, u, X, div_X
    return phi


__all__ = ["lifted_heat_method"]
