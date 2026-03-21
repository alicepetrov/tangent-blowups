"""
Lifted Heat Method
------------------
Approximate geodesic distances on a lifted point cloud using the heat method
(Crane et al. 2013), with the Laplacian, gradient, and divergence operators
defined in the tangent blow-up's lifted space.

Algorithm:
1) Heat diffusion:   (I + t L) u = delta
2) Gradient flow:    X = -grad(u) / ||grad(u)||
3) Poisson solve:    L phi = -div(X)

All three operators (L, grad, div) are computed on the lifted point cloud via
the discrete operators defined in ``geometry.kernels``, ensuring consistency
with the paper (Sections 5.1--5.2).
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from ..geometry.iterated_grassmann import BlowUpLevel
from ..geometry.kernels import (
    lifted_affinity,
    affinity_to_laplacian,
    lifted_gradient,
    lifted_divergence,
)

# Keep old import available for backward-compatible input coercion
from ..geometry.grassmann import BlownUpSample


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
    """Estimate the diffusion time step from median edge distance."""
    coo = W.tocoo()
    mask = coo.row != coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    if rows.size == 0:
        return float(t_scale)

    # Use original spatial positions for the time scale so that
    # geodesic distances remain in spatial units.
    positions = level.embedded[:, :level.n_orig]
    d = positions[rows] - positions[cols]
    dist = np.linalg.norm(d, axis=1)
    dist = dist[dist > 0.0]
    if dist.size == 0:
        return float(t_scale)
    return float(t_scale) * float(np.median(dist) ** 2)


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
            basis = basis[:, :, np.newaxis]

    level = BlowUpLevel.from_point_tangents(spatial, basis)

    # Lift to level 1 when alpha > 0 so the Laplacian, gradient, and
    # divergence all operate on the sheet-aware Chordal-Sasaki embedding.
    if alpha > 0.0:
        level = level.lift(alpha=alpha, k=k, lam=lam)

    return level


def lifted_heat_method(
    points_or_sample: BlowUpLevel | np.ndarray | BlownUpSample,
    subspace_basis: np.ndarray | None = None,
    *,
    source_index: int | Sequence[int] | np.ndarray = 0,
    k: int | None = 30,
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
    discrete Laplacian, gradient, and divergence from the tangent blow-up
    framework (paper Sections 5.1--5.2).

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
        h:  Bandwidth for self-tuning kernel (``"local"``, ``None``, or float).
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

    # --- Build Laplacian on the (possibly lifted) level ---
    W = lifted_affinity(level, k=k_eff, h=h, symmetrize=symmetrize)
    L, W, _ = affinity_to_laplacian(W, normalized=use_normalized)
    W = W.tocsr()

    # --- Step 1: Heat diffusion  (I + tL) u = delta ---
    if t is None:
        t = _estimate_time_step(level, W, t_scale)

    rhs = np.zeros(n_points, dtype=float)
    rhs[src_idx] = 1.0

    A = sparse.eye(n_points, format="csr") + float(t) * L
    u = spla.spsolve(A, rhs)

    # --- Step 2: Normalised gradient  X = -grad(u) / |grad(u)| ---
    grad_u = lifted_gradient(level, u, W, lam=gradient_lam)
    grad_norms = np.linalg.norm(grad_u, axis=1)
    grad_norms = np.clip(grad_norms, gradient_eps, None)
    X = -grad_u / grad_norms[:, None]

    # --- Step 3: Poisson solve  L phi = -div(X) ---
    div_X = lifted_divergence(level, X, W, lam=gradient_lam)

    L_poisson, div_rhs = _apply_dirichlet_constraint(
        L,
        -div_X,
        anchor_index=int(anchor_index),
        anchor_value=0.0,
    )
    phi = spla.spsolve(L_poisson, div_rhs)

    if return_intermediate:
        return phi, u, X, div_X
    return phi


__all__ = ["lifted_heat_method"]
