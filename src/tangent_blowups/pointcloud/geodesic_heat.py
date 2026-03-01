"""
Lifted Heat Method
------------------
Approximate geodesic distances on a lifted point cloud using the heat method.

Algorithm (Crane et al., adapted to lifted Laplacian):
1) Heat diffusion:   (I + t L) u = delta          
2) Gradient flow:    X = -grad(u) / ||grad(u)||
3) Poisson solve:    L phi = -div(X)              
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from ..geometry.grassmann import BlownUpSample
from .laplacian import lifted_pointcloud_laplacian

SubspaceKind = Literal["auto", "tangent", "normal"]
SubspaceMetric = Literal["chordal", "geodesic"]


def _coerce_blown_up(
    points_or_sample: np.ndarray | BlownUpSample,
    subspace_basis: np.ndarray | None,
) -> BlownUpSample:
    if isinstance(points_or_sample, BlownUpSample):
        if subspace_basis is not None:
            raise ValueError("subspace_basis must be None when passing BlownUpSample.")
        return points_or_sample
    if subspace_basis is None:
        raise ValueError("subspace_basis is required when passing raw points.")
    return BlownUpSample(points_or_sample, subspace_basis)


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


def _resolve_subspace_kind(sample: BlownUpSample, kind: SubspaceKind) -> SubspaceKind:
    if kind != "auto":
        return kind

    if sample.k == sample.n - 1:
        return "tangent"
    if sample.k == 1 and sample.n == 2:
        return "tangent"
    if sample.k == 1:
        return "normal"
    return "tangent"


def _tangent_projectors(sample: BlownUpSample, kind: SubspaceKind) -> np.ndarray:
    P = sample.projectors
    if kind == "tangent":
        return P
    if kind == "normal":
        eye = np.eye(sample.n, dtype=float)
        return eye[None, :, :] - P
    raise ValueError(f"Unknown subspace_kind '{kind}'.")


def _estimate_time_step(points: np.ndarray, W: sparse.spmatrix, t_scale: float) -> float:
    coo = W.tocoo()
    mask = coo.row != coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    if rows.size == 0:
        return float(t_scale)

    d = points[rows] - points[cols]
    dist = np.linalg.norm(d, axis=1)
    dist = dist[dist > 0.0]
    if dist.size == 0:
        return float(t_scale)
    return float(t_scale) * float(np.median(dist) ** 2)


def _weighted_least_squares_gradient(
    points: np.ndarray,
    values: np.ndarray,
    W: sparse.csr_matrix,
    projectors: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, dim = points.shape
    grads = np.zeros((n, dim), dtype=float)

    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        js = W.indices[start:end]
        ws = W.data[start:end]

        if js.size == 0:
            continue

        mask = js != i
        if not np.any(mask):
            continue
        js = js[mask]
        ws = ws[mask]

        disp = points[js] - points[i]
        disp = disp @ projectors[i]
        if disp.size == 0:
            continue

        rhs = values[js] - values[i]
        w_sqrt = np.sqrt(ws)
        A = disp * w_sqrt[:, None]
        b = rhs * w_sqrt

        try:
            g, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue

        g = projectors[i] @ g
        grads[i] = g

    norms = np.linalg.norm(grads, axis=1)
    norms = np.clip(norms, eps, None)
    return grads, norms


def _graph_divergence(
    points: np.ndarray,
    vectors: np.ndarray,
    W: sparse.csr_matrix,
    projectors: np.ndarray,
) -> np.ndarray:
    """
    Compute the integrated graph divergence using edge flux.
    div_i = sum_j W_{ij} * ( (X_i + X_j)/2 * (x_j - x_i) )
    """
    n, dim = points.shape
    div = np.zeros(n, dtype=float)

    for i in range(n):
        start, end = W.indptr[i], W.indptr[i + 1]
        js = W.indices[start:end]
        ws = W.data[start:end]

        if js.size == 0:
            continue

        mask = js != i
        if not np.any(mask):
            continue
        js = js[mask]
        ws = ws[mask]

        # The outgoing edge vector, projected to tangent space
        disp = points[js] - points[i]
        disp = disp @ projectors[i]

        # The average vector field on the edge
        avg_X = 0.5 * (vectors[i] + vectors[js])
        avg_X = avg_X @ projectors[i]

        # Flux is the dot product of the average vector and the displacement
        flux = np.sum(avg_X * disp, axis=1)

        # Divergence is the weighted sum of outgoing flux
        div[i] = np.sum(ws * flux)

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


def lifted_heat_method(
    points_or_sample: np.ndarray | BlownUpSample,
    subspace_basis: np.ndarray | None = None,
    *,
    source_index: int | Sequence[int] | np.ndarray = 0,
    k: int | None = 16,
    radius: float | None = None,
    h: float | Literal["local"] | None = "local",
    alpha: float = 1.0,
    subspace_metric: SubspaceMetric = "chordal",
    laplacian_normalized: bool = False,
    symmetrize: bool = True,
    include_self: bool = False,
    t: float | None = None,
    t_scale: float = 1.0,
    subspace_kind: SubspaceKind = "auto",
    gradient_eps: float = 1e-12,
    anchor_index: Optional[int] = None,
    return_intermediate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    
    sample = _coerce_blown_up(points_or_sample, subspace_basis)
    n_points = sample.N
    if n_points == 0:
        empty = np.zeros((0,), dtype=float)
        return (empty, empty, empty, empty) if return_intermediate else empty

    src_idx = _resolve_source_indices(source_index, n_points)
    if anchor_index is None:
        anchor_index = int(src_idx[0])

    kind = _resolve_subspace_kind(sample, subspace_kind)
    projectors = _tangent_projectors(sample, kind)

    L, W, _ = lifted_pointcloud_laplacian(
        sample,
        k=k,
        radius=radius,
        h=h,
        alpha=alpha,
        subspace_metric=subspace_metric,
        normalized=laplacian_normalized,
        symmetrize=symmetrize,
        include_self=include_self,
        return_parts=True,
    )

    W = W.tocsr()
    points = np.asarray(sample.spatial, dtype=float)

    if t is None:
        t = _estimate_time_step(points, W, t_scale)

    rhs = np.zeros(n_points, dtype=float)
    rhs[src_idx] = 1.0

    A = sparse.eye(n_points, format="csr") + float(t) * L
    u = spla.spsolve(A, rhs)

    grads, grad_norms = _weighted_least_squares_gradient(
        points,
        u,
        W,
        projectors,
        eps=gradient_eps,
    )
    X = -grads / grad_norms[:, None]

    div = _graph_divergence(
        points,
        X,
        W,
        projectors,
    )

    L_poisson, div_rhs = _apply_dirichlet_constraint(
        L,
        -div,
        anchor_index=int(anchor_index),
        anchor_value=0.0,
    )
    phi = spla.spsolve(L_poisson, div_rhs)

    if return_intermediate:
        return phi, u, X, div
    return phi


__all__ = ["lifted_heat_method"]