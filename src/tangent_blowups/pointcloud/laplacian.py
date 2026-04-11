"""
Point Cloud Laplacians
----------------------
Builds graph Laplacians from point clouds using a Gaussian (heat kernel) weight.
Also supports Laplacians on lifted points in the product space R^n x G(k, n).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import warnings

from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence

from .neighbors import knn_edges, radius_edges
from ..geometry.iterated_grassmann import BlowUpLevel

LaplacianReturn = (
    sparse.csr_matrix
    | tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]
)


def _as_points(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"points must be 2D (N, d). Got {P.shape}.")
    if not np.isfinite(P).all():
        raise ValueError("points must be finite.")
    return P


def _pick_bandwidth(
    n: int,
    rows: np.ndarray,
    dist2: np.ndarray,
    h: float | Literal["local"] | None,
) -> float | np.ndarray:
    """
    Determine the bandwidth for the Gaussian kernel.
    
    Returns:
        float: A single global bandwidth.
        np.ndarray: An array of shape (n,) containing local bandwidths per point.
    """
    # 1. User specified a fixed global bandwidth
    if isinstance(h, (int, float)):
        h_val = float(h)
        if h_val <= 0.0:
            raise ValueError("h must be positive.")
        return h_val

    # Edge case: empty graph
    if dist2.size == 0:
        return 1.0

    # 2. Local self-tuning bandwidth (Zelnik-Manor & Perona)
    if h == "local":
        h_sq = np.zeros(n, dtype=float)
        # For a k-NN graph, the max distance in a row is the distance to the k-th neighbor.
        np.maximum.at(h_sq, rows, dist2)
        
        h_local = np.sqrt(h_sq)
        
        # Safety fallback: If duplicate points exist, distance to k-th neighbor might be 0.
        # We replace 0s with the global median to prevent division by zero.
        zero_mask = h_local == 0
        if zero_mask.any():
            global_median = float(np.median(np.sqrt(dist2[dist2 > 0]))) if (dist2 > 0).any() else 1.0
            h_local[zero_mask] = global_median
            
        return h_local

    # 3. Global median bandwidth (h = None)
    dist = np.sqrt(dist2)
    dist = dist[dist > 0.0]
    if dist.size == 0:
        return 1.0
    return float(np.median(dist))


def _build_weight_matrix(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    dist2: np.ndarray,
    *,
    h: float | np.ndarray,
    symmetrize: bool,
) -> sparse.csr_matrix:
    if n == 0 or rows.size == 0:
        return sparse.csr_matrix((n, n), dtype=float)

    # Apply the bandwidth(s)
    if isinstance(h, np.ndarray):
        # Local scaling: W_ij = exp(- d_ij^2 / (h_i * h_j))
        h_i = h[rows]
        h_j = h[cols]
        weights = np.exp(-dist2 / (h_i * h_j))
    else:
        # Global scaling: W_ij = exp(- d_ij^2 / h^2)
        weights = np.exp(-dist2 / (h * h))

    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    W.sum_duplicates()

    if symmetrize:
        W = W.maximum(W.T)

    return W


def _laplacian_from_weight(
    W: sparse.csr_matrix,
    *,
    normalized: bool,
    eps: float,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    n = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sparse.diags(deg, format="csr")

    if not normalized:
        L = D - W
        return L, W, D

    inv_sqrt = np.zeros_like(deg)
    mask = deg > eps
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv = sparse.diags(inv_sqrt, format="csr")
    L = sparse.eye(n, format="csr") - D_inv @ W @ D_inv
    return L, W, D

def pointcloud_laplacian(
    points: np.ndarray,
    *,
    k: int | None = 20,
    radius: float | None = None,
    h: float | Literal["local"] | None = "local",
    symmetrize: bool = True,
    include_self: bool = False,
    normalized: bool = False,
    return_parts: bool = False,
    workers: int = -1,
    eps: float = 1e-12,
) -> LaplacianReturn:
    """
    Build a Laplacian for a point cloud in Euclidean space.

    Args:
        points: (N, d) array.
        k: number of nearest neighbors (ignored if radius is provided).
        radius: neighborhood radius for radius graph (overrides k).
        h: Gaussian kernel bandwidth. "local" uses distance to k-th neighbor.
           If None, uses global median neighbor distance.
        symmetrize: ensure symmetric weights (recommended for kNN).
        include_self: include self edges with weight exp(0)=1.
        normalized: if True, returns symmetric normalized Laplacian.
        return_parts: if True, returns (L, W, D).
        workers: cKDTree parallelism (workers=-1 uses all cores).
        eps: numerical tolerance for normalized Laplacian.
    """
    P = _as_points(points)
    n = P.shape[0]
    if n == 0:
        empty = sparse.csr_matrix((0, 0), dtype=float)
        if return_parts:
            return empty, empty, empty
        return empty

    if radius is not None:
        rows, cols, dist2 = radius_edges(
            P,
            radius,
            include_self=include_self,
            workers=workers,
        )
    else:
        if k is None:
            raise ValueError("Either k or radius must be provided.")
        rows, cols, dist2 = knn_edges(
            P,
            k,
            include_self=include_self,
            workers=workers,
        )

    # Use the updated bandwidth picker
    h_use = _pick_bandwidth(n, rows, dist2, h)
    
    # Pass h_use directly to the updated weight builder
    W = _build_weight_matrix(
        n, rows, cols, dist2, h=h_use, symmetrize=symmetrize
    )
    L, W, D = _laplacian_from_weight(W, normalized=normalized, eps=eps)

    if return_parts:
        return L, W, D
    return L

def bilateral_pointcloud_laplacian(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 20,
    sigma_x: float | None = None,
    sigma_n: float | None = None,
    symmetrize: bool = True,
    normalized: bool = False,
    return_parts: bool = False,
    workers: int = -1,
    eps: float = 1e-12,
) -> LaplacianReturn:
    """Build a bilateral graph Laplacian weighted by spatial AND normal similarity.

    The affinity between points *i* and *j* is::

        W_ij = exp(-||x_i - x_j||^2 / sigma_x^2)
             * exp(-||n_i - n_j||^2 / sigma_n^2)

    This is the standard normal-aware baseline for point cloud processing.
    It distinguishes nearby points with different normals (e.g. at sharp
    edges) but, unlike the lifted Laplacian, cannot separate overlapping
    sheets that share similar normals at different spatial locations.

    Args:
        points:  (N, d) spatial coordinates.
        normals: (N, d) unit surface normals.
        k:       k-NN neighbourhood size (built in spatial coordinates).
        sigma_x: Spatial bandwidth.  If None, set to the median spatial
                 k-NN distance.
        sigma_n: Normal bandwidth.  If None, set to the median normal
                 k-NN distance.
        symmetrize: Symmetrise the affinity matrix.
        normalized: Return symmetric-normalised Laplacian if True.
        return_parts: If True, return (L, W, D).
        workers: cKDTree parallelism (-1 = all cores).
        eps:     Numerical tolerance for normalised Laplacian.
    """
    from scipy.spatial import cKDTree

    P = _as_points(points)
    N_pts = np.asarray(normals, dtype=float)
    if P.shape != N_pts.shape:
        raise ValueError(
            f"points {P.shape} and normals {N_pts.shape} must match.")
    n = P.shape[0]
    if n == 0:
        empty = sparse.csr_matrix((0, 0), dtype=float)
        if return_parts:
            return empty, empty, empty
        return empty

    tree = cKDTree(P)
    _, nn_idx = tree.query(P, k=min(k + 1, n), workers=workers)
    nn_idx = nn_idx[:, 1:]  # drop self

    rows = np.repeat(np.arange(n), nn_idx.shape[1])
    cols = nn_idx.ravel()

    dx = P[cols] - P[rows]
    d2_x = np.einsum("ij,ij->i", dx, dx)

    dn = N_pts[cols] - N_pts[rows]
    d2_n = np.einsum("ij,ij->i", dn, dn)

    # Auto-bandwidths from median k-NN distances
    if sigma_x is None:
        pos = d2_x[d2_x > 0]
        sigma_x = float(np.median(np.sqrt(pos))) if pos.size > 0 else 1.0
    if sigma_n is None:
        pos = d2_n[d2_n > 0]
        sigma_n = float(np.median(np.sqrt(pos))) if pos.size > 0 else 1.0

    w = np.exp(-d2_x / (sigma_x ** 2)) * np.exp(-d2_n / (sigma_n ** 2))

    W = sparse.csr_matrix((w, (rows, cols)), shape=(n, n))
    if symmetrize:
        W = W + W.T
        W.data *= 0.5

    L, W, D = _laplacian_from_weight(W, normalized=normalized, eps=eps)
    if return_parts:
        return L, W, D
    return L


def _coerce_blown_up(
    points_or_sample: np.ndarray | BlowUpLevel,
    subspace_basis: np.ndarray | None,
) -> BlowUpLevel:
    if isinstance(points_or_sample, BlowUpLevel):
        if subspace_basis is not None:
            raise ValueError("subspace_basis must be None when passing BlowUpLevel.")
        return points_or_sample
    if subspace_basis is None:
        raise ValueError("subspace_basis is required when passing raw points.")
    return BlowUpLevel.from_point_tangents(points_or_sample, subspace_basis)


def _edges_from_distance_matrix(
    dist: np.ndarray,
    *,
    k: int | None,
    radius: float | None,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = dist.shape[0]
    if n == 0:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
        )

    dist2 = dist * dist
    if not include_self:
        np.fill_diagonal(dist2, np.inf)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    dist2_list: list[np.ndarray] = []

    if radius is not None:
        r2 = float(radius) ** 2
        for i in range(n):
            js = np.flatnonzero(dist2[i] <= r2)
            if js.size == 0:
                continue
            rows_list.append(np.full(js.size, i, dtype=int))
            cols_list.append(js.astype(int, copy=False))
            dist2_list.append(dist2[i, js].astype(float, copy=False))
    else:
        if k is None or k <= 0:
            return (
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )
        k_eff = min(k, n - (0 if include_self else 1))
        if k_eff <= 0:
            return (
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=float),
            )
        for i in range(n):
            row = dist2[i]
            idx = np.argpartition(row, k_eff - 1)[:k_eff]
            idx = idx[np.isfinite(row[idx])]
            if idx.size == 0:
                continue
            rows_list.append(np.full(idx.size, i, dtype=int))
            cols_list.append(idx.astype(int, copy=False))
            dist2_list.append(row[idx].astype(float, copy=False))

    if not rows_list:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
        )

    return (
        np.concatenate(rows_list),
        np.concatenate(cols_list),
        np.concatenate(dist2_list),
    )


def lifted_pointcloud_laplacian(
    points_or_sample: np.ndarray | BlowUpLevel,
    subspace_basis: np.ndarray | None = None,
    *,
    k: int | None = 20,
    radius: float | None = None,
    h: float | Literal["local"] | None = "local",
    alpha: float = 1.0,
    subspace_metric: Literal["chordal", "geodesic"] = "chordal",
    symmetrize: bool = True,
    include_self: bool = False,
    normalized: bool = False,
    return_parts: bool = False,
    workers: int = -1,
    eps: float = 1e-12,
) -> LaplacianReturn:
    """
    Build a Laplacian on lifted points in the product space R^n x G(k, n).

    The product metric is:
        d^2 = ||x_i - x_j||^2 + alpha * d_subspace(U_i, U_j)^2

    Args:
        points_or_sample: (N, n) points or a BlowUpLevel.
        subspace_basis: (N, n, k) basis data if points_or_sample is raw points.
        k: number of nearest neighbors (ignored if radius is provided).
        radius: neighborhood radius for radius graph (overrides k).
        h: Gaussian kernel bandwidth. "local" uses distance to k-th neighbor.
           If None, uses global median neighbor distance.
        alpha: weight on the subspace term.
        subspace_metric: "chordal" (fast embedding) or "geodesic" (full matrix).
        symmetrize: ensure symmetric weights (recommended for kNN).
        include_self: include self edges with weight exp(0)=1.
        normalized: if True, returns symmetric normalized Laplacian.
        return_parts: if True, returns (L, W, D).
        workers: cKDTree parallelism (workers=-1 uses all cores).
        eps: numerical tolerance for normalized Laplacian.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    sample = _coerce_blown_up(points_or_sample, subspace_basis)
    n = sample.N
    if n == 0:
        empty = sparse.csr_matrix((0, 0), dtype=float)
        if return_parts:
            return empty, empty, empty
        return empty

    if subspace_metric == "chordal":
        embed = sample.embedding_vector(alpha=alpha)
        if radius is not None:
            rows, cols, dist2 = radius_edges(
                embed,
                radius,
                include_self=include_self,
                workers=workers,
            )
        else:
            if k is None:
                raise ValueError("Either k or radius must be provided.")
            rows, cols, dist2 = knn_edges(
                embed,
                k,
                include_self=include_self,
                workers=workers,
            )
    else:
        dist = sample.distance_matrix(
            alpha=alpha,
            subspace_metric=subspace_metric,
        )
        rows, cols, dist2 = _edges_from_distance_matrix(
            dist,
            k=k,
            radius=radius,
            include_self=include_self,
        )

    # Use the updated bandwidth picker
    h_use = _pick_bandwidth(n, rows, dist2, h)
    
    # Pass h_use directly to the updated weight builder
    W = _build_weight_matrix(
        n, rows, cols, dist2, h=h_use, symmetrize=symmetrize
    )
    L, W, D = _laplacian_from_weight(W, normalized=normalized, eps=eps)

    if return_parts:
        return L, W, D
    return L


def laplacian_spectrum(
    L: sparse.spmatrix | np.ndarray,
    *,
    k: int = 6,
    which: Literal["SM", "LM"] = "SM",
    sigma: float = -1e-5,
    drop_first: bool = False,
    return_eigenvalues: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Compute eigenpairs of a Laplacian matrix.

    For large sparse matrices, finding the smallest magnitude ('SM') eigenvalues 
    directly is highly unstable. This function automatically uses shift-invert 
    mode (via `sigma`) to robustly and efficiently find the bottom of the spectrum.

    Args:
        L: Laplacian matrix (sparse or dense).
        k: number of eigenpairs to compute.
        which: "SM" (smallest magnitude) or "LM" (largest magnitude).
        sigma: shift applied for shift-invert mode when computing "SM" (default: -1e-5).
        drop_first: drop the first eigenpair (often the trivial constant eigenvector).
        return_eigenvalues: if False, return only eigenvectors.

    Returns:
        (evals, evecs) or evecs only. Eigenvectors are in columns.
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    n = int(L.shape[0])
    if n == 0:
        empty_vals = np.zeros((0,), dtype=float)
        empty_vecs = np.zeros((0, 0), dtype=float)
        return (empty_vals, empty_vecs) if return_eigenvalues else empty_vecs

    if L.shape[0] != L.shape[1]:
        raise ValueError("L must be square.")

    k_eff = min(k + (1 if drop_first else 0), n)
    use_sparse = sparse.issparse(L)

    # For small matrices, dense eigh is faster and avoids ARPACK convergence issues
    # (e.g. nearly singular Laplacians from tight product kernels).
    DENSE_THRESHOLD = 2000

    if use_sparse:
        Ls = L.tocsr()
        # ARPACK cannot compute k >= N - 1 eigenvalues.
        # Fallback to dense solver if requesting the full spectrum of a small graph.
        if n <= DENSE_THRESHOLD or k_eff >= n - 1:
            Ld = Ls.toarray()
            evals, evecs = np.linalg.eigh(Ld)
        else:
            if which == "SM":
                # SHIFT-INVERT MODE: Finds eigenvalues closest to `sigma`.
                # We ask for "LM" of the shifted operator, which yields the "SM" of the original.
                # Use a larger Krylov subspace (ncv) than the default min(max(2k+1,20), n).
                # Densely-packed spectra (e.g. product kernels with near-zero weights) need
                # more Lanczos vectors to resolve individual eigenpairs.
                ncv = min(max(4 * k_eff + 1, 40), n - 1)
                try:
                    evals, evecs = spla.eigsh(
                        Ls, k=k_eff, sigma=sigma, which="LM", ncv=ncv
                    )
                except ArpackNoConvergence as exc:
                    if exc.eigenvalues.size >= k_eff:
                        # Enough eigenpairs converged despite the exception.
                        evals, evecs = exc.eigenvalues, exc.eigenvectors
                    else:
                        # Genuine failure: fall back to dense eigh.
                        warnings.warn(
                            f"ARPACK did not converge ({exc.eigenvalues.size}/{k_eff} "
                            "eigenpairs). Falling back to dense eigh — may be slow for "
                            "large N. Consider reducing N (--max-points) or adjusting "
                            "kernel parameters.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                        evals, evecs = np.linalg.eigh(Ls.toarray())
            else:
                # Standard mode for largest magnitude eigenvalues
                evals, evecs = spla.eigsh(Ls, k=k_eff, which="LM")
    else:
        Ld = np.asarray(L, dtype=float)
        evals, evecs = np.linalg.eigh(Ld)

    # Sort the results consistently
    order = np.argsort(evals)
    if which == "LM":
        order = order[::-1]  # Sort descending if we wanted largest first
        
    evals = evals[order]
    evecs = evecs[:, order]

    # Handle dropping the trivial component (the 0 eigenvalue)
    if drop_first and evals.size > 0:
        evals = evals[1:]
        evecs = evecs[:, 1:]

    # Truncate to exactly k if we fetched extra
    evals = evals[:k]
    evecs = evecs[:, :k]

    if return_eigenvalues:
        return evals, evecs
    return evecs

__all__ = [
    "pointcloud_laplacian",
    "bilateral_pointcloud_laplacian",
    "lifted_pointcloud_laplacian",
    "laplacian_spectrum",
]
