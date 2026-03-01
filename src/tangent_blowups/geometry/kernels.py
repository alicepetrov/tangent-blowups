"""
Lifted Kernels and Affinity Matrices
--------------------------------------
Kernels on the product space R^n x Gr(d, n) arising from the tangent blow-up
construction.  Three families are provided, all operating on BlowUpLevel data:

1. **Lifted Gaussian** (single bandwidth):
       K_sigma((p, U), (q, V)) = exp(-d_M^2 / sigma^2)
   where d_M^2 = ||Phi_i - Phi_j||^2 is the squared Chordal-Sasaki distance
   already encoded in BlowUpLevel.embedded.

2. **Product kernel** (independent spatial and angular bandwidths):
       K((p, U), (q, V)) = exp(-||x_i - x_j||^2 / sigma_x^2)
                         * exp(-||P_i - P_j||_F^2 / sigma_u^2)
   where x_i = level.embedded and P_i = level.projectors are the CURRENT
   level's embedded positions and tangent projectors respectively.

3. **Self-tuning affinity** (Zelnik-Manor & Perona 2004):
       W_ij = exp(-d_M^2 / (h_i * h_j))
   with h_i = distance from Phi_i to its k-th nearest neighbour in the
   Chordal-Sasaki metric.  This adapts the bandwidth pointwise to the local
   density of the lifted point cloud.

Positive definiteness notes:
- The fixed-bandwidth Gaussian (kernel 1) is positive definite: it is a
  Gaussian RBF on the isometric Euclidean embedding, so PD follows from
  Bochner's theorem.
- The product kernel (kernel 2) is positive definite by the Schur product
  theorem: the pointwise product of two PD kernels is PD.
- The self-tuning kernel (kernel 3) with pointwise-varying h_i h_j is NOT
  guaranteed positive semidefinite.  The adaptive bandwidth breaks the
  stationarity required by Bochner's theorem.
- In all cases, only the full (dense) Gram matrix inherits these PD
  guarantees.  The sparse k-NN approximation is not generally PSD; the
  graph Laplacian L = D - W is guaranteed PSD via D_ii = sum_j W_ij >= 0.

References
----------
Zelnik-Manor & Perona, "Self-Tuning Spectral Clustering", NeurIPS 2004.
Björck & Golub, "Numerical Methods for Computing Angles Between Subspaces",
    Mathematics of Computation, 1973.
Sasaki, "On the Differential Geometry of Tangent Bundles of Riemannian
    Manifolds", Tohoku Mathematical Journal, 1958.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .iterated_grassmann import BlowUpLevel


# ---------------------------------------------------------------------------
# k-NN graph construction
# ---------------------------------------------------------------------------

def _knn_edges(
    embed: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a directed k-NN graph in Euclidean embedding space.

    Returns:
        rows, cols:  Edge indices, each of length N*k.
        dist2:       Squared distances along each edge.
    """
    N = embed.shape[0]
    k_eff = min(k + 1, N)   # +1 because query includes the point itself
    tree = cKDTree(embed)
    dist, idx = tree.query(embed, k=k_eff)
    # Drop self (first column is always distance 0 to self)
    dist = dist[:, 1:]
    idx  = idx[:, 1:]
    rows  = np.repeat(np.arange(N, dtype=int), idx.shape[1])
    cols  = idx.ravel().astype(int)
    dist2 = dist.ravel() ** 2
    return rows, cols, dist2


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def _local_bandwidths(
    N: int,
    rows: np.ndarray,
    dist2: np.ndarray,
) -> np.ndarray:
    """
    Zelnik-Manor & Perona pointwise bandwidth.

    h_i = distance from point i to its k-th nearest neighbour
         = max_{j in kNN(i)} d(i, j)

    For a k-NN graph built by _knn_edges, the maximum distance in each row is
    the distance to the k-th (farthest) neighbour in the neighbourhood.
    """
    h_sq = np.zeros(N, dtype=float)
    np.maximum.at(h_sq, rows, dist2)
    h = np.sqrt(h_sq)
    # Safety: replace zeros (duplicate points) with global median
    zero = h == 0
    if zero.any():
        pos_d = np.sqrt(dist2[dist2 > 0])
        fallback = float(np.median(pos_d)) if pos_d.size > 0 else 1.0
        h[zero] = fallback
    return h


def _resolve_bandwidth(
    N: int,
    rows: np.ndarray,
    dist2: np.ndarray,
    h: float | Literal["local"] | None,
) -> float | np.ndarray:
    """
    Resolve the bandwidth parameter to a scalar or per-point array.

    Args:
        N:      Number of points.
        rows:   Edge source indices from _knn_edges.
        dist2:  Squared distances from _knn_edges.
        h:      "local" -> Zelnik-Manor pointwise,
                None   -> global median neighbour distance,
                float  -> fixed global bandwidth.

    Returns:
        float or (N,) array.
    """
    if isinstance(h, (int, float)):
        h_val = float(h)
        if h_val <= 0.0:
            raise ValueError(f"Bandwidth h must be positive, got {h_val}.")
        return h_val
    if dist2.size == 0:
        return 1.0
    if h == "local":
        return _local_bandwidths(N, rows, dist2)
    # h is None: global median
    d = np.sqrt(dist2)
    d = d[d > 0.0]
    return float(np.median(d)) if d.size > 0 else 1.0


# ---------------------------------------------------------------------------
# Weight matrix helpers
# ---------------------------------------------------------------------------

def _weights_from_bandwidth(
    dist2: np.ndarray,
    h: float | np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Gaussian kernel weights given (possibly pointwise) bandwidth."""
    if isinstance(h, np.ndarray):
        return np.exp(-dist2 / (h[rows] * h[cols]))
    return np.exp(-dist2 / (h * h))


def _assemble_affinity(
    N: int,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    *,
    symmetrize: bool,
) -> sparse.csr_matrix:
    if N == 0 or rows.size == 0:
        return sparse.csr_matrix((N, N), dtype=float)
    W = sparse.coo_matrix((weights, (rows, cols)), shape=(N, N)).tocsr()
    W.sum_duplicates()
    if symmetrize:
        W = W.maximum(W.T)
    return W


# ---------------------------------------------------------------------------
# Public API: affinity builders
# ---------------------------------------------------------------------------

def lifted_affinity(
    level: BlowUpLevel,
    *,
    k: int = 16,
    h: float | Literal["local"] | None = "local",
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """
    Self-tuning affinity matrix in the Chordal-Sasaki metric.

    Builds a k-NN graph using the Euclidean metric on ``level.embedded``
    (which isometrically encodes the Chordal-Sasaki product metric at that
    blow-up level), then assigns edge weights:

        W_ij = exp(-||Phi_i - Phi_j||^2 / (h_i * h_j))

    where h_i is the distance to the k-th nearest neighbour of point i
    (Zelnik-Manor & Perona 2004 self-tuning).

    For ``h="local"`` the bandwidth adapts pointwise to the local density;
    for a fixed float it applies a global scale; for ``h=None`` the global
    median neighbour distance is used.

    Note: the self-tuning ``h="local"`` variant with pointwise h_i h_j is
    NOT guaranteed positive semidefinite.  A fixed scalar ``h`` recovers a
    standard Gaussian RBF which is PD (Bochner), but the sparse k-NN
    approximation returned here is not generally PSD either way; the
    resulting graph Laplacian L = D - W is PSD by construction.

    Args:
        level:      BlowUpLevel at any level (0, 1, 2, ...).
        k:          k-NN neighbourhood size (also determines self-tuning h_i).
        h:          Bandwidth strategy.  See :func:`_resolve_bandwidth`.
        symmetrize: Make W symmetric by taking elementwise maximum.

    Returns:
        Sparse symmetric (N, N) affinity matrix.
    """
    N = level.N
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    rows, cols, dist2 = _knn_edges(level.embedded, k)
    h_use = _resolve_bandwidth(N, rows, dist2, h)
    weights = _weights_from_bandwidth(dist2, h_use, rows, cols)
    return _assemble_affinity(N, rows, cols, weights, symmetrize=symmetrize)


def lifted_gaussian_affinity(
    level: BlowUpLevel,
    sigma: float,
    *,
    k: int = 16,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """
    Fixed-bandwidth lifted Gaussian affinity in the Chordal-Sasaki metric:

        W_ij = exp(-||Phi_i - Phi_j||^2 / sigma^2)

    This is the single-bandwidth kernel from Section "Lifted Kernels, Part 4"
    of the theoretical notes.  The underlying kernel is positive definite
    (Gaussian RBF on the isometric Euclidean embedding, by Bochner's theorem),
    but the sparse k-NN approximation returned here is not generally PSD.
    The graph Laplacian L = D - W built from this matrix is PSD by construction.

    Args:
        level:      BlowUpLevel.
        sigma:      Bandwidth (> 0).
        k:          k-NN neighbourhood size.
        symmetrize: Make W symmetric.

    Returns:
        Sparse symmetric (N, N) affinity matrix.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    N = level.N
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    rows, cols, dist2 = _knn_edges(level.embedded, k)
    weights = np.exp(-dist2 / (sigma * sigma))
    return _assemble_affinity(N, rows, cols, weights, symmetrize=symmetrize)


def product_affinity(
    level: BlowUpLevel,
    sigma_x: float,
    sigma_u: float,
    *,
    k: int = 16,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """
    Product-kernel affinity with independent spatial and angular bandwidths:

        W_ij = exp(-||x_i - x_j||^2      / sigma_x^2)
             * exp(-||P_i - P_j||_F^2    / sigma_u^2)

    where:
      - ``x_i = level.embedded[i, :level.n_orig]``  -- original spatial position
        (the first n_orig coordinates of the Chordal-Sasaki embedding, which are
        always the unmodified spatial coordinates)
      - ``P_i = level.projectors[i]``   -- tangent projector at the current
        level (encoding the NEXT order of geometry: curvature at level 0,
        curvature-of-curvature at level 1, etc.)

    The k-NN graph is built in the full ``level.embedded`` (Chordal-Sasaki
    metric) so that cross-component neighbours are naturally suppressed near
    tangential intersection points, but the spatial factor uses only the
    original position coordinates to keep the two bandwidths truly independent.

    The underlying kernel is positive definite: each factor is a Gaussian RBF
    on a Euclidean space (Bochner's theorem), and the Schur product theorem
    gives PD for their pointwise product.  As with all kernels here, the
    sparse k-NN approximation is not generally PSD; L = D - W is PSD.

    Args:
        level:      BlowUpLevel at any level.  Typical usage:

                    - ``level0`` -> sigma_x controls spatial scale,
                      sigma_u controls tangent-plane scale.
                    - ``level1`` -> sigma_x controls Chordal-Sasaki scale,
                      sigma_u controls curvature-direction scale.

        sigma_x:    Bandwidth for the original spatial positions (> 0).
        sigma_u:    Bandwidth for the tangent projector (Frobenius) part (> 0).
        k:          k-NN neighbourhood size.
        symmetrize: Make W symmetric.

    Returns:
        Sparse symmetric (N, N) affinity matrix.
    """
    if sigma_x <= 0.0:
        raise ValueError(f"sigma_x must be positive, got {sigma_x}.")
    if sigma_u <= 0.0:
        raise ValueError(f"sigma_u must be positive, got {sigma_u}.")
    N = level.N
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    # k-NN in the Chordal-Sasaki embedded space
    rows, cols, dist2_embed = _knn_edges(level.embedded, k)

    # Spatial factor: ||x_i - x_j||^2 in the ORIGINAL ambient space only.
    # level.embedded = (x, sqrt(alpha/2)*vec(P), ...) so the first n_orig coords
    # are always the original spatial positions.  Using dist2_embed would
    # double-count the projector distance already captured by the angular factor.
    positions = level.embedded[:, :level.n_orig]
    dx = positions[rows] - positions[cols]
    dist2_spatial = np.einsum("ij,ij->i", dx, dx)
    kx = np.exp(-dist2_spatial / (sigma_x * sigma_x))

    # Angular factor: ||P_i - P_j||_F^2 via the chordal identity
    #
    #   ||P_i - P_j||_F^2 = 2d - 2 ||U_i^T U_j||_F^2
    #
    # where U_i = level.frame[i] has shape (D, d).  This avoids materialising
    # the (N, D, D) projector matrices and the (E, D, D) edge-difference tensors.
    # Cost:  O(E * D * d)  vs  O(E * D^2)  — for D=12, d=2: 6x cheaper;
    # memory: O(E * d^2)  vs  O(E * D^2)  — 36x smaller intermediates.
    U = level.frame                                    # (N, D, d)
    M = np.einsum("eka,ekb->eab", U[rows], U[cols])   # (E, d, d): U_i^T U_j
    inner_sq = np.einsum("eab,eab->e", M, M)          # ||U_i^T U_j||_F^2
    proj_dist2 = 2.0 * level.d - 2.0 * inner_sq
    ku = np.exp(-proj_dist2 / (sigma_u * sigma_u))

    weights = kx * ku
    return _assemble_affinity(N, rows, cols, weights, symmetrize=symmetrize)


# ---------------------------------------------------------------------------
# Public API: Laplacian construction
# ---------------------------------------------------------------------------

def affinity_to_laplacian(
    W: sparse.csr_matrix,
    *,
    normalized: bool = False,
    eps: float = 1e-12,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build a graph Laplacian from an affinity matrix.

    Unnormalized (default):
        L = D - W,   D_{ii} = sum_j W_{ij}

    Symmetric normalized:
        L_sym = I - D^{-1/2} W D^{-1/2}

    Both are positive semidefinite and have a zero eigenvalue for each
    connected component of the graph.

    Args:
        W:          Sparse (N, N) affinity matrix (symmetric, non-negative).
        normalized: If True, return the symmetric normalized Laplacian.
        eps:        Threshold below which degree is treated as zero.

    Returns:
        (L, W, D) — Laplacian, affinity, diagonal degree matrix.
    """
    n = W.shape[0]
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sparse.diags(deg, format="csr")

    if not normalized:
        return D - W, W, D

    inv_sqrt = np.zeros_like(deg)
    mask = deg > eps
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_isqrt = sparse.diags(inv_sqrt, format="csr")
    L = sparse.eye(n, format="csr") - D_isqrt @ W @ D_isqrt
    return L, W, D


# ---------------------------------------------------------------------------
# Convenience: affinity → Laplacian in one call
# ---------------------------------------------------------------------------

def lifted_laplacian(
    level: BlowUpLevel,
    *,
    kernel: Literal["self_tuning", "gaussian", "product"] = "self_tuning",
    k: int = 16,
    h: float | Literal["local"] | None = "local",
    sigma: float | None = None,
    sigma_x: float | None = None,
    sigma_u: float | None = None,
    normalized: bool = False,
    symmetrize: bool = True,
    eps: float = 1e-12,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build a graph Laplacian on a BlowUpLevel in one call.

    Thin wrapper that selects the appropriate affinity builder and passes
    the result to :func:`affinity_to_laplacian`.

    Args:
        level:      BlowUpLevel (any level).
        kernel:     "self_tuning" (default), "gaussian", or "product".
        k:          k-NN neighbourhood size.
        h:          Bandwidth for "self_tuning" kernel.
        sigma:      Bandwidth for "gaussian" kernel.
        sigma_x:    Spatial bandwidth for "product" kernel.
        sigma_u:    Angular bandwidth for "product" kernel.
        normalized: Symmetric normalized Laplacian if True.
        symmetrize: Symmetrize the affinity matrix.
        eps:        Degree threshold for normalized Laplacian.

    Returns:
        (L, W, D) — Laplacian, affinity matrix, degree matrix.
    """
    if kernel == "self_tuning":
        W = lifted_affinity(level, k=k, h=h, symmetrize=symmetrize)
    elif kernel == "gaussian":
        if sigma is None:
            raise ValueError("sigma is required for the 'gaussian' kernel.")
        W = lifted_gaussian_affinity(level, sigma, k=k, symmetrize=symmetrize)
    elif kernel == "product":
        if sigma_x is None or sigma_u is None:
            raise ValueError(
                "sigma_x and sigma_u are both required for the 'product' kernel."
            )
        W = product_affinity(level, sigma_x, sigma_u, k=k, symmetrize=symmetrize)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'self_tuning', 'gaussian', or 'product'.")

    return affinity_to_laplacian(W, normalized=normalized, eps=eps)


__all__ = [
    # Affinity builders
    "lifted_affinity",
    "lifted_gaussian_affinity",
    "product_affinity",
    # Laplacian
    "affinity_to_laplacian",
    "lifted_laplacian",
]
