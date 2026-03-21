"""
Lifted Kernels and Affinity Matrices
--------------------------------------
Kernels on the product space R^n x Gr(d, n) arising from the tangent blow-up
construction.  Three families are provided, all operating on BlowUpLevel data:

1. **Lifted Gaussian** (single bandwidth):
       K_sigma((p, U), (q, V)) = exp(-d_M^2 / sigma^2)
   where d_M^2 = ||Phi_i - Phi_j||^2 is the squared Chordal-Sasaki distance
   already encoded in BlowUpLevel.embedded.

2. **Product kernel** (multi-scale, independent spatial and angular bandwidths):
       K = exp(-||x_i - x_j||^2 / sigma_x^2)
         * prod_{m=0}^{ell} exp(-||P_i^(m) - P_j^(m)||_F^2 / sigma_m^2)
   where x_i are the original spatial positions and P_i^(m) is the
   tangent projector at level m.  At level 0 this reduces to a two-factor
   kernel.  At higher levels, one angular factor per blow-up level is
   included, each with an independent bandwidth sigma_m.

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
    k: int = 30,
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
    k: int = 30,
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
    sigma_u: float | list[float] | np.ndarray,
    *,
    k: int = 30,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """
    Multi-scale product-kernel affinity with independent spatial and angular
    bandwidths:

        W_ij = exp(-||x_i - x_j||^2 / sigma_x^2)
             * prod_{m=0}^{ell} exp(-||P_i^(m) - P_j^(m)||_F^2 / sigma_m^2)

    where:
      - ``x_i = level.embedded[i, :level.n_orig]``  -- original spatial position
      - ``P_i^(m)`` is the tangent projector at level *m* (levels 0 through
        ell-1 are extracted from the embedding, level ell from
        ``level.projectors``)
      - ``sigma_m`` is the bandwidth for the *m*-th angular factor

    At level 0 this reduces to the two-factor kernel (spatial x angular).
    At higher levels it includes one angular factor per blow-up level,
    matching the multi-scale product kernel from the paper (Section 5.1).

    The k-NN graph is built in the full ``level.embedded`` (Chordal-Sasaki
    metric) so that cross-component neighbours are naturally suppressed near
    tangential intersection points.

    The underlying kernel is positive definite (Schur product theorem).
    The sparse k-NN approximation is not generally PSD; L = D - W is PSD.

    Args:
        level:      BlowUpLevel at any level.
        sigma_x:    Bandwidth for the original spatial positions (> 0).
        sigma_u:    Angular bandwidth(s).  A single float applies the same
                    bandwidth to all projector levels.  A list/array of length
                    ``level.level + 1`` assigns an independent bandwidth per
                    level.
        k:          k-NN neighbourhood size.
        symmetrize: Make W symmetric.

    Returns:
        Sparse symmetric (N, N) affinity matrix.
    """
    if sigma_x <= 0.0:
        raise ValueError(f"sigma_x must be positive, got {sigma_x}.")

    n_angular = level.level + 1   # number of angular factors (levels 0..ell)
    if isinstance(sigma_u, (int, float)):
        sigma_u_val = float(sigma_u)
        if sigma_u_val <= 0.0:
            raise ValueError(f"sigma_u must be positive, got {sigma_u_val}.")
        sigmas = [sigma_u_val] * n_angular
    else:
        sigmas = [float(s) for s in sigma_u]
        if len(sigmas) != n_angular:
            raise ValueError(
                f"sigma_u sequence has length {len(sigmas)}, expected "
                f"{n_angular} (one per projector level 0..{level.level})."
            )
        if any(s <= 0.0 for s in sigmas):
            raise ValueError("All sigma_u entries must be positive.")

    N = level.N
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    # k-NN in the Chordal-Sasaki embedded space
    rows, cols, dist2_embed = _knn_edges(level.embedded, k)

    # --- Spatial factor: ||x_i - x_j||^2 in the ORIGINAL ambient space ---
    positions = level.embedded[:, :level.n_orig]
    dx = positions[rows] - positions[cols]
    dist2_spatial = np.einsum("ij,ij->i", dx, dx)
    weights = np.exp(-dist2_spatial / (sigma_x * sigma_x))

    # --- Angular factors for levels 0..ell-1 (from embedded blocks) ---
    for m, (start, ncols, scale) in enumerate(level._proj_blocks):
        block_diff = (level.embedded[rows, start:start + ncols]
                      - level.embedded[cols, start:start + ncols])
        # block_diff = scale * (vec(P_i^(m)) - vec(P_j^(m)))
        # so ||block_diff||^2 = scale^2 * ||P^(m)_i - P^(m)_j||_F^2
        scaled_dist2 = np.einsum("ij,ij->i", block_diff, block_diff)
        # exp(-||P||_F^2 / sigma_m^2) = exp(-scaled_dist2 / (scale^2 * sigma_m^2))
        denom = scale * scale * sigmas[m] * sigmas[m]
        weights *= np.exp(-scaled_dist2 / denom)

    # --- Angular factor for current level ell (from live projectors) ---
    # Use chordal identity: ||P_i - P_j||_F^2 = 2d - 2||U_i^T U_j||_F^2
    U = level.frame                                    # (N, D, d)
    M = np.einsum("eka,ekb->eab", U[rows], U[cols])   # (E, d, d): U_i^T U_j
    inner_sq = np.einsum("eab,eab->e", M, M)          # ||U_i^T U_j||_F^2
    proj_dist2 = 2.0 * level.d - 2.0 * inner_sq
    weights *= np.exp(-proj_dist2 / (sigmas[-1] * sigmas[-1]))

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
    k: int = 30,
    h: float | Literal["local"] | None = "local",
    sigma: float | None = None,
    sigma_x: float | None = None,
    sigma_u: float | list[float] | None = None,
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
        sigma_u:    Angular bandwidth(s) for "product" kernel.  A single
                    float applies the same bandwidth to all projector levels;
                    a list assigns independent bandwidths per level.
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


# ---------------------------------------------------------------------------
# Discrete gradient and divergence
# ---------------------------------------------------------------------------

def lifted_gradient(
    level: BlowUpLevel,
    f: np.ndarray,
    W: sparse.csr_matrix,
    *,
    lam: float = 1e-3,
) -> np.ndarray:
    """
    Discrete gradient of a scalar field on a BlowUpLevel.

    Given a function *f* sampled at the lifted points, computes the
    tangent-plane gradient at each vertex via weighted least-squares
    regression of neighbour value differences against projected
    displacements.

    Algorithm at each vertex *i*:

    1. Project neighbour displacements into the tangent plane:
       ``delta_ij = U_i^T (Phi_j - Phi_i)``  in ``R^d``.
    2. Solve the weighted normal equations:
       ``g_i = S_i^{-1} r_i`` where
       ``S_i = sum_j w_ij delta_ij delta_ij^T`` and
       ``r_i = sum_j w_ij delta_ij (f_j - f_i)``.
    3. Lift back to the ambient space:
       ``(grad f)_i = U_i g_i``  in ``R^D``.

    The matrix ``S_i`` is the kernel-weighted covariance of the projected
    displacements; as N -> inf it converges to a scalar multiple of the
    metric tensor, so ``S_i^{-1}`` performs the discrete analogue of
    raising the index with ``G^{-1}``.

    Args:
        level:  BlowUpLevel at any level.
        f:      Scalar field, shape ``(N,)``.
        W:      Sparse (N, N) symmetric affinity matrix (e.g. from
                :func:`lifted_affinity`).  Only the sparsity pattern and
                values are used as regression weights.
        lam:    Tikhonov regularisation added to ``S_i`` as
                ``lam * tr(S_i)/d * I``.

    Returns:
        ``(N, D)`` array of ambient-space gradient vectors lying in
        ``col(U_i)`` at each point.
    """
    N, D, d = level.N, level.D, level.d
    f = np.asarray(f, dtype=float).ravel()
    if f.shape[0] != N:
        raise ValueError(f"f has length {f.shape[0]}, expected {N}.")

    Phi = level.embedded    # (N, D)
    U   = level.frame       # (N, D, d)
    grad_ambient = np.zeros((N, D), dtype=float)

    W_csr = W.tocsr()
    for i in range(N):
        js = W_csr[i].indices
        ws = np.asarray(W_csr[i].data, dtype=float)
        if js.size == 0:
            continue

        # Projected displacements in tangent plane
        d_phi = Phi[js] - Phi[i]           # (k, D)
        delta = d_phi @ U[i]               # (k, d)

        # Value differences
        df = f[js] - f[i]                  # (k,)

        # Weighted normal equations: S g = r
        w_delta = ws[:, np.newaxis] * delta  # (k, d)
        S = w_delta.T @ delta               # (d, d)
        r = w_delta.T @ df                  # (d,)

        # Relative ridge regularisation
        tr_S = np.trace(S)
        ridge = lam * (tr_S / d) if tr_S > 0 else lam
        S += ridge * np.eye(d)

        try:
            g_tan = np.linalg.solve(S, r)  # (d,)
        except np.linalg.LinAlgError:
            continue

        grad_ambient[i] = U[i] @ g_tan     # (D,)

    return grad_ambient


def lifted_divergence(
    level: BlowUpLevel,
    X: np.ndarray,
    W: sparse.csr_matrix,
    *,
    lam: float = 1e-3,
) -> np.ndarray:
    """
    Discrete divergence of a tangent vector field on a BlowUpLevel.

    Given a vector field ``X_i in col(U_i)`` (i.e. tangent to the lifted
    manifold at each point), estimate the tangent-plane Jacobian by
    regressing neighbour differences of the tangent-plane coordinates
    against projected displacements, and take the trace.

    Algorithm at each vertex *i*:

    1. Express ``X_i`` in tangent-plane coordinates:
       ``X_bar_i = U_i^T X_i``  in ``R^d``.
    2. For each tangent component *a*, regress
       ``X_bar_j^a - X_bar_i^a`` against ``delta_ij``:
       ``J_i^a = S_i^{-1} R_i^a``
       where ``R_i^a = sum_j w_ij delta_ij (X_bar_j^a - X_bar_i^a)``.
    3. Take the trace:
       ``(div X)_i = sum_a J_i^{a,a} = tr(S_i^{-1} R_i)``.

    Args:
        level:  BlowUpLevel at any level.
        X:      Vector field, shape ``(N, D)``.  Each ``X[i]`` should lie
                in ``col(U_i)``.
        W:      Sparse (N, N) symmetric affinity matrix.
        lam:    Tikhonov regularisation (relative scaling).

    Returns:
        ``(N,)`` array of divergence values.
    """
    N, D, d = level.N, level.D, level.d
    X = np.asarray(X, dtype=float)
    if X.shape != (N, D):
        raise ValueError(f"X has shape {X.shape}, expected ({N}, {D}).")

    Phi = level.embedded    # (N, D)
    U   = level.frame       # (N, D, d)

    div = np.zeros(N, dtype=float)

    W_csr = W.tocsr()
    for i in range(N):
        js = W_csr[i].indices
        ws = np.asarray(W_csr[i].data, dtype=float)
        if js.size == 0:
            continue

        Ui = U[i]                              # (D, d)

        # Projected displacements in tangent plane of point i
        d_phi = Phi[js] - Phi[i]               # (k, D)
        delta = d_phi @ Ui                     # (k, d)

        # Weighted covariance
        w_delta = ws[:, np.newaxis] * delta    # (k, d)
        S = w_delta.T @ delta                  # (d, d)

        # Relative ridge regularisation
        tr_S = np.trace(S)
        ridge = lam * (tr_S / d) if tr_S > 0 else lam
        S += ridge * np.eye(d)

        # Project ALL vectors into point i's tangent frame so that
        # the difference is computed in a single coordinate system.
        Xi_bar = Ui.T @ X[i]                   # (d,)
        Xj_bar = X[js] @ Ui                    # (k, d)  = U_i^T X_j for each j
        dX = Xj_bar - Xi_bar                   # (k, d)

        # R[b, a] = sum_j w_j delta_j^b dX_j^a
        R = w_delta.T @ dX                    # (d, d)

        # J = S^{-1} R;  div = tr(J)
        try:
            J = np.linalg.solve(S, R)          # (d, d)
        except np.linalg.LinAlgError:
            continue

        div[i] = np.trace(J)

    return div


__all__ = [
    # Affinity builders
    "lifted_affinity",
    "lifted_gaussian_affinity",
    "product_affinity",
    # Laplacian
    "affinity_to_laplacian",
    "lifted_laplacian",
    # Gradient / Divergence
    "lifted_gradient",
    "lifted_divergence",
]
