"""
Lifted Kernels and Affinity Matrices
--------------------------------------
Kernels on the product space R^n x Gr(d, n) arising from the tangent blow-up
construction.  Three families are provided, all operating on BlowUpLevel data:

1. **Product kernel** (multi-scale, independent spatial and angular bandwidths):
       K = exp(-||x_i - x_j||^2 / sigma_x^2)
         * prod_{m=0}^{ell} exp(-||P_i^(m) - P_j^(m)||_F^2 / sigma_m^2)
   where x_i are the original spatial positions and P_i^(m) is the
   tangent projector at level m.  At level 0 this reduces to a two-factor
   kernel.  At higher levels, one angular factor per blow-up level is
   included, each with an independent bandwidth sigma_m.

2. **Uniform** (binary adjacency):
       W_ij = 1  for all k-NN edges.
   Gives the standard combinatorial graph Laplacian L = D - W.

3. **Self-tuning affinity** (Zelnik-Manor & Perona 2004):
       W_ij = exp(-d_M^2 / (h_i * h_j))
   with h_i = distance from Phi_i to its k-th nearest neighbour in the
   Chordal-Sasaki metric.  This adapts the bandwidth pointwise to the local
   density of the lifted point cloud.

Positive definiteness notes:
- The product kernel (kernel 1) is positive definite by the Schur product
  theorem: the pointwise product of two PD kernels is PD.
- The self-tuning kernel (kernel 2) with pointwise-varying h_i h_j is NOT
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
from .weight_config import WeightConfig, product_weights


# ---------------------------------------------------------------------------
# k-NN graph construction
# ---------------------------------------------------------------------------

def _knn_edges(
    embed: np.ndarray,
    k: int,
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a directed k-NN graph in Euclidean embedding space.

    Args:
        embed:  (N, D) point embeddings.
        k:      Number of nearest neighbours.
        device: ``"cuda"`` uses GPU brute-force; ``"cpu"`` uses cKDTree.

    Returns:
        rows, cols:  Edge indices, each of length N*k.
        dist2:       Squared distances along each edge.
    """
    if device == "cuda":
        try:
            from ._gpu_ops import knn_edges_gpu
            return knn_edges_gpu(embed, k)
        except ImportError:
            pass

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


def estimate_product_bandwidths(
    level: BlowUpLevel,
    k: int,
) -> tuple[float, list[float]]:
    """Estimate sigma_x and per-lift sigma_u from median k-NN distances.

    The product metric at level ell decomposes into ell + 1 factors:
    spatial + one per embedded projector block (one per lift).  This
    function returns one angular bandwidth per factor of the product
    metric:

    - Level 1 (1 lift):  ``[sigma_tangent]``
    - Level 2 (2 lifts): ``[sigma_tangent, sigma_curvature]``

    Each bandwidth is the median k-NN distance in the corresponding
    block, scaled by ``sqrt(n_factors)`` to compensate for the
    multiplicative effect of multiple kernel factors.

    Args:
        level: BlowUpLevel at any level.
        k:     k-NN neighbourhood size.

    Returns:
        ``(sigma_x, sigma_u_list)`` -- spatial bandwidth and a list of
        ``level.level`` angular bandwidths (one per lift).
    """
    rows, cols, _ = _knn_edges(level.embedded, k)

    # Spatial distances
    positions = level.embedded[:, :level.n_orig]
    dx = positions[rows] - positions[cols]
    dist_x = np.sqrt(np.einsum("ij,ij->i", dx, dx))
    dist_x = dist_x[dist_x > 0.0]
    sigma_x = float(np.median(dist_x)) if dist_x.size > 0 else 1.0

    # One angular bandwidth per embedded projector block (one per lift)
    sigmas_u: list[float] = []
    for _m, (start, ncols, scale) in enumerate(level._proj_blocks):
        # alpha=0 lift: the angular factor is skipped in product_affinity,
        # so sigma_u is unused.  Emit a placeholder to keep the list length
        # equal to the number of lifts.
        if scale == 0.0:
            sigmas_u.append(1.0)
            continue
        diff = (level.embedded[rows, start:start + ncols]
                - level.embedded[cols, start:start + ncols])
        raw_dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        true_dist = raw_dist / scale
        true_dist = true_dist[true_dist > 0.0]
        sigmas_u.append(
            float(np.median(true_dist)) if true_dist.size > 0 else 1.0
        )

    return sigma_x, sigmas_u


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
    k: int = 20,
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


def uniform_affinity(
    level: BlowUpLevel,
    *,
    k: int = 20,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """
    Binary k-NN affinity matrix (all edge weights = 1).

    Builds a k-NN graph in the Chordal-Sasaki embedding, then assigns
    unit weight to every edge.  The resulting graph Laplacian L = D - W
    is the standard combinatorial Laplacian.

    Args:
        level:      BlowUpLevel at any level.
        k:          k-NN neighbourhood size.
        symmetrize: Make W symmetric.

    Returns:
        Sparse symmetric (N, N) affinity matrix with entries in {0, 1}.
    """
    N = level.N
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    rows, cols, _ = _knn_edges(level.embedded, k)
    weights = np.ones(len(rows), dtype=float)
    return _assemble_affinity(N, rows, cols, weights, symmetrize=symmetrize)


def product_affinity(
    level: BlowUpLevel,
    sigma_x: float | None = None,
    sigma_u: float | list[float] | np.ndarray | None = None,
    *,
    k: int = 20,
    self_tuning: bool = False,
    symmetrize: bool = True,
    device: str = "cpu",
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
    n_lifts = level.level   # one angular factor per lift (= len(_proj_blocks))
    if self_tuning:
        sigmas: list[float] = []  # unused in self-tuning branch
    else:
        if sigma_x is None or sigma_x <= 0.0:
            raise ValueError(
                f"sigma_x must be positive when self_tuning=False, got {sigma_x}."
            )
        if isinstance(sigma_u, (int, float)):
            sigma_u_val = float(sigma_u)
            if sigma_u_val <= 0.0:
                raise ValueError(f"sigma_u must be positive, got {sigma_u_val}.")
            sigmas = [sigma_u_val] * max(n_lifts, 1)
        elif sigma_u is None:
            raise ValueError("sigma_u is required when self_tuning=False.")
        else:
            sigmas = [float(s) for s in sigma_u]
            if len(sigmas) != n_lifts:
                raise ValueError(
                    f"sigma_u sequence has length {len(sigmas)}, expected "
                    f"{n_lifts} (one per lift)."
                )
            if any(s <= 0.0 for s in sigmas):
                raise ValueError("All sigma_u entries must be positive.")

    N = level.N
    if N == 0:
        return sparse.csr_matrix((0, 0), dtype=float)

    # k-NN in the Chordal-Sasaki embedded space (always CPU -- cKDTree is
    # faster than brute-force GPU for moderate D)
    rows, cols, dist2_embed = _knn_edges(level.embedded, k, device="cpu")

    # --- GPU fast path for weight computation ---
    if device == "cuda":
        try:
            from ._gpu_ops import product_affinity_gpu
            weights = product_affinity_gpu(
                level, sigma_x, sigmas, k, rows, cols,
                self_tuning=self_tuning,
            )
            return _assemble_affinity(
                N, rows, cols, weights, symmetrize=symmetrize,
            )
        except ImportError:
            pass

    # --- Spatial factor: ||x_i - x_j||^2 in the ORIGINAL ambient space ---
    positions = level.embedded[:, :level.n_orig]
    dx = positions[rows] - positions[cols]
    dist2_spatial = np.einsum("ij,ij->i", dx, dx)
    if self_tuning:
        h_x = _local_bandwidths(N, rows, dist2_spatial)
        weights = np.exp(-dist2_spatial / (h_x[rows] * h_x[cols]))
    else:
        weights = np.exp(-dist2_spatial / (sigma_x * sigma_x))

    # --- Angular factors: one per lift (from the product metric) ---
    # Each _proj_block corresponds to one factor of the product space.
    # At level ell the embedding is [x, scale*vec(P^0), ..., scale*vec(P^{ell-1})]
    # with scale = sqrt(alpha/2). The product metric decomposes into ell+1
    # factors (spatial + ell angular).
    _BLOCK_CHUNK = 32
    E = rows.shape[0]
    for m, (start, ncols, scale) in enumerate(level._proj_blocks):
        # alpha=0 lift -> scale=0 -> the projector block is identically zero
        # and the angular factor reduces to exp(0)=1.  Skip to avoid 0/0.
        if scale == 0.0:
            continue
        scaled_dist2 = np.zeros(E, dtype=float)
        for c0 in range(0, ncols, _BLOCK_CHUNK):
            c1 = min(c0 + _BLOCK_CHUNK, ncols)
            diff = (level.embedded[rows, start + c0:start + c1]
                    - level.embedded[cols, start + c0:start + c1])
            scaled_dist2 += np.einsum("ij,ij->i", diff, diff)
        if self_tuning:
            # Use intrinsic ||P_diff||^2 = scaled_dist2 / scale^2 so the
            # Zelnik-Manor bandwidth is in the same units as sigma_u from
            # estimate_product_bandwidths. alpha cancels both numerator and
            # denominator here but still affects the k-NN topology above.
            intrinsic_dist2 = scaled_dist2 / (scale * scale)
            h_u = _local_bandwidths(N, rows, intrinsic_dist2)
            weights *= np.exp(-intrinsic_dist2 / (h_u[rows] * h_u[cols]))
        else:
            # scaled_dist2 = (alpha/2) * ||P_diff||^2 -- alpha visibly sharpens
            # the angular factor. sigma_u is intrinsic (see
            # estimate_product_bandwidths, which divides raw distance by scale),
            # so the denominator is sigma_u^2, not scale^2 * sigma_u^2.
            weights *= np.exp(-scaled_dist2 / (sigmas[m] * sigmas[m]))

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
    kernel: Literal["product", "uniform", "lifted"] = "product",
    k: int = 20,
    self_tuning: bool = True,
    sigma_x: float | None = None,
    sigma_u: float | list[float] | None = None,
    h: float | Literal["local"] | None = "local",
    normalized: bool = True,
    symmetrize: bool = True,
    eps: float = 1e-12,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build a graph Laplacian on a BlowUpLevel in one call.

    Thin wrapper that selects the appropriate affinity builder and passes
    the result to :func:`affinity_to_laplacian`.

    Args:
        level:        BlowUpLevel (any level).
        kernel:       "product" (default), "uniform", or "lifted" (self-tuning
                      Gaussian on the full Chordal-Sasaki embedding).
        k:            k-NN neighbourhood size.
        self_tuning:  For kernel="product", use Zelnik-Manor pointwise bandwidths
                      per factor (default True). When False, `sigma_x`/`sigma_u`
                      are required.
        sigma_x:      Spatial bandwidth for non-self-tuning "product" kernel.
        sigma_u:      Angular bandwidth(s) for non-self-tuning "product" kernel.
        h:            Bandwidth strategy for kernel="lifted".
        normalized:   Symmetric normalized Laplacian if True (default).
        symmetrize:   Symmetrize the affinity matrix.
        eps:          Degree threshold for normalized Laplacian.

    Returns:
        (L, W, D) -- Laplacian, affinity matrix, degree matrix.
    """
    if kernel == "product":
        W = product_affinity(
            level, sigma_x, sigma_u,
            k=k, self_tuning=self_tuning, symmetrize=symmetrize,
        )
    elif kernel == "uniform":
        W = uniform_affinity(level, k=k, symmetrize=symmetrize)
    elif kernel == "lifted":
        W = lifted_affinity(level, k=k, h=h, symmetrize=symmetrize)
    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Choose 'product', 'uniform', or 'lifted'."
        )

    return affinity_to_laplacian(W, normalized=normalized, eps=eps)


# ---------------------------------------------------------------------------
# Fast scatter-add (bincount is C-level; np.add.at is Python-level)
# ---------------------------------------------------------------------------

def _scatter_sum_1d(idx, vals, N):
    """Sum vals into N bins given by idx.  vals: (E,)."""
    return np.bincount(idx, weights=vals, minlength=N)


def _scatter_sum_2d(idx, vals, N):
    """Sum vals into N bins given by idx.  vals: (E, d)."""
    d = vals.shape[1]
    out = np.empty((N, d), dtype=float)
    for a in range(d):
        out[:, a] = np.bincount(idx, weights=vals[:, a], minlength=N)
    return out


def _scatter_sum_3d(idx, vals, N):
    """Sum vals into N bins given by idx.  vals: (E, d, d)."""
    d = vals.shape[1]
    out = np.empty((N, d, d), dtype=float)
    for a in range(d):
        for b in range(d):
            out[:, a, b] = np.bincount(
                idx, weights=vals[:, a, b], minlength=N
            )
    return out


# ---------------------------------------------------------------------------
# Shared edge-level precomputation for vectorised gradient / divergence
# ---------------------------------------------------------------------------

def _precompute_edges(level, W, lam):
    """
    Precompute per-edge projected displacements and per-vertex S_i^{-1}.

    Returns a dict with:
        rows, cols  : (E,) edge source / target indices
        weights     : (E,) affinity weights
        delta       : (E, d) tangent-plane projected displacements
        w_delta     : (E, d) weight * delta
        S_inv       : (N, d, d) regularised inverse covariance per vertex
    """
    N, D, d = level.N, level.D, level.d
    Phi = level.embedded   # (N, D)
    U   = level.frame      # (N, D, d)

    W_coo = W.tocoo()
    rows = W_coo.row.astype(int)
    cols = W_coo.col.astype(int)
    ws   = np.asarray(W_coo.data, dtype=float)

    # Edge displacements projected into source tangent plane
    d_phi = Phi[cols] - Phi[rows]                       # (E, D)
    delta = np.einsum("eD,eDd->ed", d_phi, U[rows])    # (E, d)

    w_delta = ws[:, np.newaxis] * delta                 # (E, d)

    # Scatter-add to build S_i = sum_j w_ij delta_ij delta_ij^T
    S_edges = np.einsum("ea,eb->eab", w_delta, delta)   # (E, d, d)
    S = _scatter_sum_3d(rows, S_edges, N)

    # Relative ridge regularisation with absolute floor.
    # For well-connected vertices, ridge = lam * tr(S) / d (relative).
    # For isolated / near-zero vertices, use a small absolute floor so
    # S is always safely invertible (even when lam=0).
    tr_S = np.trace(S, axis1=1, axis2=2)                # (N,)
    pos_tr = tr_S[tr_S > 0]
    fallback_scale = float(np.median(pos_tr)) if pos_tr.size > 0 else 1.0
    ridge = np.maximum(lam * tr_S / d, lam * fallback_scale / d)
    ridge = np.maximum(ridge, 1e-12 * fallback_scale / d)
    S += ridge[:, np.newaxis, np.newaxis] * np.eye(d)

    # Batch invert (np.linalg.inv is fine for small d x d)
    S_inv = np.linalg.inv(S)                            # (N, d, d)

    return dict(
        rows=rows, cols=cols, weights=ws,
        delta=delta, w_delta=w_delta, S_inv=S_inv,
    )


# ---------------------------------------------------------------------------
# Discrete gradient and divergence (vectorised)
# ---------------------------------------------------------------------------

def lifted_gradient(
    level: BlowUpLevel,
    f: np.ndarray,
    W: sparse.csr_matrix,
    *,
    lam: float = 0.0,
    _edge_cache: dict | None = None,
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
        level:      BlowUpLevel at any level.
        f:          Scalar field, shape ``(N,)``.
        W:          Sparse (N, N) symmetric affinity matrix (e.g. from
                    :func:`lifted_affinity`).  Only the sparsity pattern and
                    values are used as regression weights.
        lam:        Tikhonov regularisation added to ``S_i`` as
                    ``lam * tr(S_i)/d * I``.
        _edge_cache: Pre-computed edge data from :func:`_precompute_edges`.
                    If None, computed internally.  Pass this when calling
                    gradient and divergence repeatedly on the same level/W
                    (e.g. inside a LinearOperator matvec).

    Returns:
        ``(N, D)`` array of ambient-space gradient vectors lying in
        ``col(U_i)`` at each point.
    """
    N, D, d = level.N, level.D, level.d
    f = np.asarray(f, dtype=float).ravel()
    if f.shape[0] != N:
        raise ValueError(f"f has length {f.shape[0]}, expected {N}.")

    ec = _edge_cache or _precompute_edges(level, W, lam)
    rows, cols = ec["rows"], ec["cols"]
    w_delta, S_inv = ec["w_delta"], ec["S_inv"]

    # r_i = sum_j w_ij delta_ij (f_j - f_i)
    df = f[cols] - f[rows]                                  # (E,)
    r_edges = w_delta * df[:, np.newaxis]                   # (E, d)
    r = _scatter_sum_2d(rows, r_edges, N)                   # (N, d)

    # g_i = S_i^{-1} r_i
    g_tan = np.einsum("iab,ib->ia", S_inv, r)              # (N, d)

    # Lift back to ambient: grad_i = U_i g_i
    U = level.frame                                          # (N, D, d)
    grad_ambient = np.einsum("iDd,id->iD", U, g_tan)       # (N, D)
    return grad_ambient


def lifted_divergence(
    level: BlowUpLevel,
    X: np.ndarray,
    W: sparse.csr_matrix,
    *,
    lam: float = 0.0,
    _edge_cache: dict | None = None,
) -> np.ndarray:
    """
    Discrete divergence of a tangent vector field on a BlowUpLevel.

    Given a vector field ``X_i in col(U_i)`` (i.e. tangent to the lifted
    manifold at each point), estimate the tangent-plane Jacobian by
    regressing neighbour differences of the tangent-plane coordinates
    against projected displacements, and take the trace.

    All vectors are projected into point *i*'s tangent frame before
    differencing.  This choice of a single frame per stencil is essential:
    using each point's own frame introduces a connection-like error that
    destroys the approximation.

    Algorithm at each vertex *i*:

    1. Project the vector field at *i* and at each neighbour *j* into
       point *i*'s frame:
       ``X_bar_k^{(i)} = U_i^T X_k``  for ``k in {i} cup N(i)``.
    2. Regress frame-consistent differences against projected displacements:
       ``R_i^a = sum_j w_ij delta_ij (X_bar_j^{(i)} - X_bar_i^{(i)})^a``.
    3. Take the trace:
       ``(div X)_i = tr(S_i^{-1} R_i)``.

    Args:
        level:      BlowUpLevel at any level.
        X:          Vector field, shape ``(N, D)``.  Each ``X[i]`` should lie
                    in ``col(U_i)``.
        W:          Sparse (N, N) symmetric affinity matrix.
        lam:        Tikhonov regularisation (relative scaling).
        _edge_cache: Pre-computed edge data from :func:`_precompute_edges`.

    Returns:
        ``(N,)`` array of divergence values.
    """
    N, D, d = level.N, level.D, level.d
    X = np.asarray(X, dtype=float)
    if X.shape != (N, D):
        raise ValueError(f"X has shape {X.shape}, expected ({N}, {D}).")

    ec = _edge_cache or _precompute_edges(level, W, lam)
    rows, cols = ec["rows"], ec["cols"]
    w_delta, S_inv = ec["w_delta"], ec["S_inv"]
    U = level.frame                                          # (N, D, d)

    # Project ALL vectors into the SOURCE point's frame (point i's frame).
    Xi_bar = np.einsum("iDd,iD->id", U, X)                 # (N, d)
    # U_i^T X_j for each edge (i, j) — use source frame
    Xj_bar_edge = np.einsum("eDd,eD->ed", U[rows], X[cols])  # (E, d)
    dX = Xj_bar_edge - Xi_bar[rows]                          # (E, d)

    # R_i[b, a] = sum_j w_j delta_j^b dX_j^a
    R_edges = np.einsum("eb,ea->eba", w_delta, dX)          # (E, d, d)
    R = _scatter_sum_3d(rows, R_edges, N)                    # (N, d, d)

    # J_i = S_i^{-1} R_i;  div_i = tr(J_i)
    J = np.einsum("iab,ibc->iac", S_inv, R)                 # (N, d, d)
    div = np.trace(J, axis1=1, axis2=2)                      # (N,)
    return div


__all__ = [
    # Weight configuration
    "WeightConfig",
    "product_weights",
    # Bandwidth estimation
    "estimate_product_bandwidths",
    # Affinity builders
    "lifted_affinity",
    "product_affinity",
    "uniform_affinity",
    # Laplacian
    "affinity_to_laplacian",
    "lifted_laplacian",
    # Gradient / Divergence
    "lifted_gradient",
    "lifted_divergence",
]
