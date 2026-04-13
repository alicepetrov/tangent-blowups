"""
GPU Edge Operations
-------------------
PyTorch CUDA implementations of the edge-level einsum, scatter-add,
and batch-inverse operations used by kernels.py.
"""
from __future__ import annotations

import torch
import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Scatter-add helpers (replace _scatter_sum_* from kernels.py)
# ---------------------------------------------------------------------------

def scatter_sum_2d(idx: torch.Tensor, vals: torch.Tensor, N: int) -> torch.Tensor:
    """Sum vals (E, d) into N bins given by idx (E,)."""
    d = vals.shape[1]
    out = torch.zeros(N, d, dtype=vals.dtype, device=vals.device)
    idx_exp = idx.unsqueeze(1).expand_as(vals)
    out.scatter_add_(0, idx_exp, vals)
    return out


def scatter_sum_3d(idx: torch.Tensor, vals: torch.Tensor, N: int) -> torch.Tensor:
    """Sum vals (E, d, d) into N bins given by idx (E,)."""
    d1, d2 = vals.shape[1], vals.shape[2]
    flat = vals.reshape(-1, d1 * d2)
    idx_exp = idx.unsqueeze(1).expand(-1, d1 * d2)
    out = torch.zeros(N, d1 * d2, dtype=vals.dtype, device=vals.device)
    out.scatter_add_(0, idx_exp, flat)
    return out.reshape(N, d1, d2)


# ---------------------------------------------------------------------------
# Batch matrix inverse (closed-form for d=2, general for d>2)
# ---------------------------------------------------------------------------

def batch_inverse(S: torch.Tensor) -> torch.Tensor:
    """Batch inverse of (N, d, d) matrices. Uses closed-form for d <= 2."""
    d = S.shape[1]
    if d == 1:
        return 1.0 / S
    if d == 2:
        a, b = S[:, 0, 0], S[:, 0, 1]
        c, d_ = S[:, 1, 0], S[:, 1, 1]
        det = a * d_ - b * c
        det = torch.where(det.abs() < 1e-30, torch.ones_like(det) * 1e-30, det)
        inv = torch.empty_like(S)
        inv[:, 0, 0] = d_ / det
        inv[:, 0, 1] = -b / det
        inv[:, 1, 0] = -c / det
        inv[:, 1, 1] = a / det
        return inv
    return torch.linalg.inv(S)


# ---------------------------------------------------------------------------
# Precompute edges on GPU (mirrors _precompute_edges in kernels.py)
# ---------------------------------------------------------------------------

def precompute_edges_gpu(level, W, lam):
    """GPU version of _precompute_edges. Returns dict of torch CUDA tensors."""
    device = torch.device("cuda")
    N, D, d = level.N, level.D, level.d

    Phi = torch.tensor(level.embedded, dtype=torch.float64, device=device)
    U = torch.tensor(level.frame, dtype=torch.float64, device=device)

    W_coo = W.tocoo()
    rows = torch.tensor(W_coo.row, dtype=torch.long, device=device)
    cols = torch.tensor(W_coo.col, dtype=torch.long, device=device)
    ws = torch.tensor(W_coo.data, dtype=torch.float64, device=device)

    # Edge displacements projected into source tangent plane
    d_phi = Phi[cols] - Phi[rows]                           # (E, D)
    delta = torch.einsum("eD,eDd->ed", d_phi, U[rows])     # (E, d)

    w_delta = ws.unsqueeze(1) * delta                       # (E, d)

    # Scatter-add to build S_i = sum_j w_ij delta_ij delta_ij^T
    S_edges = torch.einsum("ea,eb->eab", w_delta, delta)    # (E, d, d)
    S = scatter_sum_3d(rows, S_edges, N)                    # (N, d, d)

    # Ridge regularisation
    tr_S = torch.diagonal(S, dim1=1, dim2=2).sum(dim=1)    # (N,)
    pos_tr = tr_S[tr_S > 0]
    fallback_scale = float(pos_tr.median().item()) if pos_tr.numel() > 0 else 1.0
    ridge = torch.clamp(lam * tr_S / d, min=lam * fallback_scale / d)
    ridge = torch.clamp(ridge, min=1e-12 * fallback_scale / d)
    eye_d = torch.eye(d, dtype=torch.float64, device=device)
    S = S + ridge.unsqueeze(1).unsqueeze(2) * eye_d

    S_inv = batch_inverse(S)                                # (N, d, d)

    return dict(
        rows=rows, cols=cols, weights=ws,
        delta=delta, w_delta=w_delta, S_inv=S_inv,
        Phi=Phi, U=U,
    )


def lifted_gradient_gpu(level, f_np, W, *, lam=0.0, _edge_cache=None):
    """GPU version of lifted_gradient. Returns numpy (N, D) array."""
    device = torch.device("cuda")
    N, D, d = level.N, level.D, level.d

    ec = _edge_cache or precompute_edges_gpu(level, W, lam)
    rows, cols = ec["rows"], ec["cols"]
    w_delta, S_inv = ec["w_delta"], ec["S_inv"]
    U = ec["U"]

    f = torch.tensor(f_np, dtype=torch.float64, device=device)

    # r_i = sum_j w_ij delta_ij (f_j - f_i)
    df = f[cols] - f[rows]                                  # (E,)
    r_edges = w_delta * df.unsqueeze(1)                     # (E, d)
    r = scatter_sum_2d(rows, r_edges, N)                    # (N, d)

    # g_i = S_i^{-1} r_i
    g_tan = torch.einsum("iab,ib->ia", S_inv, r)           # (N, d)

    # Lift back to ambient: grad_i = U_i g_i
    grad_ambient = torch.einsum("iDd,id->iD", U, g_tan)    # (N, D)
    return grad_ambient.cpu().numpy()


def edge_divergence_gpu(level, X_np, W):
    """GPU version of _edge_divergence from geodesic_heat.py."""
    device = torch.device("cuda")
    N = level.N
    n_orig = level.n_orig
    positions = torch.tensor(
        level.embedded[:, :n_orig], dtype=torch.float64, device=device,
    )

    W_coo = W.tocoo()
    rows = torch.tensor(W_coo.row, dtype=torch.long, device=device)
    cols = torch.tensor(W_coo.col, dtype=torch.long, device=device)
    ws = torch.tensor(W_coo.data, dtype=torch.float64, device=device)

    edge_vec = positions[cols] - positions[rows]            # (E, n_orig)

    X = torch.tensor(X_np[:, :n_orig], dtype=torch.float64, device=device)
    X_avg = 0.5 * (X[rows] + X[cols])                      # (E, n_orig)
    X_edge = (X_avg * edge_vec).sum(dim=1)                  # (E,)

    div = torch.zeros(N, dtype=torch.float64, device=device)
    div.scatter_add_(0, rows, ws * X_edge)
    return div.cpu().numpy()


# ---------------------------------------------------------------------------
# Product affinity on GPU
# ---------------------------------------------------------------------------

def _local_bandwidths_torch(N, rows, dist2):
    """Per-point k-NN bandwidth (max neighbor distance). Mirrors kernels._local_bandwidths."""
    h_sq = torch.zeros(N, dtype=dist2.dtype, device=dist2.device)
    h_sq.scatter_reduce_(0, rows, dist2, reduce="amax", include_self=True)
    h = torch.sqrt(h_sq)
    zero = h == 0
    if zero.any():
        pos = dist2[dist2 > 0]
        fallback = float(torch.median(torch.sqrt(pos)).item()) if pos.numel() > 0 else 1.0
        h = torch.where(zero, torch.full_like(h, fallback), h)
    return h


def product_affinity_gpu(level, sigma_x, sigmas, k, rows_np, cols_np, *, self_tuning=False):
    """Compute product-kernel weights on GPU given precomputed k-NN edges.

    Mirrors the CPU implementation in `kernels.product_affinity`, including the
    self-tuning branch (Zelnik-Manor pointwise bandwidth per factor).
    """
    device = torch.device("cuda")
    N = level.N

    rows = torch.tensor(rows_np, dtype=torch.long, device=device)
    cols = torch.tensor(cols_np, dtype=torch.long, device=device)
    embedded = torch.tensor(level.embedded, dtype=torch.float64, device=device)

    positions = embedded[:, :level.n_orig]
    dx = positions[rows] - positions[cols]
    dist2_spatial = (dx * dx).sum(dim=1)
    if self_tuning:
        h_x = _local_bandwidths_torch(N, rows, dist2_spatial)
        weights = torch.exp(-dist2_spatial / (h_x[rows] * h_x[cols]))
    else:
        weights = torch.exp(-dist2_spatial / (sigma_x * sigma_x))

    for m, (start, ncols_, scale) in enumerate(level._proj_blocks):
        if scale == 0.0:
            continue
        diff = embedded[rows, start:start + ncols_] - embedded[cols, start:start + ncols_]
        scaled_dist2 = (diff * diff).sum(dim=1)
        if self_tuning:
            intrinsic_dist2 = scaled_dist2 / (scale * scale)
            h_u = _local_bandwidths_torch(N, rows, intrinsic_dist2)
            weights = weights * torch.exp(-intrinsic_dist2 / (h_u[rows] * h_u[cols]))
        else:
            weights = weights * torch.exp(-scaled_dist2 / (sigmas[m] * sigmas[m]))

    return weights.cpu().numpy()


# ---------------------------------------------------------------------------
# GPU k-NN (brute-force via torch.cdist + topk)
# ---------------------------------------------------------------------------

def knn_edges_gpu(embed_np, k):
    """Brute-force k-NN on GPU using chunked cdist.

    Args:
        embed_np: (N, D) numpy array.
        k:        Number of nearest neighbours.

    Returns:
        rows, cols, dist2 as numpy arrays.
    """
    device = torch.device("cuda")
    N, D = embed_np.shape
    k_eff = min(k + 1, N)
    X = torch.tensor(embed_np, dtype=torch.float64, device=device)

    chunk = 4096
    all_idx = []
    all_dist2 = []

    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        X_q = X[i:end]                                      # (chunk, D)
        # Squared Euclidean distances
        D2 = torch.cdist(X_q, X, p=2.0).pow(2)             # (chunk, N)
        # Set self-distance to inf
        diag_start = i
        diag_end = end
        for j in range(diag_end - diag_start):
            D2[j, diag_start + j] = float("inf")
        d2, idx = D2.topk(k_eff - 1, largest=False, dim=1)  # exclude self
        all_idx.append(idx.cpu())
        all_dist2.append(d2.cpu())

    idx_all = torch.cat(all_idx, dim=0).numpy()              # (N, k)
    dist2_all = torch.cat(all_dist2, dim=0).numpy()

    rows = np.repeat(np.arange(N, dtype=np.intp), idx_all.shape[1])
    cols = idx_all.ravel().astype(np.intp)
    dist2 = dist2_all.ravel().astype(np.float64)
    return rows, cols, dist2


__all__ = [
    "precompute_edges_gpu",
    "lifted_gradient_gpu",
    "edge_divergence_gpu",
    "product_affinity_gpu",
    "knn_edges_gpu",
    "scatter_sum_2d",
    "scatter_sum_3d",
    "batch_inverse",
]
