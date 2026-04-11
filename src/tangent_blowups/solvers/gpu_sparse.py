"""
GPU Sparse Solvers
------------------
Preconditioned conjugate gradient (PCG) for SPD sparse systems on CUDA,
with scipy CSR <-> torch sparse CSR conversion utilities.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse


def scipy_csr_to_torch(
    A: sparse.csr_matrix,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a scipy CSR matrix to a torch sparse CSR tensor."""
    A = A.tocsr()
    A.sort_indices()
    crow = torch.tensor(A.indptr, dtype=torch.int64, device=device)
    col = torch.tensor(A.indices, dtype=torch.int64, device=device)
    val = torch.tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_csr_tensor(crow, col, val, size=A.shape, device=device)


def _spmv(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse matrix-vector product for CSR tensors."""
    return A.matmul(x)


def pcg_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    M_inv: torch.Tensor | None = None,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> torch.Tensor:
    """Preconditioned conjugate gradient for SPD systems on GPU.

    Args:
        A:      Sparse CSR tensor (N, N), SPD.
        b:      Dense RHS vector (N,).
        x0:     Initial guess (warm-start). If None, uses zeros.
        M_inv:  Diagonal preconditioner (N,). If None, uses Jacobi (1/diag(A)).
        tol:    Relative residual tolerance.
        max_iter: Maximum iterations.

    Returns:
        Solution vector x (N,).
    """
    N = b.shape[0]
    if x0 is not None:
        x = x0.clone()
    else:
        x = torch.zeros(N, dtype=b.dtype, device=b.device)

    r = b - _spmv(A, x)
    b_norm = torch.linalg.norm(b)
    if b_norm == 0.0:
        return x

    if M_inv is None:
        # Jacobi preconditioner from diagonal of A
        diag_A = _extract_diagonal(A, N)
        M_inv = 1.0 / torch.clamp(diag_A.abs(), min=1e-14)

    z = M_inv * r
    p = z.clone()
    rz = torch.dot(r, z)

    for _ in range(max_iter):
        Ap = _spmv(A, p)
        pAp = torch.dot(p, Ap)
        if pAp <= 0.0:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.linalg.norm(r)
        if r_norm / b_norm < tol:
            break

        z = M_inv * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x


def _extract_diagonal(A: torch.Tensor, N: int) -> torch.Tensor:
    """Extract diagonal of a sparse CSR tensor."""
    crow = A.crow_indices()
    col = A.col_indices()
    val = A.values()
    diag = torch.zeros(N, dtype=val.dtype, device=val.device)
    for i in range(N):
        start, end = crow[i].item(), crow[i + 1].item()
        cols_i = col[start:end]
        mask = cols_i == i
        if mask.any():
            diag[i] = val[start:end][mask][0]
    return diag


def _extract_diagonal_fast(A: sparse.csr_matrix) -> np.ndarray:
    """Extract diagonal from scipy CSR (fast, on CPU)."""
    return np.asarray(A.diagonal(), dtype=float)


def sparse_solve_gpu(
    A_scipy: sparse.csr_matrix,
    b_numpy: np.ndarray,
    *,
    x0_numpy: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> np.ndarray:
    """Solve Ax = b on GPU using PCG. A must be SPD.

    Args:
        A_scipy: Scipy CSR matrix (SPD).
        b_numpy: RHS vector (N,).
        x0_numpy: Optional warm-start initial guess.
        tol:     Relative residual tolerance.
        max_iter: Maximum CG iterations.

    Returns:
        Solution vector x as numpy array (N,).
    """
    device = torch.device("cuda")
    dtype = torch.float64

    A_t = scipy_csr_to_torch(A_scipy, device, dtype)
    b_t = torch.tensor(b_numpy, dtype=dtype, device=device)

    # Precompute Jacobi preconditioner on CPU (faster diagonal extraction)
    diag_np = _extract_diagonal_fast(A_scipy)
    M_inv = torch.tensor(
        1.0 / np.clip(np.abs(diag_np), 1e-14, None),
        dtype=dtype, device=device,
    )

    x0_t = None
    if x0_numpy is not None:
        x0_t = torch.tensor(x0_numpy, dtype=dtype, device=device)

    x_t = pcg_solve(A_t, b_t, x0=x0_t, M_inv=M_inv, tol=tol, max_iter=max_iter)
    return x_t.cpu().numpy()


__all__ = ["sparse_solve_gpu", "pcg_solve", "scipy_csr_to_torch"]
