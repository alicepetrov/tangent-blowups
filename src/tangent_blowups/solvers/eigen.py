"""
Eigensolvers for graph Laplacians.

`laplacian_spectrum` uses ARPACK shift-invert for the bottom of the spectrum,
with a robust dense fallback for small matrices or convergence failures.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence


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
        if n <= DENSE_THRESHOLD or k_eff >= n - 1:
            Ld = Ls.toarray()
            evals, evecs = np.linalg.eigh(Ld)
        else:
            if which == "SM":
                ncv = min(max(4 * k_eff + 1, 40), n - 1)
                try:
                    evals, evecs = spla.eigsh(
                        Ls, k=k_eff, sigma=sigma, which="LM", ncv=ncv
                    )
                except ArpackNoConvergence as exc:
                    if exc.eigenvalues.size >= k_eff:
                        evals, evecs = exc.eigenvalues, exc.eigenvectors
                    else:
                        warnings.warn(
                            f"ARPACK did not converge ({exc.eigenvalues.size}/{k_eff} "
                            "eigenpairs). Falling back to dense eigh -- may be slow for "
                            "large N. Consider reducing N or adjusting kernel parameters.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                        evals, evecs = np.linalg.eigh(Ls.toarray())
            else:
                evals, evecs = spla.eigsh(Ls, k=k_eff, which="LM")
    else:
        Ld = np.asarray(L, dtype=float)
        evals, evecs = np.linalg.eigh(Ld)

    order = np.argsort(evals)
    if which == "LM":
        order = order[::-1]

    evals = evals[order]
    evecs = evecs[:, order]

    if drop_first and evals.size > 0:
        evals = evals[1:]
        evecs = evecs[:, 1:]

    evals = evals[:k]
    evecs = evecs[:, :k]

    if return_eigenvalues:
        return evals, evecs
    return evecs


__all__ = ["laplacian_spectrum"]
