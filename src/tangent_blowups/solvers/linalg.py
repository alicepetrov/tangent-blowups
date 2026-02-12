import numpy as np


def normalize_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise vector normalization with safe divide.

    Works with both (N, d) arrays and higher-rank arrays where the last axis is
    treated as the vector dimension.
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def ensure_orthonormal(U: np.ndarray) -> np.ndarray:
    """
    Orthonormalizes the columns of U using QR decomposition.
    """
    Q, _ = np.linalg.qr(U)
    return Q

__all__ = ["normalize_vectors", "ensure_orthonormal"]
