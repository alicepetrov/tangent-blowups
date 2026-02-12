"""
Projector Utilities
-------------------
Handles subspace operations using the projection matrix representation P = UU^T.
This is the "coordinate-free" way to handle subspaces, avoiding basis ambiguities.
"""
import numpy as np

def basis_to_projector(U: np.ndarray) -> np.ndarray:
    """
    Converts orthonormal basis U (..., n, k) to Projector P (..., n, n).
    Supports batching.
    """
    # Einstein summation for batch matrix multiplication: U @ U.T
    # ...nij, ...nlj -> ...nil (Contract over inner dimension k)
    return np.einsum('...ik,...jk->...ij', U, U)

def chordal_distance(P1: np.ndarray, P2: np.ndarray) -> float:
    """
    Computes Chordal distance between two subspaces via their projectors.
    d_c(P1, P2) = || P1 - P2 ||_F / sqrt(2)
    This is the Euclidean distance in the embedding space of matrices.
    """
    # Frobenius norm of difference
    diff = P1 - P2
    norm_sq = np.sum(diff**2) # faster than linalg.norm for pure arrays
    return np.sqrt(norm_sq) / np.sqrt(2)

def principal_angles_from_projectors(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Computes principal angles strictly from Projectors.
    Use this if you don't have the bases U1, U2 handy.
    
    Theory:
    Trace(P1 @ P2) = sum(cos^2(theta_i))
    This gives us the "average" alignment, but recovering individual angles
    requires an eigendecomposition of the product P1 P2 P1.
    """
    # For full angles, it is usually numerically better to recover a basis 
    # (eigenvectors of P) and use the SVD method in grassmann.py.
    # But for a quick "alignment score":
    alignment = np.trace(P1 @ P2)
    return alignment

def mean_projector(projectors: np.ndarray) -> np.ndarray:
    """
    Computes the Karcher mean (or Fréchet mean) of a set of subspaces.
    For the Chordal metric, this is simply the SVD of the sum of projectors.
    
    Args:
        projectors: (N, n, n) array of P matrices.
        
    Returns:
        P_mean: (n, n) projector of the average subspace.
    """
    # 1. Arithmetic mean of projectors (Extrinsic mean)
    # This matrix is symmetric but not a projection (eigenvalues not {0,1}).
    P_avg = np.mean(projectors, axis=0)
    
    # 2. Project back to the Grassmannian
    # The closest rank-k projector to a matrix A is formed by its top k eigenvectors.
    # We need to know 'k'. We can infer it from the trace of the input projectors.
    k_approx = int(np.round(np.trace(projectors[0])))
    
    # 3. Eigendecomposition
    vals, vecs = np.linalg.eigh(P_avg)
    
    # 4. Select top k eigenvectors
    # eigh returns eigenvalues in ascending order, so take the last k
    U_mean = vecs[:, -k_approx:]
    
    return basis_to_projector(U_mean)