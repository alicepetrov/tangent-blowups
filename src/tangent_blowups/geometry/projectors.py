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

def subspace_alignment(P1: np.ndarray, P2: np.ndarray) -> float:
    """
    Returns the alignment score Tr(P1 @ P2) = sum(cos^2(theta_i)),
    where theta_i are the principal angles between the two subspaces.

    This is a scalar summary of subspace proximity, not the principal
    angles themselves. To recover individual angles, use
    grassmann.principal_angles() with orthonormal bases.
    """
    return float(np.trace(P1 @ P2))

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


# -------------------------------------------------------------------------
# Grassmannian distances from orthonormal bases
# -------------------------------------------------------------------------

def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces.

    Args:
        U1, U2: (n, k) orthonormal matrices.

    Returns:
        (k,) angles in radians in [0, pi/2].
    """
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def dist_geodesic(U1: np.ndarray, U2: np.ndarray) -> float:
    """Riemannian (geodesic) distance on G(k, n)."""
    return float(np.linalg.norm(principal_angles(U1, U2)))