"""
Grassmannian Geometry Module
----------------------------
Handles operations on the Grassmann manifold G(k, n) and the product space R^n x G(k, n).

Mathematical Context:
- A point in G(k, n) is a subspace. We represent it via an orthonormal frame U (n x k).
- Because U and UQ (where Q is orthogonal k x k) represent the same subspace, 
  we often operate on the projector P = UU^T, which is unique and sign-invariant.
- Tangent vectors at U are matrices Delta (n x k) such that U^T Delta = 0.
"""
from functools import cached_property
import numpy as np
from typing import Literal, Union
from tangent_blowups.solvers.linalg import normalize_vectors, ensure_orthonormal
from . import projectors

# -------------------------------------------------------------------------
# Grassmannian Distances
# -------------------------------------------------------------------------

def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Computes the principal angles between two subspaces U1 and U2.
    Theta = arccos(singular_values(U1^T @ U2))
    
    Args:
        U1, U2: (n, k) orthonormal matrices.
        
    Returns:
        angles: (k,) array of angles in radians [0, pi/2].
    """
    # Clip for numerical stability in arccos
    M = U1.T @ U2
    # Singular values of the overlap matrix measure the alignment
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0) 
    return np.arccos(s)

def dist_geodesic(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    The intrinsic Riemannian distance on G(k, n).
    d_geo(U1, U2) = || theta ||_2 (L2 norm of principal angles).
    """
    thetas = principal_angles(U1, U2)
    return float(np.linalg.norm(thetas))

def dist_chordal(U1, U2):
    P1 = projectors.basis_to_projector(U1)
    P2 = projectors.basis_to_projector(U2)
    return projectors.chordal_distance(P1, P2)

# -------------------------------------------------------------------------
# Exponential and Logarithmic Maps (Tangent Space Ops)
# -------------------------------------------------------------------------

def log_map(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Riemannian Logarithm: Log_U(V).
    Computes the tangent vector Delta at U that 'points' towards V.
    
    Algorithm:
    1. Project V onto orthogonal complement of U: U_perp_V = (I - UU^T)V
    2. Compute SVD: U_sig @ S @ V_sig.T = U_perp_V
    3. Compute principal angles: theta = arccos(singular_values(U^T V))
    4. Recombine: Delta = U_sig @ diag(theta) @ V_sig.T
    
    Note: This implementation is a "Soft Log" approximation suitable for 
    optimization/smoothing.
    """
    # 1. Project V onto orthogonal complement of U
    # This captures the "direction" in which V differs from U
    U_perp_V = V - U @ (U.T @ V)
    
    # 2. SVD of the projection
    # U_sig @ S @ V_sig.T = U_perp_V
    u_sig, s_vals, v_sig_t = np.linalg.svd(U_perp_V, full_matrices=False)
    
    # 3. Principal angles 
    # Using overlap is safer for small angles
    overlap = U.T @ V
    _, c_vals, _ = np.linalg.svd(overlap)
    
    c_vals = np.clip(c_vals, -1.0, 1.0)
    thetas = np.arccos(c_vals)
    
    # If thetas are very small, return zero vector
    if np.allclose(thetas, 0):
        return np.zeros_like(U)

    # 4. Construct Tangent Vector
    return u_sig @ np.diag(thetas) @ v_sig_t

def exp_map(U: np.ndarray, Delta: np.ndarray) -> np.ndarray:
    """
    Riemannian Exponential: Exp_U(Delta).
    Moves U along the geodesic defined by tangent vector Delta for time 1.
    
    Delta must be a valid tangent vector (U^T Delta = 0).
    
    Formula:
    U(t) = U V cos(Sigma t) V^T + U_perp sin(Sigma t) V^T
    Where Delta = U_perp Sigma V^T (compact SVD of tangent vector)
    """
    # 1. Compact SVD of tangent vector Delta
    u_perp, sigmas, vt = np.linalg.svd(Delta, full_matrices=False)
    
    # 2. Geodesic flow
    cos_sig = np.diag(np.cos(sigmas))
    sin_sig = np.diag(np.sin(sigmas))
    
    # Re-assemble: Base retraction + extension
    term1 = U @ (vt.T @ cos_sig @ vt)
    term2 = u_perp @ sin_sig @ vt
    
    return term1 + term2

# -------------------------------------------------------------------------
# Product Space Handling (Blown-up Space)
# -------------------------------------------------------------------------

class BlownUpPoint:
    """
    A container for a single point in the product space R^n x G(k, n).
    Useful for debugging and unit tests.
    """
    def __init__(self, x: np.ndarray, U: np.ndarray):
        """
        Args:
            x: (n,) spatial coordinate
            U: (n, k) subspace basis (will be orthogonalized)
        """
        self.x = np.array(x)
        self.n = self.x.shape[0]
        self.U = ensure_orthonormal(U)
        self.k = self.U.shape[1]
        
        if self.U.shape[0] != self.n:
            raise ValueError(f"Dimension mismatch: x is {self.n}, U is {self.U.shape}")

    @cached_property
    def P(self) -> np.ndarray:
        """
        Returns the unique projection matrix P = UU^T representing the subspace.
        Cached for efficiency.
        
        Returns:
            (n, n) symmetric projection matrix.
        """
        return projectors.basis_to_projector(self.U)

    def distance_to(
        self,
        other: 'BlownUpPoint',
        alpha: float = 1.0,
        subspace_metric: Literal['chordal', 'geodesic'] = 'chordal',
    ) -> float:
        """
        Computes the product metric distance.
        d^2 = ||x1 - x2||^2 + alpha * d_subspace(U1, U2)^2

        Args:
            other: Another BlownUpPoint.
            alpha: Weight on the Grassmannian distance term.
            subspace_metric: Which Grassmannian distance to use:
                - "chordal" (default)
                - "geodesic"
        """
        dist_x = np.linalg.norm(self.x - other.x)
        if subspace_metric == 'chordal':
            dist_u = dist_chordal(self.U, other.U)
        elif subspace_metric == 'geodesic':
            dist_u = dist_geodesic(self.U, other.U)
        else:
            raise ValueError(
                f"Unknown subspace_metric '{subspace_metric}'. "
                "Expected 'chordal' or 'geodesic'."
            )
        return np.sqrt(dist_x**2 + alpha * (dist_u**2))
    
    def vectorized_embedding(self, alpha: float = 1.0) -> np.ndarray:
        """
        Maps this point into the Euclidean embedding space R^D.
        V = [x, sqrt(alpha/2) * flatten(P)]
        """
        P_flat = self.P.flatten()
        scale = np.sqrt(alpha / 2.0)
        return np.hstack([self.x, scale * P_flat])

    def __repr__(self):
        return f"BlownUpPoint(x={self.x}, G({self.k},{self.n}))"


class BlownUpSample:
    """
    Vectorized container for N points in the product space R^n x G(k, n).
    Handles Nash Blowups (Point + Tangent) and Gauss Maps (Point + Normal).
    """
    def __init__(self, spatial: np.ndarray, subspace_basis: np.ndarray):
        """
        Direct constructor. 
        Args:
            spatial: (N, n) coordinates
            subspace_basis: (N, n, k) orthonormal frames defining the subspace.
        """
        self.spatial = np.ascontiguousarray(spatial)
        self.basis = np.ascontiguousarray(subspace_basis)
        
        # Validation
        self.N, self.n = self.spatial.shape
        
        # Handle the case where user passes (N, n) vectors (k=1)
        if self.basis.ndim == 2:
            self.basis = self.basis[:, :, np.newaxis]
            
        _, self.n_b, self.k = self.basis.shape
        
        assert self.n == self.n_b, \
            f"Dimension mismatch: Spatial is R^{self.n}, Subspace is in R^{self.n_b}"
        
    def __getitem__(self, idx) -> Union['BlownUpSample', 'BlownUpPoint']:
        """
        Smart indexing:
        - If idx is an integer, returns a lightweight BlownUpPoint (View).
        - If idx is a slice/array, returns a new BlownUpSample (Subset).
        """
        if isinstance(idx, (int, np.integer)):
            # Return the object representation for easy "p.distance_to(q)" logic
            return BlownUpPoint(self.spatial[idx], self.basis[idx])
        
        # Return the vectorized container for slicing
        return BlownUpSample(self.spatial[idx], self.basis[idx])

    @property
    def projectors(self) -> np.ndarray:
        """Returns (N, n, n) array of projection matrices P = UU^T."""
        return np.einsum('nik,njk->nij', self.basis, self.basis)

    def embedding_vector(self, alpha: float = 1.0) -> np.ndarray:
        """
        Maps the non-Euclidean product space R^n x G(k, n) into a Euclidean space R^D
        such that the Euclidean distance in R^D approximates the Product Metric.
        
        Metric Target: d^2 = ||dx||^2 + alpha * d_chordal(P1, P2)^2
        Recall: d_chordal^2 = ||P1 - P2||_F^2 / 2
        
        Embedding: V = [x,  sqrt(alpha/2) * flatten(P)]
        
        Then: ||V1 - V2||^2 = ||dx||^2 + (alpha/2) * ||dP||^2
                            = ||dx||^2 + alpha * d_chordal^2
        """
        # (N, n*n)
        P_flat = self.projectors.reshape(self.N, -1)
        
        # Scale factor to match the requested alpha weight exactly
        scale = np.sqrt(alpha / 2.0)
        
        return np.hstack([self.spatial, scale * P_flat])

    def distance_matrix(
        self,
        other: Union['BlownUpSample', None] = None,
        *,
        alpha: float = 1.0,
        subspace_metric: Literal['chordal', 'geodesic'] = 'chordal',
    ) -> np.ndarray:
        """
        Computes pairwise product-metric distances between two samples.

        If other is None, returns the (N, N) distance matrix for self.
        Otherwise returns (N, M) distances between self and other.

        d^2 = ||x_i - y_j||^2 + alpha * d_subspace(U_i, V_j)^2
        """
        if other is None:
            other = self

        if self.n != other.n or self.k != other.k:
            raise ValueError(
                f"Dimension mismatch: self is G({self.k},{self.n}), "
                f"other is G({other.k},{other.n})"
            )

        # Spatial distances ||x_i - y_j||^2
        A = self.spatial
        B = other.spatial
        a2 = np.sum(A * A, axis=1)[:, np.newaxis]
        b2 = np.sum(B * B, axis=1)[np.newaxis, :]
        dist_x_sq = a2 + b2 - 2.0 * (A @ B.T)
        dist_x_sq = np.maximum(dist_x_sq, 0.0)

        # Subspace distances
        if subspace_metric == 'chordal':
            P = self.projectors
            Q = other.projectors
            # inner_{ij} = trace(P_i P_j)
            inner = np.einsum('nij,mij->nm', P, Q)
            dist_u_sq = (self.k + other.k - 2.0 * inner) / 2.0
            dist_u_sq = np.maximum(dist_u_sq, 0.0)
        elif subspace_metric == 'geodesic':
            dist_u_sq = np.empty((self.N, other.N), dtype=float)
            for i in range(self.N):
                Ui = self.basis[i]
                for j in range(other.N):
                    d = dist_geodesic(Ui, other.basis[j])
                    dist_u_sq[i, j] = d * d
        else:
            raise ValueError(
                f"Unknown subspace_metric '{subspace_metric}'. "
                "Expected 'chordal' or 'geodesic'."
            )

        return np.sqrt(dist_x_sq + alpha * dist_u_sq)

    # -----------------------------------------------------------------------
    #  Factories for Specific Geometric Inputs
    # -----------------------------------------------------------------------

    @classmethod
    def from_normals(cls, points: np.ndarray, normals: np.ndarray):
        """
        Creates the lift from Normal vectors.
        
        Math Note:
        For clustering/distance purposes, the Grassmannian of Normals G(1, n)
        is isometric to the Grassmannian of Tangent Planes G(n-1, n).
        We can store the Normal (k=1) directly as it is more memory efficient
        than storing the Tangent Plane basis (k=n-1).
        """
        # Normals are (N, D). This is a valid G(1, D) representation.
        return cls(points, normalize_vectors(normals))

    @classmethod
    def from_tangents(cls, points: np.ndarray, tangents: np.ndarray):
        """
        Creates the lift from Tangent information.
        
        Accepts:
        - 2D Tangent Vectors: (N, 2) -> Stores as G(1, 2)
        - 3D Tangent Planes:  (N, 3, 2) -> Stores as G(2, 3)
        """
        return cls(points, tangents)

    @classmethod
    def from_mixed(cls, points: np.ndarray, data: np.ndarray, kind: Literal['tangent', 'normal']):
        """
        Generic entry point if you are piping data from a config file.
        """
        if kind == 'normal':
            return cls.from_normals(points, data)
        elif kind == 'tangent':
            return cls.from_tangents(points, data)
        else:
            raise ValueError(f"Unknown geometric kind: {kind}")

    # -----------------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------------

    def dualize(self):
        """
        Converts between Tangent and Normal representations.
        Returns a NEW BlownUpSample with k_new = n - k.
        """
        # 1. Construct Projector onto Orthogonal Complement
        # P_perp = I - UU^T
        # We can do this in batch.
        I = np.eye(self.n) # (n, n)
        P = self.projectors    # (N, n, n)
        
        # Broadcast subtraction: (1, n, n) - (N, n, n)
        P_perp = I[np.newaxis, :, :] - P
        
        # 2. Extract Basis for P_perp via SVD
        # The columns of U_svd corresponding to the largest singular values span the range.
        # Since P_perp is a projector of rank (n-k), we expect (n-k) ones and k zeros.
        # np.linalg.svd handles batches: (N, n, n) -> (N, n, n), (N, n), (N, n, n)
        u_svd, s_svd, _ = np.linalg.svd(P_perp)
        
        # 3. Select the top (n-k) vectors
        target_rank = self.n - self.k
        dual_basis = u_svd[:, :, :target_rank]
        
        return BlownUpSample(self.spatial, dual_basis)