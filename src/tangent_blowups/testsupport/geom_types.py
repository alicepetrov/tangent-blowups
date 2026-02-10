import abc
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union, Any
from scipy.spatial import cKDTree
import numpy as np

# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

ParametricFunc = Callable[..., np.ndarray]

@dataclass
class Sample:
    """
    A pure data container for discrete geometric representations.
    """
    points: np.ndarray  # (N, 3) for curves/clouds, (Nu, Nv, 3) for grids
    params: Union[np.ndarray, Tuple[np.ndarray, ...]]
    
    tangents: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    
    # Metadata for analysis
    singular_mask: Optional[np.ndarray] = None
    singular_indices: Optional[np.ndarray] = None


class GroundTruth(abc.ABC):
    """
    Abstract base class for continuous geometric representations.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def evaluate(
        self, 
        *coords: np.ndarray, 
        with_tangents: bool = True, 
        with_normals: bool = True
    ) -> Sample:
        pass


class ParametricCurve(GroundTruth):
    """
    1D Curve in 3D space: f(t) -> (x, y, z)
    """
    def __init__(
        self, 
        position: ParametricFunc, 
        tangent: Optional[ParametricFunc] = None, 
        normal: Optional[ParametricFunc] = None,
    ):
        self.position = position
        self.tangent = tangent
        self.normal = normal

    def evaluate(
        self, 
        *coords: np.ndarray, 
        with_tangents: bool = True, 
        with_normals: bool = True, 
        singularity_tol: float = None
    ) -> Sample:
        t = np.asarray(coords[0])
        
        points = self.position(t)
        tans = self.tangent(t) if with_tangents and self.tangent else None
        norms = self.normal(t) if with_normals and self.normal else None
        
        # Singularity Detection
        # Strategy: Use geometric check to find self-intersections
        mask, idx = None, None
        if with_tangents and tans is not None and singularity_tol is not None:
            mask, idx = _detect_self_intersections(
                points, tans
            )

        return Sample(points, t, tans, norms, mask, idx)


class ParametricSurface(GroundTruth):
    """
    2D Surface in 3D space: f(u, v) -> (x, y, z)
    
    Note: 'evaluate' supports both Grid inputs (from meshgrid) 
    and Point Cloud inputs (flat arrays) via broadcasting.
    """
    def __init__(
        self, 
        position: ParametricFunc, 
        tangent: Optional[ParametricFunc] = None, 
        normal: Optional[ParametricFunc] = None
    ):
        self.position = position
        self.tangent = tangent
        self.normal = normal

    def evaluate(
        self, 
        *coords: np.ndarray, 
        with_tangents: bool = True, 
        with_normals: bool = True, 
        singularity_tol: float = None
    ) -> Sample:
        u, v = coords
        
        points = self.position(u, v)
        tans = self.tangent(u, v) if with_tangents and self.tangent else None
        norms = self.normal(u, v) if with_normals and self.normal else None
        
        # Singularity Detection
        mask, idx = None, None
        
        # Only perform geometric check if we have tangents and data isn't massive
        if with_tangents and tans is not None:
            # We must flatten the arrays to perform the N^2 check
            N_pts = points.shape[0] * (points.shape[1] if points.ndim > 2 else 1)
            
            if singularity_tol: # Only check for singularities if a tolerance is provided
                flat_pts = points.reshape(-1, 3)
                flat_tans = tans.reshape(-1, 3)
                # Helper uses dummy params for surfaces (index-based check only)
                # We pass a dummy 2D param array to skip the 1D check logic in helper
                dummy_params = np.zeros((N_pts, 2)) 
                
                flat_mask, flat_idx = _detect_self_intersections(
                    flat_pts, flat_tans
                )
                
                if flat_mask is not None:
                    # Reshape mask back to original grid shape
                    mask = flat_mask.reshape(points.shape[:-1])
                    # Note: indices will refer to the flattened array
                    idx = flat_idx 

        return Sample(points, (u, v), tans, norms, mask, idx)
    
# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# TODO add max distance cutoff and handle normals
def _detect_self_intersections(
    points: np.ndarray,
    tangents: np.ndarray,
    *,
    k: int = 5,
    min_index_separation: int = 10,
    min_angle_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(points, dtype=float)
    T = np.asarray(tangents, dtype=float)

    if P.ndim != 2 or T.ndim != 2 or P.shape != T.shape:
        raise ValueError(f"`points` and `tangents` must both be shape (N, d). Got {P.shape=} {T.shape=}")

    N = P.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=bool), np.array([], dtype=int)

    norms = np.linalg.norm(T, axis=1)
    valid = (norms > 1e-12) & np.all(np.isfinite(P), axis=1) & np.all(np.isfinite(T), axis=1)

    Tn = np.zeros_like(T)
    Tn[valid] = T[valid] / norms[valid, None]

    # cKDTree kNN: returns self as first neighbor when querying same set
    kk = min(k + 1, N)
    tree = cKDTree(P)
    _, neighs = tree.query(P, k=kk, workers=-1)  # (N, kk) int

    # Normalize shape when kk == 1 (SciPy can return (N,))
    if neighs.ndim == 1:
        neighs = neighs[:, None]

    dot_thresh = float(np.cos(np.deg2rad(min_angle_deg)))

    I = np.repeat(np.arange(N), neighs.shape[1])
    J = neighs.reshape(-1)

    m = (I != J)
    if min_index_separation > 0:
        m &= (np.abs(I - J) >= min_index_separation)
    m &= valid[I] & valid[J]

    if not np.any(m):
        return np.zeros(N, dtype=bool), np.array([], dtype=int)

    I2 = I[m]
    J2 = J[m]

    dots = np.abs(np.einsum("ij,ij->i", Tn[I2], Tn[J2]))
    hit = dots <= dot_thresh

    hit_mask = np.zeros(N, dtype=bool)
    if np.any(hit):
        hit_mask[I2[hit]] = True
        hit_mask[J2[hit]] = True

    idx = np.flatnonzero(hit_mask)
    return hit_mask, idx