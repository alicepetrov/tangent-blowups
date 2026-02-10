import abc
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from geom_types import GroundTruth, Sample

# -----------------------------------------------------------------------------
# Base Interface
# -----------------------------------------------------------------------------

class SamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray, ...]:
        """Returns parameter arrays ready for evaluation."""
        pass

    def __call__(self, gt: GroundTruth) -> Tuple[np.ndarray, ...]:
        return self.generate(gt)


def sample(
    gt: GroundTruth,
    strategy: SamplingStrategy,
    with_tangents: bool = False,
    with_normals: bool = False,
) -> Sample:
    
    params = strategy(gt)
    return gt.evaluate(
        *params,
        with_tangents=with_tangents,
        with_normals=with_normals
    )

# -----------------------------------------------------------------------------
# Parametric Strategies
# -----------------------------------------------------------------------------

@dataclass
class UniformCurve(SamplingStrategy):
    n: int
    t_min: float = 0.0
    t_max: float = 2.0 * np.pi
    endpoint: bool = True

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray]:
        return (np.linspace(self.t_min, self.t_max, self.n, endpoint=self.endpoint),)


@dataclass
class RandomCurve(SamplingStrategy):
    n: int
    t_min: float = 0.0
    t_max: float = 2.0 * np.pi
    rng: Optional[np.random.Generator] = None

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray]:
        rng = self.rng or np.random.default_rng()
        return (rng.uniform(self.t_min, self.t_max, size=self.n),)


@dataclass
class UniformSurface(SamplingStrategy):
    """Generates a Grid (Mesh) of points."""
    nu: int
    nv: int
    u_bounds: Tuple[float, float] = (0.0, 1.0)
    v_bounds: Tuple[float, float] = (0.0, 1.0)

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray, np.ndarray]:
        u = np.linspace(self.u_bounds[0], self.u_bounds[1], self.nu)
        v = np.linspace(self.v_bounds[0], self.v_bounds[1], self.nv)
        return np.meshgrid(u, v, indexing='ij')


@dataclass
class RandomSurface(SamplingStrategy):
    """Generates a Point Cloud (List) of random points."""
    n: int # Total number of points
    u_bounds: Tuple[float, float] = (0.0, 1.0)
    v_bounds: Tuple[float, float] = (0.0, 1.0)
    rng: Optional[np.random.Generator] = None

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng or np.random.default_rng()
        u = rng.uniform(self.u_bounds[0], self.u_bounds[1], size=self.n)
        v = rng.uniform(self.v_bounds[0], self.v_bounds[1], size=self.n)
        return (u, v)
    
# -----------------------------------------------------------------------------
# Grid Based Strategies
# -----------------------------------------------------------------------------

@dataclass
class JitteredCurve(SamplingStrategy):
    """
    Stratified Sampling for 1D Curves.
    Divides the domain into 'n' segments and picks a random point 
    within each segment.
    
    Why use this? 
    It reduces clumping compared to pure random sampling (Blue Noise),
    providing better coverage for integration or collision detection.
    """
    n: int
    t_min: float = 0.0
    t_max: float = 2.0 * np.pi
    rng: Optional[np.random.Generator] = None

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray]:
        rng = self.rng or np.random.default_rng()
        
        # 1. Segment width
        dt = (self.t_max - self.t_min) / self.n
        
        # 2. Base indices (0, 1, ..., n-1)
        indices = np.arange(self.n)
        
        # 3. Add random offset in [0, 1) to each segment
        # t = min + (index + random_offset) * step
        noise = rng.random(size=self.n)
        t = self.t_min + (indices + noise) * dt
        
        return (t,)


@dataclass
class ChebyshevCurve(SamplingStrategy):
    """
    Generates non-uniform points clustered at the start and end of the curve.
    
    Why use this?
    If you plan to fit a polynomial or spline to the curve data, 
    Chebyshev nodes minimize Runge's phenomenon (oscillation errors at edges).
    """
    n: int
    t_min: float = 0.0
    t_max: float = 2.0 * np.pi

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray]:
        # Roots of Chebyshev polynomial of the first kind
        # Standard nodes in [-1, 1]
        k = np.arange(1, self.n + 1)
        x_std = np.cos((2 * k - 1) / (2 * self.n) * np.pi)
        
        # Map [-1, 1] to [t_min, t_max]
        half_width = (self.t_max - self.t_min) / 2.0
        center = (self.t_max + self.t_min) / 2.0
        
        t = center + half_width * x_std
        
        # Sort for clean plotting (optional, but usually desired for curves)
        t.sort()
        
        return (t,)

@dataclass
class JitteredSurface(SamplingStrategy):
    """
    Stratified Sampling: divides domain into a grid, but picks a random 
    parameter within each grid cell.
    
    Returns:
        (u, v) grids of shape (nu, nv).
    """
    nu: int
    nv: int
    u_bounds: Tuple[float, float] = (0.0, 1.0)
    v_bounds: Tuple[float, float] = (0.0, 1.0)
    rng: Optional[np.random.Generator] = None

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng or np.random.default_rng()
        
        # 1. Calculate step sizes
        du = (self.u_bounds[1] - self.u_bounds[0]) / self.nu
        dv = (self.v_bounds[1] - self.v_bounds[0]) / self.nv
        
        # 2. Generate base grid indices (0, 1, ..., N-1)
        # We use meshgrid to create the integer coordinates for every cell
        i, j = np.meshgrid(
            np.arange(self.nu), 
            np.arange(self.nv), 
            indexing='ij'
        )
        
        # 3. Add random offset in [0, 1) and scale to step size
        # param = min + (index + random_offset) * step
        u_noise = rng.random(size=(self.nu, self.nv))
        v_noise = rng.random(size=(self.nu, self.nv))
        
        u = self.u_bounds[0] + (i + u_noise) * du
        v = self.v_bounds[0] + (j + v_noise) * dv
        
        return u, v

@dataclass
class ChebyshevGrid(SamplingStrategy):
    """
    Generates a non-uniform grid using Chebyshev nodes.
    Points are clustered at the edges of the domain.
    Best for polynomial interpolation/fitting tasks.
    """
    nu: int
    nv: int
    u_bounds: Tuple[float, float] = (0.0, 1.0)
    v_bounds: Tuple[float, float] = (0.0, 1.0)

    def _chebyshev_nodes(self, n: int, bounds: Tuple[float, float]) -> np.ndarray:
        # Roots of Chebyshev polynomial of the first kind
        k = np.arange(1, n + 1)
        # Standard nodes in [-1, 1]
        x_std = np.cos((2 * k - 1) / (2 * n) * np.pi)
        
        # Map [-1, 1] to [min, max]
        half_width = (bounds[1] - bounds[0]) / 2.0
        center = (bounds[1] + bounds[0]) / 2.0
        return center + half_width * x_std

    def generate(self, gt: GroundTruth) -> Tuple[np.ndarray, np.ndarray]:
        # Generate 1D nodes
        u_nodes = self._chebyshev_nodes(self.nu, self.u_bounds)
        v_nodes = self._chebyshev_nodes(self.nv, self.v_bounds)
        
        # Create 2D Grid
        # Note: Chebyshev nodes are often sorted descending by default cos(),
        # so we sort them to be monotonic if preferred for visualization.
        u_nodes.sort()
        v_nodes.sort()
        
        return np.meshgrid(u_nodes, v_nodes, indexing='ij')

# -----------------------------------------------------------------------------
# Singularity-Aware Strategies
# -----------------------------------------------------------------------------

# TODO