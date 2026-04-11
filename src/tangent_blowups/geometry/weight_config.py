"""
Regression Weight Configuration
--------------------------------
Shared weight scheme for all discrete operators (gradient, divergence,
curvature regression).  Two options:

1. **Product kernel** (default): decoupled spatial and angular bandwidths.
       w_ij = exp(-||x_i - x_j||^2 / sigma_x^2) * exp(-||P_i - P_j||_F^2 / sigma_u^2)

2. **Uniform**: all neighbours weighted equally (w_ij = 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for per-vertex regression weights.

    Attributes:
        kernel:  ``"product"`` (default) or ``"uniform"``.
        sigma_x: Spatial bandwidth for the product kernel.
        sigma_u: Angular bandwidth for the product kernel.
    """
    kernel: Literal["product", "uniform"] = "product"
    sigma_x: float = 0.5
    sigma_u: float = 0.5


def product_weights(
    spatial_dist2: np.ndarray,
    angular_dist2: np.ndarray,
    config: WeightConfig,
) -> np.ndarray:
    """Product-kernel weights from pre-computed squared distances.

    Args:
        spatial_dist2: (k,) squared spatial distances ||x_i - x_j||^2.
        angular_dist2: (k,) squared angular distances ||P_i - P_j||_F^2.
        config:        Weight configuration.

    Returns:
        (k,) non-negative weight array.
    """
    if config.kernel == "uniform":
        return np.ones(len(spatial_dist2), dtype=float)
    return (np.exp(-spatial_dist2 / (config.sigma_x ** 2))
            * np.exp(-angular_dist2 / (config.sigma_u ** 2)))
