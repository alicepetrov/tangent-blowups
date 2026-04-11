from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.interpolate import splprep, splev

from ...geometry.iterated_grassmann import BlowUpLevel

SubspaceMetric = Literal["chordal", "geodesic"]


@dataclass(frozen=True)
class BSplineResult:
    """
    Container for a B-spline fit in blown-up space with spatial projection.

    Attributes:
        tck: SciPy B-spline representation (knots, coefficients, degree).
        u: Parameter values associated with the input samples.
        embedding: (N, D) Euclidean embedding of the blown-up sample.
        curve_u: Parameter values used to evaluate the fitted spline.
        curve_embedding: (M, D) fitted spline in the embedding space.
        curve_spatial: (M, n) fitted spline projected back to spatial coordinates.
        alpha: Product metric weight on the subspace term.
        subspace_metric: Which Grassmann metric was assumed in the embedding.
        degree: Spline degree k.
        smoothing: Smoothing factor passed to splprep.
        periodic: Whether a periodic spline was fit.
    """

    tck: tuple
    u: np.ndarray
    embedding: np.ndarray
    curve_u: np.ndarray
    curve_embedding: np.ndarray
    curve_spatial: np.ndarray
    alpha: float
    subspace_metric: SubspaceMetric
    degree: int
    smoothing: Optional[float]
    periodic: bool


def _product_metric_embedding(
    sample: BlowUpLevel,
    *,
    alpha: float,
    subspace_metric: SubspaceMetric,
) -> np.ndarray:
    """
    Map R^n x G(k,n) into Euclidean space to preserve the product metric.

    The BlowUpLevel.embedding_vector() is exact for the chordal metric.
    """
    if subspace_metric != "chordal":
        raise NotImplementedError(
            "Only the 'chordal' subspace metric has an explicit Euclidean embedding."
        )
    return sample.embedding_vector(alpha=alpha)


def evaluate_bspline(tck: tuple, u: np.ndarray) -> np.ndarray:
    """
    Evaluate a SciPy B-spline representation and return an (M, D) array.
    """
    u = np.atleast_1d(u)
    coords = splev(u, tck)
    return np.stack(coords, axis=-1)


def project_bspline_to_spatial(
    tck: tuple,
    *,
    u: np.ndarray,
    spatial_dim: int,
) -> np.ndarray:
    """
    Evaluate a B-spline in the embedding and project to the first spatial_dim axes.
    """
    curve = evaluate_bspline(tck, u)
    return curve[:, :spatial_dim]


def fit_bspline_curve(
    sample: BlowUpLevel,
    *,
    alpha: float = 1.0,
    subspace_metric: SubspaceMetric = "chordal",
    degree: int = 3,
    smoothing: Optional[float] = 0.0,
    periodic: bool = False,
    weights: Optional[np.ndarray] = None,
    parameter: Optional[np.ndarray] = None,
    n_eval: int = 400,
) -> BSplineResult:
    """
    Fit a B-spline in the blown-up embedding and project back to spatial space.

    The spline is fit in the Euclidean embedding that matches the product
    metric (for the chordal Grassmann distance). The resulting curve is
    projected to the original spatial coordinates by dropping the subspace
    embedding dimensions.
    """
    if sample.N <= 0:
        raise ValueError("Sample must contain at least one point.")
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    if sample.N <= degree:
        raise ValueError(
            f"Need at least degree+1 points for spline fit (N={sample.N}, degree={degree})."
        )
    if n_eval < 2:
        raise ValueError("n_eval must be >= 2.")
    if weights is not None and len(weights) != sample.N:
        raise ValueError("weights must have the same length as the sample.")
    if parameter is not None and len(parameter) != sample.N:
        raise ValueError("parameter must have the same length as the sample.")

    embedding = _product_metric_embedding(
        sample,
        alpha=alpha,
        subspace_metric=subspace_metric,
    )

    coords = embedding.T  # splprep expects (dim, N)
    tck, u = splprep(
        coords,
        w=weights,
        u=parameter,
        k=degree,
        s=smoothing,
        per=periodic,
    )

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    curve_u = np.linspace(u_min, u_max, n_eval)
    curve_embedding = evaluate_bspline(tck, curve_u)
    curve_spatial = curve_embedding[:, : sample.n_orig]

    return BSplineResult(
        tck=tck,
        u=u,
        embedding=embedding,
        curve_u=curve_u,
        curve_embedding=curve_embedding,
        curve_spatial=curve_spatial,
        alpha=alpha,
        subspace_metric=subspace_metric,
        degree=degree,
        smoothing=smoothing,
        periodic=periodic,
    )


__all__ = [
    "BSplineResult",
    "evaluate_bspline",
    "fit_bspline_curve",
    "project_bspline_to_spatial",
]
