"""
Test-support front door for synthetic geometry generation and perturbation.
"""

from .geom_types import GroundTruth, ParametricCurve, ParametricSurface, Sample
from .samplers import (
    SamplingStrategy,
    sample,
    UniformCurve,
    RandomCurve,
    UniformSurface,
    RandomSurface,
    JitteredCurve,
    ChebyshevCurve,
    JitteredSurface,
    ChebyshevGrid,
)
from .examples_2D import circle, figure8, lissajous
from .examples_3D import (
    helix,
    trefoil_knot,
    space_lissajous,
    figure8_space,
    whitney_umbrella,
    monkey_saddle,
    cone,
)
from .modifiers import (
    add_point_noise,
    jitter_points,
    jitter_tangents,
    jitter_normals,
    flip_tangent_orientations,
    flip_normal_orientations,
    flip_orientations,
)

__all__ = [
    "GroundTruth",
    "ParametricCurve",
    "ParametricSurface",
    "Sample",
    "SamplingStrategy",
    "sample",
    "UniformCurve",
    "RandomCurve",
    "UniformSurface",
    "RandomSurface",
    "JitteredCurve",
    "ChebyshevCurve",
    "JitteredSurface",
    "ChebyshevGrid",
    "circle",
    "figure8",
    "lissajous",
    "helix",
    "trefoil_knot",
    "space_lissajous",
    "figure8_space",
    "whitney_umbrella",
    "monkey_saddle",
    "cone",
    "add_point_noise",
    "jitter_points",
    "jitter_tangents",
    "jitter_normals",
    "flip_tangent_orientations",
    "flip_normal_orientations",
    "flip_orientations",
]
