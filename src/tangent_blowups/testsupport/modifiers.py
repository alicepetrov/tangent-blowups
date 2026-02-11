"""
Modifiers for sampled geometric data.

All functions operate on ``Sample`` and return a modified ``Sample``.
By default they return a new object; set ``inplace=True`` to mutate.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .geom_types import Sample
from ..solvers.linalg import normalize_vectors


def _get_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _copy_or_view(arr: Optional[np.ndarray], inplace: bool) -> Optional[np.ndarray]:
    if arr is None:
        return None
    return arr if inplace else np.array(arr, copy=True)


def _new_sample(
    src: Sample,
    *,
    points: np.ndarray,
    tangents: Optional[np.ndarray],
    normals: Optional[np.ndarray],
) -> Sample:
    return Sample(
        points=points,
        params=src.params,
        tangents=tangents,
        normals=normals,
        singular_mask=src.singular_mask,
        singular_indices=src.singular_indices,
    )


def _prepare(
    sample: Sample,
    *,
    inplace: bool,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    points = _copy_or_view(sample.points, inplace)
    tangents = _copy_or_view(sample.tangents, inplace)
    normals = _copy_or_view(sample.normals, inplace)
    return points, tangents, normals


def add_point_noise(
    sample: Sample,
    sigma: float,
    *,
    rng: Optional[np.random.Generator] = None,
    inplace: bool = False,
) -> Sample:
    """
    Add Gaussian noise N(0, sigma^2) to point coordinates.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    rng = _get_rng(rng)
    points, tangents, normals = _prepare(sample, inplace=inplace)
    points += rng.normal(loc=0.0, scale=sigma, size=points.shape)

    if inplace:
        sample.points = points
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


def jitter_points(
    sample: Sample,
    max_jitter: float,
    *,
    rng: Optional[np.random.Generator] = None,
    inplace: bool = False,
) -> Sample:
    """
    Add bounded uniform jitter to points in [-max_jitter, max_jitter].
    """
    if max_jitter < 0.0:
        raise ValueError("max_jitter must be non-negative")

    rng = _get_rng(rng)
    points, tangents, normals = _prepare(sample, inplace=inplace)
    points += rng.uniform(low=-max_jitter, high=max_jitter, size=points.shape)

    if inplace:
        sample.points = points
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


def jitter_tangents(
    sample: Sample,
    sigma: float,
    *,
    rng: Optional[np.random.Generator] = None,
    renormalize: bool = True,
    inplace: bool = False,
) -> Sample:
    """
    Add Gaussian jitter to tangent vectors, optionally re-normalizing.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    points, tangents, normals = _prepare(sample, inplace=inplace)
    if tangents is None:
        return sample if inplace else _new_sample(sample, points=points, tangents=None, normals=normals)

    rng = _get_rng(rng)
    tangents += rng.normal(loc=0.0, scale=sigma, size=tangents.shape)
    if renormalize:
        tangents = normalize_vectors(tangents)

    if inplace:
        sample.tangents = tangents
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


def jitter_normals(
    sample: Sample,
    sigma: float,
    *,
    rng: Optional[np.random.Generator] = None,
    renormalize: bool = True,
    inplace: bool = False,
) -> Sample:
    """
    Add Gaussian jitter to normal vectors, optionally re-normalizing.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    points, tangents, normals = _prepare(sample, inplace=inplace)
    if normals is None:
        return sample if inplace else _new_sample(sample, points=points, tangents=tangents, normals=None)

    rng = _get_rng(rng)
    normals += rng.normal(loc=0.0, scale=sigma, size=normals.shape)
    if renormalize:
        normals = normalize_vectors(normals)

    if inplace:
        sample.normals = normals
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


def _flip_vectors(
    vectors: Optional[np.ndarray],
    *,
    flip_probability: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if vectors is None:
        return None

    if not (0.0 <= flip_probability <= 1.0):
        raise ValueError("flip_probability must be in [0, 1]")

    signs = np.where(
        rng.random(size=vectors.shape[:-1]) < flip_probability,
        -1.0,
        1.0,
    )
    return vectors * signs[..., None]


def flip_tangent_orientations(
    sample: Sample,
    *,
    flip_probability: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    inplace: bool = False,
) -> Sample:
    """
    Randomly flip tangent orientation (v -> -v) with given probability.
    """
    rng = _get_rng(rng)
    points, tangents, normals = _prepare(sample, inplace=inplace)
    tangents = _flip_vectors(tangents, flip_probability=flip_probability, rng=rng)

    if inplace:
        sample.tangents = tangents
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


def flip_normal_orientations(
    sample: Sample,
    *,
    flip_probability: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    inplace: bool = False,
) -> Sample:
    """
    Randomly flip normal orientation (n -> -n) with given probability.
    """
    rng = _get_rng(rng)
    points, tangents, normals = _prepare(sample, inplace=inplace)
    normals = _flip_vectors(normals, flip_probability=flip_probability, rng=rng)

    if inplace:
        sample.normals = normals
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


def flip_orientations(
    sample: Sample,
    *,
    flip_probability: float = 0.5,
    coupled: bool = False,
    rng: Optional[np.random.Generator] = None,
    inplace: bool = False,
) -> Sample:
    """
    Flip tangent and normal orientations.

    If coupled=True, tangent and normal at the same sample are flipped together.
    """
    if not (0.0 <= flip_probability <= 1.0):
        raise ValueError("flip_probability must be in [0, 1]")

    rng = _get_rng(rng)
    points, tangents, normals = _prepare(sample, inplace=inplace)

    if not coupled:
        tangents = _flip_vectors(tangents, flip_probability=flip_probability, rng=rng)
        normals = _flip_vectors(normals, flip_probability=flip_probability, rng=rng)
    else:
        ref = tangents if tangents is not None else normals
        if ref is not None:
            signs = np.where(
                rng.random(size=ref.shape[:-1]) < flip_probability,
                -1.0,
                1.0,
            )
            if tangents is not None:
                tangents = tangents * signs[..., None]
            if normals is not None:
                normals = normals * signs[..., None]

    if inplace:
        sample.tangents = tangents
        sample.normals = normals
        return sample
    return _new_sample(sample, points=points, tangents=tangents, normals=normals)


__all__ = [
    "add_point_noise",
    "jitter_points",
    "jitter_tangents",
    "jitter_normals",
    "flip_tangent_orientations",
    "flip_normal_orientations",
    "flip_orientations",
]
