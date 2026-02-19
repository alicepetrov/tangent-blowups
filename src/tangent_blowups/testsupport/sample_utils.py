from __future__ import annotations

from typing import Optional

import numpy as np

from .geom_types import Sample
from ..solvers.linalg import normalize_vectors


def _reshape_optional(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def flatten_sample(s: Sample) -> Sample:
    """
    Flatten a Sample produced from grid sampling into a point-cloud Sample.
    """
    points = np.asarray(s.points, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])

    tangents = _reshape_optional(s.tangents)
    normals = _reshape_optional(s.normals)

    mask = s.singular_mask
    if mask is not None and mask.ndim > 1:
        mask = mask.reshape(-1)

    params = s.params
    if isinstance(params, tuple):
        params = tuple(np.asarray(p).reshape(-1) for p in params)
    else:
        arr = np.asarray(params)
        params = arr.reshape(-1) if arr.ndim > 1 else params

    return Sample(
        points=points,
        params=params,
        tangents=tangents,
        normals=normals,
        singular_mask=mask,
        singular_indices=s.singular_indices,
    )


def _resolve_singular_candidates(s: Sample) -> Optional[np.ndarray]:
    if s.singular_indices is not None:
        return np.asarray(s.singular_indices, dtype=int)
    if s.singular_mask is not None:
        return np.flatnonzero(s.singular_mask)
    return None


def _filter_singular_indices(s: Sample, valid_mask: np.ndarray) -> np.ndarray:
    candidates = _resolve_singular_candidates(s)
    if candidates is None or candidates.size == 0:
        return np.array([], dtype=int)

    valid_indices = np.flatnonzero(valid_mask)
    singular_in_valid = np.intersect1d(valid_indices, candidates, assume_unique=False)
    return np.searchsorted(valid_indices, singular_in_valid)


def filter_sample_vectors(
    s: Sample,
    vectors: np.ndarray,
    *,
    eps: float = 1e-8,
    name: str = "vectors",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter invalid/degenerate vectors and map singular indices to the filtered set.

    Returns:
        points, vectors, singular_indices
    """
    points = np.asarray(s.points, dtype=float)
    vectors = np.asarray(vectors, dtype=float)

    if points.ndim != 2 or vectors.ndim != 2:
        raise ValueError("points and vectors must be 2D arrays after flattening.")
    if points.shape[0] != vectors.shape[0]:
        raise ValueError("points and vectors must have the same length.")

    valid = np.isfinite(points).all(axis=1) & np.isfinite(vectors).all(axis=1)
    norms = np.linalg.norm(vectors, axis=1)
    valid &= norms > eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate {name}.")

    singular_filtered = _filter_singular_indices(s, valid)
    points = points[valid]
    vectors = normalize_vectors(vectors[valid])
    return points, vectors, singular_filtered


def pick_query_indices_with_singular(
    n_points: int,
    singular_indices: np.ndarray,
    n_queries: int,
    seed: int,
) -> list[int]:
    if n_points <= 0 or n_queries <= 0:
        return []

    rng = np.random.default_rng(seed)
    chosen: list[int] = []

    if singular_indices is not None and singular_indices.size > 0:
        singular_idx = int(rng.choice(singular_indices))
        chosen.append(singular_idx)

    if n_queries <= len(chosen):
        return chosen[:n_queries]

    candidates = np.setdiff1d(np.arange(n_points), np.array(chosen, dtype=int), assume_unique=False)
    extra = rng.choice(candidates, size=n_queries - len(chosen), replace=False).tolist()
    return chosen + extra


__all__ = [
    "flatten_sample",
    "filter_sample_vectors",
    "pick_query_indices_with_singular",
]
