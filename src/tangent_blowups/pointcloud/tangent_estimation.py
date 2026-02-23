"""
Local PCA Tangent/Normal Estimation
----------------------------------
Estimate tangents or normals on a point cloud using local PCA neighborhoods.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.spatial import cKDTree

from ..solvers.linalg import normalize_vectors
from ..testsupport.geom_types import Sample


def _as_points(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"points must be 2D (N, d). Got {P.shape}.")
    if not np.isfinite(P).all():
        raise ValueError("points must be finite.")
    return P


def _coerce_sample(
    points_or_sample: np.ndarray | Sample,
) -> tuple[np.ndarray, Optional[Sample]]:
    if isinstance(points_or_sample, Sample):
        pts = _as_points(points_or_sample.points)
        return pts, points_or_sample
    return _as_points(points_or_sample), None


def _neighbors(
    points: np.ndarray,
    *,
    k: int | None,
    radius: float | None,
    include_self: bool,
    workers: int,
) -> list[np.ndarray]:
    n = points.shape[0]
    if n == 0:
        return [np.array([], dtype=int) for _ in range(0)]

    tree = cKDTree(points)

    if radius is not None:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        neighbors = tree.query_ball_point(points, r=float(radius), workers=workers)
        out: list[np.ndarray] = []
        for i, js in enumerate(neighbors):
            if not include_self:
                js = [j for j in js if j != i]
            out.append(np.asarray(js, dtype=int))
        return out

    if k is None or k <= 0:
        raise ValueError("k must be positive when radius is not provided.")

    if include_self:
        k_eff = min(k, n)
        dists, idxs = tree.query(points, k=k_eff, workers=workers)
        if k_eff == 1:
            idxs = idxs[:, None]
        return [idxs[i].astype(int, copy=False) for i in range(n)]

    k_eff = min(k + 1, n)
    dists, idxs = tree.query(points, k=k_eff, workers=workers)
    if k_eff == 1:
        return [np.array([], dtype=int) for _ in range(n)]
    if idxs.ndim == 1:
        idxs = idxs[:, None]

    out = []
    for i in range(n):
        js = idxs[i]
        js = js[js != i]
        if js.size > k:
            js = js[:k]
        out.append(js.astype(int, copy=False))
    return out


def _local_pca(
    points: np.ndarray,
    *,
    k: int | None,
    radius: float | None,
    center: Literal["mean", "point"],
    include_self: bool,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, dim = points.shape
    evals = np.zeros((n, dim), dtype=float)
    evecs = np.zeros((n, dim, dim), dtype=float)

    neighbors = _neighbors(
        points,
        k=k,
        radius=radius,
        include_self=include_self,
        workers=workers,
    )

    for i, js in enumerate(neighbors):
        if js.size < 2:
            continue

        local = points[js]
        if center == "mean":
            local = local - local.mean(axis=0, keepdims=True)
        elif center == "point":
            local = local - points[i]
        else:
            raise ValueError(f"Unknown center mode '{center}'.")

        cov = (local.T @ local) / max(local.shape[0], 1)
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            vals = np.zeros(dim, dtype=float)
            vecs = np.zeros((dim, dim), dtype=float)

        evals[i] = vals
        evecs[i] = vecs

    return evals, evecs


def _update_sample(
    sample: Sample,
    *,
    tangents: np.ndarray | None,
    normals: np.ndarray | None,
) -> Sample:
    return Sample(
        points=sample.points,
        params=sample.params,
        tangents=tangents if tangents is not None else sample.tangents,
        normals=normals if normals is not None else sample.normals,
        singular_mask=sample.singular_mask,
        singular_indices=sample.singular_indices,
    )


def estimate_normals_pca(
    points_or_sample: np.ndarray | Sample,
    *,
    k: int | None = 16,
    radius: float | None = None,
    center: Literal["mean", "point"] = "mean",
    include_self: bool = True,
    workers: int = -1,
) -> np.ndarray | Sample:
    """
    Estimate normals using local PCA (smallest eigenvector).

    If a Sample is provided, returns a new Sample with normals filled in.
    If a raw point array is provided, returns normals as (N, d).
    """
    points, sample = _coerce_sample(points_or_sample)
    _, evecs = _local_pca(
        points,
        k=k,
        radius=radius,
        center=center,
        include_self=include_self,
        workers=workers,
    )
    normals = normalize_vectors(evecs[:, :, 0])
    if sample is None:
        return normals
    return _update_sample(sample, tangents=None, normals=normals)


def estimate_tangents_pca(
    points_or_sample: np.ndarray | Sample,
    *,
    k: int | None = 16,
    radius: float | None = None,
    tangent_dim: int | None = None,
    center: Literal["mean", "point"] = "mean",
    include_self: bool = True,
    workers: int = -1,
) -> np.ndarray | Sample:
    """
    Estimate tangents using local PCA (largest eigenvectors).

    If a Sample is provided, returns a new Sample with tangents filled in.
    If a raw point array is provided, returns tangents as (N, d) for
    tangent_dim=1 or (N, d, tangent_dim) otherwise.
    """
    points, sample = _coerce_sample(points_or_sample)
    _, evecs = _local_pca(
        points,
        k=k,
        radius=radius,
        center=center,
        include_self=include_self,
        workers=workers,
    )

    dim = points.shape[1]
    if tangent_dim is None:
        tangent_dim = max(1, dim - 1)
    if tangent_dim <= 0 or tangent_dim > dim:
        raise ValueError("tangent_dim must be between 1 and point dimension.")

    tangents = evecs[:, :, -tangent_dim:]
    if tangent_dim == 1:
        tangents = tangents[:, :, 0]

    if sample is None:
        return tangents
    return _update_sample(sample, tangents=tangents, normals=None)


def estimate_normals_and_tangents_pca(
    points_or_sample: np.ndarray | Sample,
    *,
    k: int | None = 16,
    radius: float | None = None,
    tangent_dim: int | None = None,
    center: Literal["mean", "point"] = "mean",
    include_self: bool = True,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray] | Sample:
    """
    Estimate both normals and tangents in one PCA pass.

    If a Sample is provided, returns a new Sample with normals and tangents.
    If a raw point array is provided, returns (normals, tangents).
    """
    points, sample = _coerce_sample(points_or_sample)
    _, evecs = _local_pca(
        points,
        k=k,
        radius=radius,
        center=center,
        include_self=include_self,
        workers=workers,
    )

    dim = points.shape[1]
    if tangent_dim is None:
        tangent_dim = max(1, dim - 1)
    if tangent_dim <= 0 or tangent_dim > dim:
        raise ValueError("tangent_dim must be between 1 and point dimension.")

    normals = normalize_vectors(evecs[:, :, 0])
    tangents = evecs[:, :, -tangent_dim:]
    if tangent_dim == 1:
        tangents = tangents[:, :, 0]

    if sample is None:
        return normals, tangents
    return _update_sample(sample, tangents=tangents, normals=normals)


__all__ = [
    "estimate_normals_pca",
    "estimate_tangents_pca",
    "estimate_normals_and_tangents_pca",
]
