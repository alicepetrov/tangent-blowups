"""
Neighborhood construction utilities for point clouds.

Provides kNN and radius graphs using a KDTree backend.
Returns edge lists (rows, cols, dist2) suitable for sparse adjacency matrices.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


def _as_points(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"points must be 2D (N, d). Got {P.shape}.")
    if not np.isfinite(P).all():
        raise ValueError("points must be finite.")
    return P


def _empty_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0,), dtype=int),
        np.zeros((0,), dtype=int),
        np.zeros((0,), dtype=float),
    )


def knn_edges(
    points: np.ndarray,
    k: int,
    *,
    include_self: bool = False,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a kNN edge list for points.

    Args:
        points: (N, d) array.
        k: number of neighbors per point.
        include_self: include i -> i edges if True.
        workers: cKDTree parallelism (workers=-1 uses all cores).

    Returns:
        rows, cols, dist2 (all 1D arrays).
    """
    P = _as_points(points)
    n = P.shape[0]
    if n == 0 or k <= 0:
        return _empty_edges()

    tree = cKDTree(P)

    if include_self:
        k_eff = min(k, n)
        dists, idxs = tree.query(P, k=k_eff, workers=workers)
        if k_eff == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        rows = np.repeat(np.arange(n, dtype=int), k_eff)
        cols = idxs.reshape(-1).astype(int, copy=False)
        dist2 = (dists.reshape(-1) ** 2).astype(float, copy=False)
        return rows, cols, dist2

    k_eff = min(k + 1, n)
    dists, idxs = tree.query(P, k=k_eff, workers=workers)
    if k_eff == 1:
        return _empty_edges()
    if idxs.ndim == 1:
        idxs = idxs[:, None]
        dists = dists[:, None]

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    dist2_list: list[np.ndarray] = []

    for i in range(n):
        js = idxs[i]
        ds = dists[i]
        mask = js != i
        js = js[mask]
        ds = ds[mask]
        if js.size > k:
            js = js[:k]
            ds = ds[:k]
        if js.size:
            rows_list.append(np.full(js.size, i, dtype=int))
            cols_list.append(js.astype(int, copy=False))
            dist2_list.append((ds * ds).astype(float, copy=False))

    if not rows_list:
        return _empty_edges()

    return (
        np.concatenate(rows_list),
        np.concatenate(cols_list),
        np.concatenate(dist2_list),
    )


def radius_edges(
    points: np.ndarray,
    radius: float,
    *,
    include_self: bool = False,
    workers: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a radius-neighborhood edge list for points.

    Args:
        points: (N, d) array.
        radius: neighborhood radius.
        include_self: include i -> i edges if True.
        workers: cKDTree parallelism (workers=-1 uses all cores).

    Returns:
        rows, cols, dist2 (all 1D arrays).
    """
    P = _as_points(points)
    n = P.shape[0]
    if n == 0 or radius <= 0.0:
        return _empty_edges()

    tree = cKDTree(P)
    neighbors: Iterable[list[int]] = tree.query_ball_point(
        P, r=radius, workers=workers
    )

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    dist2_list: list[np.ndarray] = []

    for i, js in enumerate(neighbors):
        if not include_self:
            js = [j for j in js if j != i]
        if not js:
            continue
        js_arr = np.asarray(js, dtype=int)
        diff = P[js_arr] - P[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        rows_list.append(np.full(js_arr.size, i, dtype=int))
        cols_list.append(js_arr)
        dist2_list.append(dist2.astype(float, copy=False))

    if not rows_list:
        return _empty_edges()

    return (
        np.concatenate(rows_list),
        np.concatenate(cols_list),
        np.concatenate(dist2_list),
    )


__all__ = ["knn_edges", "radius_edges"]
