"""
Intersecting Planes Lifted Heat Method Example
----------------------------------------------
Compute lifted heat-method distances on two intersecting planes in 3D.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components

from tangent_blowups.pointcloud import lifted_heat_method
from tangent_blowups.pointcloud.laplacian import lifted_pointcloud_laplacian
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import RandomSurface, plane_cross, sample


def _flatten(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def build_plane_cross_points_and_normals(
    *,
    nu: int = 120,
    nv: int = 120,
    u_bounds: Tuple[float, float] = (0.0, 2.0),
    v_bounds: Tuple[float, float] = (0.0, 1.0),
    scale: float = 1.0,
    jitter: float = 0.0,
    seed: int = 7,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    surface = plane_cross(scale=scale)
    rng = np.random.default_rng(seed)
    strategy = RandomSurface(
        n=nu * nv,
        u_bounds=u_bounds,
        v_bounds=v_bounds,
        rng=rng,
    )
    s = sample(surface, strategy, with_tangents=False, with_normals=True)

    points = _flatten(s.points)
    normals = _flatten(s.normals)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=jitter, size=points.shape)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    norms = np.linalg.norm(normals, axis=1)
    valid &= norms > normal_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate normals.")

    points = points[valid]
    normals = normalize_vectors(normals[valid])
    return points, normals


def _pick_source_index(points: np.ndarray) -> int:
    """
    Choose a source point near the intersection line (x-axis).
    """
    if points.size == 0:
        raise ValueError("No points available to pick a source.")
    score = points[:, 1] ** 2 + points[:, 2] ** 2 + 0.1 * (points[:, 0] ** 2)
    return int(np.argmin(score))  # Offset to avoid the exact intersection line


def _set_3d_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def main():
    nu = 120
    nv = 120
    scale = 1.0
    jitter = 0.0

    k = 20
    alpha = 1.0
    laplacian_normalized = False
    t = None
    t_scale = 1.0

    points, normals = build_plane_cross_points_and_normals(
        nu=nu,
        nv=nv,
        u_bounds=(0.0, 2.0),
        v_bounds=(0.0, 1.0),
        scale=scale,
        jitter=jitter,
        seed=7,
    )

    source_index = _pick_source_index(points)
    print(f"Using source index {source_index}")

    dist = lifted_heat_method(
        points,
        normals,
        source_index=source_index,
        k=k,
        alpha=alpha,
        laplacian_normalized=laplacian_normalized,
        t=t,
        t_scale=t_scale,
    )

    dist = dist - dist[source_index]
    dist = np.maximum(dist, 0.0)

    L, W, _ = lifted_pointcloud_laplacian(
        points,
        normals,
        k=k,
        alpha=alpha,
        normalized=laplacian_normalized,
        return_parts=True,
    )
    n_comp, labels = connected_components(W, directed=False, connection="weak")
    source_comp = labels[source_index]
    mask = labels == source_comp
    pts_plot = points[mask]
    dist_plot = dist[mask]

    pts_other = points[~mask]

    cmap = plt.get_cmap("magma").copy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    if pts_other.size:
        ax.scatter(
            pts_other[:, 0],
            pts_other[:, 1],
            pts_other[:, 2],
            s=8.0,
            c="lightgray",
            alpha=0.35,
        )
    sc = ax.scatter(
        pts_plot[:, 0],
        pts_plot[:, 1],
        pts_plot[:, 2],
        c=dist_plot,
        s=8.0,
        cmap=cmap,
        alpha=0.9,
    )
    ax.scatter(
        [points[source_index, 0]],
        [points[source_index, 1]],
        [points[source_index, 2]],
        s=80,
        c="none",
        edgecolors="k",
        linewidths=1.5,
    )
    ax.set_title("Lifted Heat-Method Distances (Intersecting Planes)")
    _set_3d_equal_aspect(ax)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
