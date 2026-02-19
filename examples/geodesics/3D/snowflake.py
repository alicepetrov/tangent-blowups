"""
Snowflake Lifted Heat Method Example
------------------------------------
Compute lifted heat-method distances on the snowflake point cloud.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components

from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud import lifted_heat_method
from tangent_blowups.pointcloud.laplacian import lifted_pointcloud_laplacian
from tangent_blowups.solvers.linalg import normalize_vectors


def _flatten(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _default_snowflake_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "thingi10k_pointcloud" / "snowflake.npz"


def load_snowflake_points_and_normals(
    path: str | Path | None = None,
    *,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        path = _default_snowflake_path()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snowflake data not found: {path}")

    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError("Snowflake point cloud is missing normals.")

    points = _flatten(sample.points)
    normals = _flatten(sample.normals)

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
    if points.size == 0:
        raise ValueError("No points available to pick a source.")
    center = np.mean(points, axis=0)
    return int(np.argmin(np.linalg.norm(points - center, axis=1)) + 20)


def _set_3d_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def _resolve_connected_mask(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    source_index: int,
    k: int,
    alpha: float,
    laplacian_normalized: bool,
) -> np.ndarray:
    _, W, _ = lifted_pointcloud_laplacian(
        points,
        normals,
        k=k,
        alpha=alpha,
        normalized=laplacian_normalized,
        return_parts=True,
    )
    _, labels = connected_components(W, directed=False, connection="weak")
    return labels == labels[source_index]


def main():
    data_path = _default_snowflake_path()

    k = 10
    alpha = 30.0
    laplacian_normalized = False
    t = None
    t_scale = 1.0

    points, normals = load_snowflake_points_and_normals(data_path)
    print(f"Loaded {points.shape[0]} points from {data_path}")

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

    mask = _resolve_connected_mask(
        points,
        normals,
        source_index=source_index,
        k=k,
        alpha=alpha,
        laplacian_normalized=laplacian_normalized,
    )
    pts_plot = points[mask]
    dist_plot = dist[mask]
    pts_other = points[~mask]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    if pts_other.size:
        ax.scatter(
            pts_other[:, 0],
            pts_other[:, 1],
            pts_other[:, 2],
            s=10.0,
            c="lightgray",
            alpha=0.35,
        )

    sc = ax.scatter(
        pts_plot[:, 0],
        pts_plot[:, 1],
        pts_plot[:, 2],
        c=dist_plot,
        s=10.0,
        cmap="magma",
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
    ax.set_title("Lifted Heat-Method Distances (Snowflake)")
    _set_3d_equal_aspect(ax)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
