"""
Snowflake Spectral Clustering Example
-------------------------------------
Compare Euclidean vs lifted spectral clustering on a snowflake point cloud.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from tangent_blowups.clustering import (
    spectral_clustering_pointcloud,
    spectral_clustering_lifted,
)
from tangent_blowups.io.load import load_pointcloud
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
    """
    Load the snowflake point cloud and return (points, normals).
    """
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


def _set_3d_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def _cluster_cmap(n_clusters: int):
    if n_clusters <= 12:
        return plt.get_cmap("Set3", n_clusters)
    if n_clusters <= 20:
        return plt.get_cmap("tab20", n_clusters)
    if n_clusters <= 60:
        colors = np.vstack(
            [
                plt.get_cmap("tab20").colors,
                plt.get_cmap("tab20b").colors,
                plt.get_cmap("tab20c").colors,
            ]
        )
        return ListedColormap(colors[:n_clusters], name="tab60")
    return plt.get_cmap("turbo", n_clusters)


def _plot_clusters(ax, points: np.ndarray, labels: np.ndarray, title: str):
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    cmap = _cluster_cmap(max(n_clusters, 1))
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=labels,
        s=10.0,
        cmap=cmap,
        vmin=-0.5,
        vmax=max(n_clusters - 0.5, 0.5),
        alpha=0.9,
    )
    ax.set_title(title)
    _set_3d_equal_aspect(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if n_clusters > 0:
        cbar.set_ticks(np.arange(n_clusters))


def _print_cluster_sizes(label: str, labels: np.ndarray):
    if labels.size == 0:
        print(f"{label}: empty labels")
        return
    n_clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_clusters)
    print(f"{label} cluster sizes: {counts.tolist()}")


def main():
    data_path = _default_snowflake_path()
    n_clusters = 61 # We know there are 61 branches in the snowflake, so let's try to recover that.
    k = 10
    alpha = 30.0
    normalized = True
    random_state = 7

    points, normals = load_snowflake_points_and_normals(
        data_path,
        normal_eps=1e-8,
    )

    print(f"Loaded {points.shape[0]} points from {data_path}")

    labels_euc, evals_euc, _, _ = spectral_clustering_pointcloud(
        points,
        n_clusters=n_clusters,
        k=k,
        laplacian_normalized=normalized,
        random_state=random_state,
    )
    labels_lift, evals_lift, _, _ = spectral_clustering_lifted(
        points,
        normals,
        n_clusters=n_clusters,
        k=k,
        alpha=alpha,
        laplacian_normalized=normalized,
        random_state=random_state,
    )

    _print_cluster_sizes("Euclidean", labels_euc)
    _print_cluster_sizes("Lifted", labels_lift)

    fig = plt.figure(figsize=(12, 6))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_clusters(ax_left, points, labels_euc, "Euclidean Spectral Clustering")
    _plot_clusters(ax_right, points, labels_lift, "Lifted Spectral Clustering")

    fig.suptitle("Snowflake Spectral Clustering: Euclidean vs Lifted")
    plt.tight_layout()
    plt.show()

    if evals_euc.size and evals_lift.size:
        print(
            "First eigenvalues (Euclidean):",
            np.array2string(evals_euc, precision=5, floatmode="fixed"),
        )
        print(
            "First eigenvalues (Lifted):",
            np.array2string(evals_lift, precision=5, floatmode="fixed"),
        )


if __name__ == "__main__":
    main()
