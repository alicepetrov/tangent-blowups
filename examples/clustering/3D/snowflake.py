"""
Snowflake Spectral Clustering Example
-------------------------------------
Compare k-means vs DBSCAN on spectral embeddings of a snowflake point cloud.
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
    labels = np.asarray(labels, dtype=int)
    noise_mask = labels < 0
    cluster_labels = labels[~noise_mask]

    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
    cmap = _cluster_cmap(max(n_clusters, 1))

    ax.set_title(title)

    if cluster_labels.size:
        sc = ax.scatter(
            points[~noise_mask, 0],
            points[~noise_mask, 1],
            points[~noise_mask, 2],
            c=cluster_labels,
            s=10.0,
            cmap=cmap,
            vmin=-0.5,
            vmax=max(n_clusters - 0.5, 0.5),
            alpha=0.9,
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if n_clusters > 0:
            cbar.set_ticks(np.arange(n_clusters))

    if noise_mask.any():
        ax.scatter(
            points[noise_mask, 0],
            points[noise_mask, 1],
            points[noise_mask, 2],
            s=8.0,
            color="lightgray",
            alpha=0.6,
            label="Noise",
        )
        ax.legend(loc="best")

    _set_3d_equal_aspect(ax)


def _print_cluster_sizes(label: str, labels: np.ndarray):
    if labels.size == 0:
        print(f"{label}: empty labels")
        return
    labels = np.asarray(labels, dtype=int)
    noise = int(np.sum(labels < 0))
    cluster_labels = labels[labels >= 0]
    if cluster_labels.size == 0:
        print(f"{label} cluster sizes: noise={noise}")
        return
    n_clusters = int(cluster_labels.max()) + 1
    counts = np.bincount(cluster_labels, minlength=n_clusters)
    if noise > 0:
        print(f"{label} cluster sizes: {counts.tolist()} (noise={noise})")
    else:
        print(f"{label} cluster sizes: {counts.tolist()}")


def _print_cluster_count(label: str, labels: np.ndarray):
    labels = np.asarray(labels, dtype=int)
    cluster_labels = labels[labels >= 0]
    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
    print(f"{label} clusters found: {n_clusters}")


def main():
    data_path = _default_snowflake_path()
    n_clusters = 62 # We know there are 62 branches in the snowflake, so let's try to recover that.
    k = 15
    alpha = 20.0
    normalized = True
    random_state = 7
    dbscan_eps = 0.2
    dbscan_min_samples = 10

    points, normals = load_snowflake_points_and_normals(
        data_path,
        normal_eps=1e-8,
    )

    print(f"Loaded {points.shape[0]} points from {data_path}")

    labels_euc_kmeans, evals_euc, _, _ = spectral_clustering_pointcloud(
        points,
        n_clusters=n_clusters,
        k=k,
        laplacian_normalized=normalized,
        random_state=random_state,
    )
    labels_euc_dbscan, _, _, _ = spectral_clustering_pointcloud(
        points,
        n_clusters=n_clusters,
        k=k,
        laplacian_normalized=normalized,
        cluster_method="dbscan",
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    labels_lift_kmeans, evals_lift, _, _ = spectral_clustering_lifted(
        points,
        normals,
        n_clusters=n_clusters,
        k=k,
        alpha=alpha,
        laplacian_normalized=normalized,
        random_state=random_state,
    )
    labels_lift_dbscan, _, _, _ = spectral_clustering_lifted(
        points,
        normals,
        n_clusters=n_clusters,
        k=k,
        alpha=alpha,
        laplacian_normalized=normalized,
        cluster_method="dbscan",
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    _print_cluster_sizes("Euclidean (KMeans)", labels_euc_kmeans)
    _print_cluster_count("Euclidean (KMeans)", labels_euc_kmeans)
    _print_cluster_sizes("Euclidean (DBSCAN)", labels_euc_dbscan)
    _print_cluster_count("Euclidean (DBSCAN)", labels_euc_dbscan)
    _print_cluster_sizes("Lifted (KMeans)", labels_lift_kmeans)
    _print_cluster_count("Lifted (KMeans)", labels_lift_kmeans)
    _print_cluster_sizes("Lifted (DBSCAN)", labels_lift_dbscan)
    _print_cluster_count("Lifted (DBSCAN)", labels_lift_dbscan)

    fig = plt.figure(figsize=(12, 10))
    ax_euc_kmeans = fig.add_subplot(2, 2, 1, projection="3d")
    ax_euc_dbscan = fig.add_subplot(2, 2, 2, projection="3d")
    ax_lift_kmeans = fig.add_subplot(2, 2, 3, projection="3d")
    ax_lift_dbscan = fig.add_subplot(2, 2, 4, projection="3d")

    _plot_clusters(
        ax_euc_kmeans,
        points,
        labels_euc_kmeans,
        "Euclidean + KMeans",
    )
    _plot_clusters(
        ax_euc_dbscan,
        points,
        labels_euc_dbscan,
        f"Euclidean + DBSCAN (eps={dbscan_eps})",
    )
    _plot_clusters(
        ax_lift_kmeans,
        points,
        labels_lift_kmeans,
        "Lifted + KMeans",
    )
    _plot_clusters(
        ax_lift_dbscan,
        points,
        labels_lift_dbscan,
        f"Lifted + DBSCAN (eps={dbscan_eps})",
    )

    fig.suptitle("Snowflake Spectral Clustering: KMeans vs DBSCAN")
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
