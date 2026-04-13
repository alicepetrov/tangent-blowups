"""
Point Cloud Spectral Clustering Example
---------------------------------------
Compare k-means, DBSCAN, and HDBSCAN on spectral embeddings of a point cloud.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.cluster import HDBSCAN, DBSCAN, KMeans

from tangent_blowups.clustering import spectral_embedding
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
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

def _thingi10k_pointcloud_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "thingi10k_pointcloud"


def _resolve_pointcloud_path(path: str | Path | None) -> Path:
    if path is None:
        return _default_snowflake_path()

    candidate = Path(path)
    if candidate.exists():
        return candidate

    if candidate.suffix == "":
        candidate = candidate.with_suffix(".npz")
    return _thingi10k_pointcloud_root() / candidate


def load_pointcloud_points_and_normals(
    path: str | Path | None = None,
    *,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a point cloud and return (points, normals).
    """
    path = _resolve_pointcloud_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError("Point cloud is missing normals.")

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
    parser = argparse.ArgumentParser(
        description="Spectral clustering on a Thingi10k point cloud."
    )
    parser.add_argument(
        "--pointcloud",
        type=Path,
        default=None,
        help=(
            "Point cloud file name or path. If a relative name is provided, it is "
            "resolved under data/thingi10k_pointcloud."
        ),
    )
    parser.add_argument(
        "--num-levels", type=int, default=1,
        help="Number of blow-up levels to compute (default: 1, try 2 for curvature).",
    )
    parser.add_argument(
        "--kernel", type=str, default="product",
        choices=["lifted", "product"],
        help="Kernel family for the lifted Laplacian (default: product).",
    )
    args = parser.parse_args()

    data_path = _resolve_pointcloud_path(args.pointcloud)
    cloud_label = data_path.stem
    num_levels = args.num_levels
    kernel = args.kernel
    n_clusters = 20
    k = 20
    alpha = 1.0
    lam = 0.0
    normalized = True
    random_state = 7
    dbscan_eps = 0.2
    dbscan_min_samples = 10
    hdbscan_min_cluster_size = 25

    points, normals = load_pointcloud_points_and_normals(
        data_path,
        normal_eps=1e-8,
    )
    print(f"Loaded {points.shape[0]} points from {data_path}")

    # ------------------------------------------------------------------
    # Helper: cluster a Laplacian with all three methods
    # ------------------------------------------------------------------
    def _cluster_laplacian(L, tag: str):
        """Return (labels_km, labels_db, labels_hdb, evals)."""
        evals, embedding = spectral_embedding(L, n_components=n_clusters)
        labels_km = KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(embedding)
        labels_db = DBSCAN(
            eps=dbscan_eps, min_samples=dbscan_min_samples,
        ).fit_predict(embedding)
        labels_hdb = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
        ).fit_predict(embedding)

        _print_cluster_sizes(f"{tag} (KMeans)", labels_km)
        _print_cluster_count(f"{tag} (KMeans)", labels_km)
        _print_cluster_sizes(f"{tag} (DBSCAN)", labels_db)
        _print_cluster_count(f"{tag} (DBSCAN)", labels_db)
        _print_cluster_sizes(f"{tag} (HDBSCAN)", labels_hdb)
        _print_cluster_count(f"{tag} (HDBSCAN)", labels_hdb)
        return labels_km, labels_db, labels_hdb, evals

    # ------------------------------------------------------------------
    # Euclidean baseline: alpha=0 lift -> pure spatial self-tuning product
    # ------------------------------------------------------------------
    l0_euc = BlowUpLevel.from_normals(points, normals).lift(
        k=k, alpha=0.0, lam=lam,
    )
    L_euc, _, _ = lifted_laplacian(
        l0_euc, kernel="product", self_tuning=True, k=k, normalized=normalized,
    )
    euc_km, euc_db, euc_hdb, evals_euc = _cluster_laplacian(L_euc, "Euclidean")

    # ------------------------------------------------------------------
    # Kernel Laplacian on iterated blow-up levels
    # ------------------------------------------------------------------
    # normals (N,3) -> tangent frames (N,3,2) via Grassmannian duality
    tangent_frames = BlowUpLevel.from_normals(points, normals).frame
    level = BlowUpLevel.from_point_tangents(points, tangent_frames)

    # level_results[lvl] = (labels_km, labels_db, labels_hdb, evals)
    level_results: dict[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    for lvl in range(1, num_levels + 1):
        level = level.lift(k=k, alpha=alpha, lam=lam)
        print(f"Level-{lvl}: embedded dim = {level.D}")

        L, _, _ = lifted_laplacian(
            level, kernel=kernel, self_tuning=True, k=k, normalized=normalized,
        )
        level_results[lvl] = _cluster_laplacian(L, f"Level-{lvl} {kernel}")

    # ------------------------------------------------------------------
    # Plot: rows = Euclidean + levels,  cols = KMeans | DBSCAN | HDBSCAN
    # ------------------------------------------------------------------
    n_rows = 1 + num_levels
    n_cols = 3
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    def _add_row(row: int, labels_km, labels_db, labels_hdb, row_label: str):
        base = row * n_cols
        ax = fig.add_subplot(n_rows, n_cols, base + 1, projection="3d")
        _plot_clusters(ax, points, labels_km, f"{row_label} + KMeans")
        ax = fig.add_subplot(n_rows, n_cols, base + 2, projection="3d")
        _plot_clusters(ax, points, labels_db,
                       f"{row_label} + DBSCAN (eps={dbscan_eps})")
        ax = fig.add_subplot(n_rows, n_cols, base + 3, projection="3d")
        _plot_clusters(ax, points, labels_hdb,
                       f"{row_label} + HDBSCAN (min={hdbscan_min_cluster_size})")

    _add_row(0, euc_km, euc_db, euc_hdb, "Euclidean")
    for lvl in range(1, num_levels + 1):
        km, db, hdb, _ = level_results[lvl]
        _add_row(lvl, km, db, hdb, f"Level-{lvl} {kernel}")

    fig.suptitle(
        f"{cloud_label} Spectral Clustering  "
        f"(alpha={alpha}, k={k}, kernel={kernel})"
    )
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Print eigenvalue spectra
    # ------------------------------------------------------------------
    print(
        "First eigenvalues (Euclidean):",
        np.array2string(evals_euc, precision=5, floatmode="fixed"),
    )
    for lvl in range(1, num_levels + 1):
        _, _, _, evals_lvl = level_results[lvl]
        print(
            f"First eigenvalues (Level-{lvl}):",
            np.array2string(evals_lvl, precision=5, floatmode="fixed"),
        )


if __name__ == "__main__":
    main()
