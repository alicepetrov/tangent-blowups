"""
Geo Faults Rift — Spectral Clustering with Product Kernel + KMeans
---------------------------------------------------------------------
Lift the rift medial axis point cloud into the Chordal-Sasaki space
and cluster into intersecting fault components.

Compares:
  1. Euclidean spectral embedding + KMeans
  2. Self-tuning lifted Laplacian + KMeans
  3. Product-kernel lifted Laplacian + KMeans

The product kernel separates spatial (sigma_x) and angular (sigma_u)
bandwidths, so the tangent-plane factor can be made tight enough to
distinguish sheets with different orientations.

The rift .npz contains tangent frames (eigv_1, eigv_2 from local PCA
on the medial axis), so no normal estimation is needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from tangent_blowups.clustering import spectral_embedding
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _geo_faults_pointcloud_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "geo_faults_pointcloud"


def _load_rift() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load rift medial axis point cloud.

    Returns (points, tangent_frames, normals).
    """
    path = _geo_faults_pointcloud_root() / "rift.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"rift.npz not found at {path}. "
            "Run export_ma_pointclouds.py first."
        )
    data = np.load(path)
    points = np.asarray(data["points"], dtype=np.float64)
    tangents = np.asarray(data["tangents"], dtype=np.float64)  # (N, 3, 2)
    normals = np.asarray(data["normals"], dtype=np.float64)    # (N, 3)
    return points, tangents, normals


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _set_3d_equal_aspect(ax) -> None:
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
    return plt.get_cmap("turbo", n_clusters)


def _plot_clusters(ax, points, labels, title, *, max_points=5000):
    labels = np.asarray(labels, dtype=int)

    # Downsample for faster rendering, preserving label distribution
    if len(labels) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(labels), size=max_points, replace=False)
        points = points[idx]
        labels = labels[idx]

    noise_mask = labels < 0
    cluster_labels = labels[~noise_mask]

    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
    cmap = _cluster_cmap(max(n_clusters, 1))
    ax.set_title(title, fontsize=9)

    if cluster_labels.size:
        sc = ax.scatter(
            points[~noise_mask, 0],
            points[~noise_mask, 1],
            points[~noise_mask, 2],
            c=cluster_labels, s=4.0, cmap=cmap,
            vmin=-0.5, vmax=max(n_clusters - 0.5, 0.5), alpha=0.8,
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if n_clusters > 0:
            cbar.set_ticks(np.arange(n_clusters))

    if noise_mask.any():
        ax.scatter(
            points[noise_mask, 0], points[noise_mask, 1], points[noise_mask, 2],
            s=2.0, color="lightgray", alpha=0.5, label="Noise",
        )
        ax.legend(loc="best")
    _set_3d_equal_aspect(ax)


def _print_cluster_sizes(label, labels):
    labels = np.asarray(labels, dtype=int)
    noise = int(np.sum(labels < 0))
    cluster_labels = labels[labels >= 0]
    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
    counts = np.bincount(cluster_labels, minlength=n_clusters).tolist()
    suffix = f" (noise={noise})" if noise > 0 else ""
    print(f"  {label}: {n_clusters} clusters, sizes={counts}{suffix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_clusters       = 2       # 2 fault planes
    k                = 20
    alpha            = 1.0
    lam = 0.0
    sigma_x          = 8000.0  # spatial bandwidth (~1.5x median k-NN dist)
    sigma_u          = 0.05     # angular bandwidth (same-plane ~0.02 -> w≈1, cross-plane ~1.4 -> w≈0)
    normalized       = True
    random_state     = 7

    n_subsample      = 1000   # subsample for speed; set to None for full data

    points, tangent_frames, normals = _load_rift()
    print(f"Loaded rift: {points.shape[0]} points, tangents {tangent_frames.shape}")

    if n_subsample is not None and points.shape[0] > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=n_subsample, replace=False)
        points = points[idx]
        tangent_frames = tangent_frames[idx]
        normals = normals[idx]
        print(f"Subsampled to {n_subsample} points")

    level = BlowUpLevel.from_point_tangents(points, tangent_frames)

    def _cluster(L: "sparse.csr_matrix") -> tuple[np.ndarray, np.ndarray]:
        evals, emb = spectral_embedding(L, n_components=n_clusters)
        labels = KMeans(
            n_clusters=n_clusters, n_init=10, random_state=random_state,
        ).fit_predict(emb)
        return labels, evals

    # ------------------------------------------------------------------
    # 1. Euclidean (alpha=0 lift): pure spatial self-tuning product kernel
    # ------------------------------------------------------------------
    print("Euclidean spectral embedding (alpha=0)...")
    l_euc = level.lift(k=k, alpha=0.0, lam=lam)
    L_euc, _, _ = lifted_laplacian(
        l_euc, kernel="product", self_tuning=True, k=k, normalized=normalized,
    )
    labels_euc, evals_euc = _cluster(L_euc)
    _print_cluster_sizes("Euclidean KMeans", labels_euc)

    # ------------------------------------------------------------------
    # 2. Self-tuning product Laplacian at level 1
    # ------------------------------------------------------------------
    print("Self-tuning product Laplacian (level 1)...")
    l1 = level.lift(k=k, alpha=alpha, lam=lam)
    print(f"  Embedded dim = {l1.D}")

    L_st, _, _ = lifted_laplacian(
        l1, kernel="product", self_tuning=True, k=k, normalized=normalized,
    )
    labels_st, evals_st = _cluster(L_st)
    _print_cluster_sizes("Self-tuning product KMeans (L1)", labels_st)

    # ------------------------------------------------------------------
    # 3. Fixed-bandwidth product kernel at level 1
    # ------------------------------------------------------------------
    print(f"Fixed-bandwidth product Laplacian (sigma_x={sigma_x}, sigma_u={sigma_u})...")
    L_prod, _, _ = lifted_laplacian(
        l1, kernel="product", self_tuning=False,
        sigma_x=sigma_x, sigma_u=sigma_u, k=k, normalized=normalized,
    )
    labels_prod, evals_prod = _cluster(L_prod)
    _print_cluster_sizes("Fixed product KMeans (L1)", labels_prod)

    # ------------------------------------------------------------------
    # 4. Self-tuning product Laplacian at level 2
    # ------------------------------------------------------------------
    print("Second iterated blow-up...")
    l2 = l1.lift(k=k, alpha=alpha, lam=lam)
    print(f"  Level-2 embedded dim = {l2.D}")

    L_l2, _, _ = lifted_laplacian(
        l2, kernel="product", self_tuning=True, k=k, normalized=normalized,
    )
    labels_l2, evals_l2 = _cluster(L_l2)
    _print_cluster_sizes("Level-2 self-tuning KMeans", labels_l2)

    # ------------------------------------------------------------------
    # Plot: 2 rows x 4 cols
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(22, 10))

    # Row 1: eigenvalue scree plots
    for col, (evals, title, color) in enumerate([
        (evals_euc,  "Eigenvalues (Euclidean)",       "steelblue"),
        (evals_st,   "Eigenvalues (Self-tuning L1)",  "darkorange"),
        (evals_prod, "Eigenvalues (Product L1)",       "forestgreen"),
        (evals_l2,   f"Eigenvalues (Product L2, D={l2.D})", "crimson"),
    ], start=1):
        ax = fig.add_subplot(2, 4, col)
        ax.bar(range(n_clusters), evals, color=color)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Index")
        if col == 1:
            ax.set_ylabel("Eigenvalue")
        ax.set_yscale("symlog", linthresh=1e-14)

    # Row 2: clustering results
    ax1 = fig.add_subplot(2, 4, 5, projection="3d")
    _plot_clusters(ax1, points, labels_euc, "Euclidean spectral + KMeans")

    ax2 = fig.add_subplot(2, 4, 6, projection="3d")
    _plot_clusters(ax2, points, labels_st,
                   f"Self-tuning L1 (alpha={alpha})\nspectral + KMeans")

    ax3 = fig.add_subplot(2, 4, 7, projection="3d")
    _plot_clusters(ax3, points, labels_prod,
                   f"Product L1 (sx={sigma_x}, su={sigma_u})\nspectral + KMeans")

    ax4 = fig.add_subplot(2, 4, 8, projection="3d")
    _plot_clusters(ax4, points, labels_l2,
                   f"Product L2 (D={l2.D})\nspectral + KMeans")

    fig.suptitle(
        f"Rift Medial Axis Clustering (k={k}, n_clusters={n_clusters})"
    )
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Print eigenvalue spectra
    # ------------------------------------------------------------------
    for label, evals in [
        ("Euclidean",     evals_euc),
        ("Self-tuning L1", evals_st),
        ("Product L1",    evals_prod),
        ("Product L2",    evals_l2),
    ]:
        print(f"Eigenvalues ({label}):",
              np.array2string(evals, precision=6, floatmode="fixed"))


if __name__ == "__main__":
    main()
