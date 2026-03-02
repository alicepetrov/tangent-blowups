"""
Cube Spectral Clustering Example
---------------------------------
Compare three approaches to spectral clustering on a cube surface:
  1. Euclidean (position only, self-tuning Laplacian)
  2. Lifted with exact normals (level-1 self-tuning Laplacian)
  3. Lifted with PCA-estimated normals (level-1 self-tuning Laplacian)

The self-tuning lifted Laplacian is built via the iterated blow-up:
  level 1  (embedded = Chordal-Sasaki: positions + scaled projectors):
           k-NN naturally separates points with different face normals,
           and the self-tuning bandwidth adapts to local density.

Six faces of the cube -> n_clusters = 6.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from tangent_blowups.clustering import (
    spectral_clustering_pointcloud,
    spectral_clustering_from_laplacian,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.pointcloud import estimate_normals_pca
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import RandomSurface, cube_surface, sample


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _flatten(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _normals_to_tangent_frames(normals: np.ndarray) -> np.ndarray:
    """
    Convert (N, 3) unit normals to (N, 3, 2) orthonormal tangent frames.

    For each normal n, the tangent frame spans the orthogonal complement of n.
    Construction: pick the canonical axis least aligned to n, project out n,
    normalise to get t1; then t2 = n x t1 (already unit).
    """
    j   = np.argmin(np.abs(normals), axis=1)          # (N,) least-aligned axis index
    e   = np.eye(3)[j]                                  # (N, 3) canonical vectors
    proj = np.einsum("ni,ni->n", e, normals)            # (N,) dot products e·n
    t1  = e - proj[:, None] * normals                   # (N, 3) orthogonal to n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2  = np.cross(normals, t1)                         # (N, 3) already unit
    return np.stack([t1, t2], axis=2)                   # (N, 3, 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_cube_points_and_normals(
    *,
    nu: int = 90,
    nv: int = 90,
    u_bounds: Tuple[float, float] = (0.0, 6.0),
    v_bounds: Tuple[float, float] = (0.0, 1.0),
    scale: float = 1.0,
    jitter: float = 0.0,
    seed: int = 7,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    surface = cube_surface(scale=scale)
    rng = np.random.default_rng(seed)
    strategy = RandomSurface(
        n=nu * nv,
        u_bounds=u_bounds,
        v_bounds=v_bounds,
        rng=rng,
    )
    s = sample(surface, strategy, with_tangents=False, with_normals=True)

    points  = _flatten(s.points)
    normals = _flatten(s.normals)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=jitter, size=points.shape)

    valid  = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    norms  = np.linalg.norm(normals, axis=1)
    valid &= norms > normal_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate normals.")

    points  = points[valid]
    normals = normalize_vectors(normals[valid])
    return points, normals


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
    if n_clusters <= 60:
        colors = np.vstack([
            plt.get_cmap("tab20").colors,
            plt.get_cmap("tab20b").colors,
            plt.get_cmap("tab20c").colors,
        ])
        return ListedColormap(colors[:n_clusters], name="tab60")
    return plt.get_cmap("turbo", n_clusters)


def _plot_clusters(ax, points: np.ndarray, labels: np.ndarray, title: str) -> None:
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    cmap = _cluster_cmap(max(n_clusters, 1))
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=labels, s=6.0, cmap=cmap,
        vmin=-0.5, vmax=max(n_clusters - 0.5, 0.5),
        alpha=0.9,
    )
    ax.set_title(title)
    _set_3d_equal_aspect(ax)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if n_clusters > 0:
        cbar.set_ticks(np.arange(n_clusters))


def _print_cluster_sizes(label: str, labels: np.ndarray) -> None:
    if labels.size == 0:
        print(f"{label}: empty labels")
        return
    n_clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_clusters)
    print(f"{label} cluster sizes: {counts.tolist()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _lifted_clustering(
    points: np.ndarray,
    normals: np.ndarray,
    n_clusters: int,
    *,
    k_blowup: int,
    k_kernel: int,
    alpha: float,
    lam: float,
    normalized: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build BlowUpLevel from normals, lift once, run self-tuning clustering."""
    frames = _normals_to_tangent_frames(normals)
    l0 = BlowUpLevel.from_point_tangents(points, frames)
    l1 = l0.lift(k=k_blowup, alpha=alpha, lam=lam)
    L, _, _ = lifted_laplacian(
        l1, kernel="self_tuning", k=k_kernel, h="local", normalized=normalized,
    )
    labels, evals, _, _ = spectral_clustering_from_laplacian(
        L, n_clusters, random_state=random_state,
    )
    return labels, evals


def main() -> None:
    nu = 90
    nv = 90
    scale = 1.0
    jitter = 0.0

    n_clusters   = 6
    k_blowup     = 20    # k-NN for curvature regression inside lift()
    k_pca        = 20    # k-NN for local PCA normal estimation
    k_kernel     = 20    # k-NN for affinity construction
    alpha        = 1.0
    lam          = 1e-3
    normalized   = True
    random_state = 7

    points, normals = build_cube_points_and_normals(
        nu=nu, nv=nv,
        u_bounds=(0.0, 6.0), v_bounds=(0.0, 1.0),
        scale=scale, jitter=jitter, seed=random_state,
    )
    print(f"Loaded {len(points)} cube surface points.")

    # ------------------------------------------------------------------
    # Euclidean spectral clustering (self-tuning on position only)
    # ------------------------------------------------------------------
    print("Running Euclidean spectral clustering...")
    labels_euc, evals_euc, _, _ = spectral_clustering_pointcloud(
        points,
        n_clusters=n_clusters,
        k=k_kernel,
        h="local",
        laplacian_normalized=normalized,
        random_state=random_state,
    )
    _print_cluster_sizes("Euclidean", labels_euc)

    # ------------------------------------------------------------------
    # Lifted: exact normals -> level-1 self-tuning Laplacian
    # ------------------------------------------------------------------
    print("Building level-1 self-tuning Laplacian (exact normals)...")
    labels_lift, evals_lift = _lifted_clustering(
        points, normals, n_clusters,
        k_blowup=k_blowup, k_kernel=k_kernel,
        alpha=alpha, lam=lam, normalized=normalized,
        random_state=random_state,
    )
    _print_cluster_sizes("Lifted exact (level 1)", labels_lift)

    # ------------------------------------------------------------------
    # Lifted: PCA-estimated normals -> level-1 self-tuning Laplacian
    # ------------------------------------------------------------------
    print(f"Estimating normals via local PCA (k={k_pca})...")
    normals_pca = estimate_normals_pca(points, k=k_pca)

    # Normal alignment (absolute cosine, handles sign ambiguity)
    align = float(np.mean(np.abs(np.einsum("ni,ni->n", normals_pca, normals))))
    print(f"  Mean |cos theta| (PCA vs exact): {align:.4f}")

    print("Building level-1 self-tuning Laplacian (PCA normals)...")
    labels_pca, evals_pca = _lifted_clustering(
        points, normals_pca, n_clusters,
        k_blowup=k_blowup, k_kernel=k_kernel,
        alpha=alpha, lam=lam, normalized=normalized,
        random_state=random_state,
    )
    _print_cluster_sizes("Lifted PCA (level 1)", labels_pca)

    # ------------------------------------------------------------------
    # Plot: 3 panels
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    _plot_clusters(ax1, points, labels_euc,  "Euclidean\n(position only)")
    _plot_clusters(ax2, points, labels_lift, "Lifted -- exact normals\n(level 1, self-tuning)")
    _plot_clusters(ax3, points, labels_pca,  f"Lifted -- PCA normals  (k={k_pca})\n(level 1, self-tuning)")

    fig.suptitle(
        f"Cube Spectral Clustering: Euclidean vs Lifted (exact vs PCA normals)\n"
        f"PCA normal alignment: mean |cos| = {align:.3f}"
    )
    plt.tight_layout()
    plt.show()

    for label, evals in [("Euclidean", evals_euc), ("Lifted exact", evals_lift), ("Lifted PCA", evals_pca)]:
        if evals.size:
            print(
                f"First eigenvalues ({label}):",
                np.array2string(evals, precision=5, floatmode="fixed"),
            )


if __name__ == "__main__":
    main()
