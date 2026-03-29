"""
Thingi10k: Iterated Blow-Up, UMAP Visualisation, and Differential Invariants
-----------------------------------------------------------------------------
Load a Thingi10k point cloud (with normals), convert normals to tangent
frames via Grassmannian duality, then compute iterated Nash blow-ups.

Figures
-------
Figure 1 -- UMAP 2D projections of each blow-up embedding, coloured by
            HDBSCAN cluster label.  Higher levels encode progressively
            higher-order geometry (level 0 = position, level 1 = tangent
            plane, level 2 = curvature).

Figure 2 -- Differential invariants on the original 3D point cloud:
            * |Mean curvature| H       (from level-1 shape operator)
            * Gaussian curvature K     (from level-1 shape operator)
            * ||grad h||_F (Codazzi)   (from level-2, if computed)

Figure 3 -- Original 3D points colored by HDBSCAN cluster at each level.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP  # type: ignore[import-untyped]
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.geometry.iterated_grassmann import (
    extract_level1,
    extract_level2,
    iterated_blowup,
)
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K_BLOWUP       = 100       # k-NN for blow-up curvature regression
ALPHA          = 5.0       # Chordal-Sasaki weight
LAM            = 1e-3      # ridge regularisation

UMAP_NEIGHBORS = 15        # UMAP: local neighbourhood size
UMAP_MIN_DIST  = 0.10      # UMAP: minimum embedding distance
UMAP_SEED      = 42

HDBSCAN_MIN_SIZE = 30      # HDBSCAN: minimum cluster size


# ---------------------------------------------------------------------------
# Point cloud loading
# ---------------------------------------------------------------------------

def _thingi10k_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "thingi10k_pointcloud"


def _default_path() -> Path:
    return _thingi10k_root() / "snowflake.npz"


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return _default_path()
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".npz")
    return _thingi10k_root() / candidate


def _load_points_normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a point cloud and return validated (points, normals)."""
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    s = load_pointcloud(path)
    if s.normals is None:
        raise ValueError("Point cloud is missing normals.")

    pts = np.asarray(s.points, dtype=float)
    nrm = np.asarray(s.normals, dtype=float)
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if nrm.ndim > 2:
        nrm = nrm.reshape(-1, nrm.shape[-1])

    valid = (np.isfinite(pts).all(axis=1)
             & np.isfinite(nrm).all(axis=1)
             & (np.linalg.norm(nrm, axis=1) > 1e-8))
    if not np.all(valid):
        print(f"Dropping {int((~valid).sum())} invalid/degenerate samples.")
    return pts[valid], normalize_vectors(nrm[valid])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_umap(X: np.ndarray) -> np.ndarray:
    """StandardScale then project to 2D with UMAP."""
    Xs = StandardScaler().fit_transform(X)
    return UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=UMAP_SEED,
    ).fit_transform(Xs)


def hdbscan_cluster(umap2: np.ndarray) -> np.ndarray:
    """HDBSCAN labels on 2D UMAP coordinates."""
    return HDBSCAN(min_cluster_size=HDBSCAN_MIN_SIZE).fit_predict(umap2)


def _label_colors(labels: np.ndarray) -> list:
    n_pos = len([u for u in set(labels) if u >= 0])
    cmap = plt.get_cmap("tab10", max(n_pos, 1))
    return [cmap(int(l)) if l >= 0 else (0.55, 0.55, 0.55, 1.0) for l in labels]


def _set_equal3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    r = max(0.5 * float(np.max(limits[:, 1] - limits[:, 0])), 1e-3)
    ax.set_xlim3d(center[0] - r, center[0] + r)
    ax.set_ylim3d(center[1] - r, center[1] + r)
    ax.set_zlim3d(center[2] - r, center[2] + r)


def plot_umap_clusters(ax, umap2: np.ndarray, labels: np.ndarray, title: str):
    n_cl = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    ax.scatter(umap2[:, 0], umap2[:, 1],
               c=_label_colors(labels), s=6, rasterized=True)
    ax.set_title(f"{title}\n({n_cl} cluster(s), {n_noise} noise pts)", fontsize=9)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")


def plot3_labels(ax, pts: np.ndarray, labels: np.ndarray, title: str):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=_label_colors(labels), s=5, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal3d(ax)


def plot3_scalar(ax, pts: np.ndarray, vals: np.ndarray, title: str,
                 cmap: str = "plasma", symmetric: bool = False):
    fin = vals[np.isfinite(vals)]
    if symmetric:
        vabs = float(np.percentile(np.abs(fin), 98)) if fin.size else 1.0
        vmin, vmax = -vabs, vabs
    else:
        vmin = float(np.percentile(fin, 2)) if fin.size else 0.0
        vmax = float(np.percentile(fin, 98)) if fin.size else 1.0
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=vals, s=5, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal3d(ax)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Iterated blow-up demo on a Thingi10k point cloud."
    )
    parser.add_argument(
        "--pointcloud", type=Path, default=None,
        help=(
            "Point cloud file name or path.  If a bare name is given it is "
            "resolved under data/thingi10k_pointcloud/."
        ),
    )
    parser.add_argument(
        "--num-levels", type=int, default=2,
        help="Number of blow-up levels (default: 2).",
    )
    args = parser.parse_args()

    data_path = _resolve_path(args.pointcloud)
    cloud_label = data_path.stem
    num_levels = args.num_levels

    # ------------------------------------------------------------------
    # 1. Load point cloud
    # ------------------------------------------------------------------
    points, normals = _load_points_normals(data_path)
    print(f"Loaded {len(points)} points from {data_path}")

    # ------------------------------------------------------------------
    # 2. Normals -> tangent frames -> iterated blow-up
    # ------------------------------------------------------------------
    tangent_frames = BlownUpSample.from_normals(points, normals).dualize().basis
    levels = iterated_blowup(
        points, tangent_frames, num_levels=num_levels,
        k=K_BLOWUP, alpha=ALPHA, lam=LAM,
    )
    dims = " -> ".join(str(l.D) for l in levels)
    print(f"Embedding dims: {dims}")

    # ------------------------------------------------------------------
    # 3. Differential invariants
    # ------------------------------------------------------------------
    H = K = grad_h_frob = None

    if num_levels >= 1:
        inv1 = extract_level1(levels[1])
        H = np.abs(inv1.mean_curvature[:, 0])
        K = np.nan_to_num(inv1.gaussian_curvature, nan=0.0)
        print(f"|H|  range: [{H.min():.4f}, {H.max():.4f}]  "
              f"median: {np.median(H):.4f}")
        print(f" K   range: [{K.min():.4f}, {K.max():.4f}]  "
              f"median: {np.median(K):.4f}")

    if num_levels >= 2:
        inv2 = extract_level2(levels[0], levels[1], inv1, k=K_BLOWUP, lam=LAM)
        grad_h_frob = np.sqrt(np.einsum(
            "nabcd,nabcd->n",
            inv2.curvature_gradient,
            inv2.curvature_gradient,
        ))
        print(f"||grad h||_F  range: [{grad_h_frob.min():.4f}, "
              f"{grad_h_frob.max():.4f}]  median: {np.median(grad_h_frob):.4f}")

    # ------------------------------------------------------------------
    # 4. UMAP projections + HDBSCAN clustering
    # ------------------------------------------------------------------
    print("Computing UMAP projections...")
    umap_results = []
    cluster_results = []
    for i, lvl in enumerate(levels):
        u = embed_umap(lvl.embedded)
        c = hdbscan_cluster(u)
        umap_results.append(u)
        cluster_results.append(c)
        n_cl = len(set(c) - {-1})
        n_noise = int((c == -1).sum())
        print(f"  Level {i}: {n_cl} cluster(s),  "
              f"{n_noise} noise pts  (embedded dim={lvl.D})")

    # ------------------------------------------------------------------
    # 5. Figure 1 -- UMAP + HDBSCAN
    # ------------------------------------------------------------------
    n_panels = len(levels)
    fig1, axes1 = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes1 = [axes1]
    for i in range(n_panels):
        plot_umap_clusters(
            axes1[i], umap_results[i], cluster_results[i],
            f"Level {i} ({levels[i].D}D)",
        )
    fig1.suptitle(
        f"{cloud_label}: UMAP + HDBSCAN at each blow-up level",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 6. Figure 2 -- Differential invariants on original 3D cloud
    # ------------------------------------------------------------------
    inv_panels: list[tuple[str, np.ndarray, str, str, bool]] = []
    if H is not None:
        inv_panels.append(("H", H, "|Mean Curvature| H", "plasma", False))
    if K is not None:
        inv_panels.append(("K", K, "Gaussian Curvature K", "inferno", True))
    if grad_h_frob is not None:
        inv_panels.append((
            "grad", grad_h_frob,
            "||grad h||_F (Codazzi tensor magnitude)",
            "viridis", False,
        ))

    if inv_panels:
        n_inv = len(inv_panels)
        fig2 = plt.figure(figsize=(6 * n_inv, 5))
        for i, (_key, vals, title, cmap, sym) in enumerate(inv_panels):
            ax = fig2.add_subplot(1, n_inv, i + 1, projection="3d")
            plot3_scalar(ax, points, vals, title, cmap=cmap, symmetric=sym)
        fig2.suptitle(
            f"{cloud_label}: Differential Invariants from Iterated Blow-Up",
            fontsize=10,
        )
        plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Figure 3 -- Original 3D points colored by cluster
    # ------------------------------------------------------------------
    fig3 = plt.figure(figsize=(6 * n_panels, 5))
    for i in range(n_panels):
        ax = fig3.add_subplot(1, n_panels, i + 1, projection="3d")
        plot3_labels(
            ax, points, cluster_results[i],
            f"Level {i} clusters\n(UMAP + HDBSCAN)",
        )
    fig3.suptitle(
        f"{cloud_label}: point cloud colored by cluster at each level",
        fontsize=10,
    )
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
