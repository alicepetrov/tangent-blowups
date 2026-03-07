"""
Spectral Clustering with Self-Tuning Kernel — 3D Examples
----------------------------------------------------------
Each figure shows a 2 x len(ALPHAS) grid:
  Rows    : blowup level (1 lift, 2 lifts)
  Columns : Chordal-Sasaki weight alpha

Toy examples (known ground truth):
  - plane_cross             transverse intersection (~90 deg)
  - cylinders_tangent       tangential, same scalar curvature, different principal directions
  - plane_paraboloid_tangent tangential, same tangent plane, different curvature

Thingi10K examples (real meshes):
  - snowflake, propeller, icicles
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import HDBSCAN

from tangent_blowups.testsupport import (
    RandomSurface, sample,
    plane_cross, cylinders_tangent, plane_paraboloid_tangent,
)
from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.clustering import spectral_embedding_from_laplacian
from tangent_blowups.pointcloud.laplacian import laplacian_spectrum
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ALPHAS   = [1.0, 5.0, 10.0, 20.0, 30.0]   # swept along columns
K        = 20
LAM      = 1e-3
N_TOY    = 6000                      # points per toy surface
N_REAL   = 12000                      # max points for thingi10k (subsampled)

N_SPECTRAL_TOY  = 6              # None = auto via eigengap; set int to override
N_SPECTRAL_REAL = 12              # None = auto via eigengap; set int to override
N_SPECTRAL_PROBE = 20               # eigenvalues to scan when auto-detecting
# ~5-7 % of expected cluster size; toy has ~N_TOY/2 pts per component
HDBSCAN_MIN_TOY  = N_TOY  // 20     # 300 for N_TOY=6000
HDBSCAN_MIN_REAL = N_REAL // 60     # 200 for N_REAL=12000

ELEV, AZIM = 25, 45                  # shared 3D viewing angle

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "thingi10k_pointcloud"

# ---------------------------------------------------------------------------
# 3D plot helpers
# ---------------------------------------------------------------------------

def _clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=ELEV, azim=AZIM)

def _cmap_for(n: int):
    if n <= 10:
        return plt.get_cmap("tab10", max(n, 1))
    colors = np.vstack([plt.get_cmap("tab20").colors,
                        plt.get_cmap("tab20b").colors,
                        plt.get_cmap("tab20c").colors])
    return ListedColormap(colors[:n])


def _scatter3(ax, pts, labels, title: str):
    labels = np.asarray(labels, dtype=int)
    noise  = labels < 0
    valid  = labels[~noise]
    n_cl   = int(valid.max()) + 1 if valid.size else 1
    cmap   = _cmap_for(n_cl)

    if (~noise).any():
        ax.scatter(pts[~noise, 0], pts[~noise, 1], pts[~noise, 2],
                   c=valid, cmap=cmap, vmin=-0.5, vmax=n_cl - 0.5,
                   s=4, linewidths=0, rasterized=True)
    if noise.any():
        ax.scatter(pts[noise, 0], pts[noise, 1], pts[noise, 2],
                   color="#CCCCCC", s=3, linewidths=0, rasterized=True)

    ax.set_title(title, fontsize=9, pad=4)
    _clean_ax(ax)

    # equal aspect approximation
    lims = np.array([[pts[:, i].min(), pts[:, i].max()] for i in range(3)])
    r = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    c = lims.mean(axis=1)
    for i, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter([c[i] - r, c[i] + r])


# ---------------------------------------------------------------------------
# Spectral clustering helpers
# ---------------------------------------------------------------------------

def _eigengap_n_components(L, cap: int) -> int:
    """
    Eigengap heuristic: scan the bottom N_SPECTRAL_PROBE eigenvalues (including
    the trivial one) and pick n = argmax of gaps between consecutive eigenvalues.

    For k well-separated components the spectrum looks like:
        [~0, ..., ~0,   gap,   larger, ...]
         <-- k near-zero -->
    argmax(diff(evals)) == k-1, so we use k-1 non-trivial eigenvectors after
    drop_first=True.  Clamped to [2, cap] so HDBSCAN always gets at least 2D.
    """
    evals, _ = laplacian_spectrum(
        L, k=N_SPECTRAL_PROBE + 1, drop_first=False, return_eigenvalues=True,
    )
    gaps = np.diff(evals)
    n    = max(int(np.argmax(gaps)), 2)
    return min(n, cap)


def _spectral_labels(level: BlowUpLevel, n_components: int | None,
                     min_cluster_size: int) -> tuple[np.ndarray, int]:
    """HDBSCAN on the drop_first spectral embedding of the self-tuning Laplacian.

    Returns (labels, n_components_used).
    """
    L, _, _ = lifted_laplacian(level, kernel="self_tuning", k=K,
                                h="local", normalized=True)
    if n_components is None:
        n_components = _eigengap_n_components(L, cap=N_SPECTRAL_PROBE)
    _, _, emb = spectral_embedding_from_laplacian(
        L, n_components=n_components, drop_first=True, normalize_rows=True,
    )
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb), n_components


def _spectral_labels_toy(level: BlowUpLevel) -> tuple[np.ndarray, int]:
    return _spectral_labels(level, N_SPECTRAL_TOY, HDBSCAN_MIN_TOY)


def _spectral_labels_real(level: BlowUpLevel) -> tuple[np.ndarray, int]:
    return _spectral_labels(level, N_SPECTRAL_REAL, HDBSCAN_MIN_REAL)


# ---------------------------------------------------------------------------
# Level-0 construction
# ---------------------------------------------------------------------------

def _level0_from_surface(surface_fn) -> tuple[np.ndarray, BlowUpLevel, np.ndarray]:
    surf = surface_fn()
    rng  = np.random.default_rng(42)
    s = sample(surf,
               RandomSurface(n=N_TOY, u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0),
                             rng=rng),
               with_tangents=True, with_normals=False)
    pts    = np.asarray(s.points,   dtype=float)   # (N, 3)
    frames = np.asarray(s.tangents, dtype=float)   # (N, 3, 2)
    # ground truth: face 0 (u<1) vs face 1 (u>=1)
    u_vals = np.asarray(s.params[0], dtype=float).ravel()
    gt     = (np.floor(u_vals) >= 1).astype(int)
    return pts, BlowUpLevel.from_point_tangents(pts, frames), gt


def _level0_from_thingi(name: str) -> tuple[np.ndarray, BlowUpLevel]:
    path = DATA_DIR / f"{name}.npz"
    pc   = load_pointcloud(path)
    pts  = np.asarray(pc.points,  dtype=float).reshape(-1, 3)
    nrm  = np.asarray(pc.normals, dtype=float).reshape(-1, 3)

    valid = np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
    valid &= np.linalg.norm(nrm, axis=1) > 1e-8
    pts, nrm = pts[valid], normalize_vectors(nrm[valid])

    # subsample
    if len(pts) > N_REAL:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N_REAL, replace=False)
        pts, nrm = pts[idx], nrm[idx]

    frames = BlownUpSample.from_normals(pts, nrm).dualize().basis  # (N, 3, 2)
    return pts, BlowUpLevel.from_point_tangents(pts, frames)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def _make_figure(pts, level0, title, fname, label_fn, gt=None):
    n_rows, n_cols = 2, len(ALPHAS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows),
        subplot_kw={"projection": "3d"},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, y=1.01)

    for row, n_lifts in enumerate([1, 2]):
        for col, alpha in enumerate(ALPHAS):
            level = level0
            for _ in range(n_lifts):
                level = level.lift(k=K, alpha=alpha, lam=LAM)
            labels, n_comp = label_fn(level)
            n_cl    = int((labels >= 0).any() and labels[labels >= 0].max() + 1)
            n_noise = int((labels < 0).sum())
            ax_title = f"level {n_lifts},  alpha={alpha}\nn_comp={n_comp}, {n_cl} clusters, {n_noise} noise"
            _scatter3(axes[row, col], pts, labels, ax_title)

    plt.tight_layout(pad=1.2)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname}")


def _make_highlight_figure(pts, level0, gt, alpha, fname):
    """1x2 figure: level-1 (fails) vs level-2 (succeeds) at a fixed alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Plane + paraboloid — tangential contact\n"
        f"Level 1 cannot separate (same tangent plane); "
        f"level 2 succeeds (curvature differs)  [alpha={alpha}]",
        fontsize=11, y=1.02,
    )

    level1 = level0.lift(k=K, alpha=alpha, lam=LAM)
    level2 = level1.lift(k=K, alpha=alpha, lam=LAM)

    for ax, level, n_lifts in zip(axes, [level1, level2], [1, 2]):
        labels, n_comp = _spectral_labels_toy(level)
        n_cl    = int((labels >= 0).any() and labels[labels >= 0].max() + 1)
        n_noise = int((labels < 0).sum())
        _scatter3(ax, pts, labels,
                  f"Level {n_lifts} — n_comp={n_comp}, {n_cl} clusters, {n_noise} noise")

    plt.tight_layout(pad=1.2)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

TOYS = [
    (plane_cross,              "Transverse (plane cross)",          "spectral_plane_cross.png"),
    (cylinders_tangent,        "Tangential (cylinders)",            "spectral_cylinders_tangent.png"),
    (plane_paraboloid_tangent, "Tangential (plane + paraboloid)",   "spectral_plane_paraboloid.png"),
]

REAL = [
    ("snowflake", "spectral_snowflake.png"),
    ("propeller", "spectral_propeller.png"),
    ("icicles",   "spectral_icicles.png"),
]

for surface_fn, title, fname in TOYS:
    print(f"\n=== {title} ===")
    pts, level0, gt = _level0_from_surface(surface_fn)
    _make_figure(pts, level0, title, fname, _spectral_labels_toy, gt=gt)

print("\n=== Highlight: plane + paraboloid (level 1 fails, level 2 succeeds) ===")
pts_pp, level0_pp, gt_pp = _level0_from_surface(plane_paraboloid_tangent)
_make_highlight_figure(pts_pp, level0_pp, gt_pp, alpha=5.0,
                       fname="spectral_paraboloid_highlight.png")

for name, fname in REAL:
    print(f"\n=== {name} ===")
    pts, level0 = _level0_from_thingi(name)
    _make_figure(pts, level0, name, fname, _spectral_labels_real)

plt.show()
