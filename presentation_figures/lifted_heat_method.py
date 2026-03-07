"""
Geodesic Heat Method on Klein Bottle (thingi10k)
-------------------------------------------------
Compares two approaches for computing approximate geodesic distances
on the Klein bottle point cloud:

  Traditional : heat method with Euclidean Laplacian (alpha=0)
  Lifted      : heat method with Chordal-Sasaki Laplacian (alpha>0)

The Klein bottle is a non-orientable self-intersecting surface in R^3.
At the self-intersection the traditional heat method bleeds across sheets,
producing corrupted geodesic rings.  The lifted method uses the tangent-frame
information to distinguish the two sheets and stays on the correct one.

Figure layout: 1 row x 3 cols
  (0) Traditional geodesic distance (level sets)
  (1) Lifted geodesic distance (level sets)
  (2) Absolute difference |phi_lifted - phi_trad|
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.spatial import cKDTree

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.pointcloud.geodesic_heat import lifted_heat_method
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K       = 20       # k-NN for Laplacian construction
ALPHA   = 10.0      # Chordal-Sasaki weight for lifted method
N_MAX   = 25000    # subsample cap
N_LEVELS = 16      # discrete geodesic rings
ELEV, AZIM = 20, 50
PT_SIZE = 3.0
CMAP    = "plasma"

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "thingi10k_pointcloud"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(name: str) -> tuple[np.ndarray, BlownUpSample]:
    path = DATA_DIR / f"{name}.npz"
    pc  = load_pointcloud(path)
    pts = np.asarray(pc.points,  dtype=float).reshape(-1, 3)
    nrm = np.asarray(pc.normals, dtype=float).reshape(-1, 3)

    valid = (np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
             & (np.linalg.norm(nrm, axis=1) > 1e-8))
    pts, nrm = pts[valid], normalize_vectors(nrm[valid])

    if len(pts) > N_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N_MAX, replace=False)
        pts, nrm = pts[idx], nrm[idx]

    # normal subspace → dualize to get tangent frame
    bup = BlownUpSample.from_normals(pts, nrm).dualize()
    return pts, bup


# ---------------------------------------------------------------------------
# Source point selection: pick the point closest to the self-intersection
# region, detected as a spatial hot-spot (high local density) with high
# tangent-frame diversity among its k-NN.
# ---------------------------------------------------------------------------

def _find_intersection_source(pts: np.ndarray, bup: BlownUpSample,
                               k_probe: int = 15) -> int:
    """
    Returns the index of a point near the self-intersection.

    Strategy: for each point compute the mean Euclidean distance to its
    k_probe nearest neighbours (small = dense = near intersection) AND the
    mean pairwise chordal distance between those neighbours' projectors
    (large = neighbours span two sheets).  Score = diversity / density.
    """
    tree = cKDTree(pts)
    dists, nn_idx = tree.query(pts, k=k_probe + 1)
    dists   = dists[:, 1:]    # exclude self
    nn_idx  = nn_idx[:, 1:]

    mean_dist = dists.mean(axis=1)

    P = bup.projectors                  # (N, 3, 3)
    diversity = np.zeros(len(pts))
    for i in range(len(pts)):
        Pi = P[nn_idx[i]]               # (k_probe, 3, 3)
        # mean pairwise Frobenius distance between projectors in neighbourhood
        diff = Pi[:, None] - Pi[None, :]  # (k, k, 3, 3)
        diversity[i] = float(np.mean(np.linalg.norm(diff.reshape(k_probe, k_probe, -1),
                                                     axis=-1)))

    score = diversity / np.clip(mean_dist, 1e-12, None)
    return int(np.argmax(score))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=ELEV, azim=AZIM)


def _equal_aspect(ax, pts):
    lims = np.array([[pts[:, i].min(), pts[:, i].max()] for i in range(3)])
    r = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    c = lims.mean(axis=1)
    for i, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter([c[i] - r, c[i] + r])


def _level_scatter(ax, pts, vals, title: str, n_levels: int = N_LEVELS,
                   vmin: float | None = None, vmax: float | None = None):
    """Scatter plot with discrete level-set colormap."""
    finite = vals[np.isfinite(vals)]
    v0 = float(finite.min()) if vmin is None else vmin
    v1 = float(finite.max()) if vmax is None else vmax
    boundaries = np.linspace(v0, v1, n_levels + 1)
    # alternate light/dark bands for geodesic ring visibility
    base = plt.get_cmap(CMAP, n_levels)(np.linspace(0, 1, n_levels))
    base[::2, :3] = np.clip(base[::2, :3] * 0.65 + 0.35, 0, 1)  # lighter bands
    cmap_lev = ListedColormap(base)
    norm = BoundaryNorm(boundaries, ncolors=n_levels, clip=True)

    c = np.where(np.isfinite(vals), vals, v0)
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=c, s=PT_SIZE, cmap=cmap_lev, norm=norm,
                    linewidths=0, rasterized=True)
    _clean_ax(ax)
    _equal_aspect(ax, pts)
    ax.set_title(title, fontsize=10, pad=4)
    return sc


def _diff_scatter(ax, pts, diff, title: str):
    """Scatter plot of absolute difference — sequential colormap."""
    finite = diff[np.isfinite(diff)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    c = np.where(np.isfinite(diff), diff, 0.0)
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=c, s=PT_SIZE, cmap="YlOrRd",
                    vmin=0.0, vmax=max(vmax, 1e-12),
                    linewidths=0, rasterized=True)
    _clean_ax(ax)
    _equal_aspect(ax, pts)
    ax.set_title(title, fontsize=10, pad=4)
    return sc


def _mark_source(ax, q: np.ndarray):
    ax.scatter([q[0]], [q[1]], [q[2]], s=120,
               c="lime", edgecolors="k", linewidths=1.2, zorder=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Loading klein_bottle_two ...")
pts, bup = _load("klein_bottle_two")
print(f"  {len(pts)} points")

print("Finding source point near self-intersection ...")
src = _find_intersection_source(pts, bup)
print(f"  Source index: {src},  position: {pts[src].round(3)}")

print("Traditional heat method (alpha=0) ...", end="", flush=True)
phi_trad = lifted_heat_method(
    bup,
    source_index=src,
    k=K,
    alpha=0.0,
    t_scale=1.0,
    laplacian_normalized=True,
)
# shift so source = 0
phi_trad -= phi_trad[src]
phi_trad = np.clip(phi_trad, 0.0, None)
print(f" done  (range [{phi_trad.min():.3g}, {phi_trad.max():.3g}])")

print("Lifted heat method (alpha=%.1f) ..." % ALPHA, end="", flush=True)
phi_lift = lifted_heat_method(
    bup,
    source_index=src,
    k=K,
    alpha=ALPHA,
    t_scale=1.0,
    laplacian_normalized=True,
)
phi_lift -= phi_lift[src]
phi_lift = np.clip(phi_lift, 0.0, None)
print(f" done  (range [{phi_lift.min():.3g}, {phi_lift.max():.3g}])")

# shared color scale so level rings are comparable
vmax_shared = float(np.percentile(
    np.concatenate([phi_trad, phi_lift]), 97
))

print("Plotting ...")
fig = plt.figure(figsize=(17, 5.5))
fig.patch.set_facecolor("white")

ax0 = fig.add_subplot(1, 3, 1, projection="3d")
ax1 = fig.add_subplot(1, 3, 2, projection="3d")
ax2 = fig.add_subplot(1, 3, 3, projection="3d")

sc0 = _level_scatter(ax0, pts, phi_trad,
                     "Traditional (Euclidean Laplacian)",
                     vmin=0.0, vmax=vmax_shared)
_mark_source(ax0, pts[src])

sc1 = _level_scatter(ax1, pts, phi_lift,
                     f"Lifted (Chordal-Sasaki, alpha={ALPHA})",
                     vmin=0.0, vmax=vmax_shared)
_mark_source(ax1, pts[src])

diff = np.abs(phi_lift - phi_trad)
sc2 = _diff_scatter(ax2, pts, diff,
                    "|Lifted - Traditional|")
_mark_source(ax2, pts[src])

fig.colorbar(sc0, ax=ax0, fraction=0.035, pad=0.05, label="geodesic dist")
fig.colorbar(sc1, ax=ax1, fraction=0.035, pad=0.05, label="geodesic dist")
fig.colorbar(sc2, ax=ax2, fraction=0.035, pad=0.05, label="|diff|")

fig.suptitle("Heat Method Geodesics — Klein Bottle (thingi10k)\n"
             "Green dot = source.  "
             "Near self-intersection the traditional method bleeds across sheets.",
             fontsize=11, y=1.01)

plt.tight_layout(pad=1.0)
fname = "heat_method_klein_bottle.png"
plt.savefig(fname, bbox_inches="tight", dpi=150)
print(f"Saved {fname}")
plt.close(fig)
print("Done.")
