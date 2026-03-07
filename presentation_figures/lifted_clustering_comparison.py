"""
Clustering in the lifted (level-1) embedding: k-means vs DBSCAN vs HDBSCAN.

Two examples:
  - Two transversely intersecting circles (~60 deg crossing): level-1 separable
  - Line + parabola (tangential contact, same tangent): level-1 NOT separable

Each figure is a 3x3 grid:
  Row 0: clustering in original 2D space
  Row 1: clustering in level-1 Chordal-Sasaki embedding
  Row 2: clustering in level-2 Chordal-Sasaki embedding
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering

from tangent_blowups.testsupport import UniformCurve, sample, tangent_parabola_line
from tangent_blowups.geometry.iterated_grassmann import iterated_blowup
from tangent_blowups.geometry.kernels import lifted_affinity

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N        = 800
K_BLOWUP = 10
ALPHA    = 1.0
LAM      = 1e-3

C0      = "#4477AA"
C1      = "#EE6677"
C_NOISE = "#AAAAAA"

# ---------------------------------------------------------------------------
# Two transversely intersecting unit circles
#   Circle 1: center (-0.5, 0),  Circle 2: center (+0.5, 0),  radius = 1
#   Intersection at (0, +/-sqrt(3)/2), tangent angle ~60 deg.
# ---------------------------------------------------------------------------
def _two_circles_sample(n: int):
    t  = np.linspace(0.0, 2.0, n, endpoint=False)
    m0 = t < 1.0

    theta0 = 2.0 * np.pi * t[m0]
    theta1 = 2.0 * np.pi * (t[~m0] - 1.0)

    px = np.empty(n); py = np.empty(n)
    tx = np.empty(n); ty = np.empty(n)

    px[m0]  = -0.5 + np.cos(theta0);  py[m0]  = np.sin(theta0)
    tx[m0]  = -np.sin(theta0);        ty[m0]  = np.cos(theta0)
    px[~m0] =  0.5 + np.cos(theta1); py[~m0] = np.sin(theta1)
    tx[~m0] = -np.sin(theta1);       ty[~m0] = np.cos(theta1)

    pts = np.stack([px, py], axis=-1)
    tan = np.stack([tx, ty], axis=-1)
    gt  = (~m0).astype(int)
    return pts, tan, gt


def _label_colors(labels: np.ndarray) -> list[str]:
    palette = [C0, C1, "#44AA77", "#CCBB44"]
    return [
        C_NOISE if l < 0 else palette[l % len(palette)]
        for l in labels
    ]


def _align(labels: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Flip 0/1 so that majority of label-0 points match gt-0."""
    mask = labels >= 0
    if mask.sum() > 0 and np.mean(labels[mask] == gt[mask]) < 0.5:
        labels = np.where(labels < 0, labels, 1 - labels)
    return labels


def _scatter(ax, pts, labels, title: str):
    ax.scatter(pts[:, 0], pts[:, 1],
               c=_label_colors(labels), s=6, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
pts_tc, tan_tc, gt_tc = _two_circles_sample(N)

s_lp = sample(tangent_parabola_line(curvature=1.0),
               UniformCurve(n=N, t_min=0.0, t_max=2.0, endpoint=False),
               with_tangents=True, with_normals=False)
pts_lp = np.asarray(s_lp.points,   dtype=float)
tan_lp = np.asarray(s_lp.tangents, dtype=float)
gt_lp  = (np.asarray(s_lp.params) >= 1.0).astype(int)

def _lift(pts, tan, num_levels):
    return iterated_blowup(pts, tan, num_levels=num_levels,
                           k=K_BLOWUP, alpha=ALPHA, lam=LAM)

levels_tc = _lift(pts_tc, tan_tc, num_levels=2)
levels_lp = _lift(pts_lp, tan_lp, num_levels=2)

emb_tc1 = levels_tc[1].embedded
emb_tc2 = levels_tc[2].embedded
emb_lp1 = levels_lp[1].embedded
emb_lp2 = levels_lp[2].embedded

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _cluster_all(X, gt, dbscan_eps):
    km  = _align(KMeans(n_clusters=2, n_init=20, random_state=0).fit_predict(X), gt)
    db  = _align(DBSCAN(eps=dbscan_eps, min_samples=5).fit_predict(X), gt)
    hdb = _align(HDBSCAN(min_cluster_size=20).fit_predict(X), gt)
    return km, db, hdb

km_tc0, db_tc0, hdb_tc0 = _cluster_all(pts_tc,  gt_tc, dbscan_eps=0.20)
km_tc1, db_tc1, hdb_tc1 = _cluster_all(emb_tc1, gt_tc, dbscan_eps=0.15)
km_tc2, db_tc2, hdb_tc2 = _cluster_all(emb_tc2, gt_tc, dbscan_eps=0.15)

km_lp0, db_lp0, hdb_lp0 = _cluster_all(pts_lp,  gt_lp, dbscan_eps=0.20)
km_lp1, db_lp1, hdb_lp1 = _cluster_all(emb_lp1, gt_lp, dbscan_eps=0.15)
km_lp2, db_lp2, hdb_lp2 = _cluster_all(emb_lp2, gt_lp, dbscan_eps=0.15)

# ---------------------------------------------------------------------------
# Figure helper
# ---------------------------------------------------------------------------
def _make_figure(pts, row0, row1, row2, title, fname):
    fig, axes = plt.subplots(3, 3, figsize=(12, 11.0))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, y=1.01)
    row_labels = ["original space", "lifted space (level 1)", "lifted space (level 2)"]
    for row, (km, db, hdb) in enumerate([row0, row1, row2]):
        _scatter(axes[row, 0], pts, km,  f"K-means\n{row_labels[row]}")
        _scatter(axes[row, 1], pts, db,  f"DBSCAN\n{row_labels[row]}  ({int((db  < 0).sum())} noise)")
        _scatter(axes[row, 2], pts, hdb, f"HDBSCAN\n{row_labels[row]}  ({int((hdb < 0).sum())} noise)")
    plt.tight_layout(pad=1.5)
    plt.savefig(fname, bbox_inches="tight", dpi=150)

_make_figure(pts_tc,
             (km_tc0, db_tc0, hdb_tc0),
             (km_tc1, db_tc1, hdb_tc1),
             (km_tc2, db_tc2, hdb_tc2),
             "Transverse intersection — two circles",
             "lifted_clustering_circles.png")

_make_figure(pts_lp,
             (km_lp0, db_lp0, hdb_lp0),
             (km_lp1, db_lp1, hdb_lp1),
             (km_lp2, db_lp2, hdb_lp2),
             "Tangential intersection — line + parabola",
             "lifted_clustering_lineparabola.png")

# ---------------------------------------------------------------------------
# Figure 3: spectral clustering on line-parabola at each blow-up level
# ---------------------------------------------------------------------------
def _spectral(level, gt):
    W = lifted_affinity(level, k=K_BLOWUP, h="local")
    labels = SpectralClustering(
        n_clusters=2, affinity="precomputed", n_init=10, random_state=0,
    ).fit_predict(W.toarray())
    return _align(labels, gt)

sc_lp0 = _spectral(levels_lp[0], gt_lp)
sc_lp1 = _spectral(levels_lp[1], gt_lp)
sc_lp2 = _spectral(levels_lp[2], gt_lp)

fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4.0))
fig3.patch.set_facecolor("white")
fig3.suptitle("Spectral clustering (self-tuning kernel) — line + parabola", fontsize=13, y=1.01)

for ax, labels, lvl in zip(axes3, [sc_lp0, sc_lp1, sc_lp2], [0, 1, 2]):
    _scatter(ax, pts_lp, labels, f"level {lvl}")

plt.tight_layout(pad=1.5)
plt.savefig("lifted_clustering_spectral.png", bbox_inches="tight", dpi=150)

plt.show()
