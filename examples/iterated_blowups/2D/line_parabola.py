"""
Line-Parabola: Iterated Blow-Up, UMAP Visualisation, and Differential Invariants
----------------------------------------------------------------------------------
Two 1D curves in R^2 are tangentially intersecting at the origin:

    Parabola:  y = k x^2   (curvature kappa = 2k  at the tangency point)
    Line:      y = 0        (curvature kappa = 0   everywhere)

They share both *position* (origin) and *tangent direction* (horizontal) at the
contact point, so neither level-0 (coordinates) nor level-1 (tangent projector)
embeddings can metrically distinguish them there.  Only the level-2 embedding
encodes curvature, pushing the two components apart.

Figures
-------
Figure 1 – UMAP 2D projections of each blow-up embedding, coloured by
           HDBSCAN cluster label:
           * Level 0 (2D)  }  shared tangent at origin → one merged cluster
           * Level 1 (6D)  }
           * Level 2 (42D) → different curvature → two clean clusters

Figure 2 – Differential invariants on the original 2D point cloud:
           * Ground-truth component (0 = parabola, 1 = line)
           * Level-1 curvature  kappa  (shape operator scalar)
           * Level-2 curvature gradient  |d(kappa)/ds|
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP  # type: ignore[import-untyped]
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from tangent_blowups.testsupport import UniformCurve, sample, tangent_parabola_line
from tangent_blowups.geometry.iterated_grassmann import (
    extract_level1,
    extract_level2,
    iterated_blowup,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_POINTS   = 1200      # total sample points (600 per component)
CURVATURE  = 1.0       # parabola: y = CURVATURE * x^2
K_BLOWUP   = 10        # k-NN for blow-up curvature regression
ALPHA      = 5.0       # Chordal-Sasaki weight
LAM        = 1e-3      # ridge regularisation

UMAP_NEIGHBORS = 15    # UMAP: local neighbourhood size
UMAP_MIN_DIST  = 0.10  # UMAP: minimum embedding distance
UMAP_SEED      = 42

HDBSCAN_MIN_SIZE = 20  # HDBSCAN: minimum cluster size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_umap(X: np.ndarray) -> np.ndarray:
    """StandardScale the embedding, then project to 2D with UMAP."""
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


def plot_umap_clusters(ax, umap2: np.ndarray, labels: np.ndarray, title: str):
    n_cl = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    ax.scatter(umap2[:, 0], umap2[:, 1],
               c=_label_colors(labels), s=6, rasterized=True)
    ax.set_title(f"{title}\n({n_cl} cluster(s), {n_noise} noise pts)", fontsize=9)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")


def plot_xy_labels(ax, pts: np.ndarray, labels: np.ndarray, title: str):
    ax.scatter(pts[:, 0], pts[:, 1],
               c=_label_colors(labels), s=6, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_xy_scalar(ax, pts: np.ndarray, vals: np.ndarray, title: str,
                   cmap: str = "RdBu_r", symmetric: bool = True):
    fin = vals[np.isfinite(vals)]
    if symmetric:
        vabs = float(np.percentile(np.abs(fin), 98)) if fin.size else 1.0
        vmin, vmax = -vabs, vabs
    else:
        vmin = float(np.percentile(fin, 2)) if fin.size else 0.0
        vmax = float(np.percentile(fin, 98)) if fin.size else 1.0
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=6,
                    cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Sample the line-parabola union
    #    Domain t in [0, 2):  t in [0, 1) -> parabola,  t in [1, 2) -> line
    # ------------------------------------------------------------------
    curve    = tangent_parabola_line(curvature=CURVATURE)
    strategy = UniformCurve(n=N_POINTS, t_min=0.0, t_max=2.0, endpoint=False)
    s        = sample(curve, strategy, with_tangents=True, with_normals=False)

    points   = np.asarray(s.points,   dtype=float)   # (N, 2)
    tangents = np.asarray(s.tangents, dtype=float)   # (N, 2)
    t_params = np.asarray(s.params,   dtype=float)   # (N,)
    gt       = (t_params >= 1.0).astype(int)          # 0 = parabola, 1 = line

    print(f"Sampled {len(points)} points  "
          f"({(gt==0).sum()} parabola, {(gt==1).sum()} line)")

    # ------------------------------------------------------------------
    # 2. Iterated blow-up
    # ------------------------------------------------------------------
    levels = iterated_blowup(points, tangents, num_levels=2,
                             k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l0, l1, l2 = levels
    print(f"Embedding dims: {l0.D} -> {l1.D} -> {l2.D}")

    # ------------------------------------------------------------------
    # 3. Differential invariants
    #    d=1, n_comp=1  =>  shape operator is a scalar (signed curvature kappa)
    # ------------------------------------------------------------------
    inv1 = extract_level1(l1)
    inv2 = extract_level2(l0, l1, inv1, k=K_BLOWUP, lam=LAM)

    kappa  = inv1.shape_operator[:, 0, 0, 0]         # (N,) signed curvature
    dkappa = inv2.curvature_gradient[:, 0, 0, 0, 0]  # (N,) d(kappa)/ds

    print(f"kappa   parabola: {kappa[gt==0].mean():+.3f}  "
          f"line: {kappa[gt==1].mean():+.3f}  "
          f"(expected: parabola~{2*CURVATURE:.1f}, line~0)")

    # ------------------------------------------------------------------
    # 4. UMAP projections + HDBSCAN clustering
    # ------------------------------------------------------------------
    print("Computing UMAP projections...")
    u0 = embed_umap(l0.embedded)
    u1 = embed_umap(l1.embedded)
    u2 = embed_umap(l2.embedded)

    c0 = hdbscan_cluster(u0)
    c1 = hdbscan_cluster(u1)
    c2 = hdbscan_cluster(u2)

    for lvl, labels in enumerate([c0, c1, c2]):
        n_cl = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        print(f"  Level {lvl}: {n_cl} cluster(s),  {n_noise} noise pts")

    # ------------------------------------------------------------------
    # 5. Figure 1 – UMAP + HDBSCAN separation
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.5))

    plot_umap_clusters(axes1[0], u0, c0, "Level 0 – original (2D)")
    plot_umap_clusters(axes1[1], u1, c1, "Level 1 – first blow-up (6D)")
    plot_umap_clusters(axes1[2], u2, c2, "Level 2 – second blow-up (42D)")

    fig1.suptitle(
        "Line-Parabola: UMAP + HDBSCAN clustering at each blow-up level\n"
        "Shared tangent at origin -> one cluster at levels 0 & 1; "
        f"curvature differs (parabola k={2*CURVATURE:.0f}, line k=0) -> two clusters at level 2",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 6. Figure 2 – Differential invariants on original 2D cloud
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

    plot_xy_labels(axes2[0], points, gt,
                   "Ground Truth\n(0 = parabola, 1 = line)")

    plot_xy_scalar(axes2[1], points, kappa,
                   f"Level-1: curvature  kappa\n"
                   f"(parabola kappa={2*CURVATURE:.1f} at origin, line kappa=0)",
                   cmap="coolwarm")

    plot_xy_scalar(axes2[2], points, np.abs(dkappa),
                   "Level-2: |d(kappa)/ds|  (curvature gradient)\n"
                   "(line has constant kappa=0 -> gradient near zero)",
                   cmap="viridis", symmetric=False)

    fig2.suptitle(
        "Line-Parabola: Differential Invariants from Iterated Blow-Up",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Figure 3 - Original points colored by cluster (per blow-up level)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))

    plot_xy_labels(axes3[0], points, c0,
                   "Level 0 clusters\n(colors from UMAP+HDBSCAN)")
    plot_xy_labels(axes3[1], points, c1,
                   "Level 1 clusters\n(colors from UMAP+HDBSCAN)")
    plot_xy_labels(axes3[2], points, c2,
                   "Level 2 clusters\n(colors from UMAP+HDBSCAN)")

    fig3.suptitle(
        "Line-Parabola: Original point cloud colored by cluster at each level",
        fontsize=10,
    )
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
