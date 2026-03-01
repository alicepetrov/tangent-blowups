"""
Plane-Paraboloid: Iterated Blow-Up, UMAP Visualisation, and Differential Invariants
--------------------------------------------------------------------------------------
Two 2D surfaces in R^3 are tangentially intersecting at the origin:

    Plane:       z = 0
    Paraboloid:  z = scale_z * (x^2 + y^2)   (kappa_1 = kappa_2 = 2*scale_z at origin)

Both surfaces share position (origin) and tangent plane (the z=0 plane) there, so
neither level-0 (coordinates) nor level-1 (tangent projector) embeddings can
metrically distinguish them at the contact point.  Only the level-2 embedding
encodes curvature, pushing the two components apart.

Figures
-------
Figure 1 – UMAP 2D projections of each blow-up embedding, coloured by
           HDBSCAN cluster label:
           * Level 0 (3D)   }  shared tangent plane at origin -> one merged cluster
           * Level 1 (12D)  }
           * Level 2 (156D) -> different curvature -> two clean clusters

Figure 2 – Differential invariants on the original 3D point cloud:
           * Ground-truth component (0 = plane, 1 = paraboloid)
           * |Mean curvature| H   (plane H=0, paraboloid H=2*scale_z at origin)
           * Gaussian curvature K (plane K=0, paraboloid K=4*scale_z^2 at origin)

Figure 3 - Original 3D points colored by cluster for each blow-up level.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP  # type: ignore[import-untyped]
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from tangent_blowups.testsupport import RandomSurface, plane_paraboloid_tangent, sample
from tangent_blowups.geometry.iterated_grassmann import (
    extract_level1,
    extract_level2,
    iterated_blowup,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_POINTS   = 2500      # total sample points (~1000 per component)
SCALE_XY   = 1.0       # spatial extent of each surface patch
SCALE_Z    = 1.0      # paraboloid: z = SCALE_Z * (x^2 + y^2)
K_BLOWUP   = 20        # k-NN for blow-up curvature regression
ALPHA      = 1.0       # Chordal-Sasaki weight
LAM        = 1e-3      # ridge regularisation
SEED       = 7

UMAP_NEIGHBORS = 10    # UMAP: local neighbourhood size
UMAP_MIN_DIST  = 0.10  # UMAP: minimum embedding distance
UMAP_SEED      = 42

HDBSCAN_MIN_SIZE = 30  # HDBSCAN: minimum cluster size (~6 % of one component)


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
    # ------------------------------------------------------------------
    # 1. Sample the plane-paraboloid union
    #    Domain: u in [0, 2), v in [0, 1]
    #    face = floor(u):  0 -> plane,  1 -> paraboloid
    # ------------------------------------------------------------------
    rng      = np.random.default_rng(SEED)
    surface  = plane_paraboloid_tangent(scale_xy=SCALE_XY, scale_z=SCALE_Z)
    strategy = RandomSurface(
        n=N_POINTS, u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0), rng=rng
    )
    s = sample(surface, strategy, with_tangents=True, with_normals=False)

    points   = np.asarray(s.points,    dtype=float)   # (N, 3)
    tangents = np.asarray(s.tangents,  dtype=float)   # (N, 3, 2)
    u_vals   = np.asarray(s.params[0], dtype=float)   # (N,)
    gt       = np.floor(u_vals).astype(int)            # 0 = plane, 1 = paraboloid

    # Drop any degenerate frames (NaN from singular Gram-Schmidt)
    flat_T = tangents.reshape(len(points), -1)
    valid  = (np.all(np.isfinite(points), axis=1) &
              np.all(np.isfinite(flat_T), axis=1))
    points, tangents, gt = points[valid], tangents[valid], gt[valid]

    print(f"Sampled {valid.sum()} points  "
          f"({(gt==0).sum()} plane, {(gt==1).sum()} paraboloid)")

    # ------------------------------------------------------------------
    # 2. Iterated blow-up
    # ------------------------------------------------------------------
    levels = iterated_blowup(points, tangents, num_levels=2,
                             k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l0, l1, l2 = levels
    print(f"Embedding dims: {l0.D} -> {l1.D} -> {l2.D}")

    # ------------------------------------------------------------------
    # 3. Differential invariants
    #    d=2, n_comp=1  =>  shape operator is a 2x2 matrix (hypersurface)
    # ------------------------------------------------------------------
    inv1 = extract_level1(l1)
    inv2 = extract_level2(l0, l1, inv1, k=K_BLOWUP, lam=LAM)

    # Mean curvature H = tr(h)/2.  Sign depends on normal orientation (QR),
    # so take |H| for display.
    H = np.abs(inv1.mean_curvature[:, 0])     # (N,)
    # Gaussian curvature K = kappa1 * kappa2.  Sign-invariant since both
    # principal curvatures flip together when the normal flips.
    K = np.nan_to_num(inv1.gaussian_curvature, nan=0.0)  # (N,)

    # Curvature gradient Frobenius norm (Codazzi tensor magnitude)
    # grad_h has shape (N, 1, 2, 2, 2) for a surface in R^3
    grad_h_frob = np.sqrt(np.einsum("nabcd,nabcd->n", inv2.curvature_gradient,
                                    inv2.curvature_gradient))  # (N,)

    print(f"|H|  plane: {H[gt==0].mean():.3f}  "
          f"paraboloid: {H[gt==1].mean():.3f}  "
          f"(expected paraboloid ~{2*SCALE_Z:.3f})")
    print(f" K   plane: {K[gt==0].mean():.3f}  "
          f"paraboloid: {K[gt==1].mean():.3f}  "
          f"(expected paraboloid ~{4*SCALE_Z**2:.4f})")

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
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

    plot_umap_clusters(axes1[0], u0, c0, "Level 0 – original (3D)")
    plot_umap_clusters(axes1[1], u1, c1, "Level 1 – first blow-up (12D)")
    plot_umap_clusters(axes1[2], u2, c2, "Level 2 – second blow-up (156D)")

    fig1.suptitle(
        "Plane-Paraboloid: UMAP + HDBSCAN clustering at each blow-up level\n"
        "Shared tangent plane at origin -> one cluster at levels 0 & 1; "
        f"curvature differs (plane k=0, paraboloid k={2*SCALE_Z:.2f}) -> two clusters at level 2",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 6. Figure 2 – Differential invariants on original 3D cloud
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(20, 5))

    ax0 = fig2.add_subplot(1, 4, 1, projection="3d")
    plot3_labels(ax0, points, gt,
                 "Ground Truth\n(0 = plane, 1 = paraboloid)")

    ax1 = fig2.add_subplot(1, 4, 2, projection="3d")
    plot3_scalar(ax1, points, H,
                 f"|Mean Curvature| H  [level 1]\n"
                 f"(plane H~0, paraboloid H~{2*SCALE_Z:.2f} at origin)",
                 cmap="plasma")

    ax2 = fig2.add_subplot(1, 4, 3, projection="3d")
    plot3_scalar(ax2, points, K,
                 f"Gaussian Curvature K  [level 1]\n"
                 f"(plane K~0, paraboloid K~{4*SCALE_Z**2:.4f} at origin)",
                 cmap="inferno")

    ax3 = fig2.add_subplot(1, 4, 4, projection="3d")
    plot3_scalar(ax3, points, grad_h_frob,
                 "||grad h||_F  [level 2]\n"
                 "(Codazzi tensor magnitude; near zero for flat/const-curv patches)",
                 cmap="viridis", symmetric=False)

    fig2.suptitle(
        "Plane-Paraboloid: Differential Invariants from Iterated Blow-Up",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 7. Figure 3 - Original points colored by cluster (per blow-up level)
    # ------------------------------------------------------------------
    fig3 = plt.figure(figsize=(15, 5))

    ax0 = fig3.add_subplot(1, 3, 1, projection="3d")
    plot3_labels(ax0, points, c0,
                 "Level 0 clusters\n(colors from UMAP+HDBSCAN)")

    ax1 = fig3.add_subplot(1, 3, 2, projection="3d")
    plot3_labels(ax1, points, c1,
                 "Level 1 clusters\n(colors from UMAP+HDBSCAN)")

    ax2 = fig3.add_subplot(1, 3, 3, projection="3d")
    plot3_labels(ax2, points, c2,
                 "Level 2 clusters\n(colors from UMAP+HDBSCAN)")

    fig3.suptitle(
        "Plane-Paraboloid: Original point cloud colored by cluster at each level",
        fontsize=10,
    )
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
