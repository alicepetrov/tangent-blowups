"""
Tangent Cylinders: Iterated Blow-Up and the Limits of Scalar Curvature
-----------------------------------------------------------------------
Two cylinders of radius r with perpendicular axes touch at the origin.
At every point on each cylinder:

    H  = 1/(2r)   [mean curvature]
    K  = 0        [Gaussian curvature]
    kappa_1 = 0,  kappa_2 = 1/r   [principal curvatures, same set]

All scalar curvature invariants are *identical* on both cylinders — yet
the surfaces are geometrically distinct because the principal *directions*
are rotated 90 degrees:

    B_1 = [[0,    0  ],    B_2 = [[1/r,  0],
           [0,  1/r  ]]           [0,    0]]

Level 0 (positions): fails near the contact — the cylinders overlap.
Level 1 (+ tangent projector P^(0)): also fails near contact — P^(0) is
  the same at the contact where both cylinders share the tangent plane.
Level 2 (+ tangent projector P^(1)): succeeds everywhere — P^(1) encodes
  the full shape-operator *matrix* B_i, which differs between the two
  cylinders everywhere (even though its eigenvalues agree).

This is the key distinction from the plane-paraboloid example: there,
scalar curvature (H, K) already differs; here it does not.  Only the
full operator -- captured by the second iterated blow-up -- suffices.

Figures
-------
Figure 1 – Input point cloud with surface normals drawn as quivers.
           Near the contact both normals point in the same direction;
           away from it they diverge in perpendicular planes.

Figure 2 – UMAP + HDBSCAN at levels 0, 1, 2.
           Separation achieved only at level 2.

Figure 3 – Scalar invariants |H| and K on the 3D point cloud.
           Both are identical on the two cylinders (H = 1/(2r), K = 0).
           The shape-operator *direction* (principal angle) encodes the
           difference; it is shown in the rightmost panel.

Figure 4 – Original 3D points coloured by cluster at each blow-up level.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP            # type: ignore[import-untyped]
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from tangent_blowups.testsupport import RandomSurface, cylinders_tangent, sample
from tangent_blowups.geometry.iterated_grassmann import (
    extract_level1,
    iterated_blowup,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RADIUS     = 1.0
HALF_AXIS  = 1.2     # axis extent on each cylinder
HALF_ANGLE = 0.55    # azimuthal sampling window (radians ~31 deg)

N_POINTS   = 5000    # total sample points (1200 per cylinder)
K_BLOWUP   = 20      # k-NN for blow-up curvature regression
ALPHA  = 1.0  # Chordal-Sasaki weight
LAM        = 1e-3    # ridge regularisation
SEED       = 42

UMAP_NEIGHBORS = 10
UMAP_MIN_DIST  = 0.10
UMAP_SEED      = 0

HDBSCAN_MIN_SIZE = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_umap(X: np.ndarray) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    return UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        n_components=2, random_state=UMAP_SEED,
    ).fit_transform(Xs)


def hdbscan_cluster(umap2: np.ndarray) -> np.ndarray:
    return HDBSCAN(min_cluster_size=HDBSCAN_MIN_SIZE).fit_predict(umap2)


def _label_colors(labels: np.ndarray) -> list:
    n_pos = len([u for u in set(labels) if u >= 0])
    cmap  = plt.get_cmap("tab10", max(n_pos, 1))
    return [cmap(int(l)) if l >= 0 else (0.55, 0.55, 0.55, 1.0) for l in labels]


def _set_equal3d(ax) -> None:
    lims   = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = lims.mean(axis=1)
    r      = max(0.5 * float(np.max(lims[:, 1] - lims[:, 0])), 1e-3)
    ax.set_xlim3d(center[0]-r, center[0]+r)
    ax.set_ylim3d(center[1]-r, center[1]+r)
    ax.set_zlim3d(center[2]-r, center[2]+r)


def plot_umap_clusters(ax, umap2, labels, title):
    n_cl    = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    ax.scatter(umap2[:, 0], umap2[:, 1],
               c=_label_colors(labels), s=6, rasterized=True)
    ax.set_title(f"{title}\n({n_cl} cluster(s), {n_noise} noise)", fontsize=9)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")


def plot3_labels(ax, pts, labels, title):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=_label_colors(labels), s=5, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_equal3d(ax)


def plot3_scalar(ax, pts, vals, title, cmap="plasma", symmetric=False):
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
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    _set_equal3d(ax)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    r = RADIUS

    # ------------------------------------------------------------------
    # 1. Sample both cylinders
    # ------------------------------------------------------------------
    rng      = np.random.default_rng(SEED)
    surface  = cylinders_tangent(radius=r, half_axis=HALF_AXIS,
                                  half_angle=HALF_ANGLE)
    strategy = RandomSurface(n=N_POINTS, u_bounds=(0.0, 2.0),
                             v_bounds=(0.0, 1.0), rng=rng)
    s        = sample(surface, strategy, with_tangents=True, with_normals=True)

    points   = np.asarray(s.points,    dtype=float)   # (N, 3)
    tangents = np.asarray(s.tangents,  dtype=float)   # (N, 3, 2)
    normals  = np.asarray(s.normals,   dtype=float)   # (N, 3)
    u_vals   = np.asarray(s.params[0], dtype=float)   # (N,)
    gt       = np.floor(u_vals).astype(int)            # 0 = C1, 1 = C2

    # Drop degenerate frames
    flat_T = tangents.reshape(len(points), -1)
    valid  = (np.all(np.isfinite(points), axis=1) &
              np.all(np.isfinite(flat_T), axis=1) &
              np.all(np.isfinite(normals), axis=1))
    points, tangents, normals, gt = (
        points[valid], tangents[valid], normals[valid], gt[valid]
    )

    print(f"Sampled {valid.sum()} points  "
          f"({(gt==0).sum()} cylinder 1,  {(gt==1).sum()} cylinder 2)")

    # ------------------------------------------------------------------
    # 2. Iterated blow-up (need level 2 for full shape-operator encoding)
    # ------------------------------------------------------------------
    levels   = iterated_blowup(points, tangents, num_levels=2,
                                k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l0, l1, l2 = levels
    print(f"Embedding dims: {l0.D} -> {l1.D} -> {l2.D}")

    # ------------------------------------------------------------------
    # 3. Scalar invariants from level 1
    #    d=2, n_comp=1 (hypersurface) -> shape operator is 2x2
    #    H = tr(B)/2,   K = det(B)
    #    Ground truth for both cylinders: H = 1/(2r), K = 0 everywhere.
    # ------------------------------------------------------------------
    inv1 = extract_level1(l1)

    H = np.abs(inv1.mean_curvature[:, 0])           # (N,)  |H|
    K = np.nan_to_num(inv1.gaussian_curvature, nan=0.0)  # (N,)

    # Principal angle: angle of the zero-curvature eigenvector in the
    # tangent plane at each point, used as a proxy for principal direction.
    # For C1 (zero-curv along e_x=e_1) -> angle ~0; for C2 -> angle ~pi/2.
    B = inv1.shape_operator[:, 0, :, :]   # (N, 2, 2) shape operator matrix
    # Eigenvector of the smaller eigenvalue = principal direction of min curvature
    evals, evecs = np.linalg.eigh(B)      # evals ascending, evecs[:,0] = min
    # Angle of the min-curvature eigenvector in (e_1, e_2) coords: atan2(v1, v0)
    v_min = evecs[:, :, 0]                # (N, 2)
    principal_angle = np.arctan2(v_min[:, 1], v_min[:, 0])   # (N,) in (-pi, pi]

    print(f"  |H|  C1: {H[gt==0].mean():.4f}  C2: {H[gt==1].mean():.4f}  "
          f"(expected both ~{1/(2*r):.4f})")
    print(f"   K   C1: {K[gt==0].mean():.4f}  C2: {K[gt==1].mean():.4f}  "
          f"(expected both ~0)")
    print(f"  principal angle  C1 mean: {np.degrees(principal_angle[gt==0]).mean():.1f} deg  "
          f"C2 mean: {np.degrees(principal_angle[gt==1]).mean():.1f} deg  "
          f"(expected ~0 and ~90)")

    # ------------------------------------------------------------------
    # 4. UMAP projections + HDBSCAN
    # ------------------------------------------------------------------
    print("Computing UMAP projections...")
    u0 = embed_umap(l0.embedded)
    u1 = embed_umap(l1.embedded)
    u2 = embed_umap(l2.embedded)

    c0 = hdbscan_cluster(u0)
    c1 = hdbscan_cluster(u1)
    c2 = hdbscan_cluster(u2)

    for lvl, labels in enumerate([c0, c1, c2]):
        n_cl    = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        print(f"  Level {lvl}: {n_cl} cluster(s),  {n_noise} noise pts")

    # ------------------------------------------------------------------
    # Figure 1 – Input point cloud with surface normals
    # ------------------------------------------------------------------
    # Subsample for legible quiver density
    rng_plot = np.random.default_rng(0)
    n_quiver = 300   # arrows per cylinder
    idx0 = rng_plot.choice(np.where(gt == 0)[0], size=min(n_quiver, (gt==0).sum()),
                            replace=False)
    idx1 = rng_plot.choice(np.where(gt == 1)[0], size=min(n_quiver, (gt==1).sum()),
                            replace=False)

    nl = 0.20   # normal arrow display length

    fig0 = plt.figure(figsize=(14, 6))

    # Left: single combined view coloured by component
    ax_l = fig0.add_subplot(1, 2, 1, projection="3d")
    for idx, col, lbl in [(idx0, "steelblue", "Cylinder 1 (axis x)"),
                           (idx1, "tomato",    "Cylinder 2 (axis y)")]:
        p = points[idx]
        n = normals[idx]
        ax_l.scatter(p[:, 0], p[:, 1], p[:, 2], s=6, color=col,
                     alpha=0.5, rasterized=True)
        ax_l.quiver(p[:, 0], p[:, 1], p[:, 2],
                    nl*n[:, 0], nl*n[:, 1], nl*n[:, 2],
                    color=col, linewidth=0.8, alpha=0.8, label=lbl)
    ax_l.set_title("Point cloud + surface normals\n(both cylinders)", fontsize=9)
    ax_l.set_xlabel("x"); ax_l.set_ylabel("y"); ax_l.set_zlabel("z")
    ax_l.legend(fontsize=8, loc="upper left")
    _set_equal3d(ax_l)

    # Right: side-by-side subplots showing each cylinder separately
    ax_r = fig0.add_subplot(1, 2, 2, projection="3d")
    for idx, col, lbl in [(idx0, "steelblue", "C1: normal rotates in yz-plane"),
                           (idx1, "tomato",    "C2: normal rotates in xz-plane")]:
        p = points[idx]
        n = normals[idx]
        ax_r.quiver(p[:, 0], p[:, 1], p[:, 2],
                    nl*n[:, 0], nl*n[:, 1], nl*n[:, 2],
                    color=col, linewidth=0.8, alpha=0.9, label=lbl)
    ax_r.set_title("Surface normals only\n(C1 rotates in yz, C2 rotates in xz)", fontsize=9)
    ax_r.set_xlabel("x"); ax_r.set_ylabel("y"); ax_r.set_zlabel("z")
    ax_r.legend(fontsize=8, loc="upper left")
    _set_equal3d(ax_r)

    fig0.suptitle(
        f"Tangent cylinders (r={r}): input geometry\n"
        "Near the contact both normals point along +z; "
        "away from it they diverge in perpendicular planes",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2 – UMAP + HDBSCAN
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

    plot_umap_clusters(axes1[0], u0, c0,
                       "Level 0 (3D positions)\nno curvature info")
    plot_umap_clusters(axes1[1], u1, c1,
                       "Level 1 (12D  pos + tangent plane)\nsame tangent plane at contact")
    plot_umap_clusters(axes1[2], u2, c2,
                       "Level 2 (156D  + shape operator)\nfull B_i separates everywhere")

    fig1.suptitle(
        f"Tangent cylinders (r={r}): UMAP + HDBSCAN at each blow-up level\n"
        "Scalar invariants identical (H = 1/(2r), K = 0) on both cylinders --"
        " only the full shape-operator matrix (level 2) separates them",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 2 – Scalar invariants (same on both) vs principal direction
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(20, 5))

    ax0 = fig2.add_subplot(1, 4, 1, projection="3d")
    plot3_labels(ax0, points, gt, "Ground truth\n(blue=C1, orange=C2)")

    ax1 = fig2.add_subplot(1, 4, 2, projection="3d")
    plot3_scalar(ax1, points, H,
                 f"|Mean curvature| H  [level 1]\n"
                 f"Both cylinders: H = 1/(2r) = {1/(2*r):.3f}",
                 cmap="plasma")

    ax2 = fig2.add_subplot(1, 4, 3, projection="3d")
    plot3_scalar(ax2, points, K,
                 "Gaussian curvature K  [level 1]\n"
                 "Both cylinders: K = 0",
                 cmap="plasma")

    ax3 = fig2.add_subplot(1, 4, 4, projection="3d")
    plot3_scalar(ax3, points, np.degrees(principal_angle),
                 "Principal direction angle (deg)  [level 1]\n"
                 "C1 ~0 deg (zero-curv along x),  C2 ~90 deg (zero-curv along y)",
                 cmap="coolwarm", symmetric=True)

    fig2.suptitle(
        "Tangent cylinders: scalar invariants are identical -- principal directions differ\n"
        "H and K alone cannot separate the two cylinders; the direction of B is decisive",
        fontsize=10,
    )
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Figure 3 – 3D clouds coloured by cluster at each level
    # ------------------------------------------------------------------
    fig3 = plt.figure(figsize=(15, 5))

    ax0 = fig3.add_subplot(1, 3, 1, projection="3d")
    plot3_labels(ax0, points, c0,
                 "Level 0 clusters\n(positions only)")

    ax1 = fig3.add_subplot(1, 3, 2, projection="3d")
    plot3_labels(ax1, points, c1,
                 "Level 1 clusters\n(pos + tangent plane)")

    ax2 = fig3.add_subplot(1, 3, 3, projection="3d")
    plot3_labels(ax2, points, c2,
                 "Level 2 clusters\n(full shape operator)")

    fig3.suptitle(
        "Tangent cylinders: original point cloud coloured by cluster at each blow-up level",
        fontsize=10,
    )
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
