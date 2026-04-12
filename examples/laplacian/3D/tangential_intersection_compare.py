"""
Tangential Intersection: Bilateral vs Blow-Up Laplacian Spectra
----------------------------------------------------------------
Two 2D surfaces in R^3 tangentially intersecting at the origin:

    Plane:       z = 0
    Paraboloid:  z = scale_z * (x^2 + y^2)

Both surfaces share position AND tangent plane at the origin, so a bilateral
kernel (spatial x angular at level 0) cannot distinguish them near the contact
point.  The blow-up Laplacian at level 2 encodes curvature, enabling separation.

Figures:
  1. Eigenvector grid: N_SHOW rows x NCOLS columns (bilateral | [level-1] | level-2).
  2. Spectral gap: first eigenvalues per method.
  3. Spectral clustering: sign of Fiedler vector vs ground truth.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
from scipy import sparse

from tangent_blowups.testsupport import RandomSurface, plane_paraboloid_tangent, sample
from tangent_blowups.geometry.iterated_grassmann import iterated_blowup
from tangent_blowups.geometry.kernels import product_affinity, affinity_to_laplacian
from tangent_blowups.pointcloud import bilateral_pointcloud_laplacian, laplacian_spectrum

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_POINTS   = 8000
SCALE_XY   = 1.0
SCALE_Z    = 1.0
K_BLOWUP   = 20
ALPHA      = 1.0
LAM        = 1e-3
SEED       = 7

# Kernel parameters (shared for fair comparison)
K_NN       = 30
SIGMA_X    = 0.4
SIGMA_U    = 1.0
NORMALIZED = False

# Layout
SHOW_L1    = True         # include level-1 product column
N_SHOW     = 3            # rows in the eigenvector figure
N_EIGS_GAP = 8            # eigenvalues for spectral gap plot

# View angle
ELEV       = 25
AZIM       = -55

# Export
SAVE_PDF   = False
OUT_PATH   = "tangential_compare.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_equal3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    r = max(0.5 * float(np.max(limits[:, 1] - limits[:, 0])), 1e-3)
    ax.set_xlim3d(center[0] - r, center[0] + r)
    ax.set_ylim3d(center[1] - r, center[1] + r)
    ax.set_zlim3d(center[2] - r, center[2] + r)


def _clean_ax3d(ax) -> None:
    """Remove all chrome from a 3D axis."""
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(pad=-5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Style ---------------------------------------------------------------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 7,
        "text.usetex": False,        # set True if LaTeX is available
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    # -- 1. Sample -----------------------------------------------------------
    rng = np.random.default_rng(SEED)
    surface = plane_paraboloid_tangent(scale_xy=SCALE_XY, scale_z=SCALE_Z)
    strategy = RandomSurface(
        n=N_POINTS, u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0), rng=rng,
    )
    s = sample(surface, strategy, with_tangents=True, with_normals=False)

    points   = np.asarray(s.points,   dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)   # (N, 3, 2)
    u_vals   = np.asarray(s.params[0], dtype=float)
    gt       = np.floor(u_vals).astype(int)

    flat_T = tangents.reshape(len(points), -1)
    valid  = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(flat_T), axis=1)
    points, tangents, gt = points[valid], tangents[valid], gt[valid]

    # Normals from tangent frame (cross product of the two tangent vectors)
    normals = np.cross(tangents[:, :, 0], tangents[:, :, 1])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print(f"Sampled {valid.sum()} points  "
          f"({(gt == 0).sum()} plane, {(gt == 1).sum()} paraboloid)")

    # -- 2. Blow-up ----------------------------------------------------------
    levels = iterated_blowup(
        points, tangents, num_levels=2, k=K_BLOWUP, alpha=ALPHA, lam=LAM,
    )
    _, l1, l2 = levels

    # -- 3. Laplacians -------------------------------------------------------
    # Bilateral: spatial x normal Gaussian (standard bilateral kernel)
    _, W_bi, _ = bilateral_pointcloud_laplacian(
        points, normals, k=K_NN, sigma_x=SIGMA_X, sigma_n=SIGMA_U,
        normalized=False, return_parts=True)
    L_bi, _, _ = affinity_to_laplacian(W_bi, normalized=NORMALIZED)

    # Level-1 product (optional)
    if SHOW_L1:
        W_l1 = product_affinity(l1, SIGMA_X, SIGMA_U, k=K_NN, symmetrize=True)
        L_l1, _, _ = affinity_to_laplacian(W_l1, normalized=NORMALIZED)

    # Level-2 product
    W_l2 = product_affinity(l2, SIGMA_X, SIGMA_U, k=K_NN, symmetrize=True)
    L_l2, _, _ = affinity_to_laplacian(W_l2, normalized=NORMALIZED)

    # -- 4. Spectra (drop trivial constant eigenvector) ----------------------
    n_eigs = max(N_SHOW, N_EIGS_GAP) + 1
    evals_bi, evecs_bi = laplacian_spectrum(
        L_bi, k=n_eigs, which="SM", drop_first=True,
    )
    if SHOW_L1:
        evals_l1, evecs_l1 = laplacian_spectrum(
            L_l1, k=n_eigs, which="SM", drop_first=True,
        )
    evals_l2, evecs_l2 = laplacian_spectrum(
        L_l2, k=n_eigs, which="SM", drop_first=True,
    )

    print(f"Bilateral eigenvalues: {np.array2string(evals_bi[:N_SHOW], precision=5)}")
    if SHOW_L1:
        print(f"Level-1   eigenvalues: {np.array2string(evals_l1[:N_SHOW], precision=5)}")
    print(f"Level-2   eigenvalues: {np.array2string(evals_l2[:N_SHOW], precision=5)}")

    # -- 5. Eigenvector figure -----------------------------------------------
    col_data = [("Bilateral", evals_bi, evecs_bi)]
    if SHOW_L1:
        col_data.append(("Level-1 product", evals_l1, evecs_l1))
    col_data.append(("Level-2 product", evals_l2, evecs_l2))

    NCOLS = len(col_data)
    fig = plt.figure(figsize=(2.2 * NCOLS + 0.3, 2.2 * N_SHOW))
    gs = GridSpec(
        N_SHOW, NCOLS + 1, figure=fig,
        width_ratios=[1] * NCOLS + [0.04],
        wspace=0.05, hspace=0.25,
    )

    for i in range(N_SHOW):
        phis = [evecs[:, i] for _, _, evecs in col_data]

        # Shared symmetric color scale across all columns per row
        vabs = max(
            *(float(np.percentile(np.abs(phi), 98)) for phi in phis),
            1e-8,
        )

        for col, phi in enumerate(phis):
            ax = fig.add_subplot(gs[i, col], projection="3d")
            sc = ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=phi, s=3, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                rasterized=True, linewidths=0,
            )
            _clean_ax3d(ax)
            _set_equal3d(ax)
            ax.view_init(elev=ELEV, azim=AZIM)

        # Row colorbar
        cax = fig.add_subplot(gs[i, NCOLS])
        fig.colorbar(sc, cax=cax)
        cax.tick_params(labelsize=5)

    # Column headers
    for col, (title, _, _) in enumerate(col_data):
        fig.axes[col].set_title(title, fontsize=9)

    # Eigenvalue annotations under each panel
    for i in range(N_SHOW):
        for col, (_, evals, _) in enumerate(col_data):
            ax = fig.axes[i * (NCOLS + 1) + col]
            ax.text2D(
                0.5, -0.02, f"$\\lambda_{{{i+1}}}$",
                transform=ax.transAxes, ha="center", va="top", fontsize=7,
                color="0.3",
            )

    if SAVE_PDF:
        fig.savefig(OUT_PATH)
        print(f"Saved {OUT_PATH}")

    # -- 6. Spectral gap plot -------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(4.0, 2.5))

    gap_data = [("Bilateral", evals_bi[:N_EIGS_GAP], "C0", "o")]
    if SHOW_L1:
        gap_data.append(("Level-1 product", evals_l1[:N_EIGS_GAP], "C1", "s"))
    gap_data.append(("Level-2 product", evals_l2[:N_EIGS_GAP], "C2", "D"))

    indices = np.arange(1, N_EIGS_GAP + 1)
    for label, evals, color, marker in gap_data:
        ax2.plot(indices, evals, marker=marker, color=color, markersize=4,
                 linewidth=1.2, label=label)

    ax2.set_xlabel("Eigenvalue index $i$")
    ax2.set_ylabel("$\\lambda_i$")
    ax2.set_xticks(indices)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_xlim(0.5, N_EIGS_GAP + 0.5)
    ax2.set_ylim(bottom=0)
    fig2.tight_layout()

    if SAVE_PDF:
        gap_path = OUT_PATH.replace(".pdf", "_gap.pdf")
        fig2.savefig(gap_path)
        print(f"Saved {gap_path}")

    # -- 7. Spectral clustering (sign of Fiedler vector) ----------------------
    clust_data = [("Ground truth", gt)]
    clust_data.append(("Bilateral", (evecs_bi[:, 0] > 0).astype(int)))
    if SHOW_L1:
        clust_data.append(("Level-1 product", (evecs_l1[:, 0] > 0).astype(int)))
    clust_data.append(("Level-2 product", (evecs_l2[:, 0] > 0).astype(int)))

    n_clust_cols = len(clust_data)
    fig3 = plt.figure(figsize=(2.2 * n_clust_cols + 0.3, 2.5))
    gs3 = GridSpec(1, n_clust_cols, figure=fig3, wspace=0.05)

    for col, (title, labels) in enumerate(clust_data):
        ax = fig3.add_subplot(gs3[0, col], projection="3d")
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=labels, s=3, cmap="coolwarm", vmin=0, vmax=1,
            rasterized=True, linewidths=0,
        )
        _clean_ax3d(ax)
        _set_equal3d(ax)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_title(title, fontsize=9)

    if SAVE_PDF:
        clust_path = OUT_PATH.replace(".pdf", "_cluster.pdf")
        fig3.savefig(clust_path)
        print(f"Saved {clust_path}")

    plt.show()


if __name__ == "__main__":
    main()
