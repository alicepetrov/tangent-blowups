"""
Tangential Intersection: Bilateral vs Blow-Up Laplacian Spectra
----------------------------------------------------------------
Two 2D surfaces in R^3 tangentially intersecting at the origin:

    Plane:       z = 0
    Paraboloid:  z = scale_z * (x^2 + y^2)

Both surfaces share position AND tangent plane at the origin, so a bilateral
kernel (spatial x angular at level 0) cannot distinguish them near the contact
point.  The blow-up Laplacian at level 2 encodes curvature, enabling separation.

Single SGP figure: N_SHOW rows x 2 columns (bilateral | blow-up).
Each row shows eigenvector phi_i colored on the point cloud, with the
eigenvalue in the row label.
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

# Spectrum
N_SHOW     = 3          # rows in the figure (eigenvectors to display)

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


def _bilateral_affinity(
    points: np.ndarray,
    normals: np.ndarray,
    sigma_x: float,
    sigma_n: float,
    k: int,
) -> sparse.csr_matrix:
    """Bilateral kernel: spatial Gaussian x normal Gaussian, sparse k-NN."""
    _, W, _ = bilateral_pointcloud_laplacian(
        points, normals, k=k, sigma_x=sigma_x, sigma_n=sigma_n,
        normalized=False, return_parts=True)
    return W


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
    _, _, l2 = levels

    # -- 3. Laplacians -------------------------------------------------------
    # Bilateral: spatial x normal Gaussian (standard bilateral kernel)
    W_bi = _bilateral_affinity(points, normals, SIGMA_X, SIGMA_U, k=K_NN)
    L_bi, _, _ = affinity_to_laplacian(W_bi, normalized=NORMALIZED)

    W_bu = product_affinity(l2, SIGMA_X, SIGMA_U, k=K_NN, symmetrize=True)
    L_bu, _, _ = affinity_to_laplacian(W_bu, normalized=NORMALIZED)

    # -- 4. Spectra (drop trivial constant eigenvector) ----------------------
    n_eigs = N_SHOW + 1  # +1 for the trivial eigenvalue we skip
    evals_bi, evecs_bi = laplacian_spectrum(
        L_bi, k=n_eigs, which="SM", drop_first=True,
    )
    evals_bu, evecs_bu = laplacian_spectrum(
        L_bu, k=n_eigs, which="SM", drop_first=True,
    )

    print(f"Bilateral eigenvalues: {np.array2string(evals_bi, precision=5)}")
    print(f"Blow-up   eigenvalues: {np.array2string(evals_bu, precision=5)}")

    # -- 5. Figure: N_SHOW rows x 2 cols + colorbar column -------------------
    fig = plt.figure(figsize=(4.5, 2.2 * N_SHOW))
    gs = GridSpec(
        N_SHOW, 3, figure=fig,
        width_ratios=[1, 1, 0.04],
        wspace=0.05, hspace=0.25,
    )

    for i in range(N_SHOW):
        phi_bi = evecs_bi[:, i]
        phi_bu = evecs_bu[:, i]

        # Shared symmetric color scale across both columns per row
        vabs = max(
            float(np.percentile(np.abs(phi_bi), 98)),
            float(np.percentile(np.abs(phi_bu), 98)),
            1e-8,
        )

        for col, phi, lam_val in [(0, phi_bi, evals_bi[i]),
                                   (1, phi_bu, evals_bu[i])]:
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
        cax = fig.add_subplot(gs[i, 2])
        fig.colorbar(sc, cax=cax)
        cax.tick_params(labelsize=5)

    # Column headers
    fig.axes[0].set_title(f"Bilateral", fontsize=9)
    fig.axes[1].set_title(f"Blow-up", fontsize=9)

    # Eigenvalue annotations under each panel
    for i in range(N_SHOW):
        for col, lam_val in [(0, evals_bi[i]), (1, evals_bu[i])]:
            ax = fig.axes[i * 3 + col]
            ax.text2D(
                0.5, -0.02, f"$\\lambda_{{{i+1}}}$", # or f"$\\lambda_{{{i+1}}}$ = {lam_val:.4f}"
                transform=ax.transAxes, ha="center", va="top", fontsize=7,
                color="0.3",
            )

    if SAVE_PDF:
        fig.savefig(OUT_PATH)
        print(f"Saved {OUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
