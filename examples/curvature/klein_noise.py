"""
Curvature estimation accuracy on the Klein bottle vs positional noise.

Two columns:
  (a) Ground-truth normals — isolates the blow-up estimator's error.
  (b) Local PCA normals     — includes normal estimation error.

Three rows of curvature quantities: K, H^2, ||h||.

X-axis:  parameter u in [0, 2pi]  (marginalised over v)
Y-axis:  median |est - gt|  (binned along u)
Lines:   one per noise level sigma.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, extract_level1
from tangent_blowups.pointcloud.tangent_estimation import estimate_normals_pca
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformSurface, klein_bottle, sample
from tangent_blowups.testsupport.geom_types import Sample


# =====================================================================
# Ground-truth curvature (copied from analytic.py to stay self-contained)
# =====================================================================

KLEIN_RADIUS, KLEIN_SCALE = 2.0, 1.0


def _fundamental_forms(dpdu, dpdv, dpuu, dpuv, dpvv):
    """(K, H) from parametric first/second fundamental forms."""
    E = np.sum(dpdu * dpdu, axis=-1)
    F = np.sum(dpdu * dpdv, axis=-1)
    G = np.sum(dpdv * dpdv, axis=-1)

    n_raw = np.cross(dpdu, dpdv)
    n_len = np.linalg.norm(n_raw, axis=-1, keepdims=True)
    n_len = np.where(n_len < 1e-30, 1e-30, n_len)
    n_hat = n_raw / n_len

    L = np.sum(dpuu * n_hat, axis=-1)
    M = np.sum(dpuv * n_hat, axis=-1)
    N = np.sum(dpvv * n_hat, axis=-1)

    det_I = E * G - F ** 2
    det_I = np.where(np.abs(det_I) < 1e-30, 1e-30, det_I)

    K = (L * N - M ** 2) / det_I
    H = (E * N - 2.0 * F * M + G * L) / (2.0 * det_I)
    return K, H


def klein_analytic(u, v, radius=KLEIN_RADIUS, scale=KLEIN_SCALE):
    u, v = np.asarray(u, float), np.asarray(v, float)
    cu, su = np.cos(u), np.sin(u)
    cuh, suh = np.cos(0.5 * u), np.sin(0.5 * u)
    sv, cv = np.sin(v), np.cos(v)
    s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)

    a = radius + cuh * sv - suh * s2v
    dadu = -0.5 * (suh * sv + cuh * s2v)
    dadv = cuh * cv - 2.0 * suh * c2v
    daduu = -0.25 * (cuh * sv - suh * s2v)
    daduv = -0.5 * (suh * cv + 2.0 * cuh * c2v)
    dadvv = -cuh * sv + 4.0 * suh * s2v

    dpdu = scale * np.stack([
        dadu * cu - a * su,
        dadu * su + a * cu,
        0.5 * (cuh * sv - suh * s2v),
    ], axis=-1)
    dpdv = scale * np.stack([
        dadv * cu, dadv * su,
        suh * cv + 2.0 * cuh * c2v,
    ], axis=-1)
    dpuu = scale * np.stack([
        daduu * cu - 2.0 * dadu * su - a * cu,
        daduu * su + 2.0 * dadu * cu - a * su,
        0.25 * (-suh * sv - cuh * s2v),
    ], axis=-1)
    dpuv = scale * np.stack([
        daduv * cu - dadv * su,
        daduv * su + dadv * cu,
        0.5 * (cuh * cv - 2.0 * suh * c2v),
    ], axis=-1)
    dpvv = scale * np.stack([
        dadvv * cu, dadvv * su,
        -suh * sv - 4.0 * cuh * s2v,
    ], axis=-1)

    K, H = _fundamental_forms(dpdu, dpdv, dpuu, dpuv, dpvv)
    H2 = H ** 2
    total = np.sqrt(np.maximum(4.0 * H2 - 2.0 * K, 0.0))
    return K, H2, total


# =====================================================================
# Sampling & curvature estimation
# =====================================================================

def _sample_surface(nu: int, nv: int, sigma: float, seed: int):
    """Returns (points, gt_normals, u_flat, v_flat) with positional noise."""
    surface = klein_bottle(radius=KLEIN_RADIUS, scale=KLEIN_SCALE)
    strategy = UniformSurface(
        nu=nu, nv=nv,
        u_bounds=(0.0, 2.0 * np.pi),
        v_bounds=(0.0, 2.0 * np.pi),
    )
    s = sample(surface, strategy, with_tangents=False, with_normals=True)

    points = np.asarray(s.points, dtype=float).reshape(-1, 3)
    normals = np.asarray(s.normals, dtype=float).reshape(-1, 3)
    U, V = s.params
    u_flat = np.asarray(U, dtype=float).ravel()
    v_flat = np.asarray(V, dtype=float).ravel()

    # Filter degenerate normals
    nrm_len = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(points).all(1) & (nrm_len > 1e-8)
    points, normals = points[valid], normals[valid]
    u_flat, v_flat = u_flat[valid], v_flat[valid]
    normals = normalize_vectors(normals)

    rng = np.random.default_rng(seed)
    points += rng.normal(scale=max(sigma, 1e-30), size=points.shape)

    return points, normals, u_flat, v_flat


def _normals_to_frames(normals: np.ndarray) -> np.ndarray:
    """Build (N, 3, 2) tangent frames from normals."""
    N = len(normals)
    frames = np.empty((N, 3, 2))
    for i in range(N):
        n = normals[i]
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, ref)
        t1 /= np.linalg.norm(t1) + 1e-30
        t2 = np.cross(n, t1)
        frames[i, :, 0] = t1
        frames[i, :, 1] = t2
    return frames


def _blowup_curvature_3d(points, normals, *, k=15, alpha=5.0, lam=1e-4):
    """Estimate K, H2, total via level-1 blow-up on a surface."""
    frames = _normals_to_frames(normals)
    l0 = BlowUpLevel.from_point_tangents(points, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam, spatial_knn=True)
    inv = extract_level1(l1)
    K = inv.gaussian_curvature
    Tc = inv.total_curvature[:, 0]
    H2 = (Tc ** 2 + 2.0 * K) / 4.0
    return {"K": K, "H2": H2, "total": Tc}


# =====================================================================
# Binning helpers
# =====================================================================

def _bin_median(u_param, values, n_bins=60):
    """Bin values by u and return (bin_centres, medians)."""
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(u_param, edges) - 1, 0, n_bins - 1)

    medians = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = values[idx == i]
        sel = sel[np.isfinite(sel)]
        if len(sel) > 2:
            medians[i] = np.median(sel)
    return centres, medians


def _sigma_label(sigma):
    return rf"$\sigma={sigma:.3f}$" if sigma < 0.01 else rf"$\sigma={sigma:.2f}$"


# =====================================================================
# Main
# =====================================================================

QUANTITIES = ["K", "H2", "total"]
_Q_LABELS = {
    "K": r"$|K - K_{\mathrm{gt}}|$",
    "H2": r"$|H^2 - H^2_{\mathrm{gt}}|$",
    "total": r"$|\,\|h\| - \|h\|_{\mathrm{gt}}|$",
}
_Q_TITLES = {
    "K": r"Gaussian $K$",
    "H2": r"Squared mean $H^2$",
    "total": r"Total $\|h\|$",
}


def main():
    nu, nv = 60, 60
    seed = 42
    k_bu = 15
    n_bins = 40
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    k_pca = 15

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    })

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(noise_levels) - 1, 1)) for i in range(len(noise_levels))]

    n_q = len(QUANTITIES)
    fig, axes = plt.subplots(n_q, 2, figsize=(7.0, 3.2 * n_q),
                             constrained_layout=True)

    for sigma, color in zip(noise_levels, colors):
        points, gt_normals, u_flat, v_flat = _sample_surface(nu, nv, sigma, seed)
        gt_K, gt_H2, gt_total = klein_analytic(u_flat, v_flat)
        gt = {"K": gt_K, "H2": gt_H2, "total": gt_total}

        # (a) Ground-truth normals
        est_gt = _blowup_curvature_3d(points, gt_normals, k=k_bu)

        # (b) PCA normals
        pca_normals = normalize_vectors(estimate_normals_pca(points, k=k_pca))
        est_pca = _blowup_curvature_3d(points, pca_normals, k=k_bu)

        label = _sigma_label(sigma)

        for row, qname in enumerate(QUANTITIES):
            # GT normals
            err_gt = np.abs(est_gt[qname] - gt[qname])
            centres, med_gt = _bin_median(u_flat, err_gt, n_bins)
            axes[row, 0].plot(centres, med_gt, lw=1.3, color=color, label=label)

            # PCA normals
            err_pca = np.abs(est_pca[qname] - gt[qname])
            centres, med_pca = _bin_median(u_flat, err_pca, n_bins)
            axes[row, 1].plot(centres, med_pca, lw=1.3, color=color, label=label)

    # ---- Formatting ----
    u_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    u_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for row, qname in enumerate(QUANTITIES):
        axes[row, 0].set_title(f"{_Q_TITLES[qname]} — GT normals")
        axes[row, 1].set_title(f"{_Q_TITLES[qname]} — PCA normals")

    for ax in axes.ravel():
        ax.set_xlabel(r"parameter $u$")
        ax.set_ylabel("median |error|")
        ax.set_yscale("log")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(u_ticks)
        ax.set_xticklabels(u_labels)
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(ticker.LogLocator(
            subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.25, lw=0.6)
        ax.legend(frameon=True, fancybox=False, edgecolor="0.8")

    fig.savefig("klein_curvature_vs_noise.pdf", bbox_inches="tight", dpi=300)
    print("Saved klein_curvature_vs_noise.pdf")
    plt.show()

    # ---- Spatial error maps ----
    _plot_spatial_error(nu=nu, nv=nv, seed=seed, k_bu=k_bu,
                        noise_levels=noise_levels, k_pca=k_pca)


def _plot_spatial_error(*, nu, nv, seed, k_bu, noise_levels, k_pca):
    """Per-point curvature error colored on the surface (green=accurate, red=inaccurate).

    Layout: for each quantity, two rows (GT normals, PCA normals) x noise columns.
    """
    n_cols = len(noise_levels)
    tangent_modes = [
        ("GT normals", True),
        ("PCA normals", False),
    ]

    # Use Gaussian K as the representative quantity for the spatial plot
    qname = "K"

    row_labels = []
    for mode_name, _ in tangent_modes:
        row_labels.append(f"Klein $K$\n{mode_name}")
    n_rows = len(row_labels)

    all_log_err = []
    panel_data = []

    for _, use_gt in tangent_modes:
        row_data = []
        for sigma in noise_levels:
            points, gt_normals, u_flat, v_flat = _sample_surface(nu, nv, sigma, seed)
            gt_K, _, _ = klein_analytic(u_flat, v_flat)

            if use_gt:
                normals = gt_normals
            else:
                normals = normalize_vectors(estimate_normals_pca(points, k=k_pca))

            est = _blowup_curvature_3d(points, normals, k=k_bu)
            err = np.abs(est[qname] - gt_K)
            log_err = np.log10(np.clip(err, 1e-6, None))
            all_log_err.append(log_err)
            row_data.append((points, log_err))
        panel_data.append(row_data)

    vmin = np.percentile(np.concatenate(all_log_err), 2)
    vmax = np.percentile(np.concatenate(all_log_err), 98)

    fig = plt.figure(figsize=(2.4 * n_cols, 2.8 * n_rows), constrained_layout=True)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1,
                                 projection="3d")
            axes[row, col] = ax
            points, log_err = panel_data[row][col]

            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=log_err, s=2, cmap="RdYlGn_r",
                vmin=vmin, vmax=vmax,
                edgecolors="none", rasterized=True, depthshade=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")

            if row == 0:
                s = noise_levels[col]
                sigma_str = f"{s:.3f}" if s < 0.01 else f"{s:.2f}"
                ax.set_title(rf"$\sigma={sigma_str}$", fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, labelpad=12)

    # Shared colorbar
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    sm = mcm.ScalarMappable(cmap="RdYlGn_r",
                            norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        fraction=0.02, pad=0.05, aspect=40, shrink=0.8)
    cbar.set_label(r"$\log_{10}|K - K_{\mathrm{gt}}|$", fontsize=9)

    fig.savefig("klein_curvature_spatial_error.pdf", bbox_inches="tight", dpi=300)
    print("Saved klein_curvature_spatial_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
