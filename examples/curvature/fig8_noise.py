"""
Curvature estimation accuracy vs positional noise.

One row per curve (figure-8, circle).  Two columns:
  (a) Ground-truth tangents — isolates the blow-up estimator's error.
  (b) Local PCA tangents     — includes tangent estimation error.

X-axis:  parameter t in [0, 2pi]
Y-axis:  median |kappa_est - kappa_gt|  (binned along t)
Lines:   one per noise level sigma.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, extract_level1
from tangent_blowups.pointcloud.tangent_estimation import estimate_tangents_pca
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformCurve, circle, figure8, sample


# =====================================================================
# Ground-truth curvature
# =====================================================================

def fig8_curvature(t: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Signed curvature kappa(t) for the figure-8  x = s*sin(t), y = s*sin(t)*cos(t).

    kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^{3/2}
    """
    t = np.asarray(t, dtype=float)
    s = scale
    dx = s * np.cos(t)
    dy = s * np.cos(2.0 * t)
    ddx = -s * np.sin(t)
    ddy = -2.0 * s * np.sin(2.0 * t)

    num = dx * ddy - dy * ddx
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom = np.where(denom < 1e-30, 1e-30, denom)
    return num / denom


def circle_curvature(t: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Constant curvature kappa = 1/radius for all t."""
    return np.full_like(t, 1.0 / radius, dtype=float)


# =====================================================================
# Sampling & curvature estimation
# =====================================================================

def _sample_curve(curve_fn, n: int, sigma: float, seed: int, t_max: float = 2.0 * np.pi):
    """Returns (points, gt_tangents, params) with positional noise added."""
    strategy = UniformCurve(n=n, t_min=0.0, t_max=t_max, endpoint=False)
    s = sample(curve_fn, strategy, with_tangents=True, with_normals=False)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)
    params = np.asarray(s.params, dtype=float)

    rng = np.random.default_rng(seed)
    points += rng.normal(scale=max(sigma, 1e-30), size=points.shape)

    tangents = normalize_vectors(tangents)
    return points, tangents, params


def _blowup_curvature_2d(points, tangents, *, k=15, alpha=5.0, lam=1e-4):
    """Estimate scalar curvature of a 2D curve via level-1 blow-up."""
    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam, spatial_knn=True)
    inv = extract_level1(l1)
    return inv.principal_curvatures[:, 0]


# =====================================================================
# Binning helpers
# =====================================================================

def _bin_median(t_param, values, n_bins=60):
    """Bin values by t_param and return (bin_centres, medians)."""
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(t_param, edges) - 1, 0, n_bins - 1)

    medians = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = values[idx == i]
        sel = sel[np.isfinite(sel)]
        if len(sel) > 2:
            medians[i] = np.median(sel)
    return centres, medians


# =====================================================================
# Main
# =====================================================================

def main():
    n = 2000
    seed = 42
    k_bu = 15
    n_bins = 50
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02]
    k_pca = 15

    curves = [
        ("Figure-8",  figure8(scale=2.0),    lambda t: fig8_curvature(t, scale=2.0)),
        ("Circle",    circle(radius=2.0),     lambda t: circle_curvature(t, radius=2.0)),
    ]

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

    n_rows = len(curves)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.0, 3.2 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (curve_name, curve_fn, kappa_fn) in enumerate(curves):
        ax_gt = axes[row, 0]
        ax_pca = axes[row, 1]

        for sigma, color in zip(noise_levels, colors):
            points, gt_tangents, params = _sample_curve(curve_fn, n, sigma, seed)
            kappa_gt = np.abs(kappa_fn(params))

            # (a) Ground-truth tangents
            kappa_est_gt = np.abs(_blowup_curvature_2d(points, gt_tangents, k=k_bu))
            err_gt = np.abs(kappa_est_gt - kappa_gt)
            centres, med_gt = _bin_median(params, err_gt, n_bins)
            label = rf"$\sigma={sigma:.3f}$" if sigma < 0.01 else rf"$\sigma={sigma:.2f}$"
            ax_gt.plot(centres, med_gt, lw=1.3, color=color, label=label)

            # (b) Local PCA tangents
            pca_tangents = estimate_tangents_pca(points, k=k_pca, tangent_dim=1)
            pca_tangents = normalize_vectors(pca_tangents)
            kappa_est_pca = np.abs(_blowup_curvature_2d(points, pca_tangents, k=k_bu))
            err_pca = np.abs(kappa_est_pca - kappa_gt)
            centres, med_pca = _bin_median(params, err_pca, n_bins)
            label_pca = rf"$\sigma={sigma:.3f}$" if sigma < 0.01 else rf"$\sigma={sigma:.2f}$"
            ax_pca.plot(centres, med_pca, lw=1.3, color=color, label=label_pca)

        ax_gt.set_title(f"{curve_name} — ground-truth tangents")
        ax_pca.set_title(f"{curve_name} — local PCA tangents")

    # ---- Formatting ----
    v_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    v_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for ax in axes.ravel():
        ax.set_xlabel(r"parameter $t$")
        ax.set_ylabel(r"median $|\kappa - \kappa_{\mathrm{gt}}|$")
        ax.set_yscale("log")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(v_ticks)
        ax.set_xticklabels(v_labels)
        ax.yaxis.set_major_locator(ticker.LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(ticker.LogLocator(
            subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.25, lw=0.6)
        ax.legend(frameon=True, fancybox=False, edgecolor="0.8")

    fig.savefig("fig8_curvature_vs_noise.pdf", bbox_inches="tight", dpi=300)
    print("Saved fig8_curvature_vs_noise.pdf")
    plt.show()

    # ---- Spatial error maps ----
    _plot_spatial_error(curves, n=n, seed=seed, k_bu=k_bu,
                        noise_levels=noise_levels, k_pca=k_pca)


def _plot_spatial_error(curves, *, n, seed, k_bu, noise_levels, k_pca):
    """Per-point curvature error colored on the curve (green=accurate, red=inaccurate).

    Layout: for each curve, two rows (GT tangents, PCA tangents) x noise columns.
    """
    n_cols = len(noise_levels)
    tangent_modes = [
        ("GT tangents", True),
        ("PCA tangents", False),
    ]
    # One row per (curve, tangent_mode) pair
    row_labels = []
    for curve_name, _, _ in curves:
        for mode_name, _ in tangent_modes:
            row_labels.append(f"{curve_name}\n{mode_name}")
    n_rows = len(row_labels)

    # Pre-compute all errors to get a shared color range
    all_log_err = []
    panel_data = []  # panel_data[row][col] = (points, log_err)

    for curve_name, curve_fn, kappa_fn in curves:
        for _, use_gt in tangent_modes:
            row_data = []
            for sigma in noise_levels:
                points, gt_tangents, params = _sample_curve(curve_fn, n, sigma, seed)
                kappa_gt = np.abs(kappa_fn(params))

                if use_gt:
                    tangents = gt_tangents
                else:
                    tangents = normalize_vectors(
                        estimate_tangents_pca(points, k=k_pca, tangent_dim=1))

                kappa_est = np.abs(_blowup_curvature_2d(points, tangents, k=k_bu))
                err = np.abs(kappa_est - kappa_gt)
                log_err = np.log10(np.clip(err, 1e-6, None))
                all_log_err.append(log_err)
                row_data.append((points, log_err))
            panel_data.append(row_data)

    # Fixed color range across all panels
    vmin = np.percentile(np.concatenate(all_log_err), 2)
    vmax = np.percentile(np.concatenate(all_log_err), 98)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.4 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            points, log_err = panel_data[row][col]

            sc = ax.scatter(
                points[:, 0], points[:, 1],
                c=log_err, s=6, cmap="RdYlGn_r",
                vmin=vmin, vmax=vmax,
                edgecolors="none", rasterized=True,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                s = noise_levels[col]
                sigma_str = f"{s:.3f}" if s < 0.01 else f"{s:.2f}"
                ax.set_title(rf"$\sigma={sigma_str}$", fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(),
                        fraction=0.02, pad=0.02, aspect=40)
    cbar.set_label(r"$\log_{10}|\kappa - \kappa_{\mathrm{gt}}|$", fontsize=9)

    fig.savefig("fig8_curvature_spatial_error.pdf", bbox_inches="tight", dpi=300)
    print("Saved fig8_curvature_spatial_error.pdf")
    plt.show()


if __name__ == "__main__":
    main()
