"""
Geodesic leakage on two intersecting planes vs positional noise.

Tests whether lifted geodesic distances "leak" across two perpendicular planes
that share an intersection line.  Two rows:
  (a) Lifted, ground-truth tangents — isolates the blow-up method's behaviour.
  (b) Lifted, PCA-estimated tangents — includes tangent estimation error.

Columns: increasing positional noise levels sigma.
All panels use the same magma colormap and colour range.

A summary line plot shows the fraction of cross-sheet points that receive
finite geodesic distance (= reachable through the lifted kernel graph).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.pointcloud import (
    lifted_heat_method, estimate_normals_pca, estimate_tangents_pca,
)
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import RandomSurface, plane_cross, sample


# =====================================================================
# Sampling
# =====================================================================

def _sample_plane_cross(
    n: int, sigma: float, seed: int, scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample two intersecting planes, add noise.

    Returns (points, gt_normals, sheet_id, src_idx).
    sheet_id[i] == 0 for the xy-plane, 1 for the xz-plane.
    Source is placed near a corner of sheet 0, far from the intersection.
    """
    surface = plane_cross(scale=scale)
    rng = np.random.default_rng(seed)
    strategy = RandomSurface(
        n=n,
        u_bounds=(0.0, 2.0),
        v_bounds=(0.0, 1.0),
        rng=rng,
    )
    s = sample(surface, strategy, with_tangents=False, with_normals=True)

    u_params = np.asarray(s.params[0], dtype=float)
    points = np.asarray(s.points, dtype=float).reshape(-1, 3)
    normals = np.asarray(s.normals, dtype=float).reshape(-1, 3)
    sheet_id = (u_params >= 1.0).astype(int)

    # Drop degenerate normals
    norms = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(points).all(axis=1) & (norms > 1e-8)
    points, normals, sheet_id = points[valid], normals[valid], sheet_id[valid]
    normals = normalize_vectors(normals)

    # Add positional noise
    rng2 = np.random.default_rng(seed + 1)
    points = points + rng2.normal(scale=max(sigma, 1e-30), size=points.shape)

    # Source: corner of sheet 0 far from intersection
    sheet0_mask = sheet_id == 0
    target = np.array([scale, scale, 0.0])
    local_idx = int(np.argmin(
        np.linalg.norm(points[sheet0_mask] - target, axis=1),
    ))
    src_idx = int(np.flatnonzero(sheet0_mask)[local_idx])

    return points, normals, sheet_id, src_idx


# =====================================================================
# Lifted geodesic
# =====================================================================

def _lifted_geodesic(
    points, normals, source_index, *, k=20, alpha=1.0,
    sigma_x=0.5, sigma_u=0.5, t_scale=1.0,
):
    frames = _normals_to_tangent_frames(normals)
    return _lifted_geodesic_from_frames(
        points, frames, source_index,
        k=k, alpha=alpha, sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
    )


def _lifted_geodesic_from_frames(
    points, frames, source_index, *, k=20, alpha=1.0,
    sigma_x=0.5, sigma_u=0.5, t_scale=1.0,
):
    """Lifted geodesic from pre-built (N, n, d) tangent frames."""
    l0 = BlowUpLevel.from_point_tangents(points, frames)
    level = l0.lift(alpha=alpha, k=k, lam=0.0)
    phi, u, _, _ = lifted_heat_method(
        level, source_index=source_index, k=k,
        kernel="product", sigma_x=sigma_x, sigma_u=sigma_u,
        t_scale=t_scale, return_intermediate=True,
    )
    phi = np.maximum(phi - phi[source_index], 0.0)

    # When the lifted kernel correctly separates sheets into disconnected
    # components, the Poisson solve anchors each component at 0, giving
    # phi ≈ 0 for unreachable points.  Detect these via the heat field:
    # if u[i] is negligible, point i was never reached by diffusion.
    u_thresh = np.max(np.abs(u)) * 1e-6
    unreachable = np.abs(u) < u_thresh
    phi[unreachable] = np.inf

    return phi


# =====================================================================
# Leakage metric
# =====================================================================

def _cross_sheet_reachable(dist, sheet_id, src_idx):
    """Fraction of other-sheet points that received finite geodesic distance."""
    other = sheet_id != sheet_id[src_idx]
    if not np.any(other):
        return 0.0
    return float(np.mean(np.isfinite(dist[other])))


# =====================================================================
# Colour helpers
# =====================================================================

CMAP = "magma"


def _dist_to_colors(dist, vmin, vmax):
    """Map distances to magma colours; unreachable (inf) points get light grey."""
    cmap = plt.get_cmap(CMAP)
    normed = np.clip((dist - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    reachable = np.isfinite(dist)
    colors = np.full((len(dist), 3), 0.85)
    colors[reachable] = cmap(normed[reachable])[:, :3]
    return colors


# =====================================================================
# 3-D plotting helpers
# =====================================================================

def _set_3d_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centre = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([centre[0] - radius, centre[0] + radius])
    ax.set_ylim3d([centre[1] - radius, centre[1] + radius])
    ax.set_zlim3d([centre[2] - radius, centre[2] + radius])


def _plot_panel(ax, points, colors, src_idx):
    """Scatter a 3-D point cloud coloured by geodesic distance."""
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=3, depthshade=True, edgecolors="none", rasterized=True,
    )
    ax.scatter(
        [points[src_idx, 0]], [points[src_idx, 1]], [points[src_idx, 2]],
        c="cyan", s=40, marker="*", zorder=10, edgecolors="k", linewidths=0.5,
    )
    _set_3d_equal_aspect(ax)
    ax.set_axis_off()


# =====================================================================
# Main
# =====================================================================

def main():
    n = 6000
    seed = 42
    k = 30
    k_pca = 30
    alpha = 1.0
    sigma_x, sigma_u = 0.5, 0.5
    t_scale = 1.0
    noise_levels = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]
    view_elev, view_azim = 25, -50

    methods = [
        ("Lifted — GT tangents", "lifted_gt"),
        ("Lifted — PCA normals", "lifted_pca"),
        ("Lifted — PCA frames", "lifted_pca_frames"),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "mathtext.fontset": "cm",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    })

    n_rows = len(methods)
    n_cols = len(noise_levels)

    # ------------------------------------------------------------------
    # Pre-compute all distances
    # ------------------------------------------------------------------
    all_finite_dist: list[np.ndarray] = []
    panel_data: list[list[tuple]] = []

    for method_name, method_key in methods:
        row_data = []
        for sigma in noise_levels:
            points, gt_normals, sheet_id, src_idx = _sample_plane_cross(
                n, sigma, seed,
            )

            if method_key == "lifted_gt":
                print(f"  [{method_name}] sigma={sigma:.3f} ...")
                dist = _lifted_geodesic(
                    points, gt_normals, src_idx,
                    k=k, alpha=alpha,
                    sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
                )

            elif method_key == "lifted_pca":
                print(f"  [{method_name}] sigma={sigma:.3f} ...")
                pca_normals = estimate_normals_pca(points, k=k_pca)
                pca_normals = normalize_vectors(pca_normals)
                dist = _lifted_geodesic(
                    points, pca_normals, src_idx,
                    k=k, alpha=alpha,
                    sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
                )

            elif method_key == "lifted_pca_frames":
                print(f"  [{method_name}] sigma={sigma:.3f} ...")
                pca_frames = estimate_tangents_pca(
                    points, k=k_pca, tangent_dim=2,
                )
                dist = _lifted_geodesic_from_frames(
                    points, pca_frames, src_idx,
                    k=k, alpha=alpha,
                    sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
                )

            row_data.append((points, dist, sheet_id, src_idx))
            finite = dist[np.isfinite(dist)]
            if finite.size:
                all_finite_dist.append(finite)
        panel_data.append(row_data)

    all_finite = np.concatenate(all_finite_dist)
    vmin = 0.0
    vmax = float(np.percentile(all_finite, 98))

    # ------------------------------------------------------------------
    # 3-D geodesic maps
    # ------------------------------------------------------------------
    row_height = 2.2
    col_width = 2.2
    cbar_height = 0.25
    fig_w = col_width * n_cols
    fig_h = row_height * n_rows + cbar_height + 0.45  # room for titles + cbar

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Leave vertical space for column titles at top and colorbar at bottom
    top_margin = 0.35 / fig_h
    bot_margin = (cbar_height + 0.15) / fig_h
    row_label_margin = 0.08  # fraction of fig width reserved for row labels
    plot_left = row_label_margin
    plot_width = 1.0 - plot_left
    plot_height = 1.0 - top_margin - bot_margin

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = plot_left + col * (plot_width / n_cols)
            y0 = 1.0 - top_margin - (row + 1) * (plot_height / n_rows)
            w = plot_width / n_cols
            h = plot_height / n_rows
            ax = fig.add_axes([x0, y0, w, h], projection="3d")
            axes[row, col] = ax

            points, dist, sheet_id, src_idx = panel_data[row][col]
            colors = _dist_to_colors(dist, vmin, vmax)
            _plot_panel(ax, points, colors, src_idx)
            ax.view_init(elev=view_elev, azim=view_azim)

    # Column titles (noise levels) above the top row
    for col in range(n_cols):
        cx = plot_left + (col + 0.5) * (plot_width / n_cols)
        s = noise_levels[col]
        sigma_str = f"{s:.3f}" if s < 0.01 else f"{s:.2f}"
        fig.text(cx, 1.0 - top_margin * 0.35, rf"$\sigma={sigma_str}$",
                 ha="center", va="center", fontsize=9)

    # Row labels to the left of each row
    row_labels = ["GT tangents", "PCA normals", "PCA frames"]
    for row in range(n_rows):
        cy = 1.0 - top_margin - (row + 0.5) * (plot_height / n_rows)
        fig.text(row_label_margin * 0.5, cy, row_labels[row],
                 ha="center", va="center", fontsize=9, rotation=90)

    # Horizontal colorbar at bottom
    cbar_ax = fig.add_axes([plot_left + 0.15, 0.04,
                            plot_width - 0.3, cbar_height * 0.4 / fig_h * 4])
    sm = plt.cm.ScalarMappable(
        cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("geodesic distance", fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig("plane_cross_geodesic_leakage.pdf", bbox_inches="tight", dpi=300)
    print("Saved plane_cross_geodesic_leakage.pdf")
    plt.show()

    # ------------------------------------------------------------------
    # Cross-sheet reachability plot
    # ------------------------------------------------------------------
    _plot_reachability(panel_data, noise_levels, methods)


def _plot_reachability(panel_data, noise_levels, methods):
    """Line plot: fraction of other-sheet points reachable vs noise level."""
    fig, ax = plt.subplots(figsize=(4.0, 2.8), constrained_layout=True)

    markers = ["o", "s", "D"]
    for row_idx, (method_name, _) in enumerate(methods):
        fracs = []
        for col in range(len(noise_levels)):
            _, dist, sheet_id, src_idx = panel_data[row_idx][col]
            fracs.append(_cross_sheet_reachable(dist, sheet_id, src_idx))
        ax.plot(
            noise_levels, fracs,
            marker=markers[row_idx], markersize=5, lw=1.5,
            label=method_name,
        )

    ax.set_xlabel(r"positional noise $\sigma$")
    ax.set_ylabel("cross-sheet reachability")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.0%}",
    ))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.8")
    ax.grid(True, which="major", alpha=0.25, lw=0.6)

    fig.savefig("plane_cross_reachability.pdf", bbox_inches="tight", dpi=300)
    print("Saved plane_cross_reachability.pdf")
    plt.show()


if __name__ == "__main__":
    main()
