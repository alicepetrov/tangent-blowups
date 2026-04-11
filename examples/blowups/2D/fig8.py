"""
SGP figure: Nash blow-up of a figure-8 curve.

Top row    — spatial coordinates (x, y)
Bottom row — tangent projector P = uu^T shown as a 2×2 matrix
"""

import numpy as np
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformCurve, figure8, sample


def build_oriented_fig8(
    *,
    n: int = 5000,
    scale: float = 2.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an oriented point cloud from a 2D figure-8 curve."""
    curve = figure8(scale=scale)
    strategy = UniformCurve(n=n, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    rng = np.random.default_rng(seed)

    s = sample(curve, strategy, with_tangents=True, with_normals=False)
    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    points += rng.normal(scale=0.02, size=points.shape)
    tangents = normalize_vectors(tangents)
    return points, tangents


def _scatter(ax, pts, values, *, cmap, vmin=None, vmax=None, s=9):
    """Scatter plot on ax with no ticks and equal aspect."""
    sc = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=values, s=s, cmap=cmap,
        vmin=vmin, vmax=vmax,
        edgecolors="none", rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def main():
    points, tangents = build_oriented_fig8(n=1200, scale=2.0, seed=7)
    lifted = BlowUpLevel.from_point_tangents(points, tangents)

    pts = lifted.embedded
    P = lifted.projectors  # (N, 2, 2)

    # -- Publication styling --
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.4,
    })

    fig = plt.figure(figsize=(5.0, 7.0), constrained_layout=True)

    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1, 1, 1],
        hspace=0.08, wspace=0.08,
    )

    # Each panel is normalized to its own [min, max] for maximum contrast.
    # A single colorbar shows the normalized scale (0 = per-panel min, 1 = max).
    cmap = "RdBu_r"

    all_axes = []
    panels = [
        (r"$x$",       pts[:, 0]),
        (r"$y$",       pts[:, 1]),
        (r"$P_{00}$",  P[:, 0, 0]),
        (r"$P_{01}$",  P[:, 0, 1]),
        (r"$P_{10}$",  P[:, 1, 0]),
        (r"$P_{11}$",  P[:, 1, 1]),
    ]
    grid_pos = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for (label, vals), (row, col) in zip(panels, grid_pos):
        ax = fig.add_subplot(gs[row, col])
        lo, hi = vals.min(), vals.max()
        _scatter(ax, pts, vals, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(label, fontsize=12, pad=6)
        all_axes.append(ax)

    # Single colorbar — shows normalised scale since each panel has its own range
    sm = mcm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=all_axes, fraction=0.03, pad=0.02, aspect=40,
                 label="normalized")

    fig.savefig("fig8_nash_blowup.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # ---- kNN comparison: ambient vs lifted near the intersection ----
    query_idx, idx_amb, dist_amb, idx_lift, dist_lift = _find_knn(lifted, k=200, alpha=1.0)

    # Plot 2: standalone kNN comparison
    _plot_knn_comparison(pts, query_idx, idx_amb, dist_amb, idx_lift, dist_lift, s=9)

    # Plot 3: combined figure — kNN on top, projector matrix on bottom
    _plot_combined(pts, P, query_idx, idx_amb, dist_amb, idx_lift, dist_lift, cmap=cmap, s=9)


def _find_knn(lifted: BlowUpLevel, *, k: int = 80, alpha: float = 1.0):
    """Find kNN indices near the self-intersection in ambient and lifted space."""
    from scipy.embedded import cKDTree

    pts = lifted.embedded
    embed = lifted.embedding_vector(alpha=alpha)

    query_idx = int(np.argmin(np.linalg.norm(pts, axis=1)))

    tree_amb = cKDTree(pts)
    dist_amb, idx_amb = tree_amb.query(pts[query_idx], k=k + 1)
    dist_amb, idx_amb = dist_amb[1:], idx_amb[1:]

    tree_lift = cKDTree(embed)
    dist_lift, idx_lift = tree_lift.query(embed[query_idx], k=k + 1)
    dist_lift, idx_lift = dist_lift[1:], idx_lift[1:]

    return query_idx, idx_amb, dist_amb, idx_lift, dist_lift


_CMAP = plt.get_cmap("RdBu_r")
_COLOR_BLUE = _CMAP(1.0)   # query point
_COLOR_RED = _CMAP(0.0)    # neighbors


def _plot_knn_panel(ax, pts, query_idx, idx_nn, dist_nn, *, title, s=9):
    """Draw a single kNN panel: gray cloud, red neighbors (gradient), blue query."""
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c="0.85",
               edgecolors="none", rasterized=True)
    # Neighbor colors: blend from full _COLOR_RED (closest) to white (farthest)
    t = dist_nn / dist_nn.max()  # 0 = closest, 1 = farthest
    white = np.array([1.0, 1.0, 1.0, 1.0])
    red = np.array(_COLOR_RED)
    colors = np.outer(1 - t, red) + np.outer(t, white)
    ax.scatter(pts[idx_nn, 0], pts[idx_nn, 1], s=s * 2, c=colors,
               edgecolors="none", zorder=2, rasterized=True)
    ax.scatter(pts[query_idx, 0], pts[query_idx, 1], s=s * 3, c=[_COLOR_BLUE],
               edgecolors="none", zorder=3)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_knn_comparison(pts, query_idx, idx_amb, dist_amb, idx_lift, dist_lift, *, s=9):
    """Standalone figure: ambient vs lifted kNN side by side."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.0, 2.8),
                                      constrained_layout=True)

    _plot_knn_panel(ax_a, pts, query_idx, idx_amb, dist_amb,
                    title=r"$k$-NN in $\mathbb{R}^2$", s=s)
    _plot_knn_panel(ax_b, pts, query_idx, idx_lift, dist_lift,
                    title=r"$k$-NN in $\mathbb{R}^2 \times G(1,2)$", s=s)

    fig.savefig("fig8_knn_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def _plot_combined(pts, P, query_idx, idx_amb, dist_amb, idx_lift, dist_lift, *, cmap, s=9):
    """Combined figure: kNN comparison on top, projector matrix on bottom."""
    fig = plt.figure(figsize=(5.0, 7.0), constrained_layout=True)

    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1],
                  hspace=0.08, wspace=0.08)

    # ---- Top row: kNN before / after ----
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_knn_panel(ax_a, pts, query_idx, idx_amb, dist_amb,
                    title=r"$k$-NN in $\mathbb{R}^2$", s=s)

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_knn_panel(ax_b, pts, query_idx, idx_lift, dist_lift,
                    title=r"$k$-NN in $\mathbb{R}^2 \times G(1,2)$", s=s)

    # ---- Bottom 2×2: projector P = uu^T ----
    proj_axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[1 + i, j])
            vals = P[:, i, j]
            lo, hi = vals.min(), vals.max()
            _scatter(ax, pts, vals, cmap=cmap, vmin=lo, vmax=hi, s=s)
            ax.set_title(rf"$P_{{{i}{j}}}$", fontsize=12, pad=6)
            proj_axes.append(ax)

    sm = mcm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=proj_axes, fraction=0.03, pad=0.02, aspect=30,
                 label="normalized")

    fig.savefig("fig8_combined.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
