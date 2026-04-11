r"""
Didactic figure: line--parabola tangential intersection.

Three panels:
  (a) point cloud  (b) with tangent lines  (c) coloured by ||II||_F

Sparse sampling + light positional noise for a clean, tikz-friendly look.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tangent_blowups.geometry.iterated_grassmann import (
    extract_level1, iterated_blowup,
)
from tangent_blowups.testsupport import tangent_parabola_line, UniformCurve, sample


def _ground_truth_curvature(
    t_params: np.ndarray, curvature: float,
) -> np.ndarray:
    r"""Analytic ||II||_F for the line-parabola curve.

    Parabola branch (t < 1):  y = c x^2,  kappa = 2c / (1 + 4c^2 x^2)^{3/2}
    Line branch     (t >= 1): kappa = 0
    """
    # Recover x-coordinate from parameter (see tangent_parabola_line impl:
    # parabola maps t in [0,1) linearly to [x_min, x_max])
    # We just use the analytic formula on the x-positions directly.
    # But we need to know which branch each point is on.
    is_line = t_params >= 1.0
    # For parabola branch, curvature of y=cx^2 at position x:
    #   kappa(x) = |2c| / (1 + (2cx)^2)^(3/2)
    # We need x; for the parabola branch the position function is
    # (x, c*x^2) where x is linearly mapped from t.
    # Since we added noise, use t to recover the *clean* x.
    c = curvature
    # t in [0,1) -> x in [x_min, x_max]; default x_min=-1.5, x_max=1.5
    x_clean = -1.5 + t_params * 3.0  # only valid for parabola branch
    kappa = np.zeros_like(t_params)
    para = ~is_line
    kappa[para] = np.abs(2.0 * c) / (1.0 + (2.0 * c * x_clean[para]) ** 2) ** 1.5
    return kappa


def _plot_curvature(ax, fig, x, y, curv, title, pt_size, vmin, vmax):
    """Curvature scatter with per-point alpha (high-curv more opaque)."""
    order = np.argsort(curv)
    cmap = plt.get_cmap("viridis")
    normed = np.clip((curv[order] - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgba = cmap(normed)
    rgba[:, 3] = 0.45 + 0.55 * normed
    ax.scatter(x[order], y[order], c=rgba,
               s=pt_size, edgecolors="none", zorder=2)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    ax.set_title(title)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=22)
    cbar.locator = ticker.MaxNLocator(integer=True)
    cbar.update_ticks()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ground-truth", action="store_true",
                      help="Plot analytic curvature instead of estimated")
    mode.add_argument("--compare", action="store_true",
                      help="Side-by-side ground truth vs estimated curvature")
    args = parser.parse_args()

    curvature = 1.0
    n_points = 120
    k_nn = 10
    alpha = 1.0
    noise = 0.015  # positional noise std
    tan_len = 0.12  # tangent tick half-length
    seed = 42

    # -- Sample --
    curve = tangent_parabola_line(curvature=curvature)
    strategy = UniformCurve(n=n_points, t_min=0.0, t_max=2.0, endpoint=False)
    s = sample(curve, strategy, with_tangents=True)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)
    t_params = np.asarray(s.params, dtype=float)

    # Add light positional noise
    rng = np.random.default_rng(seed)
    points = points + rng.normal(scale=noise, size=points.shape)

    # -- Curvature --
    curv_gt = _ground_truth_curvature(t_params, curvature)

    need_estimated = not args.ground_truth  # needed for default and --compare
    if need_estimated:
        levels = iterated_blowup(points, tangents, num_levels=1,
                                 k=k_nn, alpha=alpha, lam=0.0)
        inv1 = extract_level1(levels[1])
        curv_est = inv1.total_curvature
        curv_est = curv_est[:, 0] if curv_est.ndim > 1 else curv_est

    # -- Figure --
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 150,
    })

    x, y = points[:, 0], points[:, 1]
    tx, ty = tangents[:, 0], tangents[:, 1]
    _viridis = plt.get_cmap("viridis")
    pt_color = _viridis(0.15)   # purple end
    tan_color = _viridis(0.55)  # green end
    pt_kw = dict(s=100, edgecolors="none", alpha=0.9, zorder=2)
    pad = 0.15
    xl = (x.min() - pad, x.max() + pad)
    yl = (y.min() - pad, y.max() + pad)

    if args.compare:
        # ------- Side-by-side curvature comparison -------
        vmin = 0.0
        vmax = max(curv_gt.max(), curv_est.max())

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5),
                                 constrained_layout=True)
        _plot_curvature(axes[0], fig, x, y, curv_gt,
                        r"(a) $\|I\!I\|_F$ ground truth", pt_kw["s"],
                        vmin, vmax)
        _style(axes[0], xl, yl)

        _plot_curvature(axes[1], fig, x, y, curv_est,
                        r"(b) $\|I\!I\|_F$ estimated", pt_kw["s"],
                        vmin, vmax)
        _style(axes[1], xl, yl)

        fig.savefig("line_parabola_curvature_compare.pdf",
                    bbox_inches="tight", dpi=300)
        fig.savefig("line_parabola_curvature_compare.png",
                    bbox_inches="tight", dpi=150)
        print("Saved line_parabola_curvature_compare.pdf")

    else:
        # ------- Three-panel didactic figure -------
        curv_norm = curv_gt if args.ground_truth else curv_est
        curv_label = (r"(c) $\|I\!I\|_F$" if args.ground_truth
                      else r"(c) $\|I\!I\|_F$ (estimated)")

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5),
                                 constrained_layout=True)

        # (a) Point cloud
        axes[0].scatter(x, y, c=[pt_color], **pt_kw)
        axes[0].set_title("(a) Point cloud")
        _style(axes[0], xl, yl)

        # (b) Points + tangent lines
        for i in range(len(points)):
            dx, dy = tan_len * tx[i], tan_len * ty[i]
            axes[1].plot([x[i] - dx, x[i] + dx],
                         [y[i] - dy, y[i] + dy],
                         color=tan_color, linewidth=2, alpha=0.9, zorder=1)
        axes[1].set_title("(b) Tangent lines")
        _style(axes[1], xl, yl)

        # (c) Curvature
        _plot_curvature(axes[2], fig, x, y, curv_norm, curv_label,
                        pt_kw["s"], curv_norm.min(), curv_norm.max())
        _style(axes[2], xl, yl)

        fig.savefig("line_parabola_didactic.pdf", bbox_inches="tight", dpi=300)
        fig.savefig("line_parabola_didactic.png", bbox_inches="tight", dpi=150)
        print("Saved line_parabola_didactic.pdf")

    plt.show()


def _style(ax, xl=None, yl=None):
    ax.set_aspect("equal")
    ax.axis("off")
    if xl is not None:
        ax.set_xlim(*xl)
    if yl is not None:
        ax.set_ylim(*yl)


if __name__ == "__main__":
    main()
