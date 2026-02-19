"""
Laplacian Spectrum Visualization
--------------------------------
Visualize Laplacian eigenpairs on point clouds.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy import sparse

from ..pointcloud.laplacian import laplacian_spectrum


def _as_points(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"points must be 2D (N, d). Got {P.shape}.")
    if not np.isfinite(P).all():
        raise ValueError("points must be finite.")
    return P


def _set_3d_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def _setup_figure(
    spatial_dim: int,
    n_plots: int,
    *,
    cols: int,
    figsize: Optional[Tuple[float, float]],
    figsize_base: float,
):
    if cols <= 0:
        raise ValueError("cols must be positive.")
    cols = min(cols, n_plots)
    rows = int(np.ceil(n_plots / cols))
    if figsize is None:
        fig = plt.figure(figsize=(figsize_base * cols, figsize_base * rows))
    else:
        fig = plt.figure(figsize=figsize)

    axes = []
    for i in range(n_plots):
        if spatial_dim == 3:
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        else:
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_aspect("equal")
        axes.append(ax)
    return fig, axes


def _normalize_values(values: np.ndarray, mode: Literal["none", "maxabs", "zscore"]) -> np.ndarray:
    if mode == "none":
        return values
    if mode == "maxabs":
        denom = float(np.max(np.abs(values)))
        if denom <= 0.0:
            return values
        return values / denom
    if mode == "zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std <= 0.0:
            return values - mean
        return (values - mean) / std
    raise ValueError(f"Unknown normalize mode '{mode}'.")


def _resolve_indices(
    eigen_indices: Optional[Sequence[int]],
    k: int,
    n_available: int,
) -> np.ndarray:
    if eigen_indices is None:
        return np.arange(min(k, n_available), dtype=int)

    idx = np.asarray(list(eigen_indices), dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("eigen_indices must be a 1D non-empty sequence.")

    idx = np.where(idx < 0, idx + n_available, idx)
    if np.any(idx < 0) or np.any(idx >= n_available):
        raise ValueError("eigen_indices out of range for computed spectrum.")
    return idx


def visualize_laplacian_eigenpairs(
    points: np.ndarray,
    L: sparse.spmatrix | np.ndarray,
    *,
    k: int = 6,
    eigen_indices: Optional[Sequence[int]] = None,
    which: Literal["SM", "LM"] = "SM",
    drop_first: bool = True,
    downsample: int = 1,
    cmap: str = "coolwarm",
    normalize: Literal["none", "maxabs", "zscore"] = "maxabs",
    center_zero: bool = True,
    shared_color_scale: bool = True,
    point_size: float = 6.0,
    point_alpha: float = 0.85,
    cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    figsize_base: float = 4.0,
    show: bool = True,
):
    """
    Visualize Laplacian eigenvectors on a 2D/3D point cloud.

    Returns:
        fig, axes, evals_sel, evecs_sel
    """
    P = _as_points(points)
    n, spatial_dim = P.shape
    if spatial_dim not in (2, 3):
        raise ValueError(f"Only 2D/3D point clouds are supported. Got n={spatial_dim}.")
    if downsample < 1:
        raise ValueError("downsample must be >= 1.")

    if eigen_indices is not None:
        idx_in = np.asarray(list(eigen_indices), dtype=int)
        max_idx = idx_in[idx_in >= 0].max() if np.any(idx_in >= 0) else -1
        k_eff = max(k, max_idx + 1)
    else:
        k_eff = k

    evals, evecs = laplacian_spectrum(
        L,
        k=k_eff,
        which=which,
        drop_first=drop_first,
        return_eigenvalues=True,
    )
    idx = _resolve_indices(eigen_indices, k, evals.shape[0])

    evals_sel = evals[idx]
    evecs_sel = evecs[:, idx]

    sl = slice(None, None, downsample)
    pts = P[sl]
    n_plots = evecs_sel.shape[1]

    fig, axes = _setup_figure(
        spatial_dim,
        n_plots,
        cols=cols,
        figsize=figsize,
        figsize_base=figsize_base,
    )

    if shared_color_scale:
        vals_all = []
        for j in range(n_plots):
            vals = _normalize_values(evecs_sel[:, j], normalize)
            vals_all.append(vals)
        vals_all = np.concatenate(vals_all)
        if center_zero:
            vlim = float(np.max(np.abs(vals_all))) if vals_all.size else 1.0
            vmin, vmax = -vlim, vlim
        else:
            vmin, vmax = float(vals_all.min()), float(vals_all.max())
    else:
        vmin = vmax = None

    for plot_idx, ax in enumerate(axes):
        vals = _normalize_values(evecs_sel[:, plot_idx], normalize)
        vals_plot = vals[sl]

        if not shared_color_scale:
            if center_zero:
                vlim = float(np.max(np.abs(vals_plot))) if vals_plot.size else 1.0
                vmin, vmax = -vlim, vlim
            else:
                vmin, vmax = float(vals_plot.min()), float(vals_plot.max())

        if spatial_dim == 3:
            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=vals_plot,
                s=point_size,
                alpha=point_alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            _set_3d_equal_aspect(ax)
        else:
            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=vals_plot,
                s=point_size,
                alpha=point_alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("equal")

        eig_idx = int(idx[plot_idx])
        ax.set_title(f"eig {eig_idx} (lambda={evals_sel[plot_idx]:.3g})")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes, evals_sel, evecs_sel


def visualize_laplacian_eigenpair_comparison(
    points: np.ndarray,
    L_left: sparse.spmatrix | np.ndarray,
    L_right: sparse.spmatrix | np.ndarray,
    *,
    k: int = 6,
    eigen_indices: Optional[Sequence[int]] = None,
    which: Literal["SM", "LM"] = "SM",
    drop_first: bool = True,
    downsample: int = 1,
    cmap: str = "coolwarm",
    normalize: Literal["none", "maxabs", "zscore"] = "maxabs",
    center_zero: bool = True,
    shared_color_scale: bool = True,
    point_size: float = 6.0,
    point_alpha: float = 0.85,
    figsize: Optional[Tuple[float, float]] = None,
    figsize_base: float = 4.0,
    labels: Tuple[str, str] = ("Point Cloud", "Lifted"),
    show: bool = True,
):
    """
    Side-by-side visualization of Laplacian eigenpairs.

    Returns:
        fig, axes, evals_left, evals_right, evecs_left, evecs_right
    """
    P = _as_points(points)
    n, spatial_dim = P.shape
    if spatial_dim not in (2, 3):
        raise ValueError(f"Only 2D/3D point clouds are supported. Got n={spatial_dim}.")
    if downsample < 1:
        raise ValueError("downsample must be >= 1.")

    if eigen_indices is not None:
        idx_in = np.asarray(list(eigen_indices), dtype=int)
        max_idx = idx_in[idx_in >= 0].max() if np.any(idx_in >= 0) else -1
        k_eff = max(k, max_idx + 1)
    else:
        k_eff = k

    evals_left, evecs_left = laplacian_spectrum(
        L_left,
        k=k_eff,
        which=which,
        drop_first=drop_first,
        return_eigenvalues=True,
    )
    evals_right, evecs_right = laplacian_spectrum(
        L_right,
        k=k_eff,
        which=which,
        drop_first=drop_first,
        return_eigenvalues=True,
    )

    n_available = min(evals_left.shape[0], evals_right.shape[0])
    if n_available == 0:
        raise ValueError("No eigenpairs available to plot.")

    idx = _resolve_indices(eigen_indices, k, n_available)

    evals_l = evals_left[idx]
    evals_r = evals_right[idx]
    evecs_l = evecs_left[:, idx]
    evecs_r = evecs_right[:, idx]

    sl = slice(None, None, downsample)
    pts = P[sl]
    n_rows = idx.shape[0]

    cols = 2
    rows = n_rows
    if figsize is None:
        fig = plt.figure(figsize=(figsize_base * cols, figsize_base * rows))
    else:
        fig = plt.figure(figsize=figsize)

    axes = []
    for r in range(rows):
        row_axes = []
        for c in range(cols):
            ax_idx = r * cols + c + 1
            if spatial_dim == 3:
                ax = fig.add_subplot(rows, cols, ax_idx, projection="3d")
            else:
                ax = fig.add_subplot(rows, cols, ax_idx)
                ax.set_aspect("equal")
            row_axes.append(ax)
        axes.append(row_axes)

    for r in range(rows):
        vals_left = _normalize_values(evecs_l[:, r], normalize)
        vals_right = _normalize_values(evecs_r[:, r], normalize)

        if shared_color_scale:
            both = np.concatenate([vals_left, vals_right])
            if center_zero:
                vlim = float(np.max(np.abs(both))) if both.size else 1.0
                vmin_l = vmin_r = -vlim
                vmax_l = vmax_r = vlim
            else:
                vmin_l = vmin_r = float(both.min())
                vmax_l = vmax_r = float(both.max())
        else:
            if center_zero:
                vlim = float(np.max(np.abs(vals_left))) if vals_left.size else 1.0
                vmin_l, vmax_l = -vlim, vlim
                vlim = float(np.max(np.abs(vals_right))) if vals_right.size else 1.0
                vmin_r, vmax_r = -vlim, vlim
            else:
                vmin_l, vmax_l = float(vals_left.min()), float(vals_left.max())
                vmin_r, vmax_r = float(vals_right.min()), float(vals_right.max())

        vals_left_plot = vals_left[sl]
        vals_right_plot = vals_right[sl]

        ax_left = axes[r][0]
        ax_right = axes[r][1]

        if spatial_dim == 3:
            sc_left = ax_left.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=vals_left_plot,
                s=point_size,
                alpha=point_alpha,
                cmap=cmap,
                vmin=vmin_l,
                vmax=vmax_l,
            )
            sc_right = ax_right.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=vals_right_plot,
                s=point_size,
                alpha=point_alpha,
                cmap=cmap,
                vmin=vmin_r,
                vmax=vmax_r,
            )
            _set_3d_equal_aspect(ax_left)
            _set_3d_equal_aspect(ax_right)
        else:
            sc_left = ax_left.scatter(
                pts[:, 0],
                pts[:, 1],
                c=vals_left_plot,
                s=point_size,
                alpha=point_alpha,
                cmap=cmap,
                vmin=vmin_l,
                vmax=vmax_l,
            )
            sc_right = ax_right.scatter(
                pts[:, 0],
                pts[:, 1],
                c=vals_right_plot,
                s=point_size,
                alpha=point_alpha,
                cmap=cmap,
                vmin=vmin_r,
                vmax=vmax_r,
            )
            ax_left.set_aspect("equal")
            ax_right.set_aspect("equal")

        eig_idx = int(idx[r])
        ax_left.set_title(f"{labels[0]} eig {eig_idx} (lambda={evals_l[r]:.3g})")
        ax_right.set_title(f"{labels[1]} eig {eig_idx} (lambda={evals_r[r]:.3g})")

        fig.colorbar(sc_left, ax=ax_left, fraction=0.046, pad=0.04)
        fig.colorbar(sc_right, ax=ax_right, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes, evals_l, evals_r, evecs_l, evecs_r


__all__ = [
    "visualize_laplacian_eigenpairs",
    "visualize_laplacian_eigenpair_comparison",
]
