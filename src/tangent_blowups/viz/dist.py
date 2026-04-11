"""
Distance Visualization
----------------------
Helpers to visualize distances from a query point in:
1) the original Euclidean space R^n
2) the blown-up product space R^n x G(k, n)
"""
from __future__ import annotations

from typing import Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from ..geometry.iterated_grassmann import BlowUpLevel
from ..geometry.projectors import dist_geodesic

Metric = Literal["chordal", "geodesic"]
LevelMode = Literal["linear", "quantile"]


def _resolve_query_index(
    n_points: int,
    query_index: Optional[int],
    rng: Optional[np.random.Generator],
    seed: Optional[int],
) -> int:
    if n_points <= 0:
        raise ValueError("Sample is empty; cannot choose a query point.")

    if rng is not None and seed is not None:
        raise ValueError("Specify either rng or seed, not both.")

    if query_index is None:
        if rng is None:
            rng = np.random.default_rng(seed)
        query_index = int(rng.integers(0, n_points))

    if query_index < 0:
        query_index += n_points

    if query_index < 0 or query_index >= n_points:
        raise ValueError(f"query_index {query_index} out of range for N={n_points}")

    return int(query_index)


def _subspace_distance_sq(
    sample: BlowUpLevel,
    idx: int,
    metric: Metric,
) -> np.ndarray:
    if metric == "chordal":
        P = sample.projectors
        Pq = P[idx]
        inner = np.einsum("nij,ij->n", P, Pq)
        dist_u_sq = sample.d - inner
        return np.maximum(dist_u_sq, 0.0)

    if metric == "geodesic":
        Uq = sample.frame[idx]
        dist_u_sq = np.empty(sample.N, dtype=float)
        for i in range(sample.N):
            d = dist_geodesic(Uq, sample.frame[i])
            dist_u_sq[i] = d * d
        return dist_u_sq

    raise ValueError(
        f"Unknown subspace_metric '{metric}'. Expected 'chordal' or 'geodesic'."
    )


def compute_distance_fields(
    sample: BlowUpLevel,
    *,
    query_index: Optional[int] = None,
    alpha: float = 1.0,
    subspace_metric: Metric = "chordal",
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute distances from a query point to all points in a sample.

    Returns:
        query_index, dist_original, dist_product
    """
    idx = _resolve_query_index(sample.N, query_index, rng, seed)

    qx = sample.embedded[idx]
    dist_x = np.linalg.norm(sample.embedded - qx, axis=1)

    dist_u_sq = _subspace_distance_sq(sample, idx, subspace_metric)
    dist_product = np.sqrt(dist_x * dist_x + alpha * dist_u_sq)
    return idx, dist_x, dist_product


def _setup_axes(spatial_dim: int, figsize: Tuple[float, float]):
    fig = plt.figure(figsize=figsize)
    if spatial_dim == 3:
        ax_left = fig.add_subplot(1, 2, 1, projection="3d")
        ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_left = fig.add_subplot(1, 2, 1)
        ax_right = fig.add_subplot(1, 2, 2)
    return fig, (ax_left, ax_right)


def _set_equal_aspect(ax, spatial_dim: int):
    if spatial_dim == 3:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])
    else:
        ax.set_aspect("equal")


def _scatter_points(
    ax,
    points: np.ndarray,
    values: np.ndarray,
    spatial_dim: int,
    *,
    cmap,
    norm: Optional[BoundaryNorm],
    s: float,
    alpha: float,
    vmin: Optional[float],
    vmax: Optional[float],
):
    scatter_kwargs = dict(c=values, s=s, cmap=cmap, alpha=alpha)
    if norm is not None:
        scatter_kwargs["norm"] = norm
    else:
        if vmin is not None:
            scatter_kwargs["vmin"] = vmin
        if vmax is not None:
            scatter_kwargs["vmax"] = vmax

    if spatial_dim == 3:
        return ax.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_kwargs)
    return ax.scatter(points[:, 0], points[:, 1], **scatter_kwargs)


def _select_closest_indices(distances: np.ndarray, k: int) -> np.ndarray:
    if k >= distances.shape[0]:
        return np.arange(distances.shape[0], dtype=int)
    return np.argpartition(distances, k - 1)[:k]


def _resolve_highlight_k(
    n_plot: int,
    highlight_k: Optional[int],
    highlight_fraction: Optional[float],
) -> int:
    if highlight_k is not None and highlight_fraction is not None:
        raise ValueError("Specify only one of highlight_k or highlight_fraction.")

    if highlight_fraction is not None:
        if highlight_fraction <= 0.0 or highlight_fraction > 1.0:
            raise ValueError("highlight_fraction must be in (0, 1].")
        return max(1, int(np.ceil(n_plot * highlight_fraction)))

    if highlight_k is None:
        return n_plot

    if highlight_k <= 0:
        raise ValueError("highlight_k must be positive.")
    return min(int(highlight_k), n_plot)


def _prepare_plot_data(
    sample: BlowUpLevel,
    downsample: int,
    left_values: np.ndarray,
    right_values: np.ndarray,
    highlight_k: Optional[int],
    highlight_fraction: Optional[float],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if downsample < 1:
        raise ValueError("downsample must be >= 1.")

    spatial_dim = sample.embedded.shape[1]
    if spatial_dim not in (2, 3):
        raise ValueError(f"Only 2D/3D point clouds are supported. Got n={spatial_dim}.")

    sl = slice(None, None, downsample)
    pts = sample.embedded[sl]
    left_plot = left_values[sl]
    right_plot = right_values[sl]

    highlight_k = _resolve_highlight_k(pts.shape[0], highlight_k, highlight_fraction)
    idx_left = _select_closest_indices(left_plot, highlight_k)
    idx_right = _select_closest_indices(right_plot, highlight_k)

    return spatial_dim, pts, left_plot, right_plot, idx_left, idx_right


def _compute_level_boundaries(
    values: np.ndarray,
    n_levels: int,
    mode: LevelMode,
    vmin: Optional[float],
    vmax: Optional[float],
) -> np.ndarray:
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2.")

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, n_levels + 1)

    if vmin is None:
        vmin = float(vals.min())
    if vmax is None:
        vmax = float(vals.max())

    if mode == "quantile":
        boundaries = np.quantile(vals, np.linspace(0.0, 1.0, n_levels + 1))
    elif mode == "linear":
        boundaries = np.linspace(vmin, vmax, n_levels + 1)
    else:
        raise ValueError(f"Unknown level_mode '{mode}'. Expected 'linear' or 'quantile'.")

    boundaries = np.asarray(boundaries, dtype=float)
    if not np.all(np.isfinite(boundaries)):
        boundaries = np.linspace(vmin, vmax, n_levels + 1)

    span = float(vmax - vmin)
    if span <= 0.0:
        eps = max(1e-12, 1e-6 * (abs(vmin) + 1.0))
    else:
        eps = 1e-12 * span
    for i in range(1, boundaries.size):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + eps

    return boundaries


def _alternate_lightness(colors: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return colors
    out = np.array(colors, copy=True)
    rgb = out[:, :3]
    light_mask = (np.arange(rgb.shape[0]) % 2) == 0
    rgb[light_mask] = rgb[light_mask] * (1.0 - strength) + strength
    rgb[~light_mask] = rgb[~light_mask] * (1.0 - strength)
    out[:, :3] = np.clip(rgb, 0.0, 1.0)
    return out


def _build_level_norm(
    values: np.ndarray,
    *,
    n_levels: int,
    level_mode: LevelMode,
    cmap,
    level_alternate: bool,
    level_alternate_strength: float,
    vmin: Optional[float],
    vmax: Optional[float],
) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    boundaries = _compute_level_boundaries(values, n_levels, level_mode, vmin, vmax)
    cmap_levels = plt.get_cmap(cmap, n_levels)
    if level_alternate:
        colors = cmap_levels(np.linspace(0.0, 1.0, n_levels))
        colors = _alternate_lightness(colors, level_alternate_strength)
        name = getattr(cmap_levels, "name", "cmap")
        cmap_levels = ListedColormap(colors, name=f"{name}_levels")
    norm = BoundaryNorm(boundaries, ncolors=cmap_levels.N, clip=True)
    return cmap_levels, norm, boundaries


def _resolve_color_mapping(
    left_values: np.ndarray,
    right_values: np.ndarray,
    idx_left: np.ndarray,
    idx_right: np.ndarray,
    *,
    shared_color_scale: bool,
    cmap,
    level_sets: bool,
    n_levels: int,
    level_mode: LevelMode,
    level_alternate: bool,
    level_alternate_strength: float,
) -> tuple[
    object,
    object,
    Optional[BoundaryNorm],
    Optional[BoundaryNorm],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
]:
    vmin = vmax = None
    if shared_color_scale:
        vmin = float(np.min([left_values[idx_left].min(), right_values[idx_right].min()]))
        vmax = float(np.max([left_values[idx_left].max(), right_values[idx_right].max()]))

    if not level_sets:
        return cmap, cmap, None, None, None, None, vmin, vmax

    left_vals = left_values[idx_left]
    right_vals = right_values[idx_right]

    if shared_color_scale:
        combined = np.concatenate([left_vals, right_vals])
        cmap_levels, norm, boundaries = _build_level_norm(
            combined,
            n_levels=n_levels,
            level_mode=level_mode,
            cmap=cmap,
            level_alternate=level_alternate,
            level_alternate_strength=level_alternate_strength,
            vmin=vmin,
            vmax=vmax,
        )
        return cmap_levels, cmap_levels, norm, norm, boundaries, boundaries, vmin, vmax

    left_cmap, left_norm, left_bounds = _build_level_norm(
        left_vals,
        n_levels=n_levels,
        level_mode=level_mode,
        cmap=cmap,
        level_alternate=level_alternate,
        level_alternate_strength=level_alternate_strength,
        vmin=None,
        vmax=None,
    )
    right_cmap, right_norm, right_bounds = _build_level_norm(
        right_vals,
        n_levels=n_levels,
        level_mode=level_mode,
        cmap=cmap,
        level_alternate=level_alternate,
        level_alternate_strength=level_alternate_strength,
        vmin=None,
        vmax=None,
    )
    return (
        left_cmap,
        right_cmap,
        left_norm,
        right_norm,
        left_bounds,
        right_bounds,
        vmin,
        vmax,
    )


def _scatter_background(
    ax,
    points: np.ndarray,
    spatial_dim: int,
    *,
    point_size: float,
    background_color: str,
    background_alpha: float,
):
    if background_alpha <= 0.0:
        return
    if spatial_dim == 3:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            c=background_color,
            alpha=background_alpha,
        )
    else:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_size,
            c=background_color,
            alpha=background_alpha,
        )


def _highlight_query_point(ax, q: np.ndarray, spatial_dim: int):
    if spatial_dim == 3:
        ax.scatter([q[0]], [q[1]], [q[2]], s=80, c="none", edgecolors="k", linewidths=1.5)
    else:
        ax.scatter([q[0]], [q[1]], s=80, c="none", edgecolors="k", linewidths=1.5)


def _add_colorbar(fig, sc, ax, boundaries: Optional[np.ndarray]):
    cbar_kwargs = dict(fraction=0.046, pad=0.04)
    if boundaries is not None:
        fig.colorbar(sc, ax=ax, boundaries=boundaries, ticks=boundaries, **cbar_kwargs)
    else:
        fig.colorbar(sc, ax=ax, **cbar_kwargs)


def _plot_distance_pair(
    *,
    pts: np.ndarray,
    left_values: np.ndarray,
    right_values: np.ndarray,
    idx_left: np.ndarray,
    idx_right: np.ndarray,
    left_title: str,
    right_title: str,
    suptitle: str,
    query_point: np.ndarray,
    spatial_dim: int,
    left_cmap,
    right_cmap,
    left_norm: Optional[BoundaryNorm],
    right_norm: Optional[BoundaryNorm],
    point_size: float,
    point_alpha: float,
    background_color: str,
    background_alpha: float,
    vmin: Optional[float],
    vmax: Optional[float],
    left_boundaries: Optional[np.ndarray],
    right_boundaries: Optional[np.ndarray],
    figsize: Tuple[float, float],
    show: bool,
):
    fig, (ax_left, ax_right) = _setup_axes(spatial_dim, figsize)

    _scatter_background(
        ax_left,
        pts,
        spatial_dim,
        point_size=point_size,
        background_color=background_color,
        background_alpha=background_alpha,
    )
    _scatter_background(
        ax_right,
        pts,
        spatial_dim,
        point_size=point_size,
        background_color=background_color,
        background_alpha=background_alpha,
    )

    sc_left = _scatter_points(
        ax_left,
        pts[idx_left],
        left_values[idx_left],
        spatial_dim,
        cmap=left_cmap,
        norm=left_norm,
        s=point_size,
        alpha=point_alpha,
        vmin=vmin,
        vmax=vmax,
    )
    sc_right = _scatter_points(
        ax_right,
        pts[idx_right],
        right_values[idx_right],
        spatial_dim,
        cmap=right_cmap,
        norm=right_norm,
        s=point_size,
        alpha=point_alpha,
        vmin=vmin,
        vmax=vmax,
    )

    _highlight_query_point(ax_left, query_point, spatial_dim)
    _highlight_query_point(ax_right, query_point, spatial_dim)
    _set_equal_aspect(ax_left, spatial_dim)
    _set_equal_aspect(ax_right, spatial_dim)

    ax_left.set_title(left_title)
    ax_right.set_title(right_title)

    _add_colorbar(fig, sc_left, ax_left, left_boundaries)
    _add_colorbar(fig, sc_right, ax_right, right_boundaries)

    fig.suptitle(suptitle)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, (ax_left, ax_right)


def _visualize_distance_pair(
    sample: BlowUpLevel,
    *,
    query_index: int,
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_title: str,
    right_title: str,
    suptitle: str,
    downsample: int,
    shared_color_scale: bool,
    cmap,
    level_sets: bool,
    n_levels: int,
    level_mode: LevelMode,
    level_alternate: bool,
    level_alternate_strength: float,
    point_size: float,
    point_alpha: float,
    highlight_fraction: Optional[float],
    highlight_k: Optional[int],
    background_color: str,
    background_alpha: float,
    figsize: Tuple[float, float],
    show: bool,
):
    spatial_dim, pts, left_plot, right_plot, idx_left, idx_right = _prepare_plot_data(
        sample,
        downsample,
        left_values,
        right_values,
        highlight_k,
        highlight_fraction,
    )

    (
        left_cmap,
        right_cmap,
        left_norm,
        right_norm,
        left_boundaries,
        right_boundaries,
        vmin,
        vmax,
    ) = _resolve_color_mapping(
        left_plot,
        right_plot,
        idx_left,
        idx_right,
        shared_color_scale=shared_color_scale,
        cmap=cmap,
        level_sets=level_sets,
        n_levels=n_levels,
        level_mode=level_mode,
        level_alternate=level_alternate,
        level_alternate_strength=level_alternate_strength,
    )

    fig, axes = _plot_distance_pair(
        pts=pts,
        left_values=left_plot,
        right_values=right_plot,
        idx_left=idx_left,
        idx_right=idx_right,
        left_title=left_title,
        right_title=right_title,
        suptitle=suptitle,
        query_point=sample.embedded[query_index],
        spatial_dim=spatial_dim,
        left_cmap=left_cmap,
        right_cmap=right_cmap,
        left_norm=left_norm,
        right_norm=right_norm,
        point_size=point_size,
        point_alpha=point_alpha,
        background_color=background_color,
        background_alpha=background_alpha,
        vmin=vmin,
        vmax=vmax,
        left_boundaries=left_boundaries,
        right_boundaries=right_boundaries,
        figsize=figsize,
        show=show,
    )

    return fig, axes


def visualize_distance_comparison(
    sample: BlowUpLevel,
    *,
    query_index: Optional[int] = None,
    alpha: float = 1.0,
    subspace_metric: Metric = "chordal",
    downsample: int = 1,
    shared_color_scale: bool = True,
    cmap: str = "magma",
    level_sets: bool = True,
    n_levels: int = 8,
    level_mode: LevelMode = "linear",
    level_alternate: bool = True,
    level_alternate_strength: float = 0.35,
    point_size: float = 6.0,
    point_alpha: float = 0.85,
    highlight_fraction: Optional[float] = None,
    highlight_k: Optional[int] = None,
    background_color: str = "lightgray",
    background_alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = (12.0, 6.0),
    show: bool = True,
):
    """
    Side-by-side heatmap of distances from a query point:
    - Left: Euclidean distances in R^n
    - Right: Product-metric distances in R^n x G(k, n)

    If level_sets is True, distances are binned into discrete bands.
    If level_alternate is True, adjacent bands alternate between lighter and darker shades.
    """
    idx, dist_x, dist_prod = compute_distance_fields(
        sample,
        query_index=query_index,
        alpha=alpha,
        subspace_metric=subspace_metric,
        rng=rng,
        seed=seed,
    )

    fig, axes = _visualize_distance_pair(
        sample,
        query_index=idx,
        left_values=dist_x,
        right_values=dist_prod,
        left_title="Original Space Distances",
        right_title=f"Product Metric Distances ({subspace_metric})",
        suptitle=f"Distance Comparison (query index {idx})",
        downsample=downsample,
        shared_color_scale=shared_color_scale,
        cmap=cmap,
        level_sets=level_sets,
        n_levels=n_levels,
        level_mode=level_mode,
        level_alternate=level_alternate,
        level_alternate_strength=level_alternate_strength,
        point_size=point_size,
        point_alpha=point_alpha,
        highlight_fraction=highlight_fraction,
        highlight_k=highlight_k,
        background_color=background_color,
        background_alpha=background_alpha,
        figsize=figsize,
        show=show,
    )

    return fig, axes, idx, dist_x, dist_prod


def visualize_product_metric_comparison(
    sample: BlowUpLevel,
    *,
    query_index: Optional[int] = None,
    alpha: float = 1.0,
    downsample: int = 1,
    shared_color_scale: bool = True,
    cmap: str = "magma",
    level_sets: bool = True,
    n_levels: int = 8,
    level_mode: LevelMode = "linear",
    level_alternate: bool = True,
    level_alternate_strength: float = 0.35,
    point_size: float = 6.0,
    point_alpha: float = 0.85,
    highlight_fraction: Optional[float] = None,
    highlight_k: Optional[int] = None,
    background_color: str = "lightgray",
    background_alpha: float = 0.2,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = (12.0, 6.0),
    show: bool = True,
):
    """
    Side-by-side heatmap: product metric (geodesic) vs product metric (chordal).
    """
    idx = _resolve_query_index(sample.N, query_index, rng, seed)

    qx = sample.embedded[idx]
    dist_x = np.linalg.norm(sample.embedded - qx, axis=1)

    dist_geo = np.sqrt(dist_x * dist_x + alpha * _subspace_distance_sq(sample, idx, "geodesic"))
    dist_chord = np.sqrt(dist_x * dist_x + alpha * _subspace_distance_sq(sample, idx, "chordal"))

    fig, axes = _visualize_distance_pair(
        sample,
        query_index=idx,
        left_values=dist_geo,
        right_values=dist_chord,
        left_title="Product Metric (geodesic)",
        right_title="Product Metric (chordal)",
        suptitle=f"Product Metric Comparison (query index {idx})",
        downsample=downsample,
        shared_color_scale=shared_color_scale,
        cmap=cmap,
        level_sets=level_sets,
        n_levels=n_levels,
        level_mode=level_mode,
        level_alternate=level_alternate,
        level_alternate_strength=level_alternate_strength,
        point_size=point_size,
        point_alpha=point_alpha,
        highlight_fraction=highlight_fraction,
        highlight_k=highlight_k,
        background_color=background_color,
        background_alpha=background_alpha,
        figsize=figsize,
        show=show,
    )

    return fig, axes, idx, dist_geo, dist_chord
