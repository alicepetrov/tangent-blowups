import inspect
import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.testsupport import samplers
from tangent_blowups.testsupport import (
    cone,
    figure8_space,
    helix,
    monkey_saddle,
    space_lissajous,
    trefoil_knot,
    whitney_umbrella,
)
from tangent_blowups.testsupport import ParametricCurve, ParametricSurface
from tangent_blowups.testsupport import SamplingStrategy

# -----------------------------------------------------------------------------
# Plot Helpers
# -----------------------------------------------------------------------------

def _set_axes_equal_3d(ax, points: np.ndarray):
    """
    Make 3D axes use the same scale so geometry is not visually distorted.
    """
    pts = np.asarray(points).reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _add_unique_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


# -----------------------------------------------------------------------------
# Curve Analysis
# -----------------------------------------------------------------------------

def analyze_curve_and_plot(
    curve: ParametricCurve,
    curve_name: str,
    strategy: SamplingStrategy,
    singularity_tol: float = 0.05,
):
    params = strategy(curve)
    sample = curve.evaluate(
        *params,
        with_tangents=True,
        with_normals=False,
        singularity_tol=singularity_tol,
    )

    points = sample.points
    tangents = sample.tangents
    mask = sample.singular_mask

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Curve: {curve_name} ({strategy.__class__.__name__})")
    ax.grid(True, alpha=0.3)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=10,
        color="cornflowerblue",
        alpha=0.7,
        label="Sampled Points",
    )

    if mask is not None and np.any(mask):
        singular_pts = points[mask]
        singular_tans = tangents[mask]

        ax.scatter(
            singular_pts[:, 0],
            singular_pts[:, 1],
            singular_pts[:, 2],
            color="red",
            s=50,
            depthshade=False,
            label="Detected Singularities",
        )

        ax.quiver(
            singular_pts[:, 0],
            singular_pts[:, 1],
            singular_pts[:, 2],
            singular_tans[:, 0],
            singular_tans[:, 1],
            singular_tans[:, 2],
            color="red",
            length=0.3,
            normalize=True,
            alpha=0.85,
            label="Diverging Tangents",
        )
        print(
            f"[Curve | {curve_name} | {strategy.__class__.__name__}] "
            f"Detected {int(np.sum(mask))} singular sample points."
        )
    else:
        print(
            f"[Curve | {curve_name} | {strategy.__class__.__name__}] "
            "No singularities detected (or none met tolerances)."
        )

    _set_axes_equal_3d(ax, points)
    _add_unique_legend(ax)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Surface Analysis
# -----------------------------------------------------------------------------

def analyze_surface_and_plot(
    surface: ParametricSurface,
    surface_name: str,
    strategy: SamplingStrategy,
    singularity_tol: float = 0.03,
):
    params = strategy(surface)
    sample = surface.evaluate(
        *params,
        with_tangents=True,
        with_normals=False,
        singularity_tol=singularity_tol,
    )

    points = sample.points.reshape(-1, 3)
    tangents = sample.tangents.reshape(-1, 3) if sample.tangents is not None else None
    mask = sample.singular_mask.reshape(-1) if sample.singular_mask is not None else None

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Surface: {surface_name} ({strategy.__class__.__name__})")
    ax.grid(True, alpha=0.3)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=5,
        color="cornflowerblue",
        alpha=0.55,
        label="Sampled Points",
    )

    if mask is not None and np.any(mask):
        singular_pts = points[mask]
        ax.scatter(
            singular_pts[:, 0],
            singular_pts[:, 1],
            singular_pts[:, 2],
            color="red",
            s=45,
            depthshade=False,
            label="Detected Singularities",
        )

        if tangents is not None:
            singular_tans = tangents[mask]
            max_arrows = 250
            if singular_pts.shape[0] > max_arrows:
                step = max(1, singular_pts.shape[0] // max_arrows)
                singular_pts = singular_pts[::step]
                singular_tans = singular_tans[::step]

            ax.quiver(
                singular_pts[:, 0],
                singular_pts[:, 1],
                singular_pts[:, 2],
                singular_tans[:, 0],
                singular_tans[:, 1],
                singular_tans[:, 2],
                color="red",
                length=0.2,
                normalize=True,
                alpha=0.8,
                label="Diverging Tangents",
            )

        print(
            f"[Surface | {surface_name} | {strategy.__class__.__name__}] "
            f"Detected {int(np.sum(mask))} singular sample points."
        )
    else:
        print(
            f"[Surface | {surface_name} | {strategy.__class__.__name__}] "
            "No singularities detected (or none met tolerances)."
        )

    _set_axes_equal_3d(ax, points)
    _add_unique_legend(ax)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Strategy Discovery
# -----------------------------------------------------------------------------

def _instantiate_strategy(
    cls: type[SamplingStrategy],
    *,
    is_curve: bool,
    idx: int,
    seed: int,
) -> SamplingStrategy | None:
    sig = inspect.signature(cls)
    kwargs = {}

    if "n" in sig.parameters:
        kwargs["n"] = 1500 if is_curve else 2000
    if "nu" in sig.parameters:
        kwargs["nu"] = 35
    if "nv" in sig.parameters:
        kwargs["nv"] = 35

    if "t_min" in sig.parameters:
        kwargs["t_min"] = 0.0
    if "t_max" in sig.parameters:
        kwargs["t_max"] = 2.0 * np.pi
    if "endpoint" in sig.parameters:
        kwargs["endpoint"] = True

    if "u_bounds" in sig.parameters:
        kwargs["u_bounds"] = (-2.0, 2.0)
    if "v_bounds" in sig.parameters:
        kwargs["v_bounds"] = (-2.0, 2.0)

    if "rng" in sig.parameters:
        kwargs["rng"] = np.random.default_rng(seed + idx)

    try:
        return cls(**kwargs)
    except TypeError as exc:
        print(f"Skipping {cls.__name__}: cannot auto-instantiate ({exc})")
        return None


def build_curve_strategies(seed: int = 11) -> list[SamplingStrategy]:
    classes = inspect.getmembers(samplers, inspect.isclass)
    classes.sort(key=lambda item: item[0])

    strategies: list[SamplingStrategy] = []
    for _, cls in classes:
        if cls is SamplingStrategy:
            continue
        if not issubclass(cls, SamplingStrategy):
            continue
        if "Curve" not in cls.__name__:
            continue

        strategy = _instantiate_strategy(cls, is_curve=True, idx=len(strategies), seed=seed)
        if strategy is not None:
            strategies.append(strategy)

    return strategies


def build_surface_strategies(seed: int = 101) -> list[SamplingStrategy]:
    classes = inspect.getmembers(samplers, inspect.isclass)
    classes.sort(key=lambda item: item[0])

    strategies: list[SamplingStrategy] = []
    for _, cls in classes:
        if cls is SamplingStrategy:
            continue
        if not issubclass(cls, SamplingStrategy):
            continue
        if ("Surface" not in cls.__name__) and ("Grid" not in cls.__name__):
            continue

        strategy = _instantiate_strategy(cls, is_curve=False, idx=len(strategies), seed=seed)
        if strategy is not None:
            strategies.append(strategy)

    return strategies


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    curves = [
        ("Helix", helix(radius=2.0, pitch=0.5)),
        ("Figure8 Space", figure8_space(scale_xy=2.0, scale_z=1.2)),
        ("Space Lissajous", space_lissajous(a=3, b=4, c=5, delta_x=np.pi / 2, delta_y=np.pi / 4, scale=2.5)),
        ("Trefoil Knot", trefoil_knot(scale=1.0, z_scale=1.0)),
    ]
    surfaces = [
        ("Whitney Umbrella", whitney_umbrella(scale=1.0)),
        ("Monkey Saddle", monkey_saddle(scale_xy=1.0, scale_z=0.25)),
        ("Cone", cone(scale_r=1.0, scale_z=1.0)),
    ]

    curve_strategies = build_curve_strategies()
    surface_strategies = build_surface_strategies()

    if not curve_strategies:
        raise RuntimeError("No curve strategies found in samplers.py")
    if not surface_strategies:
        raise RuntimeError("No surface strategies found in samplers.py")

    print("\n=== Curve Strategies ===")
    for strategy in curve_strategies:
        print(f"\n--- Strategy: {strategy.__class__.__name__} ---")
        for curve_name, curve in curves:
            analyze_curve_and_plot(curve, curve_name, strategy)

    print("\n=== Surface Strategies ===")
    for strategy in surface_strategies:
        print(f"\n--- Strategy: {strategy.__class__.__name__} ---")
        for surface_name, surface in surfaces:
            analyze_surface_and_plot(surface, surface_name, strategy)


if __name__ == "__main__":
    main()
