import inspect
import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.testsupport import samplers
from tangent_blowups.testsupport import circle, figure8, lissajous
from tangent_blowups.testsupport import ParametricCurve
from tangent_blowups.testsupport import SamplingStrategy

# -----------------------------------------------------------------------------
# Test Driver
# -----------------------------------------------------------------------------

def analyze_and_plot(
    curve: ParametricCurve,
    name: str,
    strategy: SamplingStrategy,
    singularity_tol: float = 0.05,
):
    """
    Samples a curve, detects singularities, and plots sampled points.
    """
    params = strategy(curve)
    sample = curve.evaluate(
        *params,
        with_tangents=True,
        with_normals=True,
        singularity_tol=singularity_tol,
    )

    points = sample.points
    tangents = sample.tangents
    mask = sample.singular_mask

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Curve: {name} ({strategy.__class__.__name__})")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.scatter(
        points[:, 0],
        points[:, 1],
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
            color="red",
            s=50,
            zorder=5,
            label="Detected Singularities",
        )

        ax.quiver(
            singular_pts[:, 0],
            singular_pts[:, 1],
            singular_tans[:, 0],
            singular_tans[:, 1],
            color="red",
            scale=20,
            width=0.005,
            alpha=0.8,
            label="Diverging Tangents",
        )

        print(f"[{name} | {strategy.__class__.__name__}] Detected {np.sum(mask)} singular sample points.")
    else:
        print(f"[{name} | {strategy.__class__.__name__}] No singularities detected (or none met tolerances).")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


def build_curve_strategies(
    n: int = 1000,
    t_min: float = 0.0,
    t_max: float = 2.0 * np.pi,
    seed: int = 0,
) -> list[SamplingStrategy]:
    """
    Discover and instantiate every curve sampling strategy in samplers.py.
    """
    strategies: list[SamplingStrategy] = []
    classes = inspect.getmembers(samplers, inspect.isclass)
    classes.sort(key=lambda item: item[0])

    for _, cls in classes:
        if cls is SamplingStrategy:
            continue
        if not issubclass(cls, SamplingStrategy):
            continue
        if "Curve" not in cls.__name__:
            continue

        kwargs = {}
        sig = inspect.signature(cls)

        if "n" in sig.parameters:
            kwargs["n"] = n
        if "t_min" in sig.parameters:
            kwargs["t_min"] = t_min
        if "t_max" in sig.parameters:
            kwargs["t_max"] = t_max
        if "endpoint" in sig.parameters:
            kwargs["endpoint"] = True
        if "rng" in sig.parameters:
            kwargs["rng"] = np.random.default_rng(seed + len(strategies))

        try:
            strategies.append(cls(**kwargs))
        except TypeError as exc:
            print(f"Skipping {cls.__name__}: cannot auto-instantiate ({exc})")

    return strategies


def main():
    curves = [
        ("Circle (No Intersections)", circle(radius=2.0)),
        ("Figure 8 (Lemniscate)", figure8(scale=2.0)),
        ("Lissajous (3:4)", lissajous(a=3, b=4, delta=np.pi, scale=4.0)),
    ]

    strategies = build_curve_strategies(n=1000, t_min=0.0, t_max=2 * np.pi, seed=7)
    if not strategies:
        raise RuntimeError("No curve sampling strategies were found in samplers.py")

    for strategy in strategies:
        print(f"\n=== Strategy: {strategy.__class__.__name__} ===")
        for curve_name, curve in curves:
            analyze_and_plot(curve, curve_name, strategy)


if __name__ == "__main__":
    main()
