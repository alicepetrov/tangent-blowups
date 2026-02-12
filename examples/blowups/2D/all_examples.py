import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import (
    UniformCurve,
    circle,
    figure8,
    lissajous,
    sample,
    square,
    triangle,
)
from tangent_blowups.viz.nash_coordinates import visualize_components, visualize_quiver


def build_curve_sample(
    curve,
    *,
    n: int = 1200,
    seed: int = 0,
    jitter_sigma: float = 0.002,
    t_max: float = 2.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample an oriented 2D curve and return points + unit tangents.
    """
    strategy = UniformCurve(n=n, t_min=0.0, t_max=t_max, endpoint=False)
    s = sample(curve, strategy, with_tangents=True, with_normals=False)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    if jitter_sigma > 0.0:
        rng = np.random.default_rng(seed)
        points += rng.normal(scale=jitter_sigma, size=points.shape)

    tangents = normalize_vectors(tangents)
    return points, tangents


def visualize_embedding_components(
    sample: BlownUpSample,
    *,
    alpha: float = 1.0,
    downsample: int = 1,
):
    """
    Visualize the 6D Nash embedding by coloring the 2D cloud with each component.

    Uses BlownUpSample.embedding_vector(alpha=...), which returns:
        [x, y, sqrt(alpha/2) * P00, sqrt(alpha/2) * P01,
         sqrt(alpha/2) * P10, sqrt(alpha/2) * P11]
    """
    pts = sample.spatial[::downsample]
    embed = sample.embedding_vector(alpha=alpha)[::downsample]

    labels = ["x", "y", "P00", "P01", "P10", "P11"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, ax in enumerate(axes):
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=embed[:, idx],
            s=6,
            cmap="viridis",
            alpha=0.9,
        )
        ax.set_title(labels[idx])
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Nash Embedding Components (R^2 x G(1,2) -> R^6)")
    plt.tight_layout()
    plt.show()


def _lift_and_plot(label: str, points: np.ndarray, tangents: np.ndarray):
    print(f"\n=== {label} ===")
    lifted = BlownUpSample.from_tangents(points, tangents)

    visualize_quiver(lifted, scale=0.05, downsample=5)
    visualize_components(lifted, components="all", downsample=2)
    visualize_embedding_components(lifted, alpha=1.0, downsample=2)


def main():
    curves = [
        ("Circle", circle(radius=2.0), 2.0 * np.pi),
        ("Figure 8 (Lemniscate)", figure8(scale=2.0), 2.0 * np.pi),
        ("Lissajous (3:4)", lissajous(a=3, b=4, delta=np.pi, scale=3.0), 2.0 * np.pi),
        ("Square", square(side=3.0), 4.0),
        ("Triangle", triangle(side=3.0), 3.0),
    ]

    for idx, (name, curve, t_max) in enumerate(curves):
        points, tangents = build_curve_sample(
            curve, n=1200, seed=7 + idx, jitter_sigma=0.002, t_max=t_max
        )
        _lift_and_plot(name, points, tangents)


if __name__ == "__main__":
    main()
