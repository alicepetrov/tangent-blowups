import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import (
    Sample,
    UniformCurve,
    add_point_noise,
    figure8,
    jitter_tangents,
    sample,
)
from tangent_blowups.viz.nash_coordinates import visualize_components, visualize_quiver


def build_oriented_fig8(
    *,
    n: int = 1200,
    scale: float = 2.0,
    seed: int = 7,
) -> Sample:
    """
    Sample an oriented point cloud from a 2D figure-8 curve.

    Returns:
        Sample with points (N, 2) and tangents (N, 2).
    """
    curve = figure8(scale=scale)
    strategy = UniformCurve(n=n, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    rng = np.random.default_rng(seed)

    # Use the test-support sampler to keep logic centralized in src/
    s = sample(curve, strategy, with_tangents=True, with_normals=False)

    # Small random jitter so the crossing is not visually degenerate
    s = add_point_noise(s, sigma=0.002, rng=rng)

    # Ensure tangents are unit-length (required for Grassmann projector lift)
    s.tangents = normalize_vectors(np.asarray(s.tangents, dtype=float))
    return s


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


def _lift_and_plot(label: str, sample_in: Sample):
    print(f"\n=== {label} ===")
    points = np.asarray(sample_in.points, dtype=float)
    tangents = normalize_vectors(np.asarray(sample_in.tangents, dtype=float))

    # Grassmann lift (Nash blowup): R^2 x G(1,2)
    lifted = BlownUpSample.from_tangents(points, tangents)

    # 1) Visualize tangents over the curve
    visualize_quiver(lifted, scale=0.05, downsample=5)

    # 2) Visualize the Grassmannian fiber components (P_ij) over R^2
    visualize_components(lifted, components="all", downsample=2)

    # 3) Explicit 6D embedding visualization (x, y, P00, P01, P10, P11)
    visualize_embedding_components(lifted, alpha=1.0, downsample=2)


def main():
    base = build_oriented_fig8(n=1200, scale=2.0, seed=7)
    rng_points = np.random.default_rng(13)
    rng_tangents = np.random.default_rng(19)

    noisy_points = add_point_noise(base, sigma=0.05, rng=rng_points)
    noisy_tangents = jitter_tangents(base, sigma=0.35, rng=rng_tangents, renormalize=True)
    noisy_both = jitter_tangents(noisy_points, sigma=0.35, rng=rng_tangents, renormalize=True)

    _lift_and_plot("Baseline (clean tangents)", base)
    _lift_and_plot("Noisy points (clean tangents)", noisy_points)
    _lift_and_plot("Jittered tangents (clean points)", noisy_tangents)
    _lift_and_plot("Noisy points + jittered tangents", noisy_both)


if __name__ == "__main__":
    main()
