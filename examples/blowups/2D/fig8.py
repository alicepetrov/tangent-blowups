import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import (
    UniformCurve,
    figure8,
    sample,
)
from tangent_blowups.viz.nash_coordinates import visualize_components, visualize_quiver


def build_oriented_fig8(
    *,
    n: int = 1200,
    scale: float = 2.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample an oriented point cloud from a 2D figure-8 curve.

    Returns:
        points: (N, 2)
        tangents: (N, 2) unit vectors
    """
    curve = figure8(scale=scale)
    strategy = UniformCurve(n=n, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    rng = np.random.default_rng(seed)

    # Use the test-support sampler to keep logic centralized in src/
    s = sample(curve, strategy, with_tangents=True, with_normals=False)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    # Small random jitter so the crossing is not visually degenerate
    points += rng.normal(scale=0.02, size=points.shape)

    # Make sure tangents are unit-length (required for Grassmann projector lift)
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
            cmap="magma",
            alpha=0.9,
        )
        ax.set_title(labels[idx])
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Nash Embedding Components (R^2 x G(1,2) -> R^6)")
    plt.tight_layout()
    plt.show()


def main():
    points, tangents = build_oriented_fig8(n=1200, scale=2.0, seed=7)

    # Grassmann lift (Nash blowup): R^2 x G(1,2)
    lifted = BlownUpSample.from_tangents(points, tangents)

    # 1) Visualize tangents over the curve
    visualize_quiver(lifted, scale=0.05, downsample=5)

    # 2) Visualize the Grassmannian fiber components (P_ij) over R^2
    visualize_components(lifted, components="all", downsample=2)

    # 3) Explicit 6D embedding visualization (x, y, P00, P01, P10, P11)
    visualize_embedding_components(lifted, alpha=1.0, downsample=2)


if __name__ == "__main__":
    main()
