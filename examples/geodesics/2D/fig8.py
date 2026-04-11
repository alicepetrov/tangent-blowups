"""
Figure-8 Lifted Heat Method Example
----------------------------------
Compute lifted heat-method distances on a 2D figure-8 curve and visualize them.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.pointcloud import lifted_heat_method
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformCurve, figure8, sample


def build_oriented_fig8(
    *,
    n: int = 1200,
    scale: float = 2.0,
    jitter: float = 0.02,
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

    s = sample(curve, strategy, with_tangents=True, with_normals=False)
    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    if jitter > 0.0:
        points = points + rng.normal(scale=jitter, size=points.shape)

    tangents = normalize_vectors(tangents)
    return points, tangents


def _pick_source_index(points: np.ndarray) -> int:
    """
    Choose a source point near the self-intersection (closest to origin).
    """
    if points.size == 0:
        raise ValueError("No points available to pick a source.")
    return int(np.argmin(np.linalg.norm(points, axis=1)) + 20)  # Offset to avoid the exact intersection point  


def main():
    n = 1200
    scale = 2.0
    jitter = 0.02
    seed = 7

    k = 20
    alpha = 1.0
    laplacian_normalized = False
    t = None
    t_scale = 1.0

    points, tangents = build_oriented_fig8(
        n=n,
        scale=scale,
        jitter=jitter,
        seed=seed,
    )

    source_index = _pick_source_index(points)
    print(f"Using source index {source_index}")

    dist = lifted_heat_method(
        points,
        tangents,
        source_index=source_index,
        k=k,
        alpha=alpha,
        laplacian_normalized=laplacian_normalized,
        t=t,
        t_scale=t_scale,
    )

    # Anchor distance at the source and remove small numerical negatives.
    dist = dist - dist[source_index]
    dist = np.maximum(dist, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=dist,
        s=8.0,
        cmap="magma",
        alpha=0.9,
    )
    ax.scatter(
        [points[source_index, 0]],
        [points[source_index, 1]],
        s=80,
        c="none",
        edgecolors="k",
        linewidths=1.5,
    )
    ax.set_aspect("equal")
    ax.set_title("Lifted Heat-Method Distances (Figure-8)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
