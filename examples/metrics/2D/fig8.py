import numpy as np
from typing import Optional

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.testsupport import UniformCurve, figure8
from tangent_blowups.testsupport.sample_utils import (
    filter_sample_vectors,
    flatten_sample,
    pick_query_indices_with_singular,
)
from tangent_blowups.viz.dist import (
    visualize_distance_comparison,
    visualize_product_metric_comparison,
)


def build_fig8_points_and_tangents(
    *,
    n: int = 1200,
    scale: float = 2.0,
    tangent_eps: float = 1e-8,
    singularity_tol: Optional[float] = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a 2D figure-8 curve and return (points, tangents, singular_indices).
    Filters degenerate tangents for numerical stability.
    """
    curve = figure8(scale=scale)
    strategy = UniformCurve(n=n, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    params = strategy(curve)
    s = curve.evaluate(
        *params,
        with_tangents=True,
        with_normals=False,
        singularity_tol=singularity_tol,
    )
    s = flatten_sample(s)
    points, tangents, singular_filtered = filter_sample_vectors(
        s,
        s.tangents,
        eps=tangent_eps,
        name="tangents",
    )
    return points, tangents, singular_filtered


def _run_distance_plots(
    sample: BlownUpSample,
    *,
    query_index: int,
    alpha: float,
    downsample: int,
    cmap,
    highlight_fraction: float,
):
    plot_kwargs = dict(
        query_index=query_index,
        alpha=alpha,
        downsample=downsample,
        shared_color_scale=True,
        cmap=cmap,
        level_sets=True,
        n_levels=8,
        level_mode="linear",
        highlight_fraction=highlight_fraction,
    )
    for metric in ("geodesic", "chordal"):
        visualize_distance_comparison(
            sample,
            subspace_metric=metric,
            **plot_kwargs,
        )

    visualize_product_metric_comparison(
        sample,
        **plot_kwargs,
    )


def main():
    points, tangents, singular_indices = build_fig8_points_and_tangents(
        n=1200,
        scale=2.0,
    )

    lifted = BlownUpSample.from_tangents(points, tangents)

    alpha = 1.0
    downsample = 2
    highlight_fraction = 0.1
    cmap = "magma"
    query_indices = pick_query_indices_with_singular(
        lifted.N,
        singular_indices,
        n_queries=2,
        seed=7,
    )

    print(f"Total samples: {lifted.N} | Query indices: {query_indices}")

    for idx in query_indices:
        print(f"\n--- Query {idx} ---")
        _run_distance_plots(
            lifted,
            query_index=idx,
            alpha=alpha,
            downsample=downsample,
            cmap=cmap,
            highlight_fraction=highlight_fraction,
        )


if __name__ == "__main__":
    main()
