import numpy as np
from typing import Optional

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.testsupport import UniformSurface, whitney_umbrella
from tangent_blowups.testsupport.sample_utils import (
    filter_sample_vectors,
    flatten_sample,
    pick_query_indices_with_singular,
)
from tangent_blowups.viz.dist import (
    visualize_distance_comparison,
    visualize_product_metric_comparison,
)


def build_whitney_points_and_normals(
    *,
    nu: int = 80,
    nv: int = 80,
    u_bounds: tuple[float, float] = (-2.0, 2.0),
    v_bounds: tuple[float, float] = (-2.0, 2.0),
    scale: float = 1.0,
    normal_eps: float = 1e-8,
    singularity_tol: Optional[float] = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a Whitney umbrella surface and return (points, normals, singular_indices).
    Filters degenerate normals for numerical stability.
    """
    surface = whitney_umbrella(scale=scale)
    strategy = UniformSurface(nu=nu, nv=nv, u_bounds=u_bounds, v_bounds=v_bounds)
    params = strategy(surface)
    s = surface.evaluate(
        *params,
        with_tangents=True,
        with_normals=True,
        singularity_tol=singularity_tol,
    )
    s = flatten_sample(s)
    points, normals, singular_filtered = filter_sample_vectors(
        s,
        s.normals,
        eps=normal_eps,
        name="normals",
    )
    return points, normals, singular_filtered


def _run_distance_plots(
    sample: BlowUpLevel,
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
    points, normals, singular_indices = build_whitney_points_and_normals(
        nu=80,
        nv=80,
        u_bounds=(-2.0, 2.0),
        v_bounds=(-2.0, 2.0),
        scale=1.0,
    )

    lifted = BlowUpLevel.from_normals(points, normals)

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
