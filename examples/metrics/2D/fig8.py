import numpy as np
from typing import Optional

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import Sample, UniformCurve, figure8
from tangent_blowups.viz.dist import (
    visualize_distance_comparison,
    visualize_product_metric_comparison,
)


def _flatten_sample(s: Sample) -> Sample:
    points = np.asarray(s.points, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])

    tangents = s.tangents
    if tangents is not None:
        tangents = np.asarray(tangents, dtype=float)
        if tangents.ndim > 2:
            tangents = tangents.reshape(-1, tangents.shape[-1])

    normals = s.normals
    if normals is not None:
        normals = np.asarray(normals, dtype=float)
        if normals.ndim > 2:
            normals = normals.reshape(-1, normals.shape[-1])

    mask = s.singular_mask
    if mask is not None and mask.ndim > 1:
        mask = mask.reshape(-1)

    params = s.params
    if isinstance(params, tuple):
        params = tuple(np.asarray(p).reshape(-1) for p in params)
    else:
        params = np.asarray(params).reshape(-1) if np.asarray(params).ndim > 1 else params

    return Sample(
        points=points,
        params=params,
        tangents=tangents,
        normals=normals,
        singular_mask=mask,
        singular_indices=s.singular_indices,
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
    s = _flatten_sample(s)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(tangents).all(axis=1)
    norms = np.linalg.norm(tangents, axis=1)
    valid &= norms > tangent_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate tangents.")

    singular_candidates = None
    if s.singular_indices is not None:
        singular_candidates = np.asarray(s.singular_indices, dtype=int)
    elif s.singular_mask is not None:
        singular_candidates = np.flatnonzero(s.singular_mask)

    valid_indices = np.flatnonzero(valid)
    if singular_candidates is not None and singular_candidates.size > 0:
        singular_in_valid = np.intersect1d(valid_indices, singular_candidates, assume_unique=False)
        singular_filtered = np.searchsorted(valid_indices, singular_in_valid)
    else:
        singular_filtered = np.array([], dtype=int)

    points = points[valid]
    tangents = normalize_vectors(tangents[valid])
    return points, tangents, singular_filtered


def pick_query_indices_with_singular(
    n_points: int,
    singular_indices: np.ndarray,
    n_queries: int,
    seed: int,
) -> list[int]:
    if n_points <= 0:
        return []
    if n_queries <= 0:
        return []

    rng = np.random.default_rng(seed)
    chosen = []

    if singular_indices is not None and singular_indices.size > 0:
        singular_idx = int(rng.choice(singular_indices))
        chosen.append(singular_idx)

    if n_queries <= len(chosen):
        return chosen[:n_queries]

    candidates = np.setdiff1d(np.arange(n_points), np.array(chosen, dtype=int), assume_unique=False)
    extra = rng.choice(candidates, size=n_queries - len(chosen), replace=False).tolist()
    return chosen + extra


def _run_distance_plots(
    sample: BlownUpSample,
    *,
    query_index: int,
    alpha: float,
    downsample: int,
    cmap,
    highlight_fraction: float,
):
    for metric in ("geodesic", "chordal"):
        visualize_distance_comparison(
            sample,
            query_index=query_index,
            alpha=alpha,
            subspace_metric=metric,
            downsample=downsample,
            shared_color_scale=True,
            cmap=cmap,
            level_sets=True,
            n_levels=8,
            level_mode="linear",
            highlight_fraction=highlight_fraction,
        )

    visualize_product_metric_comparison(
        sample,
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


def main():
    points, tangents, singular_indices = build_fig8_points_and_tangents(
        n=1200,
        scale=2.0,
    )

    lifted = BlownUpSample.from_tangents(points, tangents)

    alpha = 1.0
    downsample = 2
    highlight_fraction = 0.1
    cmap = "viridis"
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
