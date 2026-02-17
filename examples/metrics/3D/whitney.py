import numpy as np
from typing import Optional

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import Sample, UniformSurface, whitney_umbrella
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
    s = _flatten_sample(s)

    points = np.asarray(s.points, dtype=float)
    normals = np.asarray(s.normals, dtype=float)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    norms = np.linalg.norm(normals, axis=1)
    valid &= norms > normal_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate normals.")

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
    normals = normalize_vectors(normals[valid])
    return points, normals, singular_filtered


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
    points, normals, singular_indices = build_whitney_points_and_normals(
        nu=80,
        nv=80,
        u_bounds=(-2.0, 2.0),
        v_bounds=(-2.0, 2.0),
        scale=1.0,
    )

    lifted = BlownUpSample.from_normals(points, normals)

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
