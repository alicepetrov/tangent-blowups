"""
Figure-8 Laplacian Example
--------------------------
Sample a 2D figure-8 curve, build point cloud and lifted Laplacians,
and visualize/compare their eigenpairs.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.pointcloud.laplacian import (
    pointcloud_laplacian,
    lifted_pointcloud_laplacian,
)
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformCurve, figure8, sample
from tangent_blowups.viz.laplacian import (
    visualize_laplacian_eigenpairs,
    visualize_laplacian_eigenpair_comparison,
)


def build_fig8_points_and_tangents(
    *,
    n: int = 1200,
    scale: float = 2.0,
    jitter: float = 0.02,
    seed: int = 7,
    tangent_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a 2D figure-8 curve and return (points, tangents).
    """
    curve = figure8(scale=scale)
    strategy = UniformCurve(n=n, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    s = sample(curve, strategy, with_tangents=True, with_normals=False)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=jitter, size=points.shape)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(tangents).all(axis=1)
    norms = np.linalg.norm(tangents, axis=1)
    valid &= norms > tangent_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate tangents.")

    points = points[valid]
    tangents = normalize_vectors(tangents[valid])
    return points, tangents


def compute_laplacians(
    points: np.ndarray,
    tangents: np.ndarray,
    *,
    k: int = 20,
    h: float | None = None,
    normalized: bool = True,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, BlowUpLevel]:
    """
    Build point cloud and lifted Laplacians for the same sample.
    """
    L_pc = pointcloud_laplacian(
        points,
        k=k,
        h=h,
        normalized=normalized,
        symmetrize=True,
    )

    lifted = BlowUpLevel.from_point_tangents(points, tangents)
    L_lifted = lifted_pointcloud_laplacian(
        lifted,
        k=k,
        h=h,
        alpha=alpha,
        normalized=normalized,
        symmetrize=True,
    )

    return L_pc, L_lifted, lifted


def _print_eigenvalues(label: str, evals: np.ndarray):
    formatted = np.array2string(evals, precision=5, floatmode="fixed")
    print(f"{label} eigenvalues: {formatted}")


def main():
    n = 1200
    scale = 2.0
    jitter = 0.02
    k = 20
    alpha = 1.0
    normalized = True

    eig_k = 6
    drop_first = True
    downsample = 2
    cmap = "RdBu_r"

    points, tangents = build_fig8_points_and_tangents(
        n=n,
        scale=scale,
        jitter=jitter,
        seed=7,
    )

    L_pc, L_lifted, _ = compute_laplacians(
        points,
        tangents,
        k=k,
        h=None,
        normalized=normalized,
        alpha=alpha,
    )

    fig_pc, axes_pc, evals_pc, _ = visualize_laplacian_eigenpairs(
        points,
        L_pc,
        k=eig_k,
        drop_first=drop_first,
        downsample=downsample,
        cmap=cmap,
        shared_color_scale=True,
        center_zero=True,
        show=False,
    )
    fig_pc.suptitle("Point Cloud Laplacian Eigenpairs")
    _print_eigenvalues("Point Cloud", evals_pc)
    plt.show()

    fig_lift, axes_lift, evals_lift, _ = visualize_laplacian_eigenpairs(
        points,
        L_lifted,
        k=eig_k,
        drop_first=drop_first,
        downsample=downsample,
        cmap=cmap,
        shared_color_scale=True,
        center_zero=True,
        show=False,
    )
    fig_lift.suptitle("Lifted Point Cloud Laplacian Eigenpairs")
    _print_eigenvalues("Lifted", evals_lift)
    plt.show()

    fig_cmp, axes_cmp, evals_left, evals_right, _, _ = (
        visualize_laplacian_eigenpair_comparison(
            points,
            L_pc,
            L_lifted,
            k=eig_k,
            drop_first=drop_first,
            downsample=downsample,
            cmap=cmap,
            shared_color_scale=True,
            center_zero=True,
            labels=("Point Cloud", "Lifted"),
            show=False,
        )
    )
    fig_cmp.suptitle("Laplacian Eigenpair Comparison")
    plt.show()

    if evals_pc.size == evals_lift.size and evals_pc.size > 0:
        diff = evals_lift - evals_pc
        formatted = np.array2string(diff, precision=5, floatmode="fixed")
        print(f"Eigenvalue differences (lifted - point cloud): {formatted}")


if __name__ == "__main__":
    main()
