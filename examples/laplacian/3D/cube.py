"""
Cube Laplacian Example
----------------------
Sample a cube surface, build point cloud and lifted Laplacians,
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
from tangent_blowups.testsupport import RandomSurface, cube_surface, sample
from tangent_blowups.viz.laplacian import (
    visualize_laplacian_eigenpairs,
    visualize_laplacian_eigenpair_comparison,
)


def _flatten(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def build_cube_points_and_normals(
    *,
    nu: int = 100,
    nv: int = 100,
    u_bounds: tuple[float, float] = (0.0, 6.0),
    v_bounds: tuple[float, float] = (0.0, 1.0),
    scale: float = 1.0,
    jitter: float = 0.0,
    seed: int = 7,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a cube surface and return (points, normals).
    """
    surface = cube_surface(scale=scale)
    rng = np.random.default_rng(seed)
    strategy = RandomSurface(
        n=nu * nv,
        u_bounds=u_bounds,
        v_bounds=v_bounds,
        rng=rng,
    )
    s = sample(surface, strategy, with_tangents=False, with_normals=True)

    points = _flatten(s.points)
    normals = _flatten(s.normals)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=jitter, size=points.shape)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    norms = np.linalg.norm(normals, axis=1)
    valid &= norms > normal_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate normals.")

    points = points[valid]
    normals = normalize_vectors(normals[valid])
    return points, normals


def compute_laplacians(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 20,
    h: float | None = None,
    normalized: bool = True,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, BlowUpLevel]:
    L_pc = pointcloud_laplacian(
        points,
        k=k,
        h=h,
        normalized=normalized,
        symmetrize=True,
    )

    lifted = BlowUpLevel.from_normals(points, normals)
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
    nu = 100
    nv = 100
    scale = 1.0
    jitter = 0.0
    k = 20
    alpha = 1.0
    normalized = True

    eig_k = 6
    drop_first = True
    downsample = 4
    cmap = "RdBu_r"

    points, normals = build_cube_points_and_normals(
        nu=nu,
        nv=nv,
        u_bounds=(0.0, 6.0),
        v_bounds=(0.0, 1.0),
        scale=scale,
        jitter=jitter,
        seed=7,
    )

    L_pc, L_lifted, _ = compute_laplacians(
        points,
        normals,
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
    fig_pc.suptitle("Cube Point Cloud Laplacian Eigenpairs")
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
    fig_lift.suptitle("Cube Lifted Laplacian Eigenpairs")
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
    fig_cmp.suptitle("Cube Laplacian Eigenpair Comparison")
    plt.show()

    if evals_pc.size == evals_lift.size and evals_pc.size > 0:
        diff = evals_lift - evals_pc
        formatted = np.array2string(diff, precision=5, floatmode="fixed")
        print(f"Eigenvalue differences (lifted - point cloud): {formatted}")


if __name__ == "__main__":
    main()
