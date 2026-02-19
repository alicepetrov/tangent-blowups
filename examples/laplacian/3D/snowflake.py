"""
Snowflake Laplacian Example
---------------------------
Load a snowflake point cloud, build point cloud and lifted Laplacians,
and visualize/compare their eigenpairs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud.laplacian import (
    pointcloud_laplacian,
    lifted_pointcloud_laplacian,
)
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.viz.laplacian import (
    visualize_laplacian_eigenpairs,
    visualize_laplacian_eigenpair_comparison,
)


def _flatten(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def _default_snowflake_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "thingi10k_pointcloud" / "snowflake.npz"


def load_snowflake_points_and_normals(
    path: str | Path | None = None,
    *,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the snowflake point cloud and return (points, normals).
    """
    if path is None:
        path = _default_snowflake_path()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snowflake data not found: {path}")

    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError("Snowflake point cloud is missing normals.")

    points = _flatten(sample.points)
    normals = _flatten(sample.normals)

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
) -> tuple[np.ndarray, np.ndarray, BlownUpSample]:
    L_pc = pointcloud_laplacian(
        points,
        k=k,
        h=h,
        normalized=normalized,
        symmetrize=True,
    )

    lifted = BlownUpSample.from_normals(points, normals)
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
    data_path = _default_snowflake_path()
    k = 20
    alpha = 1.0 # TODO We need to crank this up to see a difference
    normalized = True

    eig_k = 6
    drop_first = True
    downsample = 1
    cmap = "coolwarm"

    points, normals = load_snowflake_points_and_normals(
        data_path,
        normal_eps=1e-8,
    )

    print(f"Loaded {points.shape[0]} points from {data_path}")

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
    fig_pc.suptitle("Snowflake Point Cloud Laplacian Eigenpairs")
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
    fig_lift.suptitle("Snowflake Lifted Laplacian Eigenpairs")
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
    fig_cmp.suptitle("Snowflake Laplacian Eigenpair Comparison")
    plt.show()

    if evals_pc.size == evals_lift.size and evals_pc.size > 0:
        diff = evals_lift - evals_pc
        formatted = np.array2string(diff, precision=5, floatmode="fixed")
        print(f"Eigenvalue differences (lifted - point cloud): {formatted}")


if __name__ == "__main__":
    main()
