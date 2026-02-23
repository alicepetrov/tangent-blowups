from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tangent_blowups.io.load import load_stl, read_stl
from tangent_blowups.io.save import save_pointcloud


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    pts = np.asarray(points).reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _plot_points_only(
    points: np.ndarray,
    *,
    title: str,
    color: str = "slateblue",
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=3,
        color=color,
        alpha=0.65,
        label="Points",
    )

    _set_axes_equal_3d(ax, points)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_singular_points(
    points: np.ndarray,
    mask: np.ndarray | None,
    *,
    title: str,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if mask is None or not np.any(mask):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2,
            color="lightgray",
            alpha=0.25,
            label="Points",
        )
        ax.text2D(0.03, 0.95, "No singular points detected", transform=ax.transAxes)
    else:
        singular = points[mask]
        ax.scatter(
            singular[:, 0],
            singular[:, 1],
            singular[:, 2],
            s=8,
            color="crimson",
            alpha=0.8,
            label="Singular points",
        )

    _set_axes_equal_3d(ax, points)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_oriented(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    title: str,
    max_vectors: int = 2000,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=3,
        color="slateblue",
        alpha=0.6,
        label="Points",
    )

    step = max(1, points.shape[0] // max_vectors)
    pts = points[::step]
    nrms = normals[::step]

    bbox = points.max(axis=0) - points.min(axis=0)
    arrow_len = 0.03 * float(np.max(bbox))

    ax.quiver(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        nrms[:, 0],
        nrms[:, 1],
        nrms[:, 2],
        length=arrow_len,
        normalize=True,
        color="orangered",
        alpha=0.8,
        linewidth=0.6,
        label="Oriented normals",
    )

    _set_axes_equal_3d(ax, points)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_mesh(
    triangles: np.ndarray,
    *,
    title: str,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    mesh = Poly3DCollection(
        triangles,
        facecolor=(0.2, 0.6, 0.8, 0.25),
        edgecolor=(0.1, 0.3, 0.4, 0.35),
        linewidths=0.4,
    )
    ax.add_collection3d(mesh)

    _set_axes_equal_3d(ax, triangles.reshape(-1, 3))
    plt.tight_layout()
    plt.show()


def _default_stl_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "thingi10k" / "snowflake.stl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an STL mesh to a point cloud and plot it."
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=_default_stl_path(),
        help="Path to STL mesh.",
    )
    parser.add_argument(
        "--samples-per-face",
        type=int,
        default=30,
        help="Number of samples per triangle face.",
    )
    parser.add_argument(
        "--sample-mode",
        type=str,
        default="uniform",
        choices=["faces", "vertices", "dense", "uniform"], # TODO add a dense_perp_angle_tol_deg option
        help="Sampling strategy for converting mesh to points.",
    )
    parser.add_argument(
        "--merge-vertices",
        action="store_true",
        help="Merge duplicate vertices (only applies to vertex sampling).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the point cloud (.npz or .npy).",
    )
    args = parser.parse_args()

    triangles, _ = read_stl(args.stl)
    sample = load_stl(
        args.stl,
        sample_mode=args.sample_mode,
        samples_per_face=args.samples_per_face,
        merge_vertices=args.merge_vertices,
    )

    points = np.asarray(sample.points, dtype=float)
    normals = np.asarray(sample.normals, dtype=float)

    print(f"Loaded {points.shape[0]} points from {args.stl}")

    if args.save is not None:
        save_path = args.save
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".npz")
        save_pointcloud(save_path, sample=sample)
        print(f"Saved point cloud to {save_path}")

    _plot_points_only(points, title="Point Cloud (Points Only)")
    _plot_singular_points(
        points,
        sample.singular_mask,
        title="Point Cloud (Detected Singular Points)",
    )
    _plot_oriented(points, normals, title="Point Cloud (Oriented Normals)")
    _plot_mesh(triangles, title="Original Mesh")


if __name__ == "__main__":
    main()
