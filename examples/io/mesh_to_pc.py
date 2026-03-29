from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tangent_blowups.io.load import load_stl, read_stl
from tangent_blowups.io.save import save_pointcloud
from tangent_blowups.testsupport.geom_types import Sample


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


def _default_thingi10k_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "thingi10k"


def _default_thingi10k_output() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "thingi10k_pointcloud"


def _subset_params(params: np.ndarray | tuple[np.ndarray, ...], idx: np.ndarray):
    if isinstance(params, tuple):
        return tuple(p[idx] for p in params)
    return params[idx]


def _downsample_sample(
    sample: Sample,
    *,
    max_points: int | None,
    rng: np.random.Generator,
) -> tuple[Sample, tuple[int, int] | None]:
    if max_points is None:
        return sample, None
    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    n_points = sample.points.shape[0]
    if n_points <= max_points:
        return sample, (n_points, n_points)

    idx = rng.choice(n_points, size=max_points, replace=False)
    idx.sort()

    points = sample.points[idx]
    params = _subset_params(sample.params, idx)
    tangents = sample.tangents[idx] if sample.tangents is not None else None
    normals = sample.normals[idx] if sample.normals is not None else None

    if sample.singular_mask is not None:
        singular_mask = sample.singular_mask[idx]
        singular_indices = np.flatnonzero(singular_mask)
    else:
        singular_mask = None
        singular_indices = None

    return (
        Sample(
            points=points,
            params=params,
            tangents=tangents,
            normals=normals,
            singular_mask=singular_mask,
            singular_indices=singular_indices,
        ),
        (n_points, max_points),
    )


def _batch_convert_thingi10k(
    *,
    input_root: Path,
    output_root: Path,
    sample_mode: str,
    samples_per_face: int,
    merge_vertices: bool,
    max_points: int | None,
    downsample_seed: int | None,
) -> None:
    stl_files = sorted(input_root.glob("*.stl"))
    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    total = len(stl_files)
    print(f"Found {total} STL files in {input_root}")
    rng = np.random.default_rng(downsample_seed)

    for idx, stl_path in enumerate(stl_files, start=1):
        try:
            sample = load_stl(
                stl_path,
                sample_mode=sample_mode,
                samples_per_face=samples_per_face,
                merge_vertices=merge_vertices,
            )
            sample, downsampled = _downsample_sample(
                sample,
                max_points=max_points,
                rng=rng,
            )
            save_path = output_root / f"{stl_path.stem}.npz"
            save_pointcloud(save_path, sample=sample)
        except Exception as exc:
            print(f"[{idx}/{total}] Failed {stl_path.name}: {exc}")
            continue

        if idx == 1 or idx == total or idx % 100 == 0:
            if downsampled is not None and downsampled[0] != downsampled[1]:
                print(
                    f"[{idx}/{total}] Saved {save_path.name} "
                    f"(downsampled {downsampled[0]} -> {downsampled[1]})"
                )
            else:
                print(f"[{idx}/{total}] Saved {save_path.name}")


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
        default=20,
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
        "--batch-thingi10k",
        action="store_true",
        help=(
            "Process all STL files in data/thingi10k and save point clouds to "
            "data/thingi10k_pointcloud."
        ),
    )
    parser.add_argument(
        "--merge-vertices",
        action="store_true",
        help="Merge duplicate vertices (only applies to vertex sampling).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="If set, downsample point clouds to at most this many points (e.g. 25000).",
    )
    parser.add_argument(
        "--downsample-seed",
        type=int,
        default=None,
        help="Optional RNG seed for downsampling.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the point cloud (.npz or .npy).",
    )
    args = parser.parse_args()

    if args.batch_thingi10k:
        _batch_convert_thingi10k(
            input_root=_default_thingi10k_root(),
            output_root=_default_thingi10k_output(),
            sample_mode=args.sample_mode,
            samples_per_face=args.samples_per_face,
            merge_vertices=args.merge_vertices,
            max_points=args.max_points,
            downsample_seed=args.downsample_seed,
        )
        return

    triangles, _ = read_stl(args.stl)
    sample = load_stl(
        args.stl,
        sample_mode=args.sample_mode,
        samples_per_face=args.samples_per_face,
        merge_vertices=args.merge_vertices,
    )
    rng = np.random.default_rng(args.downsample_seed)
    sample, downsampled = _downsample_sample(
        sample,
        max_points=args.max_points,
        rng=rng,
    )

    points = np.asarray(sample.points, dtype=float)
    normals = np.asarray(sample.normals, dtype=float)

    print(f"Loaded {points.shape[0]} points from {args.stl}")
    if downsampled is not None and downsampled[0] != downsampled[1]:
        print(f"Downsampled {downsampled[0]} -> {downsampled[1]} points")

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
