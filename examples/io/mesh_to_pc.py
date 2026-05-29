from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tangent_blowups.io.adapters import (
    detect_normal_singularities,
    stl_mesh_to_oriented_point_cloud,
)
from tangent_blowups.io.load import load_mesh, read_mesh
from tangent_blowups.io.save import save_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_stl_path() -> Path:
    return _repo_root() / "data" / "thingi10k" / "snowflake.stl"


def _default_thingi10k_root() -> Path:
    return _repo_root() / "data" / "thingi10k"


def _default_thingi10k_output() -> Path:
    return _repo_root() / "data" / "thingi10k_pointcloud"


def _default_threedscans_root() -> Path:
    return _repo_root() / "data" / "threedscans"


def _default_threedscans_output() -> Path:
    return _repo_root() / "data" / "threedscans_pointcloud"


@dataclass(frozen=True)
class _PlyHeader:
    format_name: str
    vertex_count: int
    vertex_properties: tuple[tuple[str, str], ...]
    data_offset: int
    first_element: str | None


_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_ply_header(path: Path) -> _PlyHeader:
    format_name: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    first_element: str | None = None

    with path.open("rb") as fh:
        first = fh.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file.")

        while True:
            raw = fh.readline()
            if raw == b"":
                raise ValueError("Unexpected end of file while reading PLY header.")

            line = raw.decode("ascii", errors="strict").strip()
            if line == "end_header":
                data_offset = fh.tell()
                break
            if not line or line.startswith("comment ") or line.startswith("obj_info "):
                continue

            parts = line.split()
            if parts[0] == "format" and len(parts) >= 2:
                format_name = parts[1]
                continue

            if parts[0] == "element" and len(parts) >= 3:
                current_element = parts[1]
                if first_element is None:
                    first_element = current_element
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                continue

            if parts[0] == "property" and current_element == "vertex":
                if len(parts) >= 5 and parts[1] == "list":
                    raise ValueError("List-valued vertex properties are not supported.")
                if len(parts) >= 3:
                    property_type = parts[1].lower()
                    property_name = parts[2]
                    if property_type not in _PLY_DTYPES:
                        raise ValueError(
                            f"Unsupported PLY vertex property type '{property_type}'."
                        )
                    vertex_properties.append((property_name, property_type))

    if format_name is None:
        raise ValueError("Missing PLY format declaration.")
    if vertex_count is None:
        raise ValueError("PLY file has no vertex element.")
    if format_name not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format '{format_name}'.")

    return _PlyHeader(
        format_name=format_name,
        vertex_count=vertex_count,
        vertex_properties=tuple(vertex_properties),
        data_offset=data_offset,
        first_element=first_element,
    )


def _ply_binary_dtype(
    properties: tuple[tuple[str, str], ...],
    *,
    format_name: str,
) -> np.dtype:
    endian = "<" if format_name == "binary_little_endian" else ">"
    fields = []
    for name, type_name in properties:
        dtype_code = _PLY_DTYPES[type_name]
        dtype = np.dtype(dtype_code if dtype_code.endswith("1") else endian + dtype_code)
        fields.append((name, dtype))
    return np.dtype(fields)


def _sample_indices(
    n_points: int,
    *,
    max_points: int | None,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if max_points is None or n_points <= max_points:
        return None
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    idx = rng.choice(n_points, size=max_points, replace=False)
    idx.sort()
    return idx


def _property_indices(
    properties: tuple[tuple[str, str], ...],
) -> dict[str, int]:
    return {name: idx for idx, (name, _) in enumerate(properties)}


def _normal_property_names(property_names: set[str]) -> tuple[str, str, str] | None:
    for names in (("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")):
        if all(name in property_names for name in names):
            return names
    return None


def _sample_from_structured_vertices(
    vertices: np.ndarray,
    *,
    indices: np.ndarray | None,
    property_names: set[str],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    subset = vertices if indices is None else vertices[indices]

    points = np.column_stack(
        [
            np.asarray(subset["x"], dtype=float),
            np.asarray(subset["y"], dtype=float),
            np.asarray(subset["z"], dtype=float),
        ]
    )

    normal_names = _normal_property_names(property_names)
    normals = None
    if normal_names is not None:
        normals = np.column_stack(
            [np.asarray(subset[name], dtype=float) for name in normal_names]
        )

    params = (
        np.arange(points.shape[0], dtype=int)
        if indices is None
        else np.asarray(indices, dtype=int)
    )
    return points, normals, params


def _read_binary_ply_vertices(
    path: Path,
    header: _PlyHeader,
    *,
    max_points: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, tuple[int, int] | None]:
    if header.first_element != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported.")

    property_names = {name for name, _ in header.vertex_properties}
    if not {"x", "y", "z"}.issubset(property_names):
        raise ValueError("PLY vertex element must contain x, y, and z properties.")

    dtype = _ply_binary_dtype(
        header.vertex_properties,
        format_name=header.format_name,
    )
    vertices = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count,),
    )
    indices = _sample_indices(
        header.vertex_count,
        max_points=max_points,
        rng=rng,
    )
    points, normals, params = _sample_from_structured_vertices(
        vertices,
        indices=indices,
        property_names=property_names,
    )
    downsampled = (
        (header.vertex_count, points.shape[0])
        if indices is not None
        else None
    )
    return points, normals, params, downsampled


def _read_ascii_ply_vertices(
    path: Path,
    header: _PlyHeader,
    *,
    max_points: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, tuple[int, int] | None]:
    if header.first_element != "vertex":
        raise ValueError("Only PLY files with vertex as the first element are supported.")

    prop_idx = _property_indices(header.vertex_properties)
    if not {"x", "y", "z"}.issubset(prop_idx):
        raise ValueError("PLY vertex element must contain x, y, and z properties.")

    normal_names = _normal_property_names(set(prop_idx))
    indices = _sample_indices(
        header.vertex_count,
        max_points=max_points,
        rng=rng,
    )
    target = None if indices is None else iter(indices.tolist())
    next_target = None if target is None else next(target, None)

    points_list: list[list[float]] = []
    normals_list: list[list[float]] = []
    params_list: list[int] = []

    with path.open("rb") as fh:
        fh.seek(header.data_offset)
        for vertex_idx in range(header.vertex_count):
            raw = fh.readline()
            if raw == b"":
                raise ValueError("Unexpected end of file while reading PLY vertices.")

            if target is not None:
                if vertex_idx != next_target:
                    continue
                params_list.append(vertex_idx)
                next_target = next(target, None)
            else:
                params_list.append(vertex_idx)

            values = raw.decode("ascii", errors="strict").split()
            points_list.append(
                [
                    float(values[prop_idx["x"]]),
                    float(values[prop_idx["y"]]),
                    float(values[prop_idx["z"]]),
                ]
            )
            if normal_names is not None:
                normals_list.append([float(values[prop_idx[name]]) for name in normal_names])

            if next_target is None and target is not None:
                break

    points = np.asarray(points_list, dtype=float)
    normals = np.asarray(normals_list, dtype=float) if normals_list else None
    params = np.asarray(params_list, dtype=int)
    downsampled = (
        (header.vertex_count, points.shape[0])
        if indices is not None
        else None
    )
    return points, normals, params, downsampled


def _filter_point_sample(
    points: np.ndarray,
    normals: np.ndarray | None,
    params: np.ndarray,
    *,
    normal_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    valid = np.isfinite(points).all(axis=1)
    if normals is not None:
        valid &= np.isfinite(normals).all(axis=1)
        valid &= np.linalg.norm(normals, axis=1) > normal_eps

    if not np.all(valid):
        points = points[valid]
        params = params[valid]
        if normals is not None:
            normals = normals[valid]

    if normals is not None:
        normals = normalize_vectors(normals)

    return points, normals, params


def _estimate_normals_open3d(
    points: np.ndarray,
    *,
    normal_k: int,
    orient_normals: bool,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    if normal_k <= 0:
        raise ValueError("normal_k must be positive.")

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    knn = min(int(normal_k), points.shape[0])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )
    if orient_normals and points.shape[0] > 3:
        try:
            pcd.orient_normals_consistent_tangent_plane(knn)
        except RuntimeError as exc:
            print(f"Warning: could not consistently orient PLY normals: {exc}")

    normals = np.asarray(pcd.normals, dtype=float)
    return normalize_vectors(normals)


def _load_ply_pointcloud(
    path: Path,
    *,
    max_points: int | None,
    rng: np.random.Generator,
    estimate_normals: bool,
    normal_k: int,
    orient_normals: bool,
    detect_singularities: bool,
    singular_k: int,
    singular_angle_deg: float,
) -> tuple[Sample, tuple[int, int] | None]:
    header = _read_ply_header(path)
    if header.format_name == "ascii":
        points, normals, params, downsampled = _read_ascii_ply_vertices(
            path,
            header,
            max_points=max_points,
            rng=rng,
        )
    else:
        points, normals, params, downsampled = _read_binary_ply_vertices(
            path,
            header,
            max_points=max_points,
            rng=rng,
        )

    points, normals, params = _filter_point_sample(points, normals, params)

    if normals is None and estimate_normals:
        print(f"Estimating normals with k={normal_k}...")
        normals = _estimate_normals_open3d(
            points,
            normal_k=normal_k,
            orient_normals=orient_normals,
        )

    singular_mask = None
    singular_indices = None
    if detect_singularities and normals is not None and points.shape[0] > 0:
        singular_mask, singular_indices = detect_normal_singularities(
            points,
            normals,
            k=singular_k,
            min_angle_deg=singular_angle_deg,
        )

    return (
        Sample(
            points=points,
            params=params,
            tangents=None,
            normals=normals,
            singular_mask=singular_mask,
            singular_indices=singular_indices,
        ),
        downsampled,
    )


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


def _batch_convert(
    *,
    input_root: Path,
    output_root: Path,
    extensions: list[str],
    sample_mode: str,
    samples_per_face: int,
    merge_vertices: bool,
    max_points: int | None,
    downsample_seed: int | None,
) -> None:
    mesh_files: list[Path] = []
    for ext in extensions:
        mesh_files.extend(input_root.glob(f"*{ext}"))
    mesh_files.sort(key=lambda p: p.name)

    if not mesh_files:
        exts = ", ".join(extensions)
        raise FileNotFoundError(f"No mesh files ({exts}) found in {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    total = len(mesh_files)
    print(f"Found {total} mesh files in {input_root}")
    rng = np.random.default_rng(downsample_seed)

    for idx, mesh_path in enumerate(mesh_files, start=1):
        try:
            triangles, face_normals = read_mesh(mesh_path)
            n_faces = triangles.shape[0]
            spf = samples_per_face
            mode = sample_mode
            if max_points is not None and n_faces * spf > max_points * 4:
                # Cap allocation: no point generating far more samples than
                # we'll keep after downsampling.  Use a 4× oversampling margin
                # so the uniform distribution stays reasonable.
                spf = max(1, (max_points * 4) // n_faces)
                if spf == 1 and n_faces >= max_points:
                    mode = "vertices"
            sample = stl_mesh_to_oriented_point_cloud(
                triangles,
                face_normals,
                sample_mode=mode,
                samples_per_face=spf,
                merge_vertices=merge_vertices,
            )
            sample, downsampled = _downsample_sample(
                sample,
                max_points=max_points,
                rng=rng,
            )
            save_path = output_root / f"{mesh_path.stem}.npz"
            save_pointcloud(save_path, sample=sample)
        except Exception as exc:
            print(f"[{idx}/{total}] Failed {mesh_path.name}: {exc}")
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
        description="Convert a mesh or PLY vertex set to a point cloud and plot it."
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Path to mesh/point file (.stl, .obj, or .ply).",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        default=None,
        help="Path to STL mesh (alias for --mesh, kept for backwards compat).",
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
        "--batch-threedscans",
        action="store_true",
        help=(
            "Process all OBJ/STL files in data/threedscans and save point "
            "clouds to data/threedscans_pointcloud."
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
        default=200000,
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots after conversion.",
    )
    parser.add_argument(
        "--no-estimate-normals",
        action="store_false",
        dest="estimate_normals",
        help="For PLY files without normals, save points without estimating normals.",
    )
    parser.set_defaults(estimate_normals=True)
    parser.add_argument(
        "--normal-k",
        type=int,
        default=30,
        help="Number of nearest neighbors used when estimating PLY normals.",
    )
    parser.add_argument(
        "--orient-normals",
        action="store_true",
        help="Try to consistently orient estimated PLY normals.",
    )
    parser.add_argument(
        "--detect-singularities",
        action="store_true",
        help="Run normal-variation singularity detection for PLY point clouds.",
    )
    parser.add_argument(
        "--singular-k",
        type=int,
        default=16,
        help="Neighbors used for singularity detection.",
    )
    parser.add_argument(
        "--singular-angle-deg",
        type=float,
        default=45.0,
        help="Minimum normal angle used for singularity detection.",
    )
    args = parser.parse_args()

    batch_kwargs = dict(
        sample_mode=args.sample_mode,
        samples_per_face=args.samples_per_face,
        merge_vertices=args.merge_vertices,
        max_points=args.max_points,
        downsample_seed=args.downsample_seed,
    )

    if args.batch_thingi10k:
        _batch_convert(
            input_root=_default_thingi10k_root(),
            output_root=_default_thingi10k_output(),
            extensions=[".stl"],
            **batch_kwargs,
        )
        return

    if args.batch_threedscans:
        _batch_convert(
            input_root=_default_threedscans_root(),
            output_root=_default_threedscans_output(),
            extensions=[".obj", ".stl"],
            **batch_kwargs,
        )
        return

    mesh_path = args.mesh or args.stl or _default_stl_path()
    if mesh_path.suffix.lower() == ".ply":
        rng = np.random.default_rng(args.downsample_seed)
        sample, downsampled = _load_ply_pointcloud(
            mesh_path,
            max_points=args.max_points,
            rng=rng,
            estimate_normals=args.estimate_normals,
            normal_k=args.normal_k,
            orient_normals=args.orient_normals,
            detect_singularities=args.detect_singularities,
            singular_k=args.singular_k,
            singular_angle_deg=args.singular_angle_deg,
        )

        points = np.asarray(sample.points, dtype=float)
        normals = (
            np.asarray(sample.normals, dtype=float)
            if sample.normals is not None
            else None
        )

        print(f"Loaded {points.shape[0]} points from {mesh_path}")
        if downsampled is not None and downsampled[0] != downsampled[1]:
            print(f"Downsampled {downsampled[0]} -> {downsampled[1]} points")

        if args.save is not None:
            save_path = args.save
            if save_path.suffix == "":
                save_path = save_path.with_suffix(".npz")
            save_pointcloud(save_path, sample=sample)
            print(f"Saved point cloud to {save_path}")

        if args.no_plot:
            return

        _plot_points_only(points, title="Point Cloud (Points Only)")
        _plot_singular_points(
            points,
            sample.singular_mask,
            title="Point Cloud (Detected Singular Points)",
        )
        if normals is not None:
            _plot_oriented(points, normals, title="Point Cloud (Estimated Normals)")
        else:
            print("Skipping normal plot because the PLY point cloud has no normals.")
        return

    triangles, _ = read_mesh(mesh_path)
    sample = load_mesh(
        mesh_path,
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

    print(f"Loaded {points.shape[0]} points from {mesh_path}")
    if downsampled is not None and downsampled[0] != downsampled[1]:
        print(f"Downsampled {downsampled[0]} -> {downsampled[1]} points")

    if args.save is not None:
        save_path = args.save
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".npz")
        save_pointcloud(save_path, sample=sample)
        print(f"Saved point cloud to {save_path}")

    if args.no_plot:
        return

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
