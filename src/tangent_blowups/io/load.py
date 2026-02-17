from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from .adapters import stl_mesh_to_oriented_point_cloud
if TYPE_CHECKING:
    from ..testsupport.geom_types import Sample


def _is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False

    tri_count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + tri_count * 50
    if expected == len(data):
        return True

    header = data[:80].lstrip().lower()
    return not header.startswith(b"solid")


def _read_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    tri_count = struct.unpack("<I", data[80:84])[0]
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("v3", "<f4", (3,)),
            ("attr", "<u2"),
        ]
    )

    arr = np.frombuffer(data, dtype=dtype, count=tri_count, offset=84)
    triangles = np.stack([arr["v1"], arr["v2"], arr["v3"]], axis=1).astype(float)
    normals = arr["normal"].astype(float)
    return triangles, normals


def _read_ascii_stl(text: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    normals: list[list[float]] = []
    triangles: list[list[list[float]]] = []
    current: list[list[float]] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if (
            len(parts) >= 5
            and parts[0].lower() == "facet"
            and parts[1].lower() == "normal"
        ):
            try:
                normals.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
            continue

        if len(parts) >= 4 and parts[0].lower() == "vertex":
            try:
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                continue
            current.append(vertex)
            if len(current) == 3:
                triangles.append(current)
                current = []

    if not triangles:
        raise ValueError("No triangles found in ASCII STL.")

    tri = np.asarray(triangles, dtype=float)
    if len(normals) == len(triangles):
        return tri, np.asarray(normals, dtype=float)
    return tri, None


def read_stl(path: str | Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read an STL file from disk and return (triangles, normals).

    Triangles are shaped (F, 3, 3). Normals are (F, 3) when present,
    otherwise None (will be recomputed by adapters).
    """
    data = Path(path).read_bytes()
    if _is_binary_stl(data):
        try:
            return _read_binary_stl(data)
        except Exception:
            pass

    text = data.decode("utf-8", errors="ignore")
    try:
        return _read_ascii_stl(text)
    except Exception:
        return _read_binary_stl(data)


def load_stl(path: str | Path, **kwargs) -> Sample:
    """
    Load an STL file and convert it into an oriented point cloud Sample.

    Keyword arguments are forwarded to ``stl_mesh_to_oriented_point_cloud``.
    """
    triangles, normals = read_stl(path)
    return stl_mesh_to_oriented_point_cloud(triangles, normals, **kwargs)

def _deserialize_params(data: np.lib.npyio.NpzFile, n_points: int):
    if "params_kind" not in data:
        return np.arange(n_points, dtype=int)

    kind_raw = data["params_kind"]
    kind = str(kind_raw.item()) if np.ndim(kind_raw) == 0 else str(kind_raw)

    if kind == "tuple":
        if "params_count" not in data:
            raise ValueError("params_count missing for tuple params.")
        count = int(np.asarray(data["params_count"]).reshape(-1)[0])
        params = tuple(data[f"params_{i}"] for i in range(count))
        return params

    if kind == "array":
        if "params" not in data:
            raise ValueError("params missing for array params.")
        return data["params"]

    return np.arange(n_points, dtype=int)


def load_pointcloud(path: str | Path) -> Sample:
    """
    Load a point cloud from disk.

    Supports:
        - .npz (points + optional fields)
        - .npy (points only)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    from ..testsupport.geom_types import Sample

    if suffix == ".npy":
        points = np.load(path)
        points = np.asarray(points, dtype=float)
        params = np.arange(points.shape[0], dtype=int)
        return Sample(
            points=points,
            params=params,
            tangents=None,
            normals=None,
            singular_mask=None,
            singular_indices=None,
        )

    if suffix != ".npz":
        raise ValueError("Unsupported extension. Use '.npz' or '.npy'.")

    with np.load(path) as data:
        if "points" not in data:
            raise ValueError("Missing 'points' array in point cloud file.")

        points = np.asarray(data["points"], dtype=float)
        tangents = (
            np.asarray(data["tangents"], dtype=float)
            if "tangents" in data
            else None
        )
        normals = (
            np.asarray(data["normals"], dtype=float)
            if "normals" in data
            else None
        )
        singular_mask = (
            np.asarray(data["singular_mask"], dtype=bool)
            if "singular_mask" in data
            else None
        )
        singular_indices = (
            np.asarray(data["singular_indices"], dtype=int)
            if "singular_indices" in data
            else None
        )

        params = _deserialize_params(data, points.shape[0])

        return Sample(
            points=points,
            params=params,
            tangents=tangents,
            normals=normals,
            singular_mask=singular_mask,
            singular_indices=singular_indices,
        )


__all__ = ["load_stl", "read_stl", "load_pointcloud"]
