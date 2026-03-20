"""Load three SCEC Sierra Madre fault meshes and save as a single combined STL."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def parse_gocad_tsurf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a GOCAD TSurf (.ts) file.

    Returns
    -------
    vertices : (V, 3) float64
    faces : (F, 3) int32, 0-based indices
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    id_to_idx: dict[int, int] = {}

    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] in ("VRTX", "PVRTX"):
                vid = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                id_to_idx[vid] = len(vertices)
                vertices.append((x, y, z))
            elif parts[0] == "TRGL":
                a, b, c = int(parts[1]), int(parts[2]), int(parts[3])
                faces.append((id_to_idx[a], id_to_idx[b], id_to_idx[c]))

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def merge_meshes(
    meshes: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple (vertices, faces) meshes into one, offsetting face indices."""
    all_verts = []
    all_faces = []
    offset = 0
    for verts, faces in meshes:
        all_verts.append(verts)
        all_faces.append(faces + offset)
        offset += verts.shape[0]
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL file from vertices and face indices."""
    n_faces = faces.shape[0]
    # 80-byte header + 4-byte triangle count
    header = b"\x00" * 80
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_faces))
        for i in range(n_faces):
            v0, v1, v2 = vertices[faces[i, 0]], vertices[faces[i, 1]], vertices[faces[i, 2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            # normal (3 floats) + 3 vertices (9 floats) + attribute byte count
            f.write(struct.pack("<3f", *normal.astype(np.float32)))
            f.write(struct.pack("<3f", *v0.astype(np.float32)))
            f.write(struct.pack("<3f", *v1.astype(np.float32)))
            f.write(struct.pack("<3f", *v2.astype(np.float32)))
            f.write(struct.pack("<H", 0))


def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "scec"
    ts_files = sorted(data_dir.glob("*.ts"))

    if not ts_files:
        raise FileNotFoundError(f"No .ts files found in {data_dir}")

    print(f"Found {len(ts_files)} fault meshes in {data_dir}")

    meshes = []
    for ts_path in ts_files:
        verts, faces = parse_gocad_tsurf(ts_path)
        print(f"  {ts_path.name}: {verts.shape[0]} vertices, {faces.shape[0]} triangles")
        meshes.append((verts, faces))

    all_verts, all_faces = merge_meshes(meshes)
    print(f"Combined: {all_verts.shape[0]} vertices, {all_faces.shape[0]} triangles")

    out_path = data_dir / "sierra_madre_combined.stl"
    write_binary_stl(out_path, all_verts, all_faces)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
