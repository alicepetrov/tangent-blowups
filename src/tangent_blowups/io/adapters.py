from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.spatial import cKDTree

from ..solvers.linalg import normalize_vectors
from ..testsupport.geom_types import Sample


def _as_rng(rng: Optional[np.random.Generator | int]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _compute_face_normals(triangles: np.ndarray) -> np.ndarray:
    e0 = triangles[:, 1] - triangles[:, 0]
    e1 = triangles[:, 2] - triangles[:, 0]
    return np.cross(e0, e1)


def _prepare_triangles(
    triangles: np.ndarray,
    normals: Optional[np.ndarray],
    *,
    normal_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(triangles, dtype=float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"triangles must be (F, 3, 3). Got {tri.shape}.")

    face_normals = _compute_face_normals(tri)
    face_norms = np.linalg.norm(face_normals, axis=1)

    valid = (
        np.isfinite(tri).all(axis=(1, 2))
        & np.isfinite(face_normals).all(axis=1)
        & (face_norms > normal_eps)
    )

    if not np.any(valid):
        return tri[:0], face_normals[:0]

    tri = tri[valid]
    face_normals = face_normals[valid]

    normals_out = None
    if normals is not None:
        n_in = np.asarray(normals, dtype=float)
        if n_in.shape == face_normals.shape:
            normals_out = n_in[valid]
            n_norms = np.linalg.norm(normals_out, axis=1)
            bad = ~np.isfinite(normals_out).all(axis=1) | (n_norms <= normal_eps)
            if np.any(bad):
                normals_out[bad] = face_normals[bad]

    if normals_out is None:
        normals_out = face_normals

    normals_out = normalize_vectors(normals_out)
    return tri, normals_out


def _merge_vertices(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if tol <= 0.0:
        raise ValueError("vertex_tol must be positive.")

    quant = np.round(points / tol).astype(np.int64)
    _, inv = np.unique(quant, axis=0, return_inverse=True)

    n_unique = int(inv.max()) + 1 if inv.size else 0
    if n_unique == 0:
        return points[:0], normals[:0]

    sum_points = np.zeros((n_unique, 3), dtype=float)
    sum_normals = np.zeros((n_unique, 3), dtype=float)
    np.add.at(sum_points, inv, points)
    np.add.at(sum_normals, inv, normals)
    counts = np.bincount(inv).astype(float)

    merged_points = sum_points / counts[:, None]
    merged_normals = normalize_vectors(sum_normals)
    return merged_points, merged_normals


def _unique_vertices(
    points: np.ndarray,
    *,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if tol <= 0.0:
        raise ValueError("vertex_tol must be positive.")

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return pts[:0], np.array([], dtype=int)

    quant = np.round(pts / tol).astype(np.int64)
    _, inv = np.unique(quant, axis=0, return_inverse=True)

    n_unique = int(inv.max()) + 1 if inv.size else 0
    if n_unique == 0:
        return pts[:0], inv

    sum_points = np.zeros((n_unique, 3), dtype=float)
    np.add.at(sum_points, inv, pts)
    counts = np.bincount(inv).astype(float)

    unique_points = sum_points / counts[:, None]
    return unique_points, inv


def _sharp_face_mask(
    normals: np.ndarray,
    *,
    perp_dot_max: float,
) -> np.ndarray:
    if normals.shape[0] < 2:
        return np.zeros(normals.shape[0], dtype=bool)

    dots = np.abs(normals @ normals.T)
    np.fill_diagonal(dots, 1.0)
    return np.any(dots <= perp_dot_max, axis=1)


def _sample_points(
    triangles: np.ndarray,
    normals: np.ndarray,
    *,
    mode: Literal["faces", "vertices"],
    samples_per_face: int,
    rng: np.random.Generator,
    merge_vertices: bool,
    vertex_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "faces":
        if samples_per_face <= 0:
            raise ValueError("samples_per_face must be positive.")
        if samples_per_face <= 1:
            points = triangles.mean(axis=1)
            return points, normals

        v0 = triangles[:, 0][:, None, :]
        v1 = triangles[:, 1][:, None, :]
        v2 = triangles[:, 2][:, None, :]

        r1 = rng.random((triangles.shape[0], samples_per_face))
        r2 = rng.random((triangles.shape[0], samples_per_face))
        sqrt_r1 = np.sqrt(r1)

        w0 = 1.0 - sqrt_r1
        w1 = sqrt_r1 * (1.0 - r2)
        w2 = sqrt_r1 * r2

        points = (
            w0[..., None] * v0
            + w1[..., None] * v1
            + w2[..., None] * v2
        )
        points = points.reshape(-1, 3)
        normals_out = np.repeat(normals, samples_per_face, axis=0)
        return points, normals_out

    if mode == "vertices":
        points = triangles.reshape(-1, 3)
        normals_out = np.repeat(normals, 3, axis=0)

        if merge_vertices:
            points, normals_out = _merge_vertices(
                points, normals_out, tol=vertex_tol
            )

        return points, normals_out

    raise ValueError(f"Unknown sample_mode '{mode}'. Expected 'faces' or 'vertices'.")


def _sample_sharp_features(
    triangles: np.ndarray,
    normals: np.ndarray,
    *,
    edge_samples: int,
    vertex_tol: float,
    perp_angle_tol_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    if edge_samples < 0:
        raise ValueError("dense_edge_samples must be non-negative.")

    if triangles.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty

    if perp_angle_tol_deg <= 0.0:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty

    perp_angle = min(float(perp_angle_tol_deg), 90.0)
    perp_dot_max = float(np.sin(np.deg2rad(perp_angle)))

    unique_points, inv = _unique_vertices(triangles, tol=vertex_tol)
    if unique_points.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty

    vertex_ids = inv.reshape(triangles.shape[0], 3)

    points_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []

    if edge_samples > 0:
        edges = np.stack(
            [
                vertex_ids[:, [0, 1]],
                vertex_ids[:, [1, 2]],
                vertex_ids[:, [2, 0]],
            ],
            axis=1,
        )
        edges = np.sort(edges, axis=2)
        edges_flat = edges.reshape(-1, 2)
        face_ids = np.repeat(np.arange(vertex_ids.shape[0]), 3)

        if edges_flat.size > 0:
            order = np.lexsort((edges_flat[:, 1], edges_flat[:, 0]))
            edges_sorted = edges_flat[order]
            faces_sorted = face_ids[order]

            changes = np.any(edges_sorted[1:] != edges_sorted[:-1], axis=1)
            starts = np.concatenate(([0], np.flatnonzero(changes) + 1))
            ends = np.concatenate((starts[1:], [edges_sorted.shape[0]]))

            if edge_samples == 1:
                t = np.array([0.5], dtype=float)
            else:
                t = np.linspace(0.0, 1.0, edge_samples + 2, dtype=float)[1:-1]

            for start, end in zip(starts, ends):
                faces = faces_sorted[start:end]
                if faces.size < 2:
                    continue

                normals_group = normals[faces]
                sharp_mask = _sharp_face_mask(
                    normals_group, perp_dot_max=perp_dot_max
                )
                if not np.any(sharp_mask):
                    continue

                sharp_faces = faces[sharp_mask]
                normals_sharp = normals[sharp_faces]

                edge = edges_sorted[start]
                p0 = unique_points[edge[0]]
                p1 = unique_points[edge[1]]

                pts = (1.0 - t)[:, None] * p0 + t[:, None] * p1
                points_rep = np.repeat(pts, normals_sharp.shape[0], axis=0)
                normals_rep = np.tile(normals_sharp, (pts.shape[0], 1))

                points_list.append(points_rep)
                normals_list.append(normals_rep)

    vid_flat = vertex_ids.reshape(-1)
    face_flat = np.repeat(np.arange(vertex_ids.shape[0]), 3)

    if vid_flat.size > 0:
        order = np.argsort(vid_flat)
        vids_sorted = vid_flat[order]
        faces_sorted = face_flat[order]

        changes = vids_sorted[1:] != vids_sorted[:-1]
        starts = np.concatenate(([0], np.flatnonzero(changes) + 1))
        ends = np.concatenate((starts[1:], [vids_sorted.shape[0]]))

        for start, end in zip(starts, ends):
            faces = np.unique(faces_sorted[start:end])
            if faces.size < 2:
                continue

            normals_group = normals[faces]
            sharp_mask = _sharp_face_mask(
                normals_group, perp_dot_max=perp_dot_max
            )
            if not np.any(sharp_mask):
                continue

            sharp_faces = faces[sharp_mask]
            vertex_id = vids_sorted[start]
            pt = unique_points[vertex_id]

            points_list.append(np.repeat(pt[None, :], sharp_faces.size, axis=0))
            normals_list.append(normals[sharp_faces])

    if not points_list:
        empty = np.zeros((0, 3), dtype=float)
        return empty, empty

    points_out = np.concatenate(points_list, axis=0)
    normals_out = np.concatenate(normals_list, axis=0)
    return points_out, normals_out


def _sample_dense_points(
    triangles: np.ndarray,
    normals: np.ndarray,
    *,
    samples_per_face: int,
    edge_samples: int,
    perp_angle_tol_deg: float,
    rng: np.random.Generator,
    vertex_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    points, normals_out = _sample_points(
        triangles,
        normals,
        mode="faces",
        samples_per_face=samples_per_face,
        rng=rng,
        merge_vertices=False,
        vertex_tol=vertex_tol,
    )

    extra_points, extra_normals = _sample_sharp_features(
        triangles,
        normals,
        edge_samples=edge_samples,
        vertex_tol=vertex_tol,
        perp_angle_tol_deg=perp_angle_tol_deg,
    )

    if extra_points.shape[0] == 0:
        return points, normals_out

    points = np.concatenate([points, extra_points], axis=0)
    normals_out = np.concatenate([normals_out, extra_normals], axis=0)
    return points, normals_out


def detect_normal_singularities(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 16,
    min_angle_deg: float = 45.0,
    orientation_invariant: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mark points whose neighboring normals deviate by more than min_angle_deg.

    If orientation_invariant=True, uses |dot(n_i, n_j)| so flipped normals do
    not register as singularities on otherwise smooth surfaces.
    """
    n_points = points.shape[0]
    mask = np.zeros(n_points, dtype=bool)
    if n_points == 0 or k <= 0:
        return mask, np.array([], dtype=int)

    kk = min(k + 1, n_points)
    tree = cKDTree(points)
    _, neighbors = tree.query(points, k=kk, workers=-1)

    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]

    if neighbors.shape[1] <= 1:
        return mask, np.array([], dtype=int)

    neighbors = neighbors[:, 1:]
    n_unit = normalize_vectors(normals)

    dots = np.einsum("ij,ikj->ik", n_unit, n_unit[neighbors])
    if orientation_invariant:
        dots = np.abs(dots)

    threshold = float(np.cos(np.deg2rad(min_angle_deg)))
    mask = np.any(dots < threshold, axis=1)
    idx = np.flatnonzero(mask)
    return mask, idx


def stl_mesh_to_oriented_point_cloud(
    triangles: np.ndarray,
    normals: Optional[np.ndarray] = None,
    *,
    sample_mode: Literal["faces", "vertices", "dense"] = "faces",
    samples_per_face: int = 1,
    dense_edge_samples: int = 6,
    dense_perp_angle_tol_deg: float = 10.0,
    rng: Optional[np.random.Generator | int] = None,
    merge_vertices: bool = False,
    vertex_tol: float = 1e-6,
    detect_singularities: bool = True,
    singular_k: int = 16,
    singular_angle_deg: float = 45.0,
    orientation_invariant: bool = True,
    normal_eps: float = 1e-12,
) -> Sample:
    """
    Convert a triangle mesh (from STL) into an oriented point cloud Sample.

    Returns a Sample with points, normals, and optional singularity markers.
    Use sample_mode="dense" to add extra samples along sharp edges and corners
    (faces with nearly perpendicular normals), duplicating points per face normal.
    Perpendicular is defined as |angle - 90 deg| <= dense_perp_angle_tol_deg.
    """
    tri, face_normals = _prepare_triangles(
        triangles,
        normals,
        normal_eps=normal_eps,
    )

    if tri.shape[0] == 0:
        empty = np.zeros((0, 3), dtype=float)
        return Sample(
            points=empty,
            params=np.array([], dtype=int),
            tangents=None,
            normals=empty,
            singular_mask=np.zeros((0,), dtype=bool),
            singular_indices=np.array([], dtype=int),
        )

    rng = _as_rng(rng)
    if sample_mode == "dense":
        points, normals_out = _sample_dense_points(
            tri,
            face_normals,
            samples_per_face=samples_per_face,
            edge_samples=dense_edge_samples,
            perp_angle_tol_deg=dense_perp_angle_tol_deg,
            rng=rng,
            vertex_tol=vertex_tol,
        )
    else:
        points, normals_out = _sample_points(
            tri,
            face_normals,
            mode=sample_mode,
            samples_per_face=samples_per_face,
            rng=rng,
            merge_vertices=merge_vertices,
            vertex_tol=vertex_tol,
        )

    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals_out).all(axis=1)
    n_norms = np.linalg.norm(normals_out, axis=1)
    valid &= n_norms > normal_eps

    if not np.all(valid):
        points = points[valid]
        normals_out = normals_out[valid]

    if detect_singularities and points.shape[0] > 0:
        mask, idx = detect_normal_singularities(
            points,
            normals_out,
            k=singular_k,
            min_angle_deg=singular_angle_deg,
            orientation_invariant=orientation_invariant,
        )
    else:
        mask = np.zeros(points.shape[0], dtype=bool)
        idx = np.array([], dtype=int)

    params = np.arange(points.shape[0], dtype=int)
    return Sample(
        points=points,
        params=params,
        tangents=None,
        normals=normals_out,
        singular_mask=mask,
        singular_indices=idx,
    )


__all__ = [
    "detect_normal_singularities",
    "stl_mesh_to_oriented_point_cloud",
]
