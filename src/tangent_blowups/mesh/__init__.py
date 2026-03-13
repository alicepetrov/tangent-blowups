"""
Discrete blow-up and dual-graph construction for triangle meshes.
"""

from .blowup import (
    face_centroids,
    face_normals,
    face_projectors,
    face_tangent_frames,
    lifted_embeddings,
    build_dual_adjacency,
    dual_lifted_distances,
    MeshBlowUp,
    make_plane_grid,
    merge_meshes_weld,
)

__all__ = [
    "face_centroids",
    "face_normals",
    "face_projectors",
    "face_tangent_frames",
    "lifted_embeddings",
    "build_dual_adjacency",
    "dual_lifted_distances",
    "MeshBlowUp",
    "make_plane_grid",
    "merge_meshes_weld",
]
