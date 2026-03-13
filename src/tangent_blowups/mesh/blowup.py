"""
Discrete Iterated Blow-Up for Triangle Meshes
----------------------------------------------
Implements the *lifted dual graph* construction for non-manifold triangle meshes.

Each face f is lifted to a node

    Phi_f = (x_f,  sqrt(alpha/2) * vec(P_f))  in R^{3 + 9}

where x_f is the face centroid and P_f = I - n_f n_f^T is the tangent
projector (n_f = unit face normal).  Two nodes are connected when the
corresponding faces share a mesh edge AND their lifted distance is below
a threshold tau.

Separation theorem
------------------
Let theta_max be the maximum turning angle between adjacent co-sheet faces
(a function of mesh resolution and surface smoothness), and let delta be
the minimum dihedral angle at any cross-sheet non-manifold junction.  Then

    same-sheet edge:   ||Phi_f - Phi_g|| <= sqrt(alpha) * |sin(theta_max)| + O(h)
    cross-sheet edge:  ||Phi_f - Phi_g|| >= sqrt(alpha) * |sin(delta)|

so any threshold tau in the gap

    (sqrt(alpha) * |sin(theta_max)| + O(h),  sqrt(alpha) * |sin(delta)|)

correctly severs all cross-sheet edges while retaining all same-sheet edges.

The gap is non-empty whenever

    |sin(delta)| > |sin(theta_max)| + O(h / sqrt(alpha))

which holds for sufficiently fine meshes and/or sufficiently large alpha.

Proof sketch
------------
For a surface with unit normals n_f, n_g:

    ||P_f - P_g||_F^2 = ||n_f n_f^T - n_g n_g^T||_F^2
                      = 2(1 - (n_f . n_g)^2)
                      = 2 sin^2(theta_{fg})

where theta_{fg} is the angle between the normals.  The projector contribution
to the lifted distance is therefore sqrt(alpha/2) * sqrt(2) * |sin theta_{fg}|
= sqrt(alpha) * |sin theta_{fg}|.

For same-sheet adjacent faces on a smooth surface, theta_{fg} = O(h * kappa),
where kappa is the principal curvature and h is the triangle diameter.
For cross-sheet faces at a non-manifold junction with dihedral angle delta,
theta_{fg} = delta.

Iterated construction
---------------------
The construction generalises to multiple levels by treating each face as a
point in the lifted space and applying the point-cloud iterated blow-up
(BlowUpLevel) to the dual node embeddings.  Level 1 (implemented here) is
sufficient for component separation; higher levels resolve curvature
differences between sheets that happen to share a normal direction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face centroids.  Returns (F, 3)."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    return (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0


def face_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute face normals via the cross product.  Returns (F, 3).

    Args:
        vertices: (V, 3) vertex positions.
        faces:    (F, 3) triangle vertex indices.
        normalize: if True (default), return unit normals; otherwise area-weighted.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    e1 = v[f[:, 1]] - v[f[:, 0]]   # (F, 3)
    e2 = v[f[:, 2]] - v[f[:, 0]]   # (F, 3)
    n = np.cross(e1, e2)            # (F, 3)
    if normalize:
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms = np.where(norms > 1e-15, norms, 1.0)
        n = n / norms
    return n


def face_projectors(normals: np.ndarray) -> np.ndarray:
    """
    Compute tangent projectors P_f = I - n_f n_f^T.  Returns (F, 3, 3).

    Args:
        normals: (F, 3) unit face normals.
    """
    n = np.asarray(normals, dtype=float)
    I3 = np.eye(3, dtype=float)
    return I3[None] - np.einsum("fi,fj->fij", n, n)   # (F, 3, 3)


def lifted_embeddings(
    centroids: np.ndarray,
    projectors: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Compute Chordal-Sasaki embeddings Phi_f = (x_f, sqrt(alpha/2) * vec(P_f)).

    Returns (F, 12) for 3-D surfaces (3 position + 9 projector components).
    """
    x = np.asarray(centroids,  dtype=float)   # (F, 3)
    P = np.asarray(projectors, dtype=float)   # (F, 3, 3)
    scale = np.sqrt(alpha / 2.0)
    P_flat = P.reshape(P.shape[0], -1)         # (F, 9)
    return np.hstack([x, scale * P_flat])      # (F, 12)


# ---------------------------------------------------------------------------
# Dual graph construction
# ---------------------------------------------------------------------------

def build_dual_adjacency(
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the combinatorial dual graph for a triangle mesh.

    An edge (f, g) exists in the dual graph when faces f and g share a mesh
    edge.  Non-manifold mesh edges (shared by 3+ faces) generate all pairwise
    pairs among their incident faces.

    Args:
        faces: (F, 3) triangle vertex indices.

    Returns:
        dual_edges:      (E, 2) int — (face_i, face_j) pairs with i < j.
        edge_valence:    (E,)   int — number of faces incident to the
                                     corresponding mesh edge.
        edge_vert_pairs: (E, 2) int — (v_min, v_max) identifying each mesh edge.
    """
    f = np.asarray(faces, dtype=int)
    F = len(f)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi in range(F):
        for k in range(3):
            a, b = int(f[fi, k]), int(f[fi, (k + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append(fi)

    dual_edges_list:   list[tuple[int, int]] = []
    valences_list:     list[int]             = []
    vert_pairs_list:   list[tuple[int, int]] = []

    for (va, vb), incident in edge_to_faces.items():
        val = len(incident)
        for i in range(val):
            for j in range(i + 1, val):
                dual_edges_list.append((incident[i], incident[j]))
                valences_list.append(val)
                vert_pairs_list.append((va, vb))

    if dual_edges_list:
        dual_edges      = np.array(dual_edges_list, dtype=int)
        edge_valence    = np.array(valences_list,   dtype=int)
        edge_vert_pairs = np.array(vert_pairs_list, dtype=int)
    else:
        dual_edges      = np.zeros((0, 2), dtype=int)
        edge_valence    = np.zeros((0,),   dtype=int)
        edge_vert_pairs = np.zeros((0, 2), dtype=int)

    return dual_edges, edge_valence, edge_vert_pairs


# ---------------------------------------------------------------------------
# Main data structure
# ---------------------------------------------------------------------------

@dataclass
class MeshBlowUp:
    """
    Lifted dual graph for a triangle mesh (level-1 blow-up).

    Attributes:
        centroids:    (F, 3)    face centroids.
        normals:      (F, 3)    unit face normals.
        projectors:   (F, 3, 3) tangent projectors P_f = I - n_f n_f^T.
        phi:          (F, 12)   lifted embeddings Phi_f (Chordal-Sasaki).
        dual_edges:   (E, 2)    combinatorial dual-graph edge pairs (i < j).
        edge_valence: (E,)      number of mesh faces per mesh edge.
        lifted_dists: (E,)      ||Phi_f - Phi_g|| for each dual edge.
        labels:       (F,)      connected-component index per face.
        severed:      (E,)      bool — True for dual edges cut by the threshold.
        tau:          threshold used to compute the current labelling.
        alpha:        Chordal-Sasaki weight used for lifting.
    """

    centroids:    np.ndarray
    normals:      np.ndarray
    projectors:   np.ndarray
    phi:          np.ndarray
    dual_edges:   np.ndarray
    edge_valence: np.ndarray
    lifted_dists: np.ndarray
    labels:       np.ndarray
    severed:      np.ndarray
    tau:          float
    alpha:        float

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        alpha: float = 1.0,
        tau: float | None = None,
    ) -> "MeshBlowUp":
        """
        Construct the level-1 lifted dual graph from a triangle mesh.

        Args:
            vertices: (V, 3) vertex positions.
            faces:    (F, 3) triangle vertex indices.
            alpha:    Chordal-Sasaki weight.  Larger alpha amplifies the
                      projector contribution to the lifted metric, widening
                      the separation gap between co-sheet and cross-sheet edges.
            tau:      Lifted-distance threshold.  Dual edges with
                      ||Phi_f - Phi_g|| >= tau are severed.
                      If None, tau is set to the midpoint of the bimodal gap
                      in the lifted-distance distribution (the value minimising
                      the count in the gap histogram bin).

        Returns:
            MeshBlowUp instance with connected-component labels.
        """
        vertices = np.asarray(vertices, dtype=float)
        faces    = np.asarray(faces,    dtype=int)

        cents = face_centroids(vertices, faces)
        norms = face_normals(vertices, faces)
        projs = face_projectors(norms)
        phi   = lifted_embeddings(cents, projs, alpha=alpha)

        dual_edges, edge_valence, _ = build_dual_adjacency(faces)

        if len(dual_edges) > 0:
            fi, fj = dual_edges[:, 0], dual_edges[:, 1]
            diff = phi[fi] - phi[fj]
            lifted_dists = np.sqrt(np.einsum("ei,ei->e", diff, diff))
        else:
            lifted_dists = np.zeros(0, dtype=float)

        if tau is None:
            tau = _auto_tau(lifted_dists)

        labels, severed = _compute_components(
            len(faces), dual_edges, lifted_dists, tau,
        )

        return cls(
            centroids=cents, normals=norms, projectors=projs, phi=phi,
            dual_edges=dual_edges, edge_valence=edge_valence,
            lifted_dists=lifted_dists, labels=labels, severed=severed,
            tau=float(tau), alpha=float(alpha),
        )

    # ------------------------------------------------------------------
    # Re-thresholding
    # ------------------------------------------------------------------

    def rethreshold(self, tau: float) -> "MeshBlowUp":
        """Return a new MeshBlowUp computed with a different threshold tau."""
        import dataclasses
        labels, severed = _compute_components(
            len(self.centroids), self.dual_edges, self.lifted_dists, tau,
        )
        return dataclasses.replace(self, labels=labels, severed=severed, tau=float(tau))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of connected components after thresholding."""
        return int(self.labels.max()) + 1 if len(self.labels) > 0 else 0

    @property
    def singular_face_mask(self) -> np.ndarray:
        """
        Boolean mask (F,) — True for faces incident to at least one severed dual
        edge.  These faces adjoin the singular (non-manifold) locus.
        """
        mask = np.zeros(len(self.centroids), dtype=bool)
        if np.any(self.severed):
            fi = self.dual_edges[self.severed, 0]
            fj = self.dual_edges[self.severed, 1]
            mask[fi] = True
            mask[fj] = True
        return mask

    def distance_gap(self) -> tuple[float, float]:
        """
        Return (max_kept_dist, min_severed_dist).

        The gap (max_kept_dist, min_severed_dist) is the interval of valid
        threshold values.  A positive gap confirms correct separation.
        """
        kept_d    = self.lifted_dists[~self.severed]
        severed_d = self.lifted_dists[self.severed]
        max_kept    = float(kept_d.max())    if len(kept_d)    > 0 else 0.0
        min_severed = float(severed_d.min()) if len(severed_d) > 0 else np.inf
        return max_kept, min_severed

    def __repr__(self) -> str:
        return (
            f"MeshBlowUp(F={len(self.centroids)}, E={len(self.dual_edges)}, "
            f"components={self.n_components}, tau={self.tau:.4f}, alpha={self.alpha})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _auto_tau(lifted_dists: np.ndarray) -> float:
    """
    Estimate tau by finding the midpoint of the largest gap in the sorted
    lifted-distance values.  Falls back to the median for unimodal inputs.
    """
    if len(lifted_dists) == 0:
        return 1.0
    s = np.sort(lifted_dists)
    gaps = np.diff(s)
    if gaps.max() <= 0.0:
        return float(np.median(s)) + 1e-9
    idx = int(np.argmax(gaps))
    return float(0.5 * (s[idx] + s[idx + 1]))


def _compute_components(
    n_faces: int,
    dual_edges: np.ndarray,
    lifted_dists: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Threshold dual edges and compute connected components.

    Returns:
        labels:  (F,) int component index per face.
        severed: (E,) bool, True where lifted_dists >= tau.
    """
    severed = lifted_dists >= tau

    if n_faces == 0 or len(dual_edges) == 0:
        return np.zeros(n_faces, dtype=int), severed

    kept = ~severed
    if not np.any(kept):
        return np.arange(n_faces, dtype=int), severed

    fi = dual_edges[kept, 0]
    fj = dual_edges[kept, 1]
    data = np.ones(np.sum(kept), dtype=float)
    adj = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([fi, fj]), np.concatenate([fj, fi]))),
        shape=(n_faces, n_faces),
    )
    _, labels = connected_components(adj, directed=False)
    return labels.astype(int), severed


# ---------------------------------------------------------------------------
# Mesh construction helpers (for testing / examples)
# ---------------------------------------------------------------------------

def make_plane_grid(
    n: int = 15,
    *,
    normal_axis: int = 2,
    u_range: tuple[float, float] = (0.0, 1.0),
    v_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a triangle mesh for a flat plane.

    The plane is spanned by the two axes not equal to ``normal_axis``.

    Args:
        n:           number of subdivisions per axis (produces 2*n*n triangles).
        normal_axis: which of {0, 1, 2} is the normal direction.
        u_range:     range of the first in-plane coordinate.
        v_range:     range of the second in-plane coordinate.

    Returns:
        vertices: (V, 3) float.
        faces:    (F, 3) int.
    """
    axes = [a for a in range(3) if a != normal_axis]   # two in-plane axes
    NV = n + 1

    u = np.linspace(u_range[0], u_range[1], NV)
    v = np.linspace(v_range[0], v_range[1], NV)
    U, V = np.meshgrid(u, v, indexing="ij")   # (NV, NV)

    verts = np.zeros((NV * NV, 3), dtype=float)
    verts[:, axes[0]] = U.ravel()
    verts[:, axes[1]] = V.ravel()

    faces = []
    for i in range(n):
        for j in range(n):
            a = i * NV + j
            b = (i + 1) * NV + j
            c = i * NV + (j + 1)
            d = (i + 1) * NV + (j + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    return verts, np.array(faces, dtype=int)


def merge_meshes_weld(
    meshes: list[tuple[np.ndarray, np.ndarray]],
    *,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate multiple meshes and weld vertices that coincide within ``tol``.

    Returns:
        vertices:    (V_merged, 3) float.
        faces:       (F_total, 3) int — indices into merged vertex array.
        mesh_labels: (F_total,)   int — which input mesh each face came from.
    """
    # Concatenate raw
    all_verts = np.concatenate([v for v, _ in meshes], axis=0)
    face_parts = []
    offset = 0
    for v, f in meshes:
        face_parts.append(f + offset)
        offset += len(v)
    all_faces  = np.concatenate(face_parts, axis=0)
    mesh_labels = np.concatenate([
        np.full(len(f), i, dtype=int) for i, (_, f) in enumerate(meshes)
    ])

    # Weld duplicate vertices
    from scipy.spatial import cKDTree
    tree = cKDTree(all_verts)
    # For each vertex, find the smallest-indexed duplicate
    pairs = tree.query_pairs(r=tol, output_type="ndarray")  # (K, 2)

    remap = np.arange(len(all_verts), dtype=int)
    # Union-find to collapse groups
    for a, b in pairs:
        ra, rb = _uf_find(remap, a), _uf_find(remap, b)
        if ra != rb:
            remap[max(ra, rb)] = min(ra, rb)

    # Compress indices
    for i in range(len(remap)):
        remap[i] = _uf_find(remap, i)
    unique, inverse = np.unique(remap, return_inverse=True)
    merged_verts = all_verts[unique]
    merged_faces = inverse[all_faces]

    return merged_verts, merged_faces, mesh_labels


def _uf_find(parent: np.ndarray, x: int) -> int:
    """Path-compressed union-find root."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def face_tangent_frames(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute orthonormal tangent frames for each face.

    Returns (F, 3, 2): two orthonormal vectors spanning each face's tangent
    plane.  The first basis vector is aligned with the first edge (v1 - v0);
    the second is the cross product of the face normal and the first vector.

    Args:
        vertices: (V, 3) vertex positions.
        faces:    (F, 3) triangle vertex indices.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces,    dtype=int)

    e1 = v[f[:, 1]] - v[f[:, 0]]                          # (F, 3)
    e2_raw = v[f[:, 2]] - v[f[:, 0]]                      # (F, 3)
    n_raw  = np.cross(e1, e2_raw)                          # (F, 3) unnorm

    e1_norms = np.linalg.norm(e1, axis=1, keepdims=True)
    e1_norms = np.where(e1_norms > 1e-15, e1_norms, 1.0)
    e1 = e1 / e1_norms                                    # (F, 3) unit

    n_norms  = np.linalg.norm(n_raw, axis=1, keepdims=True)
    n_norms  = np.where(n_norms > 1e-15, n_norms, 1.0)
    n_unit   = n_raw / n_norms                             # (F, 3) unit

    e2 = np.cross(n_unit, e1)                              # (F, 3) unit (n perp e1)

    return np.stack([e1, e2], axis=2)                      # (F, 3, 2)


def dual_lifted_distances(
    dual_edges: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Compute ||Phi_f - Phi_g|| for each dual edge.

    Args:
        dual_edges: (E, 2) int pair array.
        phi:        (F, D) lifted embedding for each face.

    Returns:
        (E,) float lifted distances.
    """
    if len(dual_edges) == 0:
        return np.zeros(0, dtype=float)
    fi, fj = dual_edges[:, 0], dual_edges[:, 1]
    diff = phi[fi] - phi[fj]
    return np.sqrt(np.einsum("ei,ei->e", diff, diff))


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
