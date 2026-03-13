"""
Curvature Estimation on a Non-Manifold Mesh: Hemisphere + Plane
===============================================================
Demonstrates the blow-up's advantage over discrete curvature baselines
on a non-manifold mesh where two smooth sheets share boundary edges.

Geometry
--------
  - Sheet 0: upper unit hemisphere  (K = 1, H = 1 everywhere)
  - Sheet 1: flat disk at z = 0     (K = 0, H = 0 everywhere)

The two sheets share vertices along the equator (the unit circle at
z = 0), creating non-manifold edges.

Why discrete methods fail
-------------------------
Discrete curvature baselines (angle defect, cotangent Laplacian) operate
on vertices.  Vertices on the equator are shared between both sheets, so
the angle defect mixes angles from the curved hemisphere with angles from
the flat disk — producing corrupted K and H estimates in a band around
the junction.

Why the blow-up succeeds
-------------------------
The iterated blow-up estimates curvature per *face* via ridge regression
over a k-NN neighbourhood in the Chordal-Sasaki lifted space.  At the
equator the dihedral angle is 90°, so the lifted metric cleanly separates
same-sheet neighbours from cross-sheet ones.  The regression therefore
uses only geometrically coherent neighbours, preserving accuracy right
up to the junction.

Figures
-------
  1. 3-D mesh coloured by curvature: ground truth, blow-up, discrete.
  2. Per-face absolute error heatmap on the mesh.
  3. RMSE vs distance-from-junction profile.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tangent_blowups.geometry.iterated_grassmann import (
    BlowUpLevel,
    extract_level1,
)
from tangent_blowups.mesh.blowup import (
    face_centroids,
    face_normals,
    face_tangent_frames,
    merge_meshes_weld,
)


# ---------------------------------------------------------------------------
# Polar mesh builders
# ---------------------------------------------------------------------------

def _make_polar_mesh(
    radial_coords: np.ndarray,
    n_phi: int,
    pos_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a triangle mesh on a polar grid.

    Args:
        radial_coords: (n_rings,) array of radial parameter values for each
            ring (ring 0 is the innermost).  The pole / centre is added
            automatically at radial parameter 0.
        n_phi: number of angular subdivisions.
        pos_fn: callable(r, phi) -> (x, y, z) mapping radial and angular
            parameters to 3-D positions.  Must broadcast over arrays.

    Returns:
        vertices: (V, 3)
        faces:    (F, 3)
    """
    n_rings = len(radial_coords)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    # Vertices: pole + n_rings * n_phi ring vertices
    verts = np.empty((1 + n_rings * n_phi, 3))
    verts[0] = pos_fn(0.0, 0.0)

    for ri, r in enumerate(radial_coords):
        for ki, phi in enumerate(phi_vals):
            verts[1 + ri * n_phi + ki] = pos_fn(r, phi)

    def ring_idx(ri, ki):
        return 1 + ri * n_phi + (ki % n_phi)

    faces = []
    # Pole fan → first ring
    for k in range(n_phi):
        faces.append([0, ring_idx(0, k), ring_idx(0, k + 1)])

    # Ring strips
    for ri in range(n_rings - 1):
        for k in range(n_phi):
            a = ring_idx(ri, k)
            b = ring_idx(ri, k + 1)
            c = ring_idx(ri + 1, k)
            d = ring_idx(ri + 1, k + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])

    return verts, np.array(faces, dtype=int)


def make_hemisphere_mesh(
    n_theta: int = 15,
    n_phi: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Upper unit hemisphere, polar grid.  Equator at z = 0."""
    theta_vals = np.linspace(0, np.pi / 2, n_theta + 1)[1:]  # skip pole

    def pos_fn(r, phi):
        # r = theta here
        return np.array([
            np.sin(r) * np.cos(phi),
            np.sin(r) * np.sin(phi),
            np.cos(r),
        ])

    return _make_polar_mesh(theta_vals, n_phi, pos_fn)


def make_disk_mesh(
    r_outer: float = 2.0,
    n_rho: int = 20,
    n_phi: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat disk at z = 0 from rho = 0 to r_outer.

    The radial spacing is chosen so that rho = 1.0 falls exactly on a
    grid ring (ring index n_rho // 2 when r_outer = 2.0 and n_rho is
    even).  This guarantees that the equatorial vertices of the hemisphere
    match disk vertices for welding.
    """
    rho_vals = np.linspace(0, r_outer, n_rho + 1)[1:]  # skip centre

    def pos_fn(r, phi):
        return np.array([r * np.cos(phi), r * np.sin(phi), 0.0])

    return _make_polar_mesh(rho_vals, n_phi, pos_fn)


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def build_mesh(
    n_theta: int = 15,
    n_rho: int = 20,
    n_phi: int = 40,
    r_outer: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a non-manifold mesh: hemisphere (sheet 0) + flat disk (sheet 1).

    Returns:
        vertices:     (V, 3) merged vertex positions.
        faces:        (F, 3) triangle vertex indices.
        sheet_labels: (F,)   0 = hemisphere, 1 = disk.
    """
    hemi_v, hemi_f = make_hemisphere_mesh(n_theta, n_phi)
    disk_v, disk_f = make_disk_mesh(r_outer, n_rho, n_phi)
    verts, faces, sheet_labels = merge_meshes_weld(
        [(hemi_v, hemi_f), (disk_v, disk_f)], tol=1e-8,
    )
    return verts, faces, sheet_labels


# ---------------------------------------------------------------------------
# Ground-truth curvature
# ---------------------------------------------------------------------------

def ground_truth_curvature(
    centroids: np.ndarray,
    sheet_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-face ground-truth K and H.

    Hemisphere (sheet 0): K = 1, |H| = 1 (unit sphere).
    Disk       (sheet 1): K = 0, |H| = 0.
    """
    F = len(sheet_labels)
    K = np.where(sheet_labels == 0, 1.0, 0.0)
    H = np.where(sheet_labels == 0, 1.0, 0.0)
    return K, H


# ---------------------------------------------------------------------------
# Discrete baseline curvatures (vertex-based)
# ---------------------------------------------------------------------------

def discrete_gaussian_curvature(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Vertex Gaussian curvature via angle defect.  Returns (V,)."""
    V = len(vertices)
    angle_sum = np.zeros(V)
    area_sum = np.zeros(V)
    tri = vertices[faces]
    for k in range(3):
        e1 = tri[:, (k + 1) % 3] - tri[:, k]
        e2 = tri[:, (k + 2) % 3] - tri[:, k]
        cos_a = np.sum(e1 * e2, axis=1) / (
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30
        )
        angles = np.arccos(np.clip(cos_a, -1, 1))
        np.add.at(angle_sum, faces[:, k], angles)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fa = 0.5 * np.linalg.norm(cross, axis=1)
        np.add.at(area_sum, faces[:, k], fa / 3.0)
    area_sum = np.where(area_sum > 1e-30, area_sum, 1e-30)
    return (2.0 * np.pi - angle_sum) / area_sum


def discrete_mean_curvature(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Per-vertex |H| via cotangent Laplacian.  Returns (V,)."""
    V = len(vertices)
    Lx = np.zeros((V, 3))
    area = np.zeros(V)
    tri = vertices[faces]
    for k in range(3):
        i0, i1, i2 = k, (k + 1) % 3, (k + 2) % 3
        e1 = tri[:, i0] - tri[:, i2]
        e2 = tri[:, i1] - tri[:, i2]
        cos_a = np.sum(e1 * e2, axis=1) / (
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30
        )
        cos_a = np.clip(cos_a, -0.999, 0.999)
        sin_a = np.sqrt(1.0 - cos_a ** 2)
        cot_a = cos_a / (sin_a + 1e-30)
        edge = vertices[faces[:, i0]] - vertices[faces[:, i1]]
        contrib = 0.5 * cot_a[:, None] * edge
        np.add.at(Lx, faces[:, i0], -contrib)
        np.add.at(Lx, faces[:, i1], contrib)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fa = 0.5 * np.linalg.norm(cross, axis=1)
        np.add.at(area, faces[:, k], fa / 3.0)
    area = np.where(area > 1e-30, area, 1e-30)
    return np.linalg.norm(Lx, axis=1) / (2.0 * area)


def vertex_to_face(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Average per-vertex values to per-face."""
    return (values[faces[:, 0]] + values[faces[:, 1]] + values[faces[:, 2]]) / 3.0


# ---------------------------------------------------------------------------
# Blow-up curvature estimation
# ---------------------------------------------------------------------------

def blowup_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 20,
    alpha: float = 1.0,
    lam: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-face (K, |H|) via iterated blow-up.  Returns (F,), (F,).

    Notes:
        The blow-up's normal orientation (from ``_complement_frames``) is
        arbitrary, so the sign of the shape operator — and hence of H — is
        not intrinsic.  We return |H| to match the cotangent-Laplacian
        baseline, which also returns |H|.  Gaussian curvature K = κ₁ κ₂ is
        sign-invariant under normal flip (both principal curvatures flip).
    """
    cents = face_centroids(vertices, faces)
    frames = face_tangent_frames(vertices, faces)
    l0 = BlowUpLevel.from_point_tangents(cents, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam)
    inv = extract_level1(l1)
    return inv.gaussian_curvature, np.abs(inv.mean_curvature[:, 0])


# ---------------------------------------------------------------------------
# Distance from junction
# ---------------------------------------------------------------------------

def face_junction_distance(
    centroids: np.ndarray,
    sheet_labels: np.ndarray,
) -> np.ndarray:
    """
    Per-face distance from the equatorial junction.

    Hemisphere faces: distance ~ z-coordinate of centroid (0 at equator,
    1 at pole).  Disk faces: distance ~ |1 - rho| where rho = sqrt(x^2+y^2).
    """
    dist = np.empty(len(centroids))
    hemi = sheet_labels == 0
    disk = ~hemi

    # Hemisphere: arc distance from equator ~ pi/2 - arcsin(z) ... just use z
    dist[hemi] = centroids[hemi, 2]

    # Disk: |rho - 1|
    rho = np.sqrt(centroids[disk, 0] ** 2 + centroids[disk, 1] ** 2)
    dist[disk] = np.abs(rho - 1.0)

    return dist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_mesh(ax, verts, faces, facecolors, **kw):
    coll = Poly3DCollection(verts[faces], linewidth=kw.pop("linewidth", 0.05))
    coll.set_facecolors(facecolors)
    coll.set_edgecolors(kw.pop("edgecolors", "none"))
    ax.add_collection3d(coll)
    lo, hi = verts.min(0), verts.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def _color_from_vals(vals, cmap, vmin, vmax):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(np.clip(vals, vmin, vmax)))
    rgba[:, 3] = 0.9
    return rgba, norm


def plot_curvature_comparison(
    verts, faces, sheet_labels,
    K_gt, H_gt,
    K_bu, H_bu,
    K_disc, H_disc,
):
    """3x2 grid: Gaussian / Mean × ground truth / blow-up / discrete."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10),
                             subplot_kw={"projection": "3d"})
    fig.suptitle(
        r"Non-Manifold Mesh Curvature: Hemisphere ($K\!=\!1$, $|H|\!=\!1$)"
        r" + Flat Disk ($K\!=\!0$, $|H|\!=\!0$)",
        fontsize=13,
    )

    datasets = [
        (K_gt,   "Gaussian (ground truth)"),
        (K_bu,   "Gaussian (blow-up)"),
        (K_disc, "Gaussian (angle defect)"),
        (H_gt,   "|H| (ground truth)"),
        (H_bu,   "|H| (blow-up)"),
        (H_disc, "|H| (cotangent Lap.)"),
    ]

    for idx, (vals, title) in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        lo, hi = np.percentile(vals, [2, 98])
        is_gaussian = "Gaussian" in title
        cmap = plt.cm.RdBu_r if is_gaussian else plt.cm.viridis
        if is_gaussian:
            vmax = max(abs(lo), abs(hi), 0.01)
            vmin = -vmax
        else:
            vmin, vmax = lo, hi
        rgba, norm = _color_from_vals(vals, cmap, vmin, vmax)
        _add_mesh(ax, verts, faces, rgba)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect([1, 1, 0.6])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)

    plt.tight_layout()
    plt.savefig("hemisphere_plane_curvature.pdf", bbox_inches="tight")
    print("Saved hemisphere_plane_curvature.pdf")
    plt.show()


def plot_error_heatmap(
    verts, faces, sheet_labels,
    K_gt, H_gt,
    K_bu, H_bu,
    K_disc, H_disc,
):
    """Per-face absolute error heatmaps on the 3-D mesh."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={"projection": "3d"})
    fig.suptitle(
        "Per-Face Absolute Curvature Error\n"
        "(bright = large error, dark = accurate)",
        fontsize=13,
    )

    datasets = [
        (np.abs(K_bu - K_gt),   "K error: blow-up"),
        (np.abs(K_disc - K_gt), "K error: angle defect"),
        (np.abs(H_bu - H_gt),   "|H| error: blow-up"),
        (np.abs(H_disc - H_gt), "|H| error: cotangent Lap."),
    ]

    # Common scale per row
    k_vmax = max(np.percentile(datasets[0][0], 98),
                 np.percentile(datasets[1][0], 98))
    h_vmax = max(np.percentile(datasets[2][0], 98),
                 np.percentile(datasets[3][0], 98))
    vmaxes = [k_vmax, k_vmax, h_vmax, h_vmax]

    for idx, ((vals, title), vm) in enumerate(zip(datasets, vmaxes)):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        cmap = plt.cm.hot_r
        rgba, norm = _color_from_vals(vals, cmap, 0, vm)
        _add_mesh(ax, verts, faces, rgba)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect([1, 1, 0.6])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)

    plt.tight_layout()
    plt.savefig("hemisphere_plane_error_heatmap.pdf", bbox_inches="tight")
    print("Saved hemisphere_plane_error_heatmap.pdf")
    plt.show()


def plot_error_vs_distance(
    centroids, sheet_labels,
    K_gt, H_gt,
    K_bu, H_bu,
    K_disc, H_disc,
):
    """RMSE vs distance-from-junction profile."""
    dist = face_junction_distance(centroids, sheet_labels)
    n_bins = 15
    bin_edges = np.linspace(0, dist.max(), n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def binned_rmse(error):
        rmse = np.empty(n_bins)
        for b in range(n_bins):
            mask = (dist >= bin_edges[b]) & (dist < bin_edges[b + 1])
            if mask.sum() > 0:
                rmse[b] = np.sqrt(np.mean(error[mask] ** 2))
            else:
                rmse[b] = np.nan
        return rmse

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Curvature RMSE vs Distance from Non-Manifold Junction\n"
        "(lower is better; blow-up maintains accuracy near the junction)",
        fontsize=12,
    )

    for ax, (gt, bu, disc, name) in zip(axes, [
        (K_gt, K_bu, K_disc, "Gaussian curvature"),
        (H_gt, H_bu, H_disc, "Mean curvature |H|"),
    ]):
        rmse_bu = binned_rmse(bu - gt)
        rmse_disc = binned_rmse(disc - gt)
        ax.plot(bin_centres, rmse_bu, "o-", color="#4C72B0", linewidth=2,
                markersize=5, label="Blow-up")
        ax.plot(bin_centres, rmse_disc, "s--", color="#DD8452", linewidth=2,
                markersize=5, label="Discrete baseline")
        ax.axvspan(0, bin_edges[1], alpha=0.15, color="red",
                   label="Junction zone")
        ax.set_xlabel("Distance from junction")
        ax.set_ylabel("RMSE")
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("hemisphere_plane_rmse_profile.pdf", bbox_inches="tight")
    print("Saved hemisphere_plane_rmse_profile.pdf")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    n_theta: int = 15,
    n_rho: int = 20,
    n_phi: int = 40,
    r_outer: float = 2.0,
    k: int = 20,
    alpha: float = 1.0,
):
    # --- Build mesh -------------------------------------------------------
    print("Building non-manifold mesh (hemisphere + disk) ...")
    verts, faces, sheet_labels = build_mesh(n_theta, n_rho, n_phi, r_outer)
    F = len(faces)
    n_hemi = np.sum(sheet_labels == 0)
    n_disk = F - n_hemi
    print(f"  {len(verts)} vertices, {F} faces "
          f"({n_hemi} hemisphere + {n_disk} disk)")

    cents = face_centroids(verts, faces)

    # --- Ground truth -----------------------------------------------------
    K_gt, H_gt = ground_truth_curvature(cents, sheet_labels)

    # --- Blow-up ----------------------------------------------------------
    print(f"Blow-up curvature estimation (k={k}, alpha={alpha}) ...")
    K_bu, H_bu = blowup_curvatures(verts, faces, k=k, alpha=alpha)

    # --- Discrete baseline ------------------------------------------------
    print("Discrete baseline curvature estimation ...")
    K_vert = discrete_gaussian_curvature(verts, faces)
    H_vert = discrete_mean_curvature(verts, faces)
    K_disc = vertex_to_face(K_vert, faces)
    H_disc = vertex_to_face(H_vert, faces)

    # --- Summary ----------------------------------------------------------
    dist = face_junction_distance(cents, sheet_labels)
    near = dist < 0.15  # faces near the junction
    far = dist > 0.5    # faces far from the junction

    def _report(name, gt, est, mask, region):
        err = np.abs(est[mask] - gt[mask])
        print(f"  {name} ({region}, {mask.sum()} faces): "
              f"MAE = {err.mean():.4f},  max = {err.max():.4f}")

    print("\nGaussian curvature:")
    _report("Blow-up ", K_gt, K_bu,   near, "near junction")
    _report("Discrete", K_gt, K_disc, near, "near junction")
    _report("Blow-up ", K_gt, K_bu,   far,  "far from junction")
    _report("Discrete", K_gt, K_disc, far,  "far from junction")

    print("\nMean curvature |H|:")
    _report("Blow-up ", H_gt, H_bu,   near, "near junction")
    _report("Discrete", H_gt, H_disc, near, "near junction")
    _report("Blow-up ", H_gt, H_bu,   far,  "far from junction")
    _report("Discrete", H_gt, H_disc, far,  "far from junction")

    # --- Plots ------------------------------------------------------------
    plot_curvature_comparison(
        verts, faces, sheet_labels,
        K_gt, H_gt, K_bu, H_bu, K_disc, H_disc,
    )
    plot_error_heatmap(
        verts, faces, sheet_labels,
        K_gt, H_gt, K_bu, H_bu, K_disc, H_disc,
    )
    plot_error_vs_distance(
        cents, sheet_labels,
        K_gt, H_gt, K_bu, H_bu, K_disc, H_disc,
    )


if __name__ == "__main__":
    main()
