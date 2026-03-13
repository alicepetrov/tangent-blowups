"""
Curvature Estimation on a Klein Bottle via Iterated Blow-Up
===========================================================
Compares curvature estimated by the iterated tangent blow-up against a
discrete baseline (angle-defect Gaussian curvature + cotangent-Laplacian
mean curvature) on a Klein-bottle mesh immersed in R^3.

Pipeline
--------
1. Mesh the Klein bottle parametrisation on a periodic (u, v) grid.
2. Compute analytic curvatures from the parametric first/second fundamental
   forms as ground truth.
3. Estimate curvature via iterated blow-up:
       level 0  (centroids, tangent frames)
       level 1  = level0.lift()   -->  shape operator  -->  principal / Gaussian / mean
4. Compute discrete baseline curvatures:
       - Gaussian curvature via angle defect (2 pi - sum of angles at vertex)
       - Mean curvature via the norm of the discrete Laplace-Beltrami vector
5. Plot all three (analytic, blow-up, discrete baseline) side-by-side.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tangent_blowups.geometry.iterated_grassmann import (
    BlowUpLevel,
    extract_level1,
)
from tangent_blowups.mesh.blowup import face_normals, face_centroids


# ---------------------------------------------------------------------------
# Klein bottle parametrisation and analytic curvature
# ---------------------------------------------------------------------------

RADIUS = 2.0
SCALE = 1.0


def klein_position(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Klein bottle immersion (u, v) -> R^3.  Shape (..., 3)."""
    cu, su = np.cos(u), np.sin(u)
    cuh, suh = np.cos(0.5 * u), np.sin(0.5 * u)
    sv, s2v = np.sin(v), np.sin(2.0 * v)
    a = RADIUS + cuh * sv - suh * s2v
    x = SCALE * a * cu
    y = SCALE * a * su
    z = SCALE * (suh * sv + cuh * s2v)
    return np.stack([x, y, z], axis=-1)


def klein_partials(u: np.ndarray, v: np.ndarray):
    """First partial derivatives dp/du, dp/dv.  Each (..., 3)."""
    cu, su = np.cos(u), np.sin(u)
    cuh, suh = np.cos(0.5 * u), np.sin(0.5 * u)
    sv, cv = np.sin(v), np.cos(v)
    s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)

    a = RADIUS + cuh * sv - suh * s2v
    dadu = -0.5 * (suh * sv + cuh * s2v)
    dadv = cuh * cv - 2.0 * suh * c2v

    dpdu = SCALE * np.stack([
        dadu * cu - a * su,
        dadu * su + a * cu,
        0.5 * (cuh * sv - suh * s2v),
    ], axis=-1)
    dpdv = SCALE * np.stack([
        dadv * cu,
        dadv * su,
        suh * cv + 2.0 * cuh * c2v,
    ], axis=-1)
    return dpdu, dpdv


def klein_second_partials(u: np.ndarray, v: np.ndarray):
    """Second partial derivatives d²p/du², d²p/dudv, d²p/dv².  Each (..., 3)."""
    cu, su = np.cos(u), np.sin(u)
    cuh, suh = np.cos(0.5 * u), np.sin(0.5 * u)
    sv, cv = np.sin(v), np.cos(v)
    s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)

    a = RADIUS + cuh * sv - suh * s2v
    dadu = -0.5 * (suh * sv + cuh * s2v)
    dadv = cuh * cv - 2.0 * suh * c2v
    daduu = -0.25 * (cuh * sv - suh * s2v)
    daduv = -0.5 * (suh * cv + 2.0 * cuh * c2v)
    dadvv = -cuh * sv + 4.0 * suh * s2v

    # d²p/du²
    dpuu = SCALE * np.stack([
        daduu * cu - 2.0 * dadu * su - a * cu,
        daduu * su + 2.0 * dadu * cu - a * su,
        0.25 * (-suh * sv - cuh * s2v),
    ], axis=-1)

    # d²p/dudv
    dpuv = SCALE * np.stack([
        daduv * cu - dadv * su,
        daduv * su + dadv * cu,
        0.5 * (cuh * cv - 2.0 * suh * c2v),
    ], axis=-1)

    # d²p/dv²
    dpvv = SCALE * np.stack([
        dadvv * cu,
        dadvv * su,
        -suh * sv - 4.0 * cuh * s2v,
    ], axis=-1)

    return dpuu, dpuv, dpvv


def analytic_curvatures(u: np.ndarray, v: np.ndarray):
    """
    Compute Gaussian and mean curvature from the parametric fundamental forms.

    Returns (K, H) each of shape u.shape.
    """
    dpdu, dpdv = klein_partials(u, v)
    dpuu, dpuv, dpvv = klein_second_partials(u, v)

    # First fundamental form
    E = np.sum(dpdu * dpdu, axis=-1)
    F = np.sum(dpdu * dpdv, axis=-1)
    G = np.sum(dpdv * dpdv, axis=-1)

    # Unit normal
    n_raw = np.cross(dpdu, dpdv)
    n_norm = np.linalg.norm(n_raw, axis=-1, keepdims=True)
    n_norm = np.where(n_norm > 1e-15, n_norm, 1.0)
    n_hat = n_raw / n_norm

    # Second fundamental form
    L = np.sum(dpuu * n_hat, axis=-1)
    M = np.sum(dpuv * n_hat, axis=-1)
    N_ = np.sum(dpvv * n_hat, axis=-1)

    det_I = E * G - F * F
    det_I = np.where(np.abs(det_I) > 1e-30, det_I, 1e-30)

    K = (L * N_ - M * M) / det_I
    H = (E * N_ - 2.0 * F * M + G * L) / (2.0 * det_I)
    return K, H


# ---------------------------------------------------------------------------
# Mesh the Klein bottle on a periodic grid
# ---------------------------------------------------------------------------

def make_klein_mesh(
    nu: int = 60,
    nv: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a triangle mesh for the Klein bottle.

    The v direction is simply periodic (v ~ v + 2*pi).  The u direction
    has a twist: (u=0, v) identifies with (u=2*pi, 2*pi - v).  This is
    what makes the surface non-orientable.

    Returns:
        vertices: (V, 3)
        faces:    (F, 3)
        u_verts:  (V,) parameter u at each vertex
        v_verts:  (V,) parameter v at each vertex
    """
    u_vals = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u_vals, v_vals, indexing="ij")  # (nu, nv)

    pts = klein_position(U, V)  # (nu, nv, 3)
    verts = pts.reshape(-1, 3)
    u_flat = U.ravel()
    v_flat = V.ravel()

    def idx(i, j):
        """Vertex index with Klein bottle identification."""
        j = j % nv
        if i < nu:
            return i * nv + j
        # u wraps with v -> (nv - j) % nv  (the Klein twist)
        return 0 * nv + (nv - j) % nv

    faces = []
    for i in range(nu):
        for j in range(nv):
            a = idx(i,     j)
            b = idx(i + 1, j)
            c = idx(i,     j + 1)
            d = idx(i + 1, j + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])

    return verts, np.array(faces, dtype=int), u_flat, v_flat


# ---------------------------------------------------------------------------
# Discrete baseline curvatures
# ---------------------------------------------------------------------------

def discrete_gaussian_curvature(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """
    Vertex Gaussian curvature via angle defect: K_i = (2*pi - sum_angles) / A_i.

    Returns (V,) per-vertex Gaussian curvature.
    """
    V = len(vertices)
    angle_sum = np.zeros(V)
    area_sum = np.zeros(V)

    v = vertices[faces]  # (F, 3, 3)
    for k in range(3):
        e1 = v[:, (k + 1) % 3] - v[:, k]
        e2 = v[:, (k + 2) % 3] - v[:, k]
        cos_a = np.sum(e1 * e2, axis=1) / (
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30
        )
        cos_a = np.clip(cos_a, -1, 1)
        angles = np.arccos(cos_a)
        np.add.at(angle_sum, faces[:, k], angles)

        # Barycentric area: 1/3 of face area per vertex
        cross = np.cross(
            v[:, 1] - v[:, 0],
            v[:, 2] - v[:, 0],
        )
        face_area = 0.5 * np.linalg.norm(cross, axis=1)
        np.add.at(area_sum, faces[:, k], face_area / 3.0)

    area_sum = np.where(area_sum > 1e-30, area_sum, 1e-30)
    return (2.0 * np.pi - angle_sum) / area_sum


def discrete_mean_curvature(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """
    Per-vertex mean curvature magnitude via the cotangent Laplacian:
      H_i = |L(x_i)| / (2 * A_i)

    Returns (V,) per-vertex |H|.
    """
    V = len(vertices)
    Lx = np.zeros((V, 3))  # Laplacian of position
    area = np.zeros(V)

    v = vertices[faces]  # (F, 3, 3)
    for k in range(3):
        i0, i1, i2 = k, (k + 1) % 3, (k + 2) % 3
        e1 = v[:, i0] - v[:, i2]
        e2 = v[:, i1] - v[:, i2]
        cos_a = np.sum(e1 * e2, axis=1) / (
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30
        )
        cos_a = np.clip(cos_a, -0.999, 0.999)
        sin_a = np.sqrt(1.0 - cos_a ** 2)
        cot_a = cos_a / (sin_a + 1e-30)

        edge = vertices[faces[:, i0]] - vertices[faces[:, i1]]  # (F, 3)
        contrib = 0.5 * cot_a[:, None] * edge
        np.add.at(Lx, faces[:, i0], -contrib)
        np.add.at(Lx, faces[:, i1], contrib)

        # Mixed-Voronoi area
        cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        face_area = 0.5 * np.linalg.norm(cross, axis=1)
        np.add.at(area, faces[:, k], face_area / 3.0)

    area = np.where(area > 1e-30, area, 1e-30)
    return np.linalg.norm(Lx, axis=1) / (2.0 * area)


def vertex_to_face(vertex_values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Average per-vertex values to per-face (barycentric mean)."""
    return (vertex_values[faces[:, 0]]
            + vertex_values[faces[:, 1]]
            + vertex_values[faces[:, 2]]) / 3.0


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
):
    """
    Estimate per-face Gaussian and mean curvature via iterated tangent blow-up.

    Returns (K_blowup, H_blowup) each of shape (F,).
    """
    cents = face_centroids(vertices, faces)       # (F, 3)
    norms = face_normals(vertices, faces)         # (F, 3)

    # Tangent frames from face normals: two orthonormal tangent vectors per face
    from tangent_blowups.mesh.blowup import face_tangent_frames
    frames = face_tangent_frames(vertices, faces)  # (F, 3, 2)

    # Level 0 → Level 1
    l0 = BlowUpLevel.from_point_tangents(cents, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam)

    # Extract curvatures
    inv1 = extract_level1(l1)

    # For a surface in R^3 with d=2 tangent directions, n_comp=1 normal:
    #   principal_curvatures: (F, 2)
    #   gaussian_curvature:   (F,)
    #   mean_curvature:       (F, 1)
    K = inv1.gaussian_curvature     # (F,)
    H = inv1.mean_curvature[:, 0]   # (F,)
    return K, H


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _clamp_percentile(vals, pct=2):
    """Clamp extreme values for robust colourmap display."""
    lo, hi = np.percentile(vals, [pct, 100 - pct])
    return np.clip(vals, lo, hi), lo, hi


def plot_results(
    vertices, faces, u_face, v_face,
    K_analytic, H_analytic,
    K_blowup, H_blowup,
    K_discrete, H_discrete,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), subplot_kw={"projection": "3d"})
    fig.suptitle(
        "Klein Bottle Curvature: Analytic vs Blow-Up vs Discrete Baseline",
        fontsize=14,
    )

    datasets = [
        (K_analytic,  "Gaussian (analytic)"),
        (K_blowup,   "Gaussian (blow-up)"),
        (K_discrete,  "Gaussian (angle defect)"),
        (H_analytic,  "Mean (analytic)"),
        (H_blowup,   "Mean (blow-up)"),
        (H_discrete,  "Mean (cotangent Laplacian)"),
    ]

    for idx, (vals, title) in enumerate(datasets):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        clamped, lo, hi = _clamp_percentile(vals, pct=2)

        # Use symmetric colormap for Gaussian, sequential for mean
        if "Gaussian" in title:
            vmax = max(abs(lo), abs(hi))
            norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            norm = mcolors.Normalize(vmin=lo, vmax=hi)
            cmap = plt.cm.viridis

        # Map values to colours
        rgba = cmap(norm(clamped))
        rgba[:, 3] = 0.9

        tris = vertices[faces]
        coll = Poly3DCollection(tris, linewidth=0.05)
        coll.set_facecolors(rgba)
        coll.set_edgecolors("none")
        ax.add_collection3d(coll)

        lo_v, hi_v = vertices.min(axis=0), vertices.max(axis=0)
        ax.set_xlim(lo_v[0], hi_v[0])
        ax.set_ylim(lo_v[1], hi_v[1])
        ax.set_zlim(lo_v[2], hi_v[2])
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect([1, 1, 0.6])

        # Colourbar via a ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)

    plt.tight_layout()
    plt.savefig("klein_bottle_curvature.pdf", bbox_inches="tight")
    print("Saved klein_bottle_curvature.pdf")
    plt.show()


def plot_correlation(
    K_analytic, H_analytic,
    K_blowup, H_blowup,
    K_discrete, H_discrete,
):
    """Scatter plots comparing blow-up and discrete estimates to analytic truth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Curvature Estimation Accuracy", fontsize=13)

    def scatter(ax, truth, estimate, xlabel, ylabel, title):
        ax.scatter(truth, estimate, s=2, alpha=0.3, rasterized=True)
        lo = min(truth.min(), estimate.min())
        hi = max(truth.max(), estimate.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y = x")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        # Report correlation
        mask = np.isfinite(truth) & np.isfinite(estimate)
        if mask.sum() > 2:
            r = np.corrcoef(truth[mask], estimate[mask])[0, 1]
            ax.text(
                0.05, 0.92, f"r = {r:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )

    Kac, Kbc, Kdc = [np.clip(x, *np.percentile(x, [2, 98])) for x in [K_analytic, K_blowup, K_discrete]]
    Hac, Hbc, Hdc = [np.clip(x, *np.percentile(x, [2, 98])) for x in [H_analytic, H_blowup, H_discrete]]

    scatter(axes[0, 0], Kac, Kbc, "K analytic", "K blow-up",
            "Gaussian curvature: blow-up vs analytic")
    scatter(axes[0, 1], Kac, Kdc, "K analytic", "K discrete",
            "Gaussian curvature: discrete vs analytic")
    scatter(axes[1, 0], Hac, Hbc, "H analytic", "H blow-up",
            "Mean curvature: blow-up vs analytic")
    scatter(axes[1, 1], Hac, Hdc, "H analytic", "H discrete",
            "Mean curvature: discrete vs analytic")

    plt.tight_layout()
    plt.savefig("klein_bottle_curvature_correlation.pdf", bbox_inches="tight")
    print("Saved klein_bottle_curvature_correlation.pdf")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(nu: int = 60, nv: int = 30, k: int = 20, alpha: float = 1.0):
    print(f"Meshing Klein bottle: {nu} x {nv} grid ...")
    vertices, faces, u_verts, v_verts = make_klein_mesh(nu, nv)
    F = len(faces)
    print(f"  {len(vertices)} vertices, {F} faces")

    # Face-centre parameters for analytic curvature
    u_face = (u_verts[faces[:, 0]] + u_verts[faces[:, 1]] + u_verts[faces[:, 2]]) / 3.0
    v_face = (v_verts[faces[:, 0]] + v_verts[faces[:, 1]] + v_verts[faces[:, 2]]) / 3.0

    # --- Analytic curvatures at face centres ---
    print("Computing analytic curvatures ...")
    K_analytic, H_analytic = analytic_curvatures(u_face, v_face)

    # --- Blow-up curvatures ---
    print(f"Computing blow-up curvatures (k={k}, alpha={alpha}) ...")
    K_blowup, H_blowup = blowup_curvatures(vertices, faces, k=k, alpha=alpha)

    # --- Discrete baseline curvatures (per-vertex, then averaged to faces) ---
    print("Computing discrete baseline curvatures ...")
    K_vert = discrete_gaussian_curvature(vertices, faces)
    H_vert = discrete_mean_curvature(vertices, faces)
    K_discrete = vertex_to_face(K_vert, faces)
    H_discrete = vertex_to_face(H_vert, faces)

    # --- Summary statistics ---
    def _stats(name, truth, est):
        mask = np.isfinite(truth) & np.isfinite(est)
        if mask.sum() < 3:
            print(f"  {name}: insufficient data")
            return
        r = np.corrcoef(truth[mask], est[mask])[0, 1]
        rmse = np.sqrt(np.mean((truth[mask] - est[mask]) ** 2))
        print(f"  {name}: corr = {r:.4f},  RMSE = {rmse:.4f}")

    print("\nAccuracy vs analytic ground truth:")
    _stats("K blow-up  ", K_analytic, K_blowup)
    _stats("K discrete ", K_analytic, K_discrete)
    _stats("H blow-up  ", H_analytic, H_blowup)
    _stats("H discrete ", H_analytic, H_discrete)

    # --- Plots ---
    plot_results(
        vertices, faces, u_face, v_face,
        K_analytic, H_analytic,
        K_blowup, H_blowup,
        K_discrete, H_discrete,
    )
    plot_correlation(
        K_analytic, H_analytic,
        K_blowup, H_blowup,
        K_discrete, H_discrete,
    )


if __name__ == "__main__":
    main()
