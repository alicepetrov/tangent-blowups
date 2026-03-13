"""
Lifted Dual Graph: Two Intersecting Planes
==========================================
Demonstrates the discrete blow-up construction for non-manifold triangle meshes.

Setup
-----
Two flat planes meeting at right angles along the x-axis:
  - Sheet 0  (z = 0, the XY-plane)   normal n0 = (0, 0, 1)
  - Sheet 1  (y = 0, the XZ-plane)   normal n1 = (0, 1, 0)

The dihedral angle at the intersection is delta = 90 deg, giving

    ||P0 - P1||_F = sqrt(2 * sin^2(90)) = sqrt(2)

and a projector contribution to the lifted distance of

    sqrt(alpha / 2) * sqrt(2) = sqrt(alpha).

All same-sheet adjacent faces are co-planar (theta_max = 0), so their
lifted distance equals only the centroid-centroid spatial distance ~ h/sqrt(2)
(where h = 1/N is the triangle diameter).  The separation gap is therefore

    (h / sqrt(2),  sqrt(alpha + h^2 / 4))   ~  (h, sqrt(alpha))   for small h.

Figures
-------
  1. Mesh coloured by ground-truth sheet (before blowup).
  2. Lifted-distance histogram with the auto-selected threshold tau.
  3. Mesh coloured by recovered connected components (after blowup).
  4. Singular-locus faces (incident to severed dual edges) highlighted.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tangent_blowups.mesh.blowup import MeshBlowUp, make_plane_grid, merge_meshes_weld


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def build_mesh(n: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a merged triangle mesh of two perpendicular planes.

    Sheet 0: z = 0, u in [0, 1], v in [0, 1].
    Sheet 1: y = 0, u in [0, 1], v in [0, 1].

    Vertices on the intersection line (y = z = 0) are welded so that the
    two sheets share topological edges there, making those edges non-manifold
    (shared by one face from each sheet).

    Returns:
        vertices:     (V, 3) merged vertex positions.
        faces:        (F, 3) triangle vertex indices.
        sheet_labels: (F,)   ground-truth sheet index per face (0 or 1).
    """
    mesh0 = make_plane_grid(n, normal_axis=2)   # z=0 plane
    mesh1 = make_plane_grid(n, normal_axis=1)   # y=0 plane
    verts, faces, sheet_labels = merge_meshes_weld([mesh0, mesh1])
    return verts, faces, sheet_labels


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SHEET_COLORS  = ["#4C72B0", "#DD8452"]   # blue, orange
COMP_COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                 "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
SINGULAR_COLOR = "#E63946"


def _face_rgba(
    labels: np.ndarray,
    colors: list[str],
    alpha_val: float = 0.85,
) -> np.ndarray:
    """Map integer face labels to RGBA colours."""
    clist = [mcolors.to_rgba(c) for c in colors]
    rgba = np.array([clist[int(l) % len(clist)] for l in labels])
    rgba[:, 3] = alpha_val
    return rgba


def _add_mesh(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    facecolors: np.ndarray,
    linewidth: float = 0.05,
    edgecolors: str = "k",
) -> None:
    """Add a coloured Poly3DCollection to a 3-D axis."""
    tris = vertices[faces]   # (F, 3, 3)
    coll = Poly3DCollection(tris, linewidth=linewidth)
    coll.set_facecolors(facecolors)
    coll.set_edgecolors(edgecolors)
    ax.add_collection3d(coll)
    # Fit axis limits
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def _style_ax3d(ax, title: str, elev: float = 25, azim: float = -55) -> None:
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n: int = 20, alpha: float = 1.0) -> None:
    # ------------------------------------------------------------------
    # Build mesh and run lifted dual graph
    # ------------------------------------------------------------------
    vertices, faces, sheet_labels = build_mesh(n)
    F = len(faces)
    print(f"Mesh: {len(vertices)} vertices, {F} faces  (2 sheets x {F//2} faces each)")

    bu = MeshBlowUp.from_mesh(vertices, faces, alpha=alpha)
    print(f"Lifted dual graph: {len(bu.dual_edges)} dual edges")
    print(f"Auto-selected tau = {bu.tau:.4f}")
    print(f"Recovered {bu.n_components} component(s)")

    max_kept, min_sev = bu.distance_gap()
    print(f"Distance gap: max kept = {max_kept:.4f}, min severed = {min_sev:.4f}")
    print(f"Singular faces: {bu.singular_face_mask.sum()} of {F}")

    # Verify recovery matches ground truth (up to label permutation)
    gt_same = np.all(
        (bu.labels == bu.labels[0]) == (sheet_labels == sheet_labels[0])
    )
    print(f"Component labels match ground truth: {gt_same}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Lifted Dual Graph: Two Intersecting Planes\n"
        r"$n\!=$" + f"{n},  " + r"$\alpha\!=$" + f"{alpha},  "
        r"$\tau\!=\!$" + f"{bu.tau:.3f}",
        fontsize=13,
    )

    # ---- Panel 1: ground truth ----------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    rgba_gt = _face_rgba(sheet_labels, SHEET_COLORS)
    _add_mesh(ax1, vertices, faces, rgba_gt)
    _style_ax3d(ax1, "Ground truth\n(sheet 0 = blue, sheet 1 = orange)")

    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(color=SHEET_COLORS[0], label="Sheet 0  (z = 0)"),
        Patch(color=SHEET_COLORS[1], label="Sheet 1  (y = 0)"),
    ], loc="upper left", fontsize=8)

    # ---- Panel 2: lifted-distance histogram ---------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    dists = bu.lifted_dists
    ax2.hist(dists, bins=60, color="#888888", edgecolor="white", linewidth=0.3)
    ax2.axvline(bu.tau, color="crimson", linestyle="--", linewidth=1.5,
                label=rf"$\tau = {bu.tau:.3f}$")
    ax2.set_xlabel(r"$\|\Phi_f - \Phi_g\|$", fontsize=11)
    ax2.set_ylabel("Edge count")
    ax2.set_title("Lifted-distance distribution\n(dual edges)")
    ax2.legend(fontsize=9)

    # Annotate the two clusters
    same_d  = dists[~bu.severed]
    cross_d = dists[bu.severed]
    ylim_top = ax2.get_ylim()[1]
    if len(same_d) > 0:
        ax2.annotate(
            f"same-sheet\n({len(same_d)} edges)",
            xy=(same_d.mean(), ylim_top * 0.60),
            ha="center", fontsize=8, color=SHEET_COLORS[0],
        )
    if len(cross_d) > 0:
        ax2.annotate(
            f"cross-sheet\n({len(cross_d)} edges)",
            xy=(cross_d.mean(), ylim_top * 0.60),
            ha="center", fontsize=8, color=SHEET_COLORS[1],
        )

    # ---- Panel 3: recovered components --------------------------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    rgba_comp = _face_rgba(bu.labels, COMP_COLORS)
    _add_mesh(ax3, vertices, faces, rgba_comp)
    _style_ax3d(ax3, f"Recovered components  ({bu.n_components} found)\n"
                f"via lifted dual graph  (alpha={alpha})")

    ax3.legend(handles=[
        Patch(color=COMP_COLORS[k], label=f"Component {k}")
        for k in range(bu.n_components)
    ], loc="upper left", fontsize=8)

    # ---- Panel 4: singular locus --------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    sing_mask = bu.singular_face_mask
    colors_sing = np.where(
        sing_mask[:, None],
        np.array(mcolors.to_rgba(SINGULAR_COLOR)),
        np.array(mcolors.to_rgba("#AAAAAA")),
    )
    colors_sing[:, 3] = np.where(sing_mask, 0.95, 0.30)
    _add_mesh(ax4, vertices, faces, colors_sing)
    _style_ax3d(ax4, f"Singular-locus faces\n"
                f"({sing_mask.sum()} faces incident to severed dual edges)")

    ax4.legend(handles=[
        Patch(color=SINGULAR_COLOR, label="Singular (severed)"),
        Patch(color="#AAAAAA",      label="Regular",    alpha=0.4),
    ], loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("mesh_blowup_two_planes.pdf", bbox_inches="tight")
    print("Saved mesh_blowup_two_planes.pdf")
    plt.show()

    # ------------------------------------------------------------------
    # Second figure: threshold sweep
    # ------------------------------------------------------------------
    _threshold_sweep(bu, vertices, faces, n, alpha)


def _threshold_sweep(
    bu: MeshBlowUp,
    vertices: np.ndarray,
    faces: np.ndarray,
    n: int,
    alpha: float,
) -> None:
    """
    Show how component count and singular-face count vary with tau.
    Also overlay the lifted-distance distribution.
    """
    taus = np.linspace(0.0, bu.lifted_dists.max() * 1.05, 300)
    n_comp_arr     = []
    n_singular_arr = []

    for t in taus:
        b = bu.rethreshold(float(t))
        n_comp_arr.append(b.n_components)
        n_singular_arr.append(int(b.singular_face_mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Threshold sweep  (n={n}, alpha={alpha})\n"
        "Correct separation: 2 components, singular = x-axis faces",
        fontsize=12,
    )

    ax = axes[0]
    ax.plot(taus, n_comp_arr, color="#4C72B0", linewidth=1.5, label="# components")
    ax.axvline(bu.tau, color="crimson", linestyle="--", linewidth=1.2,
               label=rf"auto $\tau = {bu.tau:.3f}$")
    ax.axhline(2, color="#55A868", linestyle=":", linewidth=1.0, label="target (2)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("# connected components")
    ax.set_title("Components vs threshold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax2r = ax.twinx()
    ax.hist(bu.lifted_dists, bins=60, color="#CCCCCC", label="lift dist (edges)",
            alpha=0.7, density=True)
    ax2r.plot(taus, n_singular_arr, color="#DD8452", linewidth=1.5,
              label="# singular faces")
    ax.axvline(bu.tau, color="crimson", linestyle="--", linewidth=1.2,
               label=rf"auto $\tau$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Density (lift dist)")
    ax2r.set_ylabel("# singular faces", color="#DD8452")
    ax.set_title("Lifted-dist density & singular-face count")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2r.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=9)

    plt.tight_layout()
    plt.savefig("mesh_blowup_tau_sweep.pdf", bbox_inches="tight")
    print("Saved mesh_blowup_tau_sweep.pdf")
    plt.show()


if __name__ == "__main__":
    main()
