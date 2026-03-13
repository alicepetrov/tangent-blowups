"""
Iterated Mesh Blow-Up: Tangentially Intersecting Sheets
========================================================
Demonstrates that a *single* lift (level 1) may fail to separate two sheets
that meet at a small dihedral angle, while a *two-step* iterated lift
(level 2) recovers correct separation by exploiting curvature information.

Setup
-----
Two sheets share boundary edges along x = 0 and x = L:

  Sheet 0  (flat):   z(x, y) = 0
  Sheet 1  (arch):   z(x, y) = A * x * (L - x)

The dihedral angle at the shared edges is

    delta = arctan(A * L)

chosen small (A = 0.02, L = 2 -> delta ~ 2.3 deg) so that the
projector contribution at the junction (~sqrt(alpha)*sin(delta) ~ 0.04)
is below the same-sheet intra-mesh spatial variation (~h/sqrt(2) ~ 0.06
for n=25), causing level-1 to fragment into many spurious components.

At level 2 the curvature contrast between sheet 0 (kappa = 0) and sheet 1
(kappa = 2A = 0.04) is encoded in the lifted frame U^(1), amplifying the
cross-sheet lifted distance to ~0.4 while keeping same-sheet distances
below 0.3 — correctly recovering 2 components.

Construction
------------
Each face f is treated as a point-cloud sample:
  - position:      x_f  = face centroid  in R^3
  - tangent frame: U_f  in R^{3 x 2}   (two edge vectors, QR-orthonormalised)

The standard BlowUpLevel machinery then applies:
  level 0:  (x_f, U_f)           -> embedded in R^3
  level 1:  (Phi_f^(1), U_f^(1)) -> embedded in R^{3+9=12}  (+ curvature est.)
  level 2:  (Phi_f^(2), U_f^(2)) -> embedded in R^{12+144=156}

At each level the lifted dual graph is obtained by restricting the
combinatorial adjacency to edges with lifted distance < tau_{auto}.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from tangent_blowups.mesh.blowup import (
    face_centroids,
    face_tangent_frames,
    build_dual_adjacency,
    dual_lifted_distances,
    merge_meshes_weld,
    _auto_tau,
    _compute_components,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def make_flat_plane(n: int, L: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat plane z=0, x in [0,L], y in [0,1]."""
    NV = n + 1
    xs = np.linspace(0.0, L, NV)
    ys = np.linspace(0.0, 1.0, NV)
    X, Y = np.meshgrid(xs, ys, indexing="ij")   # (NV, NV)
    Z    = np.zeros_like(X)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

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


def make_arch(
    n: int,
    L: float = 2.0,
    A: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Parabolic arch z = A*x*(L-x), x in [0,L], y in [0,1]."""
    NV = n + 1
    xs = np.linspace(0.0, L, NV)
    ys = np.linspace(0.0, 1.0, NV)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z    = A * X * (L - X)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

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


# ---------------------------------------------------------------------------
# Lifted dual-graph separation at an arbitrary level
# ---------------------------------------------------------------------------

def classify_at_level(
    phi: np.ndarray,          # (F, D) embedding at some level
    dual_edges: np.ndarray,   # (E, 2)
    n_faces: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute lifted distances, auto-select tau, and return component labels.

    Returns:
        dists:   (E,) lifted distances.
        tau:     auto-selected threshold.
        labels:  (F,) component labels.
        severed: (E,) bool.
    """
    dists = dual_lifted_distances(dual_edges, phi)
    tau   = _auto_tau(dists)
    labels, severed = _compute_components(n_faces, dual_edges, dists, tau)
    return dists, tau, labels, severed


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SHEET_COLORS = ["#4C72B0", "#DD8452"]
COMP_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def _face_rgba(labels: np.ndarray, colors: list[str], a: float = 0.85) -> np.ndarray:
    clist = [mcolors.to_rgba(c) for c in colors]
    rgba  = np.array([clist[int(l) % len(clist)] for l in labels])
    rgba[:, 3] = a
    return rgba


def _add_mesh(ax, vertices, faces, facecolors, lw=0.05):
    tris = vertices[faces]
    coll = Poly3DCollection(tris, linewidth=lw)
    coll.set_facecolors(facecolors)
    coll.set_edgecolors("k")
    ax.add_collection3d(coll)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2] - 0.05, hi[2] + 0.05)


def _style3d(ax, title: str) -> None:
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect([2, 1, 0.5])


def _hist(ax, dists, tau, label, color, severed):
    def _safe_hist(ax, data, **kwargs):
        if len(data) == 0:
            return
        lo, hi = data.min(), data.max()
        eps = max(1e-9, (hi - lo) * 1e-3)
        hist_range = (lo - eps, hi + eps)
        bins = min(40, max(1, len(np.unique(data.round(decimals=8)))))
        ax.hist(data, bins=bins, range=hist_range, **kwargs)

    _safe_hist(ax, dists[~severed], color=SHEET_COLORS[0],
               label="same-sheet",  alpha=0.75, density=False)
    _safe_hist(ax, dists[severed],  color=SHEET_COLORS[1],
               label="cross-sheet", alpha=0.75, density=False)
    ax.axvline(tau, color="crimson", linestyle="--", linewidth=1.4,
               label=rf"$\tau={tau:.3f}$")
    ax.set_xlabel(r"$\|\Phi_f^{(" + label + r")} - \Phi_g^{(" + label + r")}\|$")
    ax.set_ylabel("edge count")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n: int = 25, A: float = 0.02, L: float = 2.0, alpha: float = 1.0) -> None:
    delta_deg = np.degrees(np.arctan(A * L))
    print(f"Arch amplitude A={A}, L={L}")
    print(f"Dihedral angle at shared edges: delta ~ {delta_deg:.1f} deg")

    # ------------------------------------------------------------------
    # Build merged mesh
    # ------------------------------------------------------------------
    mesh0 = make_flat_plane(n, L)
    mesh1 = make_arch(n, L, A)
    verts, faces, sheet_labels = merge_meshes_weld([mesh0, mesh1])
    F = len(faces)
    print(f"Mesh: {len(verts)} vertices, {F} faces")

    dual_edges, edge_valence, _ = build_dual_adjacency(faces)
    print(f"Dual edges: {len(dual_edges)}  "
          f"(cross-shared: {(edge_valence >= 2).sum()} edge-adjacent pairs)")

    # ------------------------------------------------------------------
    # Build iterated BlowUpLevel on face data
    # ------------------------------------------------------------------
    cents   = face_centroids(verts, faces)       # (F, 3)
    frames  = face_tangent_frames(verts, faces)  # (F, 3, 2)

    l0 = BlowUpLevel.from_point_tangents(cents, frames)
    l1 = l0.lift(k=min(20, F - 1), alpha=alpha, lam=1e-3)
    l2 = l1.lift(k=min(20, F - 1), alpha=alpha, lam=1e-3)

    print(f"Level dims: D0={l0.D}, D1={l1.D}, D2={l2.D}")

    # Classify at each level
    d0, tau0, lab0, sev0 = classify_at_level(l0.embedded,  dual_edges, F)
    d1, tau1, lab1, sev1 = classify_at_level(l1.embedded,  dual_edges, F)
    d2, tau2, lab2, sev2 = classify_at_level(l2.embedded,  dual_edges, F)

    def n_comp(labels): return int(labels.max()) + 1 if len(labels) else 0
    def gap(dists, severed):
        k, s = dists[~severed], dists[severed]
        return (float(k.max()) if len(k) else 0.0,
                float(s.min()) if len(s) else np.inf)

    print(f"Level 0: {n_comp(lab0)} components, tau={tau0:.4f}, gap={gap(d0,sev0)}")
    print(f"Level 1: {n_comp(lab1)} components, tau={tau1:.4f}, gap={gap(d1,sev1)}")
    print(f"Level 2: {n_comp(lab2)} components, tau={tau2:.4f}, gap={gap(d2,sev2)}")

    # Verify
    for lvl, lab in [("L0", lab0), ("L1", lab1), ("L2", lab2)]:
        ok = np.all((lab == lab[0]) == (sheet_labels == sheet_labels[0]))
        print(f"  {lvl} matches ground truth: {ok}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        rf"Iterated Mesh Blow-Up: Flat Plane $\cup$ Parabolic Arch"
        f"\n"
        rf"$n={n}$,  $A={A}$,  $L={L}$,  $\alpha={alpha}$,  "
        rf"dihedral $\delta \approx {delta_deg:.1f}^\circ$",
        fontsize=13,
    )

    # Row 1: 3D views
    ax_gt = fig.add_subplot(2, 3, 1, projection="3d")
    _add_mesh(ax_gt, verts, faces, _face_rgba(sheet_labels, SHEET_COLORS))
    _style3d(ax_gt, "Ground truth\n(blue=flat, orange=arch)")

    ax_l1 = fig.add_subplot(2, 3, 2, projection="3d")
    _add_mesh(ax_l1, verts, faces, _face_rgba(lab1, COMP_COLORS))
    _style3d(ax_l1, f"Level-1 components  ({n_comp(lab1)} found)")

    ax_l2 = fig.add_subplot(2, 3, 3, projection="3d")
    _add_mesh(ax_l2, verts, faces, _face_rgba(lab2, COMP_COLORS))
    _style3d(ax_l2, f"Level-2 components  ({n_comp(lab2)} found)")

    # Row 2: histograms
    ax_h0 = fig.add_subplot(2, 3, 4)
    _hist(ax_h0, d0, tau0, "0", SHEET_COLORS[0], sev0)
    ax_h0.set_title(f"Level 0  (position only)\n{n_comp(lab0)} component(s)")

    ax_h1 = fig.add_subplot(2, 3, 5)
    _hist(ax_h1, d1, tau1, "1", SHEET_COLORS[0], sev1)
    ax_h1.set_title(f"Level 1  (+projectors)\n{n_comp(lab1)} component(s)")

    ax_h2 = fig.add_subplot(2, 3, 6)
    _hist(ax_h2, d2, tau2, "2", SHEET_COLORS[0], sev2)
    ax_h2.set_title(f"Level 2  (+curvature)\n{n_comp(lab2)} component(s)")

    # Annotate gap size on histograms
    for ax, dists, severed in [(ax_h1, d1, sev1), (ax_h2, d2, sev2)]:
        mk, ms = gap(dists, severed)
        if ms < np.inf:
            ax.annotate(
                f"gap\n[{mk:.3f}, {ms:.3f}]",
                xy=((mk + ms) / 2, ax.get_ylim()[1] * 0.5),
                ha="center", fontsize=7.5, color="crimson",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

    plt.tight_layout()
    plt.savefig("mesh_blowup_tangent_arch.pdf", bbox_inches="tight")
    print("Saved mesh_blowup_tangent_arch.pdf")
    plt.show()


if __name__ == "__main__":
    main()
