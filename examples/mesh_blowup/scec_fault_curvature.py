"""
Curvature Estimation on SCEC Sierra Madre Fault Meshes
======================================================
Loads three intersecting SCEC fault meshes (GOCAD TSurf .ts files),
merges them into a single mesh, and compares per-face curvature from
the iterated tangent blow-up against two discrete baselines:

  1. **Dihedral-angle shape operator** (Rusinkiewicz 2004): fits a 2x2
     shape operator per face from the signed dihedral angles at its edges.
  2. **Local quadric fit**: fits z = au^2 + buv + cv^2 to nearby face
     centroids projected into each face's tangent plane.

Both baselines are *per-face* (no vertex averaging), handle open
boundaries gracefully, and are standard in discrete differential
geometry.

Uses PyVista for interactive 3-D viewing with titles and colour bars.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from tangent_blowups.geometry.iterated_grassmann import (
    BlowUpLevel,
    extract_level1,
)
from tangent_blowups.mesh.blowup import (
    face_centroids,
    face_normals,
    face_tangent_frames,
)


# ---------------------------------------------------------------------------
# GOCAD TSurf parser
# ---------------------------------------------------------------------------

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


def load_scec_faults() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and merge the three SCEC Sierra Madre fault meshes.

    The three fault surfaces are concatenated into a single mesh.
    They geometrically self-intersect in 3-D but do not share vertices
    at their intersection curves (the CFM triangulations are independent).
    The blow-up's k-NN on face centroids naturally picks up cross-sheet
    neighbours without needing shared mesh topology.

    Returns
    -------
    vertices : (V, 3)
    faces : (F, 3)
    sheet_labels : (F,) int  -- which original .ts file each face came from
    """
    data_dir = Path(__file__).resolve().parents[2] / "data" / "scec"
    ts_files = sorted(data_dir.glob("*.ts"))
    if not ts_files:
        raise FileNotFoundError(f"No .ts files found in {data_dir}")

    all_verts, all_faces, all_labels = [], [], []
    offset = 0
    for sheet_id, ts_path in enumerate(ts_files):
        verts, faces = parse_gocad_tsurf(ts_path)
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_labels.append(np.full(len(faces), sheet_id, dtype=int))
        offset += len(verts)

    return (
        np.concatenate(all_verts),
        np.concatenate(all_faces),
        np.concatenate(all_labels),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_polydata(verts: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Build a PyVista PolyData from (V,3) vertices and (F,3) face indices."""
    F = len(faces)
    pv_faces = np.empty((F, 4), dtype=np.int64)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces
    return pv.PolyData(verts.astype(np.float64), pv_faces.ravel())


def _face_adjacency(faces: np.ndarray) -> dict[int, list[int]]:
    """Build face adjacency via shared edges.  Returns {face_idx: [neighbours]}."""
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            e = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            edge_to_faces[e].append(fi)

    adj: dict[int, list[int]] = defaultdict(list)
    for flist in edge_to_faces.values():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                adj[flist[i]].append(flist[j])
                adj[flist[j]].append(flist[i])
    return adj


def _face_smooth(
    vals: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 1,
) -> np.ndarray:
    """Simple 1-ring Laplacian smoothing of per-face scalar values."""
    adj = _face_adjacency(faces)
    out = vals.copy()
    for _ in range(iterations):
        new = out.copy()
        for fi in range(len(faces)):
            if adj[fi]:
                new[fi] = np.mean(out[adj[fi]])
        out = new
    return out


# ---------------------------------------------------------------------------
# Baseline 1: Dihedral-angle shape operator (Rusinkiewicz 2004)
# ---------------------------------------------------------------------------

def dihedral_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face (K, |H|) via dihedral-angle shape operator.

    For each edge shared by two faces, the signed dihedral angle divided
    by the edge length gives the normal curvature in the edge direction.
    A 2x2 shape operator S is fitted per face from its (up to 3) edge
    constraints.  K = det(S), H = tr(S)/2.

    Boundary faces (< 2 interior edges) get NaN.
    """
    F = len(faces)
    normals = face_normals(vertices, faces)       # (F, 3) unit
    cents = face_centroids(vertices, faces)        # (F, 3)
    frames = face_tangent_frames(vertices, faces)  # (F, 3, 2)

    # Build edge -> [face_idx, face_idx] map
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_to_faces[min(a, b), max(a, b)].append(fi)

    # For each face, collect (tangent-plane edge direction, kappa_n) pairs
    face_obs: list[list[tuple[np.ndarray, float]]] = [[] for _ in range(F)]

    for (va, vb), flist in edge_to_faces.items():
        if len(flist) != 2:
            continue  # boundary edge — skip
        fi, fj = flist
        edge_vec = vertices[vb] - vertices[va]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_hat = edge_vec / edge_len

        # Signed dihedral angle: positive when normals "open up"
        ni, nj = normals[fi], normals[fj]
        sin_theta = np.dot(np.cross(ni, nj), edge_hat)
        cos_theta = np.dot(ni, nj)
        theta = np.arctan2(sin_theta, cos_theta)
        kappa_n = theta / edge_len

        # Project edge direction into each face's tangent frame
        for f_idx in (fi, fj):
            U = frames[f_idx]  # (3, 2)
            d = U.T @ edge_hat  # (2,)
            d_norm = np.linalg.norm(d)
            if d_norm > 1e-10:
                d = d / d_norm
                face_obs[f_idx].append((d, kappa_n))

    K_out = np.full(F, np.nan)
    H_out = np.full(F, np.nan)

    for fi in range(F):
        obs = face_obs[fi]
        if len(obs) < 2:
            continue  # underdetermined

        # Fit S = [[s11, s12], [s12, s22]] from kappa_n = d^T S d
        # Each observation: d1^2 * s11 + 2*d1*d2 * s12 + d2^2 * s22 = kappa_n
        n_obs = len(obs)
        A = np.empty((n_obs, 3))
        b = np.empty(n_obs)
        for i, (d, kn) in enumerate(obs):
            A[i] = [d[0] ** 2, 2 * d[0] * d[1], d[1] ** 2]
            b[i] = kn

        s, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        s11, s12, s22 = s
        K_out[fi] = s11 * s22 - s12 * s12
        H_out[fi] = 0.5 * (s11 + s22)

    return K_out, np.abs(H_out)


# ---------------------------------------------------------------------------
# Baseline 2: Local quadric fit
# ---------------------------------------------------------------------------

def quadric_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    n_rings: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face (K, |H|) via local quadric fitting.

    For each face, gathers centroids from its n-ring neighbourhood,
    projects them into the face's tangent plane, and fits

        h = a*u^2 + b*u*v + c*v^2

    via least squares.  Then K = 4ac - b^2, H = a + c.

    Faces with too few neighbours (< 3) get NaN.
    """
    F = len(faces)
    normals = face_normals(vertices, faces)
    cents = face_centroids(vertices, faces)
    frames = face_tangent_frames(vertices, faces)  # (F, 3, 2)

    adj = _face_adjacency(faces)

    # Expand adjacency to n_rings
    def n_ring_neighbours(fi: int, n: int) -> list[int]:
        visited = {fi}
        frontier = {fi}
        for _ in range(n):
            next_frontier: set[int] = set()
            for f in frontier:
                for nb in adj.get(f, []):
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.add(nb)
            frontier = next_frontier
        visited.discard(fi)
        return list(visited)

    K_out = np.full(F, np.nan)
    H_out = np.full(F, np.nan)

    for fi in range(F):
        nbs = n_ring_neighbours(fi, n_rings)
        if len(nbs) < 3:
            continue

        U = frames[fi]   # (3, 2)
        n_i = normals[fi]  # (3,)
        c_i = cents[fi]    # (3,)

        # Project neighbour centroids into local tangent frame
        deltas = cents[nbs] - c_i  # (M, 3)
        uv = deltas @ U            # (M, 2)
        h = deltas @ n_i           # (M,)  height above tangent plane

        # Fit h = a*u^2 + b*u*v + c*v^2
        u, v = uv[:, 0], uv[:, 1]
        A = np.column_stack([u ** 2, u * v, v ** 2])
        coeffs, _, _, _ = np.linalg.lstsq(A, h, rcond=None)
        a, b, c = coeffs

        K_out[fi] = 4 * a * c - b * b
        H_out[fi] = a + c

    return K_out, np.abs(H_out)


# ---------------------------------------------------------------------------
# Blow-up curvature estimation
# ---------------------------------------------------------------------------

def blowup_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 25,
    alpha: float = 1.0,
    lam: float = 3e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face (K, |H|) via iterated blow-up.  Returns (F,), (F,)."""
    cents = face_centroids(vertices, faces)
    frames = face_tangent_frames(vertices, faces)
    l0 = BlowUpLevel.from_point_tangents(cents, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam)
    inv = extract_level1(l1)
    return inv.gaussian_curvature, np.abs(inv.mean_curvature[:, 0])


# ---------------------------------------------------------------------------
# PyVista views
# ---------------------------------------------------------------------------

def show_sheets(mesh: pv.PolyData, sheet_labels: np.ndarray):
    """Interactive view: mesh coloured by fault sheet."""
    mesh.cell_data["Sheet"] = sheet_labels
    pl = pv.Plotter()
    pl.add_mesh(
        mesh, scalars="Sheet", cmap=["#4C72B0", "#DD8452", "#55A868"],
        show_scalar_bar=True, scalar_bar_args={"title": "Sheet ID"},
    )
    pl.add_title("Fault Sheet Identity", font_size=12)
    pl.show()


def _per_method_clim(
    vals: np.ndarray, diverging: bool,
) -> list[float]:
    """Compute colour limits from a single method's values."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return [-1, 1] if diverging else [0, 1]
    if diverging:
        vmax = max(abs(float(np.percentile(v, 2))),
                   abs(float(np.percentile(v, 98))), 1e-8)
        return [-vmax, vmax]
    else:
        return [max(float(np.percentile(v, 2)), 0),
                max(float(np.percentile(v, 98)), 1e-8)]


def show_curvature_comparison(
    mesh: pv.PolyData,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
):
    """Interactive 2-row x N-col view: K (top), |H| (bottom) for each method.

    Each column gets its own colour scale so that every method's
    curvature structure is visible at its own dynamic range.
    """
    methods = list(results.keys())
    n = len(methods)

    pl = pv.Plotter(shape=(2, n))

    for col, method in enumerate(methods):
        K, H = results[method]
        k_clim = _per_method_clim(K, diverging=True)
        h_clim = _per_method_clim(H, diverging=False)

        for row, (vals, label, cmap, clim) in enumerate([
            (K, f"K ({method})", "RdBu_r", k_clim),
            (H, f"|H| ({method})", "viridis", h_clim),
        ]):
            pl.subplot(row, col)
            m = mesh.copy()
            display_vals = np.where(np.isfinite(vals), vals, 0.0)
            m.cell_data[label] = display_vals
            pl.add_mesh(
                m, scalars=label, cmap=cmap, clim=clim,
                show_scalar_bar=True,
                scalar_bar_args={"title": label, "n_labels": 5},
                nan_opacity=0.3,
            )
            pl.add_title(label, font_size=9)

    pl.link_views()
    pl.show()


# ---------------------------------------------------------------------------
# Matplotlib 2-D charts
# ---------------------------------------------------------------------------

def plot_histograms(results: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Side-by-side histograms for K and |H| across all methods."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    methods = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvature Distribution by Method (interior faces)", fontsize=12)

    for ax, (idx, label) in zip(axes, [(0, "Gaussian curvature K"),
                                        (1, "Mean curvature |H|")]):
        for i, method in enumerate(methods):
            vals = results[method][idx]
            vals = vals[np.isfinite(vals)]
            lo, hi = np.percentile(vals, [1, 99])
            bins = np.linspace(lo, hi, 80)
            ax.hist(vals, bins=bins, alpha=0.5, label=method, color=colors[i % len(colors)])
        ax.set_xlabel(label)
        ax.set_ylabel("Face count")
        ax.set_title(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("scec_fault_curvature_histograms.png", dpi=150, bbox_inches="tight")
    print("Saved scec_fault_curvature_histograms.png")
    plt.show()


def plot_per_sheet_stats(
    sheet_labels: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
):
    """Bar chart of per-sheet median curvature from each method."""
    sheets = np.unique(sheet_labels)
    n_sheets = len(sheets)
    methods = list(results.keys())
    n_methods = len(methods)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    names = [
        "SMDD Sierra Madre",
        "SMDE Cucamonga conn.",
        "SMDE Sierra Madre",
    ]
    if len(names) < n_sheets:
        names += [f"Sheet {i}" for i in range(len(names), n_sheets)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Sheet Median Curvature", fontsize=12)
    x = np.arange(n_sheets)
    width = 0.8 / n_methods

    for ax, (idx, label) in zip(axes, [(0, "Gaussian K"), (1, "|H|")]):
        for i, method in enumerate(methods):
            vals = results[method][idx]
            medians = [np.nanmedian(vals[sheet_labels == s]) for s in sheets]
            ax.bar(x + (i - n_methods / 2 + 0.5) * width, medians, width,
                   label=method, color=colors[i % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(f"Median {label}")
        ax.set_title(label)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("scec_fault_curvature_per_sheet.png", dpi=150, bbox_inches="tight")
    print("Saved scec_fault_curvature_per_sheet.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    k: int = 25,
    alpha: float = 1.0,
    lam: float = 3e-3,
    smooth_iters: int = 1,
) -> None:
    # --- Load mesh ---------------------------------------------------------
    print("Loading SCEC Sierra Madre fault meshes ...")
    verts, faces, sheet_labels = load_scec_faults()
    F = len(faces)
    for s in np.unique(sheet_labels):
        n = np.sum(sheet_labels == s)
        print(f"  Sheet {s}: {n} faces")
    print(f"  Total: {len(verts)} vertices, {F} faces")

    # --- Normalise coordinates (metres -> unit scale) ----------------------
    center = verts.mean(axis=0)
    scale = np.max(verts.max(axis=0) - verts.min(axis=0))
    verts_n = (verts - center) / scale

    # --- Blow-up curvature -------------------------------------------------
    print(f"Computing blow-up curvatures (k={k}, alpha={alpha}, lam={lam}) ...")
    K_bu, H_bu = blowup_curvatures(verts_n, faces, k=k, alpha=alpha, lam=lam)

    if smooth_iters > 0:
        print(f"Smoothing blow-up curvatures ({smooth_iters} iterations) ...")
        K_bu = _face_smooth(K_bu, faces, iterations=smooth_iters)
        H_bu = _face_smooth(H_bu, faces, iterations=smooth_iters)

    # --- Baseline 1: Dihedral-angle shape operator -------------------------
    print("Computing dihedral-angle shape operator ...")
    K_dih, H_dih = dihedral_curvatures(verts_n, faces)
    n_valid_dih = np.sum(np.isfinite(K_dih))
    print(f"  {n_valid_dih} / {F} faces have valid dihedral estimates")

    # --- Baseline 2: Local quadric fit -------------------------------------
    print("Computing local quadric fit (2-ring) ...")
    K_quad, H_quad = quadric_curvatures(verts_n, faces, n_rings=2)
    n_valid_quad = np.sum(np.isfinite(K_quad))
    print(f"  {n_valid_quad} / {F} faces have valid quadric estimates")

    # --- Collect results ---------------------------------------------------
    results: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Blow-up": (K_bu, H_bu),
        "Dihedral": (K_dih, H_dih),
        "Quadric fit": (K_quad, H_quad),
    }

    # --- Summary statistics ------------------------------------------------
    def _stats(name, vals):
        v = vals[np.isfinite(vals)]
        if len(v) == 0:
            print(f"  {name:30s}  (no valid values)")
            return
        print(f"  {name:30s}  median={np.median(v):+10.4f}  "
              f"MAD={np.median(np.abs(v - np.median(v))):10.4f}  "
              f"[{np.percentile(v, 5):+.4f}, {np.percentile(v, 95):+.4f}]")

    print("\nGaussian curvature K:")
    for method, (K, _) in results.items():
        _stats(method, K)
    print("\nMean curvature |H|:")
    for method, (_, H) in results.items():
        _stats(method, H)

    # Pairwise correlation (on jointly valid faces)
    print("\nPairwise Pearson correlation:")
    method_names = list(results.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            mi, mj = method_names[i], method_names[j]
            Ki, Hi = results[mi]
            Kj, Hj = results[mj]
            valid = np.isfinite(Ki) & np.isfinite(Kj)
            if valid.sum() > 10:
                rK = np.corrcoef(Ki[valid], Kj[valid])[0, 1]
                rH = np.corrcoef(Hi[valid], Hj[valid])[0, 1]
                print(f"  {mi} vs {mj} ({valid.sum()} faces): "
                      f"K={rK:.4f}, |H|={rH:.4f}")

    # --- Build PyVista mesh ------------------------------------------------
    mesh = _make_polydata(verts, faces)

    # --- Interactive 3-D views (PyVista) -----------------------------------
    show_sheets(mesh, sheet_labels)
    show_curvature_comparison(mesh, results)

    # --- 2-D charts (matplotlib) -------------------------------------------
    plot_histograms(results)
    plot_per_sheet_stats(sheet_labels, results)


if __name__ == "__main__":
    main()
