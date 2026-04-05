"""
Curvature Estimation on Thingi10K Point Clouds
===============================================
Side-by-side comparison of curvature estimation methods on real-world
point clouds from the Thingi10K dataset.

Methods compared
----------------
1. **Blow-up** (ours) — level-1 iterated tangent blow-up.
2. **Jet fitting** (Cazals & Pouget 2003) — local polynomial fit.
3. **CNC** — 

Usage::

    python thingi10k.py                     # lists available point clouds
    python thingi10k.py -p ship
    python thingi10k.py -p snowflake --k-bu 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, extract_level1
from tangent_blowups.geometry.jet_fitting import jet_curvature
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors


# =====================================================================
# Paths
# =====================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pointcloud_dir() -> Path:
    return _repo_root() / "data" / "thingi10k_pointcloud"


def _mesh_dir() -> Path:
    return _repo_root() / "data" / "thingi10k"


def _available() -> list[str]:
    return sorted(p.stem for p in _pointcloud_dir().glob("*.npz"))


# =====================================================================
# Rotation
# =====================================================================

def _rotation_matrix(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c = np.cos(np.radians(angle_deg))
    s = np.sin(np.radians(angle_deg))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def _parse_rotation(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "x":
        return _rotation_matrix(90, np.array([0.0, 1.0, 0.0]))
    if spec == "y":
        return _rotation_matrix(-90, np.array([1.0, 0.0, 0.0]))
    parts = spec.split(",")
    if len(parts) == 4:
        angle = float(parts[0])
        axis = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        return _rotation_matrix(angle, axis)
    raise ValueError(f"Invalid --rotate spec: {spec!r}")


# =====================================================================
# Data loading
# =====================================================================

def _load_points_and_normals(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load point cloud, return (points, normals)."""
    path = _pointcloud_dir() / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    s = load_pointcloud(path)
    if s.normals is None:
        raise ValueError(f"{name} has no normals.")
    pts = np.asarray(s.points, dtype=float)
    nrm = np.asarray(s.normals, dtype=float)
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if nrm.ndim > 2:
        nrm = nrm.reshape(-1, nrm.shape[-1])
    valid = (
        np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
        & (np.linalg.norm(nrm, axis=1) > 1e-8)
    )
    return pts[valid], normalize_vectors(nrm[valid])


def _load_mesh(name: str):
    """Load STL mesh via trimesh.  Returns a trimesh.Trimesh."""
    import trimesh
    path = _mesh_dir() / f"{name}.stl"
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")
    return trimesh.load(path, force="mesh")


# =====================================================================
# Mesh curvature baseline via trimesh (Cohen-Steiner & Morvan)
# =====================================================================

def _mesh_curvature_radius(mesh) -> float:
    """Choose a radius for the trimesh discrete curvature measures.

    Uses twice the mean edge length — large enough to integrate a
    meaningful patch, small enough to stay local.
    """
    edges = mesh.vertices[mesh.edges_unique]
    lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
    return float(2.0 * np.mean(lengths))


def mesh_curvatures(mesh, radius: float | None = None):
    """
    Per-vertex Gaussian and mean curvature via trimesh.

    Uses ``discrete_gaussian_curvature_measure`` and
    ``discrete_mean_curvature_measure`` (Cohen-Steiner & Morvan,
    *Restricted Delaunay triangulations and normal cycle*, 2003).

    The raw outputs are curvature *measures* (integrals over a ball);
    we normalise to pointwise values by dividing by the ball area.
    """
    import trimesh.curvature as tc

    if radius is None:
        radius = _mesh_curvature_radius(mesh)

    verts = mesh.vertices
    K_measure = tc.discrete_gaussian_curvature_measure(mesh, verts, radius)
    H_measure = tc.discrete_mean_curvature_measure(mesh, verts, radius)

    ball_area = max(np.pi * radius**2, 1e-30)
    K = K_measure / ball_area
    H = H_measure / ball_area
    return K, H


def _transfer_mesh_to_pc(
    mesh_values: np.ndarray,
    mesh_vertices: np.ndarray,
    pc_points: np.ndarray,
    *,
    k_avg: int = 8,
) -> np.ndarray:
    """Transfer per-vertex mesh values to point cloud via k-NN averaging.

    Instead of snapping to the single nearest vertex (which creates a
    staircase artefact when the mesh is coarser than the point cloud),
    we average over the *k_avg* nearest mesh vertices weighted by
    inverse distance.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh_vertices)
    dists, idx = tree.query(pc_points, k=k_avg)
    dists = np.maximum(dists, 1e-30)
    weights = 1.0 / dists
    weights /= weights.sum(axis=1, keepdims=True)
    vals = mesh_values[idx]
    return np.sum(vals * weights, axis=1)


# =====================================================================
# Curvature estimation wrappers
# =====================================================================

def _normals_to_frames(normals: np.ndarray) -> np.ndarray:
    """Build (N, 3, 2) tangent frames from normals."""
    N = len(normals)
    frames = np.empty((N, 3, 2))
    for i in range(N):
        n = normals[i]
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, ref)
        t1 /= np.linalg.norm(t1) + 1e-30
        t2 = np.cross(n, t1)
        frames[i, :, 0] = t1
        frames[i, :, 1] = t2
    return frames


def _blowup_curvature(pts, normals, *, k=30, alpha=20.0, lam=1e-3):
    """Curvature via level-1 blow-up.  Returns dict of scalar fields."""
    frames = _normals_to_frames(normals)
    l0 = BlowUpLevel.from_point_tangents(pts, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam, spatial_knn=True)
    inv = extract_level1(l1)
    K = inv.gaussian_curvature
    Tc = inv.total_curvature[:, 0]
    H2 = (Tc**2 + 2.0 * K) / 4.0    # H² = (||h||² + 2K) / 4, orientation-free
    return {"K": K, "H2": H2, "total": Tc}


def _jet_curvature_all(pts, normals, *, k=30, degree=3):
    """Curvature via jet fitting.  Returns dict of scalar fields."""
    res = jet_curvature(pts, normals, k=k, degree=degree)
    K = res["gaussian_curvature"]
    k1, k2 = res["k1"], res["k2"]
    Tc = np.sqrt(np.nan_to_num(k1**2 + k2**2, nan=0.0))
    H2 = np.nan_to_num((k1 + k2)**2 / 4.0, nan=0.0)
    return {"K": K, "H2": H2, "total": Tc}


# =====================================================================
# Polyscope helpers
# =====================================================================

QUANTITIES = ["K", "H2", "total"]
CMAPS = {"K": "coolwarm", "H2": "viridis", "total": "viridis"}


def _clamp(vals, pct=5):
    """Percentile-clamp, return (clamped, vmin, vmax)."""
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return np.nan_to_num(vals, nan=0.0), -1.0, 1.0
    lo, hi = np.percentile(finite, [pct, 100 - pct])
    vmax = max(abs(lo), abs(hi), 1e-15)
    clamped = np.clip(np.nan_to_num(vals, nan=0.0), -vmax, vmax)
    return clamped, -vmax, vmax


def _show_quantity(qname, panel_names, panel_data, *, normalize=False):
    """Set the active quantity on every panel by re-adding the scalar data."""
    key = (qname, normalize)
    for name in panel_names:
        if key in panel_data[name]:
            vals, vmin, vmax = panel_data[name][key]
            ps.get_point_cloud(name).add_scalar_quantity(
                "curvature", vals, enabled=True,
                cmap=CMAPS[qname], vminmax=(vmin, vmax),
            )


# =====================================================================
# Main
# =====================================================================

def run(
    name: str,
    *,
    k_bu: int = 30,
    k_jet: int = 30,
    degree: int = 3,
    mesh: bool = True,
    n_max: int | None = None,
    rotate: np.ndarray | None = None,
):
    print(f"\n=== {name} ===")

    pts, normals = _load_points_and_normals(name)
    if rotate is not None:
        pts = pts @ rotate.T
        normals = normals @ rotate.T
    n_total = len(pts)
    if n_max is not None and n_total > n_max:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, n_max, replace=False)
        pts, normals = pts[idx], normals[idx]
        print(f"  {len(pts)} points (downsampled from {n_total})")
    else:
        print(f"  {len(pts)} points")

    print("  Computing blow-up curvature ...")
    bu = _blowup_curvature(pts, normals, k=k_bu)

    print("  Computing jet-fitting curvature ...")
    jf = _jet_curvature_all(pts, normals, k=k_jet, degree=degree)

    # Build methods dict in display order: jet (left) -> cnc -> blow-up (right)
    methods = {"jet": jf}

    # CNC baseline (if file exists)
    cnc_path = Path(f"cnc/cnc_{name}.txt")
    if cnc_path.exists():
        print(f"  Loading CNC curvatures from {cnc_path} ...")
        cnc_data = np.loadtxt(cnc_path)
        K_cnc = cnc_data[:, 0]
        H_cnc = cnc_data[:, 1]
        H2_cnc = H_cnc**2
        total_cnc = np.sqrt(4.0 * H2_cnc - 2.0 * K_cnc)
        methods["cnc"] = {"K": K_cnc, "H2": H2_cnc, "total": total_cnc}

    methods["blow-up"] = bu

    # Mesh baseline (if requested and STL available)
    mesh_path = _mesh_dir() / f"{name}.stl"
    if mesh and mesh_path.exists():
        print("  Computing mesh baseline curvature (trimesh) ...")
        mesh = _load_mesh(name)
        K_mesh_v, H_mesh_v = mesh_curvatures(mesh)
        K_mesh = _transfer_mesh_to_pc(K_mesh_v, mesh.vertices, pts)
        H_mesh = _transfer_mesh_to_pc(H_mesh_v, mesh.vertices, pts)
        H2_mesh = H_mesh**2
        methods["mesh"] = {"K": K_mesh, "H2": H2_mesh}

        def _corr(a, b, label):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                return
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            print(f"    {label}: corr = {r:.4f}")

        print("  Correlation with mesh baseline:")
        _corr(K_mesh, bu["K"], "K   blow-up vs mesh")
        _corr(K_mesh, jf["K"], "K   jet     vs mesh")
        _corr(H2_mesh, bu["H2"], "H2  blow-up vs mesh")
        _corr(H2_mesh, jf["H2"], "H2  jet     vs mesh")
        print("  Correlation blow-up vs jet:")
        _corr(bu["K"], jf["K"], "K   blow-up vs jet")
        _corr(bu["H2"], jf["H2"], "H2  blow-up vs jet")
    else:
        print("  (no STL mesh found for mesh baseline)")

    # -- Register side-by-side panels in polyscope --
    method_names = list(methods.keys())
    bbox = pts.max(axis=0) - pts.min(axis=0)
    spacing = bbox[0] * 1.4

    # Shared colour ranges (across all methods)
    shared_range: dict[str, tuple[float, float]] = {}
    for qname in QUANTITIES:
        all_v = []
        for m in method_names:
            if qname in methods[m]:
                all_v.append(methods[m][qname])
        if all_v:
            _, vmin, vmax = _clamp(np.concatenate(all_v))
            shared_range[qname] = (vmin, vmax)
        else:
            shared_range[qname] = (-1.0, 1.0)

    panel_names = []
    panel_data: dict[str, dict[tuple, tuple]] = {}
    for i, method in enumerate(method_names):
        offset = np.array([i * spacing, 0.0, 0.0])
        pname = f"{method}"
        cloud = ps.register_point_cloud(pname, pts + offset, radius=0.001)
        cloud.set_color((0.75, 0.75, 0.75))
        cloud.add_vector_quantity("normals", normals, enabled=False,
                                 color=(0.2, 0.2, 0.8))
        panel_names.append(pname)

        panel_data[pname] = {}
        for qname in QUANTITIES:
            if qname not in methods[method]:
                continue
            raw = np.nan_to_num(methods[method][qname], nan=0.0)

            # Shared range: same vminmax for all methods
            sv_min, sv_max = shared_range[qname]
            panel_data[pname][(qname, False)] = (
                np.clip(raw, sv_min, sv_max), sv_min, sv_max)

            # Per-method range: each panel gets its own vminmax
            _, pv_min, pv_max = _clamp(methods[method][qname])
            panel_data[pname][(qname, True)] = (
                np.clip(raw, pv_min, pv_max), pv_min, pv_max)

    return method_names, panel_names, panel_data, normals


def main():
    parser = argparse.ArgumentParser(
        description="Curvature estimation on Thingi10K point clouds"
    )
    parser.add_argument(
        "-p", "--pointcloud", type=str, default=None,
        help="Name of the point cloud (without extension). Omit to list available.",
    )
    parser.add_argument("--k-bu", type=int, default=30, help="k-NN for blow-up")
    parser.add_argument("--k-jet", type=int, default=30, help="k-NN for jet fitting")
    parser.add_argument("--degree", type=int, default=3, help="Jet fitting degree")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Skip mesh baseline computation")
    parser.add_argument("--n-max", type=int, default=None,
                        help="Downsample to at most this many points")
    parser.add_argument("--rotate", "-r", type=str, default=None,
                        help="Rotate point cloud: 'x', 'y', or 'angle,ax,ay,az'")
    args = parser.parse_args()

    if args.pointcloud is None:
        avail = _available()
        print("Available point clouds:")
        for name in avail:
            print(f"  {name}")
        return

    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    state = {"quantity": 0, "normalize": False, "normals": False}
    R = _parse_rotation(args.rotate) if args.rotate else None
    method_names, panel_names, panel_data, normals = run(
        args.pointcloud, k_bu=args.k_bu, k_jet=args.k_jet, degree=args.degree,
        mesh=not args.no_mesh, n_max=args.n_max, rotate=R)

    # Show K initially
    _show_quantity("K", panel_names, panel_data, normalize=False)

    print(f"Copies placed side-by-side: {' | '.join(method_names)}")

    def _refresh():
        _show_quantity(QUANTITIES[state["quantity"]], panel_names, panel_data,
                       normalize=state["normalize"])

    def callback():
        psim.TextUnformatted(
            f"{args.pointcloud}  |  Panels: {' | '.join(method_names)}")
        psim.Separator()

        changed_q, new_q = psim.Combo("Quantity", state["quantity"], QUANTITIES)
        if changed_q:
            state["quantity"] = new_q
            _refresh()

        changed_n, new_n = psim.Checkbox("Normalize per method", state["normalize"])
        if changed_n:
            state["normalize"] = new_n
            _refresh()

        changed_nrm, new_nrm = psim.Checkbox("Show normals", state["normals"])
        if changed_nrm:
            state["normals"] = new_nrm
            for pname in panel_names:
                ps.get_point_cloud(pname).add_vector_quantity(
                    "normals", normals, enabled=new_nrm,
                    color=(0.2, 0.2, 0.8))

        psim.Separator()
        if psim.Button("Screenshot"):
            fname = f"curvature_{args.pointcloud}.png"
            ps.screenshot(fname, transparent_bg=False)
            print(f"Saved {fname}")

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
