"""
Curvature Estimation on Point Clouds
=====================================
Side-by-side comparison of curvature estimation methods on real-world
point clouds (Thingi10K, 3D-Scans, or any .npz point cloud).

Methods compared
----------------
1. **Blow-up** (ours) — level-1 iterated tangent blow-up (uniform weights).
2. **CNC** (pre-computed, loaded from ``cnc/`` folder).

Usage::

    python pointcloud.py                     # lists available point clouds
    python pointcloud.py -p ship
    python pointcloud.py -p Glykon --k-bu 30
    python pointcloud.py -p /path/to/custom.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, extract_level1
from tangent_blowups.geometry.weight_config import WeightConfig
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import examples_3D as _ex3d


# =====================================================================
# Non-orientable surface registry (mirrors alpha_ablation.py)
# =====================================================================
_PI = float(np.pi)
_EPS = 1e-3
NON_ORIENTABLE_SURFACES = {
    "mobius_band_1":        (_ex3d.mobius_band_1,        (-1.0, 1.0),        (0.0, 2 * _PI)),
    "mobius_band_3":        (_ex3d.mobius_band_3,        (-1.0, 1.0),        (0.0, 2 * _PI)),
    "steiner_crosscap":     (_ex3d.steiner_crosscap,     (0.0, 2 * _PI),     (0.0, _PI / 2)),
    "steiner_crosscap_cut": (_ex3d.steiner_crosscap_cut, (0.0, _PI),         (0.0, _PI / 2)),
    "steiner_roman":        (_ex3d.steiner_roman,        (0.0, _PI),         (0.0, _PI)),
    "busser_decic":         (_ex3d.busser_decic,         (0.0, 2 * _PI),     (-_PI / 2, _PI / 2)),
    "boy_surface":          (_ex3d.boy_surface,          (0.0, 2 * _PI),     (0.0, _PI)),
    "boy_surface_5":        (_ex3d.boy_surface_5,        (0.0, 2 * _PI),     (0.0, _PI)),
    "etruscan_venus":       (_ex3d.etruscan_venus,       (_EPS, 2 * _PI - _EPS), (_EPS, 2 * _PI - _EPS)),
    "jeener_bonan":         (_ex3d.jeener_bonan,         (0.0, 2 * _PI),     (0.0, 2 * _PI)),
    "jeener_bonan_2cc":     (_ex3d.jeener_bonan_2cc,     (0.0, 2 * _PI),     (0.0, 2 * _PI)),
    "busser_nonorientable": (_ex3d.busser_nonorientable, (0.0, 2 * _PI),     (-_PI / 2, _PI / 2)),
    "petit_russian_hat":    (_ex3d.petit_russian_hat,    (0.0, 2 * _PI),     (0.0, _PI)),
    "petit_russian_hat_cut":(_ex3d.petit_russian_hat_cut,(_PI, 2 * _PI),     (0.0, _PI)),
    "banchoff_klein":       (_ex3d.banchoff_klein,       (0.0, 2 * _PI),     (0.0, 2 * _PI)),
    "banchoff_klein_3":     (_ex3d.banchoff_klein_3,     (0.0, 2 * _PI),     (0.0, 2 * _PI)),
}


def _sample_surface(name: str, nu: int, nv: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a non-orientable surface on a uniform grid. Returns (points, normals)."""
    if name not in NON_ORIENTABLE_SURFACES:
        raise ValueError(
            f"Unknown surface '{name}'. Available: {', '.join(NON_ORIENTABLE_SURFACES)}"
        )
    factory, (u_lo, u_hi), (v_lo, v_hi) = NON_ORIENTABLE_SURFACES[name]
    surf = factory()
    u = np.linspace(u_lo, u_hi, nu)
    v = np.linspace(v_lo, v_hi, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    s = surf.evaluate(U, V, with_tangents=False, with_normals=True)
    pts = np.asarray(s.points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(s.normals, dtype=float).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(nrm).all(axis=1) & (
        np.linalg.norm(nrm, axis=1) > 1e-8
    )
    if not np.all(valid):
        print(f"Dropping {int(np.sum(~valid))} degenerate samples from {name}.")
    return pts[valid], normalize_vectors(nrm[valid])


# =====================================================================
# Paths
# =====================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _available() -> dict[str, Path]:
    """Return {stem: path} for all .npz point clouds under data/ (recursive)."""
    result: dict[str, Path] = {}
    root = _repo_root() / "data"
    if root.is_dir():
        for p in root.rglob("*.npz"):
            result.setdefault(p.stem, p)
    return dict(sorted(result.items()))


def _resolve_pointcloud(name: str) -> Path:
    """Resolve a point cloud name or path to an actual file."""
    # Direct path
    p = Path(name)
    if p.suffix == ".npz" and p.exists():
        return p
    # Search known directories
    avail = _available()
    if name in avail:
        return avail[name]
    raise FileNotFoundError(
        f"Point cloud '{name}' not found. Available: {', '.join(avail)}"
    )



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
    path = _resolve_pointcloud(name)
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


def _blowup_curvature(pts, normals, *, k=20, alpha=1.0, lam=0.0,
                      weight_config=None):
    """Curvature via level-1 blow-up.  Returns dict of scalar fields."""
    frames = _normals_to_frames(normals)
    l0 = BlowUpLevel.from_point_tangents(pts, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam, weight_config=weight_config)
    inv = extract_level1(l1)
    K = inv.gaussian_curvature
    Tc = inv.total_curvature[:, 0]
    H2 = (Tc**2 + 2.0 * K) / 4.0    # H² = (||h||² + 2K) / 4, orientation-free
    return {"K": K, "H2": H2, "total": Tc}



# =====================================================================
# Polyscope helpers
# =====================================================================

QUANTITIES = ["K", "H2", "total"]
CMAPS = {"K": "coolwarm", "H2": "viridis", "total": "inferno"}


_POSITIVE_QUANTITIES = {"H2", "total"}


def _clamp(vals, pct=5):
    """Percentile-clamp (symmetric), return (clamped, -vmax, vmax)."""
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return np.nan_to_num(vals, nan=0.0), -1.0, 1.0
    lo, hi = np.percentile(finite, [pct, 100 - pct])
    vmax = max(abs(lo), abs(hi), 1e-15)
    clamped = np.clip(np.nan_to_num(vals, nan=0.0), -vmax, vmax)
    return clamped, -vmax, vmax


def _clamp_positive(vals, pct=5):
    """Percentile-clamp for non-negative values, return (clamped, 0, vmax)."""
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return np.nan_to_num(vals, nan=0.0), 0.0, 1.0
    hi = np.percentile(finite, 100 - pct)
    vmax = max(hi, 1e-15)
    clamped = np.clip(np.nan_to_num(vals, nan=0.0), 0.0, vmax)
    return clamped, 0.0, vmax


def _clamp_for(qname, vals, pct=5):
    """Dispatch to _clamp or _clamp_positive based on quantity."""
    if qname in _POSITIVE_QUANTITIES:
        return _clamp_positive(vals, pct)
    return _clamp(vals, pct)


_QUANTITY_LABELS = {
    "K": r"Gaussian curvature $K$",
    "H2": r"$H^2$",
    "total": r"Curvature magnitude $\|h\|$",
}


def _export_ply(pointcloud_name, pts, normals, panel_names, panel_data, state):
    """Export each method's coloured point cloud as a PLY file for Blender."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    qname = QUANTITIES[state["quantity"]]
    key = "per" if state["per_method"] else "shared"
    cmap = plt.get_cmap(CMAPS[qname])

    for pname in panel_names:
        if qname not in panel_data[pname][key]:
            continue
        vals, vmin, vmax = panel_data[pname][key][qname]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        rgba = cmap(norm(vals))
        rgb = (rgba[:, :3] * 255).astype(np.uint8)

        N = len(pts)
        fname = f"{pointcloud_name}_{pname}_{qname}.ply"
        with open(fname, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(N):
                f.write(f"{pts[i,0]} {pts[i,1]} {pts[i,2]} "
                        f"{normals[i,0]} {normals[i,1]} {normals[i,2]} "
                        f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")
        print(f"Saved {fname}")


def _export_colourbar(pointcloud_name, panel_names, panel_data, state):
    """Save a standalone colour bar PDF matching the current polyscope view."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    qname = QUANTITIES[state["quantity"]]
    key = "per" if state["per_method"] else "shared"
    # Get range from the first panel that has this quantity
    vmin = vmax = None
    for name in panel_names:
        if qname in panel_data[name][key]:
            _, vmin, vmax = panel_data[name][key][qname]
            break
    if vmin is None:
        print("No data for current quantity.")
        return

    cmap = plt.get_cmap(CMAPS[qname])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.5, 0.35))
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation="horizontal",
    )
    cb.set_label(_QUANTITY_LABELS.get(qname, qname))

    fname = f"colourbar_{pointcloud_name}_{qname}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def _show_quantity(qname, panel_names, panel_data, *, per_method=False):
    """Set the active quantity on every panel by re-adding the scalar data."""
    key = "per" if per_method else "shared"
    for name in panel_names:
        if qname in panel_data[name][key]:
            vals, vmin, vmax = panel_data[name][key][qname]
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
    pts: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    k_bu: int = 50,
    n_max: int | None = None,
    rotate: np.ndarray | None = None,
    weight_config: WeightConfig | None = None,
):
    """Curvature comparison. Loads from `name` unless `pts`/`normals` are given."""
    print(f"\n=== {name} ===")

    if pts is None or normals is None:
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

    wc = weight_config or WeightConfig(kernel="uniform")
    print(f"  Computing blow-up curvature (weights={wc.kernel}) ...")
    bu = _blowup_curvature(pts, normals, k=k_bu, weight_config=wc)

    # Build methods dict in display order: jet -> cnc -> blow-up
    methods: dict[str, dict[str, np.ndarray]] = {}

    # Jet fitting baseline (pre-computed, from jet/ folder)
    jet_path = Path(__file__).parent / "jet" / f"jet_{name}.txt"
    if jet_path.exists():
        print(f"  Loading jet-fitting curvatures from {jet_path} ...")
        jet_data = np.loadtxt(jet_path)
        K_jet = jet_data[:, 0]
        H_jet = jet_data[:, 1]
        H2_jet = H_jet**2
        total_jet = np.sqrt(np.maximum(4.0 * H2_jet - 2.0 * K_jet, 0.0))
        methods["jet"] = {"K": K_jet, "H2": H2_jet, "total": total_jet}

    # CNC baseline (if file exists)
    cnc_path = Path(__file__).parent / "cnc" / f"cnc_{name}.txt"
    if cnc_path.exists():
        print(f"  Loading CNC curvatures from {cnc_path} ...")
        cnc_data = np.loadtxt(cnc_path)
        K_cnc = cnc_data[:, 0]
        H_cnc = cnc_data[:, 1]
        H2_cnc = H_cnc**2
        total_cnc = np.sqrt(np.maximum(4.0 * H2_cnc - 2.0 * K_cnc, 0.0))
        methods["cnc"] = {"K": K_cnc, "H2": H2_cnc, "total": total_cnc}

    methods["blow-up"] = bu

    # -- Register side-by-side panels in polyscope --
    method_names = list(methods.keys())
    bbox = pts.max(axis=0) - pts.min(axis=0)
    spacing = bbox[0] * 1.4

    # Shared colour ranges: median of per-method upper bounds so that
    # one method's outliers don't wash out the others.
    shared_range: dict[str, tuple[float, float]] = {}
    for qname in QUANTITIES:
        bounds = []
        for m in method_names:
            if qname in methods[m]:
                _, _, vmax = _clamp_for(qname, methods[m][qname])
                bounds.append(vmax)
        if bounds:
            vmax = float(np.median(bounds))
            if qname in _POSITIVE_QUANTITIES:
                shared_range[qname] = (0.0, vmax)
            else:
                shared_range[qname] = (-vmax, vmax)
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

        panel_data[pname] = {"shared": {}, "per": {}}
        for qname in QUANTITIES:
            if qname not in methods[method]:
                continue
            raw = np.nan_to_num(methods[method][qname], nan=0.0)
            # Shared colour range
            sv_min, sv_max = shared_range[qname]
            panel_data[pname]["shared"][qname] = (
                np.clip(raw, sv_min, sv_max), sv_min, sv_max)
            # Per-method colour range
            clamped, pv_min, pv_max = _clamp_for(qname, methods[method][qname])
            panel_data[pname]["per"][qname] = (clamped, pv_min, pv_max)

    return method_names, panel_names, panel_data, normals, pts


def main():
    parser = argparse.ArgumentParser(
        description="Curvature estimation on point clouds (Thingi10K, 3D-Scans, or custom .npz)"
    )
    parser.add_argument(
        "-p", "--pointcloud", type=str, default=None,
        help="Name or path of the point cloud (.npz). Omit to list available.",
    )
    parser.add_argument(
        "-s", "--surface", type=str, default=None,
        choices=sorted(NON_ORIENTABLE_SURFACES.keys()),
        help="Sample a non-orientable surface instead of loading a .npz point cloud. "
             "Mutually exclusive with --pointcloud.",
    )
    parser.add_argument("--surface-nu", type=int, default=300,
                        help="Grid resolution along u for --surface (default 300).")
    parser.add_argument("--surface-nv", type=int, default=300,
                        help="Grid resolution along v for --surface (default 300).")
    parser.add_argument("--k-bu", type=int, default=20, help="k-NN for blow-up")
    parser.add_argument("--weights", type=str, default="uniform",
                        choices=["uniform", "product"],
                        help="Regression weight scheme (default: uniform)")
    parser.add_argument("--sigma-x", type=float, default=0.5,
                        help="Spatial bandwidth for product weights")
    parser.add_argument("--sigma-u", type=float, default=0.5,
                        help="Angular bandwidth for product weights")
    parser.add_argument("--n-max", type=int, default=None,
                        help="Downsample to at most this many points")
    parser.add_argument("--rotate", "-r", type=str, default=None,
                        help="Rotate point cloud: 'x', 'y', or 'angle,ax,ay,az'")
    args = parser.parse_args()

    if args.pointcloud is not None and args.surface is not None:
        parser.error("--pointcloud and --surface are mutually exclusive.")

    if args.pointcloud is None and args.surface is None:
        avail = _available()
        print("Available point clouds:")
        if avail:
            for name, path in avail.items():
                print(f"  {name}  ({path.parent.name})")
        else:
            print("  (none)")
        print("Available non-orientable surfaces (--surface NAME):")
        for name in sorted(NON_ORIENTABLE_SURFACES):
            print(f"  {name}")
        return

    if args.surface is not None:
        display_name = args.surface
        pts_in, normals_in = _sample_surface(
            args.surface, args.surface_nu, args.surface_nv,
        )
        print(f"Sampled {len(pts_in)} points from surface '{args.surface}' "
              f"({args.surface_nu}x{args.surface_nv} grid)")
    else:
        display_name = args.pointcloud
        pts_in = normals_in = None   # run() will load from the name

    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    state = {"quantity": 0, "normals": False, "per_method": False}
    R = _parse_rotation(args.rotate) if args.rotate else None
    wc = WeightConfig(kernel=args.weights, sigma_x=args.sigma_x, sigma_u=args.sigma_u)
    method_names, panel_names, panel_data, normals, pts = run(
        display_name, pts=pts_in, normals=normals_in, k_bu=args.k_bu,
        n_max=args.n_max, rotate=R, weight_config=wc)

    # Show K initially
    _show_quantity("K", panel_names, panel_data)

    print(f"Copies placed side-by-side: {' | '.join(method_names)}")

    def _refresh():
        _show_quantity(QUANTITIES[state["quantity"]], panel_names, panel_data,
                       per_method=state["per_method"])

    def callback():
        psim.TextUnformatted(
            f"{display_name}  |  Panels: {' | '.join(method_names)}")
        psim.Separator()

        changed_q, new_q = psim.Combo("Quantity", state["quantity"], QUANTITIES)
        if changed_q:
            state["quantity"] = new_q
            _refresh()

        changed_pm, new_pm = psim.Checkbox("Normalize per method", state["per_method"])
        if changed_pm:
            state["per_method"] = new_pm
            _refresh()

        changed_nrm, new_nrm = psim.Checkbox("Show normals", state["normals"])
        if changed_nrm:
            state["normals"] = new_nrm
            for pname in panel_names:
                ps.get_point_cloud(pname).add_vector_quantity(
                    "normals", normals, enabled=new_nrm,
                    color=(0.2, 0.2, 0.8))

        psim.Separator()
        if psim.Button("Export colour bar"):
            _export_colourbar(display_name, panel_names, panel_data, state)

        if psim.Button("Export PLY"):
            _export_ply(display_name, pts, normals,
                        panel_names, panel_data, state)

        if psim.Button("Screenshot"):
            fname = f"curvature_{display_name}.png"
            ps.screenshot(fname, transparent_bg=False)
            print(f"Saved {fname}")

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
