"""
Alpha Ablation: Heat Flow on Lifted Point Clouds
-------------------------------------------------
Visualises the effect of the lift parameter ``alpha`` on heat diffusion on
a point cloud.  For each alpha in {0, 1, 2, 5}, we build a level-1 blow-up
and run the lifted heat method, displaying the heat field as one polyscope
point cloud per alpha.  An optional ``--bilateral`` panel adds the
normal-based bilateral Laplacian for comparison.

This script mirrors the ``--spread heat`` row of ``pointcloud.py``: same
``lifted_heat_method`` call, same per-column ``precompute_heat_method``
cache, same log-scale colormap, same bandwidth auto-estimation.

Usage:
    python alpha_ablation.py -p ship --bilateral
    python alpha_ablation.py -p Glykon --rotate y --alphas 0,1,2,5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from matplotlib import colormaps

from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud import lifted_heat_method, precompute_heat_method
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import estimate_product_bandwidths
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import examples_3D as _ex3d

# Bilateral heat method lives next to this script (no src equivalent).
from scalar_field_bilateral_comparison import bilateral_heat_method

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
DEFAULT_ALPHAS = [0.0, 1, 20, 50]
_HEAT_CMAPS = ["inferno", "magma", "viridis", "plasma", "coolwarm", "hot", "YlOrRd"]

# Non-orientable surface registry: name -> (factory, u_range, v_range).
# Ranges are shrunk slightly away from coordinate singularities / seams.
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


def _sample_surface(name: str, nu: int, nv: int):
    """Sample a non-orientable surface on a uniform grid.

    Returns flattened (points, normals) of shape (nu*nv, 3) each, both finite.
    """
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


# -- Rotation ----------------------------------------------------------------
def _rotation_matrix(angle_deg, axis):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def _parse_rotation(spec):
    spec = spec.strip().lower()
    if spec == "x":
        return _rotation_matrix(90, np.array([0.0, 1.0, 0.0]))
    if spec == "y":
        return _rotation_matrix(-90, np.array([1.0, 0.0, 0.0]))
    parts = spec.split(",")
    if len(parts) == 4:
        return _rotation_matrix(float(parts[0]),
                                np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
    raise ValueError(f"Invalid --rotate spec: {spec!r}")


# -- Data loading ------------------------------------------------------------
def _available_pointclouds():
    result = {}
    if _DATA_ROOT.is_dir():
        for p in _DATA_ROOT.rglob("*.npz"):
            result.setdefault(p.stem, p)
    return dict(sorted(result.items()))


def _resolve_pointcloud(name):
    p = Path(name)
    if p.suffix == ".npz" and p.exists():
        return p
    avail = _available_pointclouds()
    if name in avail:
        return avail[name]
    raise FileNotFoundError(
        f"Point cloud '{name}' not found. Available: {', '.join(avail)}"
    )


def _load_points_and_normals(path, normal_eps=1e-8):
    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError(f"Point cloud {path.name} is missing normals.")
    points = np.asarray(sample.points, dtype=float)
    normals = np.asarray(sample.normals, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if normals.ndim > 2:
        normals = normals.reshape(-1, normals.shape[-1])
    valid = (np.isfinite(points).all(axis=1)
             & np.isfinite(normals).all(axis=1)
             & (np.linalg.norm(normals, axis=1) > normal_eps))
    if not np.all(valid):
        print(f"Dropping {int(np.sum(~valid))} invalid samples.")
    return points[valid], normalize_vectors(normals[valid])


# -- Level construction ------------------------------------------------------
def _build_level(points, normals, *, alpha, k):
    frames = _normals_to_tangent_frames(normals)
    level = BlowUpLevel.from_point_tangents(points, frames)
    return level.lift(alpha=alpha, k=k)


# -- Heat colours (verbatim from pointcloud.py) ------------------------------
def _heat_to_colors(u, n_points, cmap_name="inferno"):
    cmap = colormaps[cmap_name]
    u_abs = np.abs(u)
    u_max = u_abs.max()
    if u_max < 1e-30:
        return np.tile(np.array(cmap(0.0)[:3]), (n_points, 1))
    log_u = np.log10(np.clip(u_abs / u_max, 1e-6, 1.0))
    t = (log_u + 6.0) / 6.0
    t = np.clip(t, 0.0, 1.0)
    return cmap(t)[:, :3]


# -- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Alpha ablation for lifted heat flow")
    parser.add_argument("--pointcloud", "-p", type=str, default=None,
                        help="Name or path of the point cloud (.npz).")
    parser.add_argument("--surface", "-s", type=str, default=None,
                        choices=sorted(NON_ORIENTABLE_SURFACES.keys()),
                        help="Sample a non-orientable surface instead of loading "
                             "a .npz point cloud. Mutually exclusive with --pointcloud.")
    parser.add_argument("--surface-nu", type=int, default=200,
                        help="Grid resolution along u for --surface (default 200).")
    parser.add_argument("--surface-nv", type=int, default=200,
                        help="Grid resolution along v for --surface (default 200).")
    parser.add_argument("--alphas", type=str,
                        default=",".join(str(a) for a in DEFAULT_ALPHAS),
                        help="Comma-separated alpha values (default: 0,1,2,5).")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--t-scale", type=float, default=1.0,
                        help="Diffusion time multiplier (Crane et al. default = 1).")
    parser.add_argument("--fixed-bandwidth", type=float, nargs="?", const=0.5, default=None,
                        help="Fix sigma_x and every sigma_u to this value across all "
                             "alphas (default 0.5 when flag given with no value). "
                             "Omit for per-alpha auto-estimation.")
    parser.add_argument("--bilateral", action="store_true",
                        help="Add a bilateral-Laplacian comparison panel.")
    parser.add_argument("--sigma-n-bi", type=float, default=0.5)
    parser.add_argument("--rotate", "-r", type=str, default=None)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--downsample-seed", type=int, default=None)
    parser.add_argument("--source", type=int, default=None,
                        help="Initial source index. If omitted, Ctrl+click a panel to pick.")
    args = parser.parse_args()

    if args.pointcloud is not None and args.surface is not None:
        parser.error("--pointcloud and --surface are mutually exclusive.")

    if args.pointcloud is None and args.surface is None:
        avail = _available_pointclouds()
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

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    if args.surface is not None:
        points, normals = _sample_surface(args.surface, args.surface_nu, args.surface_nv)
        print(f"Sampled {len(points)} points from surface '{args.surface}' "
              f"({args.surface_nu}x{args.surface_nv} grid)")
    else:
        path = _resolve_pointcloud(args.pointcloud)
        points, normals = _load_points_and_normals(path)
        print(f"Loaded {len(points)} points from {path.name}")

    if args.max_points and len(points) > args.max_points:
        rng = np.random.default_rng(args.downsample_seed)
        idx = np.sort(rng.choice(len(points), size=args.max_points, replace=False))
        points, normals = points[idx], normals[idx]
        print(f"Downsampled to {len(points)} points")

    if args.rotate is not None:
        R = _parse_rotation(args.rotate)
        points = points @ R.T
        normals = normals @ R.T

    bbox = points.max(axis=0) - points.min(axis=0)
    x_spacing = float(bbox[0]) * 1.4

    # -- Build one BlowUpLevel + bandwidths per alpha (lifted panels) --
    levels = {}      # alpha -> BlowUpLevel
    band_x = {}      # alpha -> sigma_x
    band_u = {}      # alpha -> [sigma_u, ...]
    for a in alphas:
        lev = _build_level(points, normals, alpha=a, k=args.k)
        if args.fixed_bandwidth is not None:
            sx = float(args.fixed_bandwidth)
            su = [float(args.fixed_bandwidth)] * max(lev.level, 1)
        else:
            sx, su = estimate_product_bandwidths(lev, args.k)
        levels[a] = lev
        band_x[a] = sx
        band_u[a] = su
        print(f"  alpha={a:g}: D={lev.D}, sigma_x={sx:.4f}, sigma_u={[round(s,4) for s in su]}")

    # -- Panel layout --
    panels = [("alpha", a) for a in alphas]
    if args.bilateral:
        panels.append(("bilateral", None))

    def _panel_name(p):
        kind, val = p
        return f"alpha={val:g}" if kind == "alpha" else "bilateral"

    panel_offsets = {p: np.array([i * x_spacing, 0.0, 0.0])
                     for i, p in enumerate(panels)}

    # -- Per-alpha precompute cache (mirrors pointcloud.py) --
    cache = {}

    def _get_cache(alpha):
        if alpha not in cache:
            print(f"  [alpha={alpha:g}] building precompute cache "
                  f"(t_scale={state['t_scale']:g})...")
            cache[alpha] = precompute_heat_method(
                levels[alpha], k=args.k,
                sigma_x=band_x[alpha],
                sigma_u=band_u[alpha],
                t_scale=state["t_scale"],
            )
        return cache[alpha]

    # -- UI state --
    state = {
        "source_index": int(args.source) if args.source is not None else None,
        "computing": args.source is not None,
        "heat_cmap_idx": 0,
        "u": {},   # panel -> raw heat field (for cmap re-coloring)
        "t_scale": float(args.t_scale),
    }

    def _compute_panel(p, src):
        kind, val = p
        if kind == "alpha":
            pre = _get_cache(val)
            _, u, _, _ = lifted_heat_method(
                levels[val], source_index=src, k=args.k,
                kernel="product",
                sigma_x=band_x[val], sigma_u=band_u[val],
                t_scale=state["t_scale"],
                return_intermediate=True,
                _precomputed=pre,
            )
        else:
            # Bilateral uses the spatial median k-NN distance for sigma_x
            # so it has a comparable length scale to the lifted panels.
            sigma_x_bi = float(np.median([band_x[a] for a in alphas]))
            _, u = bilateral_heat_method(
                points, normals, src,
                k=args.k, sigma_x=sigma_x_bi, sigma_n=args.sigma_n_bi,
                t_scale=state["t_scale"],
            )

        state["u"][p] = u
        cmap_name = _HEAT_CMAPS[state["heat_cmap_idx"]]
        colors = _heat_to_colors(u, len(points), cmap_name)
        ps.get_point_cloud(_panel_name(p)).add_color_quantity(
            "heat", colors, enabled=True,
        )

        # Source marker
        tag = f"src_{_panel_name(p)}"
        if ps.has_point_cloud(tag):
            ps.remove_point_cloud(tag)
        src_pt = points[src:src + 1] + panel_offsets[p]
        sc = ps.register_point_cloud(tag, src_pt, radius=0.008)
        sc.set_color((0.0, 1.0, 1.0))

    def _compute_all():
        src = state["source_index"]
        print(f"Computing heat fields from source {src}...")
        for p in panels:
            print(f"  [{_panel_name(p)}] computing...")
            _compute_panel(p, src)
        print("Done")

    def callback():
        psim.TextUnformatted("Ctrl+click any panel to pick the heat source")
        src_str = ("(none -- pick one)" if state["source_index"] is None
                   else str(state["source_index"]))
        psim.TextUnformatted(f"Source: {src_str}")
        psim.Separator()

        all_panel_names = [_panel_name(p) for p in panels]
        pick = ps.get_selection()
        if pick.is_hit and pick.structure_name in all_panel_names:
            picked = pick.local_index
            psim.TextUnformatted(f"Hovered: {picked}")
            if psim.Button("Set as source"):
                state["source_index"] = int(picked)

        changed_t, new_t = psim.InputFloat("t_scale", state["t_scale"])
        if changed_t and new_t > 0:
            state["t_scale"] = float(new_t)
            cache.clear()
            if state["source_index"] is not None:
                state["computing"] = True

        if state["source_index"] is not None and psim.Button("Compute"):
            state["computing"] = True

        if state["u"]:
            changed_cm, new_idx = psim.Combo(
                "Heat colormap", state["heat_cmap_idx"], _HEAT_CMAPS,
            )
            if changed_cm:
                state["heat_cmap_idx"] = new_idx
                cmap_name = _HEAT_CMAPS[new_idx]
                for p, u in state["u"].items():
                    colors = _heat_to_colors(u, len(points), cmap_name)
                    ps.get_point_cloud(_panel_name(p)).add_color_quantity(
                        "heat", colors, enabled=True,
                    )

        if state["computing"]:
            state["computing"] = False
            _compute_all()

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    for p in panels:
        cloud = ps.register_point_cloud(
            _panel_name(p), points + panel_offsets[p], radius=0.001,
        )
        cloud.set_color((0.7, 0.7, 0.7))

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
