"""
Geodesic Comparison on Point Clouds
------------------------------------
Side-by-side comparison of geodesic distances on point clouds
(Thingi10K, 3D-Scans, or any .npz point cloud):

1. **Lifted** (ours) -- product-kernel heat method on the level-1 blow-up
2. **Robust** -- nonmanifold Laplacian from the ``robust_laplacian`` library

Usage:
    python thingi10k.py -p ship --rotate y
    python thingi10k.py -p Glykon
    python thingi10k.py                      # lists available point clouds
"""
from __future__ import annotations

import argparse
from math import gamma as math_gamma
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import robust_laplacian
from matplotlib import colormaps
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import connected_components

from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud import lifted_heat_method
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.solvers.linalg import normalize_vectors

# -- Constants ---------------------------------------------------------------
METHODS = ["lifted", "robust"]
_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
_POINTCLOUD_DIRS = ["thingi10k_pointcloud", "threedscans_pointcloud"]


# -- Rotation ----------------------------------------------------------------
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


# -- Data loading ------------------------------------------------------------
def _available_pointclouds() -> dict[str, Path]:
    """Return {stem: path} for all available point clouds across datasets."""
    result: dict[str, Path] = {}
    for dirname in _POINTCLOUD_DIRS:
        d = _DATA_ROOT / dirname
        if d.is_dir():
            for p in d.glob("*.npz"):
                result[p.stem] = p
    return dict(sorted(result.items()))


def _resolve_pointcloud(name: str) -> Path:
    """Resolve a point cloud name or path to an actual file."""
    p = Path(name)
    if p.suffix == ".npz" and p.exists():
        return p
    avail = _available_pointclouds()
    if name in avail:
        return avail[name]
    raise FileNotFoundError(
        f"Point cloud '{name}' not found. Available: {', '.join(avail)}"
    )


def _load_points_and_normals(
    path: Path, *, normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError(f"Point cloud {path.name} is missing normals.")
    points = np.asarray(sample.points, dtype=float)
    normals = np.asarray(sample.normals, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if normals.ndim > 2:
        normals = normals.reshape(-1, normals.shape[-1])
    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid &= np.linalg.norm(normals, axis=1) > normal_eps
    if not np.all(valid):
        print(f"Dropping {int(np.sum(~valid))} invalid/degenerate samples.")
    return points[valid], normalize_vectors(normals[valid])


def _build_level(points, normals, *, alpha, k, n_lifts=1):
    frames = _normals_to_tangent_frames(normals)
    level = BlowUpLevel.from_point_tangents(points, frames)
    for _ in range(n_lifts):
        level = level.lift(alpha=alpha, k=k)
    return level


# -- Generic heat method for arbitrary (L, M, W) ----------------------------
def _estimate_time_step_pts(points, W, t_scale):
    coo = W.tocoo()
    mask = coo.row != coo.col
    rows, cols = coo.row[mask], coo.col[mask]
    if rows.size == 0:
        return float(t_scale)
    d = points[rows] - points[cols]
    dist = np.linalg.norm(d, axis=1)
    dist = dist[dist > 0.0]
    if dist.size == 0:
        return float(t_scale)
    return float(t_scale) * float(np.mean(dist) ** 2)


def _estimate_mass_pts(points, W, d_manifold=2):
    N = len(points)
    coo = W.tocoo()
    mask = coo.row != coo.col
    rows, cols = coo.row[mask], coo.col[mask]
    if rows.size == 0:
        return np.ones(N)
    dx = points[rows] - points[cols]
    dist = np.linalg.norm(dx, axis=1)
    r_k = np.zeros(N)
    np.maximum.at(r_k, rows, dist)
    k_per = np.bincount(rows, minlength=N).astype(float)
    k_per = np.clip(k_per, 1.0, None)
    V_d = np.pi ** (d_manifold / 2.0) / math_gamma(d_manifold / 2.0 + 1.0)
    mass = V_d * (r_k ** d_manifold) / k_per
    zero = mass <= 0.0
    if zero.any():
        mass[zero] = float(np.median(mass[~zero])) if (~zero).any() else 1.0
    return mass


def _edge_gradient(points, f, W, lam=0.0):
    """Weighted least-squares gradient in R^3 from a k-NN graph."""
    N, n = points.shape
    coo = W.tocoo()
    rows, cols = coo.row, coo.col
    ws = np.asarray(coo.data, dtype=float)

    dx = points[cols] - points[rows]
    df = f[cols] - f[rows]
    w_dx = ws[:, None] * dx

    S = np.zeros((N, n, n))
    b = np.zeros((N, n))
    for dim in range(n):
        for dim2 in range(n):
            vals = w_dx[:, dim] * dx[:, dim2]
            np.add.at(S[:, dim, dim2], rows, vals)
        vals_b = ws * df * dx[:, dim]
        np.add.at(b[:, dim], rows, vals_b)

    tr = np.trace(S, axis1=1, axis2=2)
    pos_tr = tr[tr > 0]
    fallback = float(np.median(pos_tr)) / n if pos_tr.size > 0 else 1.0
    ridge = np.maximum(lam * tr / n, 1e-12 * fallback)
    S += ridge[:, None, None] * np.eye(n)

    grad = np.linalg.solve(S, b)
    return grad


def _edge_divergence_pts(points, X, W):
    """Edge-compatible divergence of vector field X using spatial edges."""
    N = len(points)
    coo = W.tocoo()
    rows, cols = coo.row, coo.col
    ws = np.asarray(coo.data, dtype=float)
    edge_vec = points[cols] - points[rows]
    X_avg = 0.5 * (X[rows] + X[cols])
    X_edge = np.einsum("ij,ij->i", X_avg, edge_vec)
    div = np.zeros(N)
    np.add.at(div, rows, ws * X_edge)
    return div


def _anchor_components(L, rhs, W, anchor_index):
    """Pin one node per connected component to make Poisson non-singular."""
    n_comp, labels = connected_components(W, directed=False, connection="weak")
    L = L.tolil(copy=True)
    rhs = np.array(rhs, copy=True)
    L.rows[anchor_index] = [anchor_index]
    L.data[anchor_index] = [1.0]
    rhs[anchor_index] = 0.0
    if n_comp > 1:
        src_label = labels[anchor_index]
        for c in range(n_comp):
            if c == src_label:
                continue
            idx = int(np.flatnonzero(labels == c)[0])
            L.rows[idx] = [idx]
            L.data[idx] = [1.0]
            rhs[idx] = 0.0
    return L.tocsr(), rhs


def generic_heat_method(
    points, L, M, W, source_index, *, t_scale=15.0, gradient_eps=1e-12,
):
    """Three-step heat method with an arbitrary graph Laplacian."""
    N = len(points)
    src = np.atleast_1d(np.asarray(source_index, dtype=int))

    # Crane et al. 2013 / geometry-central: RHS is the indicator vector
    # e_i (Dirac delta as FEM functional).
    rhs = np.zeros(N)
    rhs[src] = 1.0

    t = _estimate_time_step_pts(points, W, t_scale)

    # Step 1: Heat diffusion  (M + t L) u = delta
    A = M + float(t) * L
    u = spla.spsolve(A.tocsr(), rhs)

    # Step 2: Normalised gradient
    grad_u = _edge_gradient(points, u, W)
    norms = np.clip(np.linalg.norm(grad_u, axis=1), gradient_eps, None)
    X = -grad_u / norms[:, None]

    # Step 3: Poisson solve  L phi = -div(X)
    div_X = _edge_divergence_pts(points, X, W)
    L_p, rhs_p = _anchor_components(L, -div_X, W, int(src[0]))
    phi = spla.spsolve(L_p, rhs_p)

    return phi, u


# -- Colour helpers ----------------------------------------------------------
def _dist_to_colors(dist, u, n_points, n_bands=15):
    """Compute smooth and band colors from distances + heat field."""
    u_thresh = np.max(np.abs(u)) * 1e-6
    mask = np.abs(u) > u_thresh

    dist_reach = dist[mask].copy()
    if dist_reach.size > 0 and dist_reach.max() > 0:
        p99 = float(np.percentile(dist_reach[dist_reach > 0], 99))
        dist_reach = np.minimum(dist_reach, p99)

    cmap = colormaps["magma"]
    far_color = np.array(cmap(1.0)[:3])  # pale yellow at the end of the colormap
    dmax = float(dist_reach.max()) if dist_reach.size > 0 and dist_reach.max() > 0 else 1.0

    smooth = np.tile(far_color, (n_points, 1))
    if dist_reach.size > 0:
        smooth[mask] = cmap(dist_reach / dmax)[:, :3]

    bands = np.tile(far_color, (n_points, 1))
    if dist_reach.size > 0:
        bw = dmax / n_bands
        b = np.clip(np.floor(dist_reach / bw).astype(int), 0, n_bands - 1)
        base = cmap(b / (n_bands - 1))[:, :3]
        darken = np.where(b % 2 == 1, 0.4, 1.0)[:, None]
        bands[mask] = base * darken

    return smooth, bands


# -- Per-method geodesic computation -----------------------------------------
def _compute_lifted(level, points, source_index, *, k, sigma_x, sigma_u,
                    t_scale, uniform_regression=True, diffusion_steps=1):
    print(f"    lifted: sigma_x={sigma_x}, sigma_u={sigma_u}, "
          f"uniform_reg={uniform_regression}, steps={diffusion_steps}")
    dist, u, _, _ = lifted_heat_method(
        level, source_index=source_index, k=k,
        kernel="product", sigma_x=sigma_x, sigma_u=sigma_u,
        t_scale=t_scale, return_intermediate=True,
        uniform_regression=uniform_regression,
        diffusion_steps=diffusion_steps,
    )
    dist = np.maximum(dist - dist[source_index], 0.0)
    return dist, u


def _compute_robust(points, source_index, *, k, t_scale):
    L, M = robust_laplacian.point_cloud_laplacian(points, n_neighbors=k)
    # Build a k-NN affinity for edge structure (robust_laplacian doesn't expose W)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k + 1, len(points)))
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(len(points)), idx.shape[1])
    cols = idx.ravel()
    vals = np.ones(len(rows))
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(points), len(points)))
    W = W + W.T
    dist, u = generic_heat_method(points, L, M, W, source_index, t_scale=t_scale)
    dist = np.maximum(dist - dist[source_index], 0.0)
    return dist, u


# -- Display -----------------------------------------------------------------
def _compute_and_display(
    method: str,
    level, points, source_index, state,
    *, k, sigma_x, sigma_u, t_scale, method_points,
    uniform_regression=True, diffusion_steps=1,
):
    """Compute geodesics for one method and update its point cloud."""
    print(f"  [{method}] computing...")
    if method == "lifted":
        dist, u = _compute_lifted(
            level, points, source_index,
            k=k, sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
            uniform_regression=uniform_regression,
            diffusion_steps=diffusion_steps,
        )
    elif method == "robust":
        dist, u = _compute_robust(
            points, source_index, k=k, t_scale=t_scale,
        )
    else:
        raise ValueError(method)

    smooth, bands = _dist_to_colors(dist, u, len(points))

    state[f"{method}_smooth"] = smooth
    state[f"{method}_bands"] = bands

    cloud = ps.get_point_cloud(method)
    cloud.add_color_quantity("geodesic", smooth, enabled=True)

    # Source marker
    src_name = f"source_{method}"
    if ps.has_point_cloud(src_name):
        ps.remove_point_cloud(src_name)
    src_pt = method_points[method][source_index : source_index + 1]
    src_cloud = ps.register_point_cloud(src_name, src_pt, radius=0.005)
    src_cloud.set_color((0.0, 1.0, 1.0))

    print(f"  [{method}] done")


# -- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Geodesic comparison on point clouds")
    parser.add_argument(
        "--pointcloud", "-p", type=str, default=None,
        help="Name or path of the point cloud (.npz). Omit to list available.",
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--levels", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Blow-up levels to show: 1, 2, or both.")
    parser.add_argument("--spread", type=str, default="partial",
                        choices=["partial", "full", "both"],
                        help="Which rows to show: partial, full, or both.")
    parser.add_argument("--sigma-x", type=float, default=0.5)
    parser.add_argument("--sigma-u", type=float, default=0.5)
    parser.add_argument("--t-scale-lifted", type=float, default=1.0,
                        help="Time-scale multiplier for the lifted method.")
    parser.add_argument("--t-scale-robust", type=float, default=1.0,
                        help="Time-scale multiplier for the robust method.")
    parser.add_argument("--partial-steps", type=int, default=1,
                        help="Diffusion steps for partial spread (top row).")
    parser.add_argument("--full-steps", type=int, default=20,
                        help="Diffusion steps for full spread (bottom row).")
    parser.add_argument(
        "--rotate", "-r", type=str, default=None,
        help="Rotation: 'x', 'y', or 'ANGLE,AX,AY,AZ'",
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="If set, uniformly downsample to at most this many points.",
    )
    parser.add_argument(
        "--downsample-seed", type=int, default=None,
        help="Optional RNG seed for downsampling.",
    )
    args = parser.parse_args()

    if args.pointcloud is None:
        avail = _available_pointclouds()
        if not avail:
            print("No point clouds found. Run mesh_to_pc.py first.")
            return
        print("Available point clouds:")
        for name, path in avail.items():
            print(f"  {name}  ({path.parent.name})")
        print("\nUsage: python thingi10k.py --pointcloud <name>")
        return

    data_path = _resolve_pointcloud(args.pointcloud)
    k = args.k
    alpha = args.alpha
    sigma_x = args.sigma_x
    sigma_u = args.sigma_u

    points, normals = _load_points_and_normals(data_path)
    n_original = points.shape[0]
    print(f"Loaded {n_original} points from {data_path.name}")

    if args.max_points is not None and args.max_points > 0 and n_original > args.max_points:
        rng = np.random.default_rng(args.downsample_seed)
        idx = rng.choice(n_original, size=args.max_points, replace=False)
        idx.sort()
        points = points[idx]
        normals = normals[idx]
        print(f"Downsampled {n_original} -> {len(points)} points")

    if args.rotate is not None:
        R = _parse_rotation(args.rotate)
        points = points @ R.T
        normals = normals @ R.T
        print(f"Applied rotation: --rotate {args.rotate}")

    # -- Build levels --
    level1 = level2 = None
    if args.levels in ("1", "both"):
        level1 = _build_level(points, normals, alpha=alpha, k=k, n_lifts=1)
        print(f"Built level-1: N={level1.N}, D={level1.D}, d={level1.d}")
    if args.levels in ("2", "both"):
        level2 = _build_level(points, normals, alpha=alpha, k=k, n_lifts=2)
        print(f"Built level-2: N={level2.N}, D={level2.D}, d={level2.d}")

    # -- Build panel grid --
    columns = []
    if level1 is not None:
        columns.append(("lift1", level1))
    if level2 is not None:
        columns.append(("lift2", level2))
    columns.append(("robust", None))

    rows = []
    if args.spread in ("partial", "both"):
        rows.append(("partial", args.partial_steps))
    if args.spread in ("full", "both"):
        rows.append(("full", args.full_steps))

    # Panel names: "row/col"
    all_panels = []
    for row_name, _ in rows:
        for col_name, _ in columns:
            all_panels.append(f"{row_name}/{col_name}")

    n_cols = len(columns)
    n_rows = len(rows)

    # Spacing
    bbox_x = points[:, 0].max() - points[:, 0].min()
    bbox_z = points[:, 2].max() - points[:, 2].min()
    x_spacing = bbox_x * 1.5
    y_spacing = max(bbox_z, bbox_x) * 1.5

    # Place panels in a grid
    panel_offsets = {}
    for ri, (row_name, _) in enumerate(rows):
        for ci, (col_name, _) in enumerate(columns):
            pname = f"{row_name}/{col_name}"
            panel_offsets[pname] = np.array([ci * x_spacing, -ri * y_spacing, 0.0])

    # -- Mutable UI state --
    state = {
        "source_index": None,
        "computing": False,
        "show_bands": False,
        "t_scale_lifted": args.t_scale_lifted,
        "t_scale_robust": args.t_scale_robust,
        "uniform_regression": False,
        "partial_steps": args.partial_steps,
        "full_steps": args.full_steps,
    }
    for p in all_panels:
        state[f"{p}_smooth"] = None
        state[f"{p}_bands"] = None

    def _compute_panel(panel_name, source_index):
        row_name, col_name = panel_name.split("/")
        diff_steps = state["partial_steps"] if row_name == "partial" else state["full_steps"]
        _, level = [(c, l) for c, l in columns if c == col_name][0]

        if col_name == "robust":
            t_s = state["t_scale_robust"]
            dist, u = _compute_robust(points, source_index, k=k, t_scale=t_s)
        else:
            t_s = state["t_scale_lifted"]
            dist, u = _compute_lifted(
                level, points, source_index,
                k=k, sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_s,
                uniform_regression=state["uniform_regression"],
                diffusion_steps=diff_steps,
            )

        smooth, bands = _dist_to_colors(dist, u, len(points))
        state[f"{panel_name}_smooth"] = smooth
        state[f"{panel_name}_bands"] = bands

        cloud = ps.get_point_cloud(panel_name)
        cloud.add_color_quantity("geodesic", smooth, enabled=True)

        # Source marker
        src_tag = f"src_{panel_name.replace('/', '_')}"
        if ps.has_point_cloud(src_tag):
            ps.remove_point_cloud(src_tag)
        src_pt = (points[source_index:source_index + 1]
                  + panel_offsets[panel_name])
        sc = ps.register_point_cloud(src_tag, src_pt, radius=0.005)
        sc.set_color((0.0, 1.0, 1.0))

    def callback():
        psim.TextUnformatted("Ctrl+click any copy to select source")
        psim.Separator()

        pick = ps.get_selection()
        have = pick.is_hit and pick.structure_name in all_panels

        if have:
            picked = pick.local_index
            psim.TextUnformatted(f"Selected point: {picked}  "
                                 f"({points[picked].round(2)})")
        elif state["source_index"] is not None:
            idx = state["source_index"]
            psim.TextUnformatted(f"Source: {idx}  ({points[idx].round(2)})")

        _, state["t_scale_lifted"] = psim.InputFloat(
            "t_scale (lifted)", state["t_scale_lifted"],
        )
        _, state["t_scale_robust"] = psim.InputFloat(
            "t_scale (robust)", state["t_scale_robust"],
        )
        _, state["uniform_regression"] = psim.Checkbox(
            "Uniform regression (w=1)", state["uniform_regression"],
        )
        if any(r == "partial" for r, _ in rows):
            _, state["partial_steps"] = psim.InputInt(
                "Partial steps", state["partial_steps"],
            )
        if any(r == "full" for r, _ in rows):
            _, state["full_steps"] = psim.InputInt(
                "Full steps", state["full_steps"],
            )

        if have:
            if psim.Button("Compute geodesics"):
                state["source_index"] = pick.local_index
                state["computing"] = True

        # Level-set toggle
        has_results = any(state[f"{p}_smooth"] is not None for p in all_panels)
        if has_results:
            changed, new_val = psim.Checkbox("Level sets", state["show_bands"])
            if changed:
                state["show_bands"] = new_val
                for p in all_panels:
                    sm = state[f"{p}_smooth"]
                    bd = state[f"{p}_bands"]
                    if sm is not None:
                        c = bd if new_val else sm
                        ps.get_point_cloud(p).add_color_quantity(
                            "geodesic", c, enabled=True,
                        )

        if state["computing"]:
            state["computing"] = False
            idx = state["source_index"]
            print(f"Computing geodesics from point {idx}...")
            for p in all_panels:
                print(f"  [{p}] computing...")
                _compute_panel(p, idx)
                print(f"  [{p}] done")

        if psim.Button("Screenshot"):
            fname = f"geodesic_{args.pointcloud}.png"
            ps.screenshot(fname, transparent_bg=False)
            print(f"Saved {fname}")

    # -- Polyscope setup --
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    for p in all_panels:
        cloud = ps.register_point_cloud(
            p, points + panel_offsets[p], radius=0.001,
        )
        cloud.set_color((0.7, 0.7, 0.7))

    col_labels = [c for c, _ in columns]
    row_labels = [f"{r} ({s} steps)" for r, s in rows]
    print(f"Grid: {' | '.join(col_labels)}  x  {' / '.join(row_labels)}")
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
