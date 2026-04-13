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
from tangent_blowups.pointcloud import lifted_heat_method, precompute_heat_method
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.geometry.kernels import estimate_product_bandwidths
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.solvers.linalg import normalize_vectors

# -- Constants ---------------------------------------------------------------
METHODS = ["lifted", "robust"]
_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


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
    if _DATA_ROOT.is_dir():
        for p in _DATA_ROOT.rglob("*.npz"):
            result.setdefault(p.stem, p)
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
    diffusion_steps=1,
):
    """Three-step heat method with an arbitrary graph Laplacian."""
    N = len(points)
    src = np.atleast_1d(np.asarray(source_index, dtype=int))

    # Crane et al. 2013 / geometry-central: RHS is the indicator vector
    # e_i (Dirac delta as FEM functional).
    rhs = np.zeros(N)
    rhs[src] = 1.0

    t = _estimate_time_step_pts(points, W, t_scale)

    # Step 1: Heat diffusion  (M + t L)^n u = delta
    A = M + float(t) * L
    u = rhs.copy()
    for _ in range(max(diffusion_steps, 1)):
        u = spla.spsolve(A.tocsr(), u)

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


_HEAT_CMAPS = ["inferno", "magma", "viridis", "plasma", "coolwarm", "hot", "YlOrRd"]


def _heat_to_colors(u, n_points, cmap_name="inferno"):
    """Colormap the raw heat field u (log-scale for visibility)."""
    cmap = colormaps[cmap_name]
    u_abs = np.abs(u)
    u_max = u_abs.max()
    if u_max < 1e-30:
        return np.tile(np.array(cmap(0.0)[:3]), (n_points, 1))
    # Log-scale: map log(u/u_max) to [0, 1] over ~6 decades
    log_u = np.log10(np.clip(u_abs / u_max, 1e-6, 1.0))
    t = (log_u + 6.0) / 6.0  # -6 -> 0, 0 -> 1
    t = np.clip(t, 0.0, 1.0)
    colors = cmap(t)[:, :3]
    return colors


# -- Export helpers ----------------------------------------------------------
def _export_ply(pointcloud_name, pts, normals, panel_name, colors_rgb):
    """Save a coloured point cloud as ASCII PLY."""
    N = len(pts)
    tag = panel_name.replace("/", "_")
    fname = f"{pointcloud_name}_{tag}.ply"
    rgb = (np.clip(colors_rgb, 0.0, 1.0) * 255).astype(np.uint8)
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


def _export_colourbar(pointcloud_name, panel_name, *, vmin, vmax,
                      cmap_name, label, log_scale=False):
    """Save a standalone colour bar PDF."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = plt.get_cmap(cmap_name)
    if log_scale:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-30), vmax=max(vmax, 1e-30))
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(4.5, 0.35))
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation="horizontal",
    )
    cb.set_label(label)

    tag = panel_name.replace("/", "_")
    fname = f"colourbar_{pointcloud_name}_{tag}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# -- Per-method geodesic computation -----------------------------------------
def _compute_lifted(level, points, source_index, *, k, sigma_x, sigma_u,
                    t_scale, uniform_regression=True, diffusion_steps=1,
                    device=None, _precomputed=None):
    print(f"    lifted: sigma_x={sigma_x}, sigma_u={sigma_u}, "
          f"uniform_reg={uniform_regression}, steps={diffusion_steps}")
    dist, u, _, _ = lifted_heat_method(
        level, source_index=source_index, k=k,
        kernel="product", sigma_x=sigma_x, sigma_u=sigma_u,
        t_scale=t_scale, return_intermediate=True,
        uniform_regression=uniform_regression,
        diffusion_steps=diffusion_steps,
        device=device,
        _precomputed=_precomputed,
    )
    dist = np.maximum(dist - dist[source_index], 0.0)
    return dist, u


def _compute_robust(points, source_index, *, k, t_scale, diffusion_steps=1):
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
    dist, u = generic_heat_method(
        points, L, M, W, source_index,
        t_scale=t_scale, diffusion_steps=diffusion_steps,
    )
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
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--levels", type=str, default="1",
                        choices=["1", "2", "both"],
                        help="Blow-up levels to show: 1, 2, or both.")
    parser.add_argument("--spread", type=str, default="geodesic",
                        choices=["geodesic", "heat", "both"],
                        help="Which rows to show: geodesic, heat, or both.")
    parser.add_argument("--sigma-x", type=float, default=None,
                        help="Spatial bandwidth (None = auto-estimate).")
    parser.add_argument("--sigma-u", type=float, default=None,
                        help="Angular bandwidth (None = auto-estimate).")
    parser.add_argument("--t-scale-lifted", type=float, default=1.0,
                        help="Time-scale multiplier for the lifted method.")
    parser.add_argument("--t-scale-robust", type=float, default=1.0,
                        help="Time-scale multiplier for the robust method.")
    parser.add_argument("--steps", type=int, default=1,
                        help="Initial diffusion steps per method.")
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
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Compute device: 'cuda' for GPU, 'cpu' for CPU, or omit for auto-detect.",
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

    # -- Per-column parameters --
    # Each column (lift1, lift2, robust) gets its own t_scale, alpha,
    # and bandwidths.  Alpha changes require rebuilding the BlowUpLevel.
    _BW_NAMES = ["tangent", "curvature", "d-curvature"]
    _N_LIFTS = {"lift1": 1, "lift2": 2}

    def _estimate_bandwidths(lev):
        """Auto-estimate bandwidths for a level, respecting CLI overrides."""
        if sigma_x is not None and sigma_u is not None:
            return sigma_x, [sigma_u] * lev.level
        auto_sx, auto_su = estimate_product_bandwidths(lev, k)
        sx = sigma_x if sigma_x is not None else auto_sx
        su = [sigma_u] * lev.level if sigma_u is not None else auto_su
        return sx, su

    def _print_params(col_name, cp):
        su_parts = []
        for i, s in enumerate(cp.get("sigma_u", [])):
            name = _BW_NAMES[i] if i < len(_BW_NAMES) else f"level-{i+1}"
            su_parts.append(f"{name}={s:.4f}")
        bw_str = f"  spatial={cp['sigma_x']:.4f}  {', '.join(su_parts)}" if su_parts else ""
        print(f"Params [{col_name}]: t_scale={cp['t_scale']}  "
              f"steps={cp['steps']}{bw_str}")

    col_params = {}
    for col_name, lev in [("lift1", level1), ("lift2", level2)]:
        if lev is None:
            continue
        auto_sx, auto_su = _estimate_bandwidths(lev)
        col_params[col_name] = {
            "t_scale": args.t_scale_lifted,
            "steps": args.steps,
            "sigma_x": auto_sx,
            "sigma_u": auto_su,
        }
        _print_params(col_name, col_params[col_name])
    col_params["robust"] = {
        "t_scale": args.t_scale_robust,
        "steps": args.steps,
    }

    # -- Build panel grid --
    # Levels are stored in a mutable dict so they can be rebuilt when
    # alpha changes.
    levels = {}
    columns = []
    if level1 is not None:
        levels["lift1"] = level1
        columns.append(("lift1", level1))
    if level2 is not None:
        levels["lift2"] = level2
        columns.append(("lift2", level2))
    levels["robust"] = None
    columns.append(("robust", None))

    rows = []
    if args.spread in ("geodesic", "both"):
        rows.append("geodesic")
    if args.spread in ("heat", "both"):
        rows.append("heat")

    # Panel names: "row/col"
    all_panels = []
    for row_name in rows:
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
    for ri, row_name in enumerate(rows):
        for ci, (col_name, _) in enumerate(columns):
            pname = f"{row_name}/{col_name}"
            panel_offsets[pname] = np.array([ci * x_spacing, -ri * y_spacing, 0.0])

    # -- Mutable UI state --
    state = {
        "source_index": None,
        "computing": False,
        "show_bands": False,
        "uniform_regression": False,
        "col": col_params,
        "heat_cmap_idx": 0,
    }
    for p in all_panels:
        state[f"{p}_smooth"] = None
        state[f"{p}_bands"] = None

    # Precomputed cache per lifted column (built on first click)
    _cache = {}

    def _get_or_build_cache(col_name, level):
        """Build the precomputed heat-method cache once per lifted column."""
        cp = state["col"][col_name]
        cache_key = (col_name, cp["t_scale"],
                     state["uniform_regression"],
                     cp["sigma_x"], tuple(cp["sigma_u"]))
        if cache_key not in _cache:
            print(f"    [{col_name}] building cache "
                  f"(t_scale={cp['t_scale']}, "
                  f"sigma_x={cp['sigma_x']:.4f}, "
                  f"sigma_u={[round(s,4) for s in cp['sigma_u']]})...")
            _cache.clear()
            _cache[cache_key] = precompute_heat_method(
                level, k=k,
                sigma_x=cp["sigma_x"],
                sigma_u=cp["sigma_u"],
                t_scale=cp["t_scale"],
                uniform_regression=state["uniform_regression"],
                device=args.device,
            )
            print(f"    [{col_name}] cache ready")
        return _cache[cache_key]

    def _compute_panel(panel_name, source_index):
        row_name, col_name = panel_name.split("/")
        level = levels[col_name]

        cp = state["col"][col_name]
        print(f"    [{col_name}] steps={cp['steps']}")
        if col_name == "robust":
            dist, u = _compute_robust(
                points, source_index, k=k, t_scale=cp["t_scale"],
                diffusion_steps=cp["steps"],
            )
        else:
            pre = _get_or_build_cache(col_name, level)
            dist, u = _compute_lifted(
                level, points, source_index,
                k=k, sigma_x=cp["sigma_x"],
                sigma_u=cp["sigma_u"],
                t_scale=cp["t_scale"],
                uniform_regression=state["uniform_regression"],
                diffusion_steps=cp["steps"],
                device=args.device,
                _precomputed=pre,
            )

        state[f"{panel_name}_dist"] = dist
        state[f"{panel_name}_u"] = u
        if row_name == "heat":
            cmap_name = _HEAT_CMAPS[state["heat_cmap_idx"]]
            colors = _heat_to_colors(u, len(points), cmap_name)
            state[f"{panel_name}_smooth"] = colors
            state[f"{panel_name}_bands"] = colors
        else:
            smooth, bands = _dist_to_colors(dist, u, len(points))
            state[f"{panel_name}_smooth"] = smooth
            state[f"{panel_name}_bands"] = bands
            colors = smooth

        cloud = ps.get_point_cloud(panel_name)
        cloud.add_color_quantity("geodesic", colors, enabled=True)

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

        _, state["uniform_regression"] = psim.Checkbox(
            "Uniform regression (w=1)", state["uniform_regression"],
        )

        # -- Per-column controls (steps, t_scale, bandwidths) --
        _ANGULAR_LABELS = ["tangent", "curvature", "d-curvature"]
        psim.Separator()
        for col_name, _ in columns:
            cp = state["col"][col_name]
            if psim.TreeNode(col_name):
                _, cp["steps"] = psim.InputInt(
                    f"steps##{col_name}", cp["steps"],
                )
                _, cp["t_scale"] = psim.InputFloat(
                    f"t_scale##{col_name}", cp["t_scale"],
                )
                if col_name != "robust":
                    changed_sx, new_sx = psim.InputFloat(
                        f"spatial##{col_name}", cp["sigma_x"],
                    )
                    if changed_sx and new_sx > 0:
                        cp["sigma_x"] = new_sx
                    for i, s in enumerate(cp["sigma_u"]):
                        label = _ANGULAR_LABELS[i] if i < len(_ANGULAR_LABELS) else f"level-{i+1}"
                        changed_su, new_su = psim.InputFloat(
                            f"{label}##{col_name}", s,
                        )
                        if changed_su and new_su > 0:
                            cp["sigma_u"][i] = new_su
                psim.TreePop()
        psim.Separator()

        if have:
            state["source_index"] = pick.local_index
        if state["source_index"] is not None:
            if psim.Button("Compute"):
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

        # Heat colormap selector
        heat_panels = [p for p in all_panels if p.startswith("heat/")]
        has_heat = any(state.get(f"{p}_u") is not None for p in heat_panels)
        if heat_panels and has_heat:
            changed_cm, new_idx = psim.Combo(
                "Heat colormap", state["heat_cmap_idx"], _HEAT_CMAPS,
            )
            if changed_cm:
                state["heat_cmap_idx"] = new_idx
                cmap_name = _HEAT_CMAPS[new_idx]
                for p in heat_panels:
                    u_cached = state.get(f"{p}_u")
                    if u_cached is not None:
                        colors = _heat_to_colors(u_cached, len(points), cmap_name)
                        state[f"{p}_smooth"] = colors
                        state[f"{p}_bands"] = colors
                        ps.get_point_cloud(p).add_color_quantity(
                            "geodesic", colors, enabled=True,
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

        if has_results and psim.Button("Export PLY + colorbar"):
            for p in all_panels:
                sm = state[f"{p}_smooth"]
                if sm is None:
                    continue
                _export_ply(args.pointcloud, points, normals, p, sm)
                row_name = p.split("/")[0]
                if row_name == "heat":
                    u_p = state.get(f"{p}_u")
                    if u_p is not None:
                        u_abs = np.abs(u_p)
                        _export_colourbar(
                            args.pointcloud, p,
                            vmin=u_abs[u_abs > 0].min() if (u_abs > 0).any() else 1e-6,
                            vmax=u_abs.max(),
                            cmap_name=_HEAT_CMAPS[state["heat_cmap_idx"]],
                            label="Heat $u$",
                            log_scale=True,
                        )
                else:
                    dist_p = state.get(f"{p}_dist")
                    if dist_p is not None:
                        _export_colourbar(
                            args.pointcloud, p,
                            vmin=0.0, vmax=float(np.percentile(
                                dist_p[dist_p > 0], 99)) if (dist_p > 0).any() else 1.0,
                            cmap_name="magma",
                            label="Geodesic distance",
                        )

    # -- Polyscope setup --
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    for p in all_panels:
        cloud = ps.register_point_cloud(
            p, points + panel_offsets[p], radius=0.001,
        )
        cloud.set_color((0.7, 0.7, 0.7))

    col_labels = [c for c, _ in columns]
    print(f"Grid: {' | '.join(col_labels)}  x  {' / '.join(rows)}")
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
