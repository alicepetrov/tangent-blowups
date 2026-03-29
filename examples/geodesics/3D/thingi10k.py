"""
Thingi10k Geodesic Comparison
-----------------------------
Side-by-side comparison of geodesic distances:

1. **Lifted** (ours) -- product-kernel heat method on the level-1 blow-up
2. **Robust** -- nonmanifold Laplacian from the ``robust_laplacian`` library

Usage:
    python thingi10k.py -p ship --rotate y
    python thingi10k.py -p pan_pipes
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
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "thingi10k_pointcloud"


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
def _available_pointclouds() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.npz"))


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


def _build_level(points, normals, *, alpha, k):
    frames = _normals_to_tangent_frames(normals)
    level0 = BlowUpLevel.from_point_tangents(points, frames)
    return level0.lift(alpha=alpha, k=k, lam=1e-3)


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


def _edge_gradient(points, f, W, lam=1e-3):
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
    ridge = np.where(tr > 0, lam * tr / n, lam)
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

    M_diag = np.asarray(M.diagonal()).ravel() if sparse.issparse(M) else np.diag(M)
    rhs = np.zeros(N)
    rhs[src] = 1.0 / M_diag[src]

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
    dmax = float(dist_reach.max()) if dist_reach.size > 0 and dist_reach.max() > 0 else 1.0

    smooth = np.full((n_points, 3), 0.7)
    if dist_reach.size > 0:
        smooth[mask] = cmap(dist_reach / dmax)[:, :3]

    bands = np.full((n_points, 3), 0.7)
    if dist_reach.size > 0:
        bw = dmax / n_bands
        b = np.clip(np.floor(dist_reach / bw).astype(int), 0, n_bands - 1)
        base = cmap(b / (n_bands - 1))[:, :3]
        darken = np.where(b % 2 == 1, 0.4, 1.0)[:, None]
        bands[mask] = base * darken

    return smooth, bands


# -- Per-method geodesic computation -----------------------------------------
def _compute_lifted(level, points, source_index, *, k, sigma_x, sigma_u, t_scale):
    print(f"    lifted bandwidths: sigma_x={sigma_x}, sigma_u={sigma_u}")
    dist, u, _, _ = lifted_heat_method(
        level, source_index=source_index, k=k,
        kernel="product", sigma_x=sigma_x, sigma_u=sigma_u,
        t_scale=t_scale, return_intermediate=True,
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
):
    """Compute geodesics for one method and update its point cloud."""
    print(f"  [{method}] computing...")
    if method == "lifted":
        dist, u = _compute_lifted(
            level, points, source_index,
            k=k, sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
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
    parser = argparse.ArgumentParser(description="Thingi10k geodesic comparison")
    parser.add_argument(
        "--pointcloud", "-p", type=str, default=None,
        help="Name of the point cloud (without .npz extension)",
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--sigma-x", type=float, default=0.5)
    parser.add_argument("--sigma-u", type=float, default=0.5)
    parser.add_argument("--t-scale", type=float, default=50.0)
    parser.add_argument(
        "--rotate", "-r", type=str, default=None,
        help="Rotation: 'x', 'y', or 'ANGLE,AX,AY,AZ'",
    )
    args = parser.parse_args()

    available = _available_pointclouds()
    if args.pointcloud is None:
        print("Available point clouds:")
        for name in available:
            print(f"  {name}")
        print("\nUsage: python thingi10k.py --pointcloud <name>")
        return

    if args.pointcloud not in available:
        print(f"Unknown: {args.pointcloud!r}. Available: {', '.join(available)}")
        return

    data_path = DATA_DIR / f"{args.pointcloud}.npz"
    k = args.k
    alpha = args.alpha
    t_scale = args.t_scale
    sigma_x = args.sigma_x
    sigma_u = args.sigma_u

    points, normals = _load_points_and_normals(data_path)
    print(f"Loaded {points.shape[0]} points from {data_path.name}")

    if args.rotate is not None:
        R = _parse_rotation(args.rotate)
        points = points @ R.T
        normals = normals @ R.T
        print(f"Applied rotation: --rotate {args.rotate}")

    level = _build_level(points, normals, alpha=alpha, k=k)
    print(f"Built level-1 BlowUpLevel: N={level.N}, D={level.D}, d={level.d}")

    # Spacing between copies
    bbox_x = points[:, 0].max() - points[:, 0].min()
    spacing = bbox_x * 3

    # Pre-compute positioned points per method (offset along x).
    method_points = {}
    for i, method in enumerate(METHODS):
        offset = np.array([i * spacing, 0.0, 0.0])
        method_points[method] = points + offset

    # -- Mutable UI state --
    state = {
        "source_index": None,
        "computing": False,
        "show_bands": False,
    }
    for m in METHODS:
        state[f"{m}_smooth"] = None
        state[f"{m}_bands"] = None

    def callback():
        psim.TextUnformatted("Ctrl+click any copy to select source")
        psim.Separator()

        pick = ps.get_selection()
        have = pick.is_hit and pick.structure_name in METHODS

        if have:
            picked = pick.local_index
            psim.TextUnformatted(f"Selected point: {picked}  "
                                 f"({points[picked].round(2)})")
        elif state["source_index"] is not None:
            idx = state["source_index"]
            psim.TextUnformatted(f"Source: {idx}  ({points[idx].round(2)})")

        if have:
            if psim.Button("Compute geodesics"):
                state["source_index"] = pick.local_index
                state["computing"] = True

        # Level-set toggle
        has_results = any(state[f"{m}_smooth"] is not None for m in METHODS)
        if has_results:
            changed, new_val = psim.Checkbox("Level sets", state["show_bands"])
            if changed:
                state["show_bands"] = new_val
                for m in METHODS:
                    sm = state[f"{m}_smooth"]
                    bd = state[f"{m}_bands"]
                    if sm is not None:
                        c = bd if new_val else sm
                        ps.get_point_cloud(m).add_color_quantity(
                            "geodesic", c, enabled=True,
                        )

        if state["computing"]:
            state["computing"] = False
            idx = state["source_index"]
            print(f"Computing geodesics from point {idx}...")
            for m in METHODS:
                _compute_and_display(
                    m, level, points, idx, state,
                    k=k, sigma_x=sigma_x, sigma_u=sigma_u,
                    t_scale=t_scale, method_points=method_points,
                )

    # -- Polyscope setup --
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    for method in METHODS:
        cloud = ps.register_point_cloud(method, method_points[method], radius=0.001)
        cloud.set_color((0.7, 0.7, 0.7))

    print(f"Copies placed side-by-side: {' | '.join(METHODS)}")
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
