"""
Scalar Field Bilateral Comparison
----------------------------------
Demonstrates the advantage of the lifted Laplacian (tangent-projector-based)
over a bilateral Laplacian (normal-based) and a robust Laplacian (purely
spatial) for geodesic distance computation:

1. **Thin tube**: A thin cylindrical shell ~3 layers thick with inward
   normals on the inner wall and outward normals on the outer wall.  The
   bilateral kernel splits inner/outer walls; the lifted kernel uses
   projectors (P = UU^T) which are orientation-invariant.

2. **Figure-8 Mobius band**: A self-intersecting, non-orientable band whose
   medial axis traces a figure-8 (lemniscate).  The normal-flip seam is
   offset from the crossing so the two failure modes are visually distinct.
   The bilateral kernel fails at both; the lifted kernel handles both.

Three copies are shown side by side: bilateral (left), robust (centre),
lifted (right).  Sources are auto-selected; Ctrl+click to override.
"""
from __future__ import annotations

import argparse
from math import gamma as math_gamma

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import robust_laplacian
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import connected_components

from tangent_blowups.pointcloud import (
    lifted_heat_method, bilateral_pointcloud_laplacian,
)
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import product_affinity, affinity_to_laplacian
from tangent_blowups.solvers.linalg import normalize_vectors

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K = 30
ALPHA = 1.0
SIGMA_X = 0.5
SIGMA_U = 0.5
SIGMA_X_BI = 0.5
SIGMA_N_BI = 0.5
T_SCALE = 1000.0

METHODS = ["bilateral", "lifted", "robust"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _nearest_index(points: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(points - target, axis=1)))


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def _make_thin_tube(
    n_theta: int = 80,
    n_z: int = 30,
    n_layers: int = 3,
    radius: float = 0.5,
    height: float = 2.0,
    thickness: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Thin cylindrical shell with inward/outward normals.

    Returns (points, normals, src_a_index, src_b_index).
    Source A near the top, source B near the bottom.  The bilateral kernel
    splits inner/outer walls; lifted correctly partitions by distance.
    """
    rng = np.random.default_rng(seed)

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0, height, n_z)
    r_layers = np.linspace(radius - thickness / 2, radius + thickness / 2, n_layers)

    all_pts = []
    all_normals = []
    for r in r_layers:
        TT, ZZ = np.meshgrid(theta, z)
        TT, ZZ = TT.ravel(), ZZ.ravel()
        n_grid = len(TT)
        # Small jitter
        TT = TT + rng.normal(0, 0.005, n_grid)
        ZZ = ZZ + rng.normal(0, 0.005, n_grid)
        X = r * np.cos(TT)
        Y = r * np.sin(TT)
        pts = np.column_stack([X, Y, ZZ])

        # Normal: radially outward for outer layers, inward for inner
        sign = 1.0 if r >= radius else -1.0
        normals = np.column_stack([
            sign * np.cos(TT),
            sign * np.sin(TT),
            np.zeros(n_grid),
        ])
        all_pts.append(pts)
        all_normals.append(normals)

    points = np.concatenate(all_pts, axis=0)
    normals = np.concatenate(all_normals, axis=0)

    # Auto-select: A near top of outer wall, B near bottom of outer wall
    outer_mask = normals[:, 0]**2 + normals[:, 1]**2 > 0.5  # all points qualify
    src_a = _nearest_index(points, np.array([radius, 0.0, height]))
    src_b = _nearest_index(points, np.array([radius, 0.0, 0.0]))

    return points, normals, int(src_a), int(src_b)


def _figure8_evaluate(tt, vv, scale, twist_phase=0.0):
    """Evaluate figure-8 Mobius band position and tangent vectors.

    Centerline: lemniscate of Gerono  r(t) = (a sin t, a sin 2t / 2, 0).
    Sweep direction with Mobius half-twist, offset by ``twist_phase`` so
    the normal-flip seam is separated from the self-intersection at the origin.
    """
    # Centerline
    cx = scale * np.sin(tt)
    cy = scale * np.sin(2 * tt) / 2

    # Unit tangent in xy-plane
    dcx = scale * np.cos(tt)
    dcy = scale * np.cos(2 * tt)
    t_len = np.clip(np.sqrt(dcx**2 + dcy**2), 1e-12, None)
    Tx = dcx / t_len
    Ty = dcy / t_len

    # In-plane normal (90 deg CCW)
    Nx = -Ty
    Ny = Tx

    # Mobius twist with phase offset
    twist_angle = (tt + twist_phase) / 2
    c2 = np.cos(twist_angle)
    s2 = np.sin(twist_angle)
    dx = s2 * Nx
    dy = s2 * Ny
    dz = c2

    # Surface
    px = cx + vv * dx
    py = cy + vv * dy
    pz = vv * dz
    points = np.column_stack([px, py, pz])

    # dS/dv = d(t)
    tv = np.column_stack([dx, dy, dz])

    # dS/dt via central finite differences
    eps = 1e-6

    def _pos(t, v):
        _cx = scale * np.sin(t)
        _cy = scale * np.sin(2 * t) / 2
        _dcx = scale * np.cos(t)
        _dcy = scale * np.cos(2 * t)
        _tl = np.clip(np.sqrt(_dcx**2 + _dcy**2), 1e-12, None)
        _Nx = -_dcy / _tl
        _Ny = _dcx / _tl
        _ta = (t + twist_phase) / 2
        _c2 = np.cos(_ta)
        _s2 = np.sin(_ta)
        return np.column_stack([
            _cx + v * _s2 * _Nx,
            _cy + v * _s2 * _Ny,
            v * _c2,
        ])

    tu = (_pos(tt + eps, vv) - _pos(tt - eps, vv)) / (2 * eps)

    return points, tu, tv


# Normal-flip seam at t = pi - twist_phase.  With twist_phase = 3*pi/4 the
# seam lands at t = pi/4 (on the right lobe, away from the self-intersection
# at t = 0, pi and from the source points at t = pi/2, 3*pi/2).
TWIST_PHASE = 1 * np.pi / 4 # TODO test


def _make_figure8_mobius(
    n_target: int = 2000,
    scale: float = 1.0,
    half_width: float = 0.12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Self-intersecting Mobius band on a figure-8 centerline.

    Area-uniform sampling via dense jittered grid + area-weighted subsampling.
    Returns (points, normals, src_a_index, src_b_index).
    Sources are placed on opposite lobes of the figure-8.
    """
    rng = np.random.default_rng(seed)

    # Dense grid (4x oversampling)
    oversample = 4
    n_total = n_target * oversample
    n_t = int(np.sqrt(n_total * 2 * np.pi / (2 * half_width)))
    n_v = max(n_total // n_t, 4)
    t_arr = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    v_arr = np.linspace(-half_width, half_width, n_v)
    tt, vv = np.meshgrid(t_arr, v_arr)
    tt, vv = tt.ravel(), vv.ravel()

    # Jitter
    dt = 2 * np.pi / n_t
    dv = 2 * half_width / n_v
    tt = tt + rng.uniform(-0.3 * dt, 0.3 * dt, tt.size)
    vv = vv + rng.uniform(-0.3 * dv, 0.3 * dv, vv.size)
    vv = np.clip(vv, -half_width, half_width)

    pts, tu, tv = _figure8_evaluate(tt, vv, scale, twist_phase=TWIST_PHASE)

    # Area element
    cross = np.cross(tu, tv)
    area_elem = np.linalg.norm(cross, axis=1)

    # Weighted sub-sample
    prob = area_elem / area_elem.sum()
    idx = rng.choice(len(pts), size=n_target, replace=False, p=prob)

    points = pts[idx]
    normals = cross[idx]
    norms = np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
    normals = normals / norms

    # Auto-select: asymmetric diagonal placement so the Voronoi boundary
    # does NOT coincide with the self-intersection at the origin.
    # t ~ pi/3  -> upper-right lobe  (sin(pi/3), sin(2pi/3)/2) ~ (0.87, 0.43)
    # t ~ 5pi/3 -> lower-left lobe   (sin(5pi/3), sin(10pi/3)/2) ~ (-0.87, -0.43)
    t_a, t_b = np.pi / 3, 5 * np.pi / 3
    target_a = np.array([scale * np.sin(t_a), scale * np.sin(2 * t_a) / 2, 0.0])
    target_b = np.array([scale * np.sin(t_b), scale * np.sin(2 * t_b) / 2, 0.0])
    src_a = _nearest_index(points, target_a)
    src_b = _nearest_index(points, target_b)

    return points, normals, int(src_a), int(src_b)


# ---------------------------------------------------------------------------
# Bilateral Laplacian + heat method
# ---------------------------------------------------------------------------

def _bilateral_affinity(
    points: np.ndarray,
    normals: np.ndarray,
    sigma_x: float,
    sigma_n: float,
    k: int,
) -> sparse.csr_matrix:
    """Bilateral kernel: spatial Gaussian x normal Gaussian, sparse k-NN."""
    _, W, _ = bilateral_pointcloud_laplacian(
        points, normals, k=k, sigma_x=sigma_x, sigma_n=sigma_n,
        normalized=False, return_parts=True)
    return W


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


def _edge_gradient_pts(points, f, W, lam=0.0):
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
    return np.linalg.solve(S, b)


def _edge_divergence_pts(points, X, W):
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


def _generic_heat_method(
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
    grad_u = _edge_gradient_pts(points, u, W)
    norms = np.clip(np.linalg.norm(grad_u, axis=1), gradient_eps, None)
    X = -grad_u / norms[:, None]

    # Step 3: Poisson solve  L phi = -div(X)
    div_X = _edge_divergence_pts(points, X, W)
    L_p, rhs_p = _anchor_components(L, -div_X, W, int(src[0]))
    phi = spla.spsolve(L_p, rhs_p)

    return phi, u


def bilateral_heat_method(
    points: np.ndarray,
    normals: np.ndarray,
    source_index: int,
    *,
    k: int = 20,
    sigma_x: float = 0.5,
    sigma_n: float = 0.5,
    t_scale: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Heat-method geodesics using a bilateral (spatial x normal) kernel."""
    N = len(points)
    print(f"  bilateral bandwidths: sigma_x={sigma_x:.4f}, sigma_n={sigma_n:.4f}")

    W = _bilateral_affinity(points, normals, sigma_x, sigma_n, k)
    L, W_sym, _ = affinity_to_laplacian(W, normalized=False)
    W_sym = W_sym.tocsr()

    mass = _estimate_mass_pts(points, W_sym)
    M = sparse.diags(mass)

    phi, u = _generic_heat_method(
        points, L, M, W_sym, source_index, t_scale=t_scale,
    )
    return phi, u


def robust_heat_method(
    points: np.ndarray,
    source_index: int,
    *,
    k: int = 20,
    t_scale: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Heat-method geodesics using the robust_laplacian library."""
    L, M = robust_laplacian.point_cloud_laplacian(points, n_neighbors=k)
    # Build a k-NN affinity for edge structure
    tree = cKDTree(points)
    _, idx = tree.query(points, k=min(k + 1, len(points)))
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(len(points)), idx.shape[1])
    cols = idx.ravel()
    vals = np.ones(len(rows))
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(points), len(points)))
    W = W + W.T

    phi, u = _generic_heat_method(
        points, L, M, W, source_index, t_scale=t_scale,
    )
    return phi, u


# ---------------------------------------------------------------------------
# Build BlowUpLevel from points + normals
# ---------------------------------------------------------------------------

def _build_level(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    alpha: float,
    k: int,
) -> BlowUpLevel:
    frames = _normals_to_tangent_frames(normals)
    level0 = BlowUpLevel.from_point_tangents(points, frames)
    return level0.lift(alpha=alpha, k=k, lam=0.0)


# ---------------------------------------------------------------------------
# Geodesic distance computation per method
# ---------------------------------------------------------------------------

def _compute_distances(
    method: str,
    level: BlowUpLevel,
    points: np.ndarray,
    normals: np.ndarray,
    source_index: int,
    *,
    k: int,
    t_scale: float,
    sigma_x: float = 0.5,
    sigma_u: float = 0.5,
    sigma_x_bi: float = 0.5,
    sigma_n_bi: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, heat_field) for the given method."""
    if method == "bilateral":
        phi, u = bilateral_heat_method(
            points, normals, source_index,
            k=k, sigma_x=sigma_x_bi, sigma_n=sigma_n_bi,
            t_scale=t_scale,
        )
    elif method == "robust":
        phi, u = robust_heat_method(
            points, source_index,
            k=k, t_scale=t_scale,
        )
    elif method == "lifted":
        phi, u, _, _ = lifted_heat_method(
            level, source_index=source_index, k=k,
            kernel="product", sigma_x=sigma_x, sigma_u=sigma_u,
            t_scale=t_scale, return_intermediate=True,
        )
    else:
        raise ValueError(method)

    phi = np.maximum(phi - phi[source_index], 0.0)
    return phi, u


# ---------------------------------------------------------------------------
# Scalar field colouring: Voronoi partition {0, 2} from two sources
# ---------------------------------------------------------------------------

def _voronoi_colors(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
) -> np.ndarray:
    """Assign blue to points closer to source A, red to source B.

    Every point is coloured -- the boundary blends smoothly via a sigmoid
    on the distance difference.
    """
    blue = np.array([0.25, 0.40, 0.85])
    red = np.array([0.85, 0.25, 0.25])

    # Soft blend: t in [0, 1] where 0 = source-A (blue), 1 = source-B (red)
    diff = dist_a - dist_b   # negative -> closer to A
    scale = 0.5 * (np.mean(np.abs(dist_a)) + np.mean(np.abs(dist_b))) + 1e-12
    sharpness = 10.0
    t = 1.0 / (1.0 + np.exp(-sharpness * diff / scale))

    colors = (1 - t[:, None]) * blue[None, :] + t[:, None] * red[None, :]
    return colors


# ---------------------------------------------------------------------------
# Geometry registry
# ---------------------------------------------------------------------------

GEOMETRIES = {
    "thin_tube": {
        "generator": _make_thin_tube,
        "description": "Thin tube (opposite normals)",
    },
    "figure8_mobius": {
        "generator": _make_figure8_mobius,
        "description": "Figure-8 Mobius band (self-intersecting)",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bilateral vs robust vs lifted geodesic comparison",
    )
    parser.add_argument(
        "--n-tube", type=int, default=None,
        help="Approx point count for the thin tube (default: ~7200)",
    )
    parser.add_argument(
        "--n-mobius", type=int, default=8000,
        help="Point count for the figure-8 Mobius band (default: 8000)",
    )
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--sigma-x", type=float, default=SIGMA_X)
    parser.add_argument("--sigma-u", type=float, default=SIGMA_U)
    parser.add_argument("--sigma-x-bi", type=float, default=SIGMA_X_BI)
    parser.add_argument("--sigma-n-bi", type=float, default=SIGMA_N_BI)
    parser.add_argument("--t-scale", type=float, default=T_SCALE)
    args = parser.parse_args()

    # Override globals from CLI
    k = args.k
    alpha = args.alpha
    sigma_x = args.sigma_x
    sigma_u = args.sigma_u
    sigma_x_bi = args.sigma_x_bi
    sigma_n_bi = args.sigma_n_bi
    t_scale = args.t_scale

    # Geometry-specific point counts
    tube_kwargs = {}
    if args.n_tube is not None:
        # Distribute across n_theta x n_z x n_layers ≈ n_tube
        n_layers = 3
        n_total_per_layer = args.n_tube // n_layers
        n_theta = max(int(np.sqrt(n_total_per_layer * 2 * np.pi)), 10)
        n_z = max(n_total_per_layer // n_theta, 5)
        tube_kwargs = {"n_theta": n_theta, "n_z": n_z, "n_layers": n_layers}
    mobius_kwargs = {"n_target": args.n_mobius}

    generator_kwargs = {
        "thin_tube": tube_kwargs,
        "figure8_mobius": mobius_kwargs,
    }

    # -- Build all geometries and levels up front --
    data = {}
    for name, spec in GEOMETRIES.items():
        points, normals, src_a, src_b = spec["generator"](
            **generator_kwargs.get(name, {}),
        )
        normals = normalize_vectors(normals)
        level = _build_level(points, normals, alpha=alpha, k=k)
        data[name] = {
            "points": points,
            "normals": normals,
            "level": level,
            "default_src_a": src_a,
            "default_src_b": src_b,
        }
        print(f"[{name}] {len(points)} points, level D={level.D} d={level.d}")
        print(f"  auto sources: A={src_a} ({points[src_a].round(3)}), "
              f"B={src_b} ({points[src_b].round(3)})")

    # -- Mutable UI state --
    geom_names = list(GEOMETRIES.keys())
    state = {
        "geom_idx": 0,
        "source_a": None,
        "source_b": None,
        "computing": False,
        "results": {},
    }

    def _current_geom():
        return geom_names[state["geom_idx"]]

    def _get_spacing():
        pts = data[_current_geom()]["points"]
        bbox = pts.max(axis=0) - pts.min(axis=0)
        return float(bbox.max()) * 1.5

    def _register_geometry(geom_name: str, *, auto_compute: bool = True):
        """Register point clouds and optionally auto-compute with defaults."""
        for m in METHODS:
            if ps.has_point_cloud(m):
                ps.remove_point_cloud(m)
        for m in METHODS:
            for tag in (f"source_a_{m}", f"source_b_{m}"):
                if ps.has_point_cloud(tag):
                    ps.remove_point_cloud(tag)

        pts = data[geom_name]["points"]
        spacing = _get_spacing()

        for i, method in enumerate(METHODS):
            offset = np.array([i * spacing, 0.0, 0.0])
            cloud = ps.register_point_cloud(
                method, pts + offset, radius=0.002,
            )
            cloud.set_color((0.7, 0.7, 0.7))

        # Auto-select default sources
        state["source_a"] = data[geom_name]["default_src_a"]
        state["source_b"] = data[geom_name]["default_src_b"]
        state["results"] = {}

        if auto_compute:
            state["computing"] = True

    def _compute_and_display():
        """Compute scalar fields for all methods and display."""
        geom = _current_geom()
        pts = data[geom]["points"]
        normals = data[geom]["normals"]
        level = data[geom]["level"]
        idx_a = state["source_a"]
        idx_b = state["source_b"]
        spacing = _get_spacing()

        for i, method in enumerate(METHODS):
            print(f"  [{method}] computing from sources {idx_a} and {idx_b}...")
            dist_a, _ = _compute_distances(
                method, level, pts, normals, idx_a,
                k=k, t_scale=t_scale,
                sigma_x=sigma_x, sigma_u=sigma_u,
                sigma_x_bi=sigma_x_bi, sigma_n_bi=sigma_n_bi,
            )
            dist_b, _ = _compute_distances(
                method, level, pts, normals, idx_b,
                k=k, t_scale=t_scale,
                sigma_x=sigma_x, sigma_u=sigma_u,
                sigma_x_bi=sigma_x_bi, sigma_n_bi=sigma_n_bi,
            )
            colors = _voronoi_colors(dist_a, dist_b)
            state["results"][method] = colors

            cloud = ps.get_point_cloud(method)
            cloud.add_color_quantity("scalar_field", colors, enabled=True)

            # Source markers
            offset = np.array([i * spacing, 0.0, 0.0])
            for tag, idx, color in [
                (f"source_a_{method}", idx_a, (0.0, 0.2, 0.7)),
                (f"source_b_{method}", idx_b, (0.85, 0.25, 0.25)),
            ]:
                if ps.has_point_cloud(tag):
                    ps.remove_point_cloud(tag)
                src = ps.register_point_cloud(
                    tag, (pts[idx:idx+1] + offset), radius=0.008,
                )
                src.set_color(color)

            print(f"  [{method}] done")

    def callback():
        geom = _current_geom()

        psim.TextUnformatted(
            "Left: BILATERAL   Centre: LIFTED (ours)   Right: ROBUST"
        )
        psim.Separator()

        # Geometry selector
        changed, new_idx = psim.Combo(
            "Geometry",
            state["geom_idx"],
            [GEOMETRIES[g]["description"] for g in geom_names],
        )
        if changed:
            state["geom_idx"] = new_idx
            _register_geometry(_current_geom())

        psim.Separator()

        # Show current selections
        pts = data[geom]["points"]
        if state["source_a"] is not None:
            idx = state["source_a"]
            psim.TextUnformatted(
                f"Source A (blue):  {idx}  ({pts[idx].round(3)})"
            )
        if state["source_b"] is not None:
            idx = state["source_b"]
            psim.TextUnformatted(
                f"Source B (red):   {idx}  ({pts[idx].round(3)})"
            )

        # Handle picking to override sources
        pick = ps.get_selection()
        have = pick.is_hit and pick.structure_name in METHODS

        if have:
            picked = pick.local_index
            psim.TextUnformatted(
                f"Hovered: {picked}  ({pts[picked].round(3)})"
            )
            if psim.Button("Set as Source A"):
                state["source_a"] = picked
            psim.SameLine()
            if psim.Button("Set as Source B"):
                state["source_b"] = picked

        # Recompute button
        if state["source_a"] is not None and state["source_b"] is not None:
            if psim.Button("Recompute"):
                state["computing"] = True

        # Deferred compute
        if state["computing"]:
            state["computing"] = False
            print("Computing scalar fields...")
            _compute_and_display()

    # -- Polyscope setup --
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    _register_geometry(geom_names[0], auto_compute=False)

    print("Copies placed side-by-side: " + " | ".join(METHODS))
    print("Auto-computing with default sources...")

    ps.set_user_callback(callback)
    state["computing"] = True
    ps.show()


if __name__ == "__main__":
    main()
