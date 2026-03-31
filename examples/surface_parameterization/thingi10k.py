"""
Surface Parameterization via Spectral Segmentation
----------------------------------------------------
Two comparisons for an SGP figure:

**Segmentation** (3 methods, same DBSCAN on spectral embedding):
  - Euclidean   — graph Laplacian on spatial positions only
  - Bilateral   — kernel weighted by spatial distance AND normal similarity
  - Lifted      — product kernel on the level-1 Chordal-Sasaki blow-up

**Parameterization** (SCP on each):
  - Global SCP  — no segmentation, Euclidean Laplacian (baseline)
  - Lifted-seg + SCP — per-segment SCP using our lifted segmentation

Polyscope layout (six side-by-side copies):
  (1) Original
  (2) Euclidean segmentation
  (3) Bilateral segmentation
  (4) Lifted segmentation
  (5) Global SCP (checkerboard)
  (6) Lifted-seg + SCP (checkerboard)

Usage:
    python thingi10k.py -p klein_bottle_two
    python thingi10k.py -p ship --rotate y
    python thingi10k.py -p cats
    python thingi10k.py                           # list available point clouds
"""
from __future__ import annotations

import argparse
from math import gamma as math_gamma
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from matplotlib.colors import hsv_to_rgb
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse import linalg as spla

from tangent_blowups.clustering import spectral_clustering_from_laplacian
from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud import (
    pointcloud_laplacian, bilateral_pointcloud_laplacian,
)
from tangent_blowups.solvers.linalg import normalize_vectors

# -- Constants ---------------------------------------------------------------
PANELS = ["original",
          "Euclidean seg.", "bilateral seg.", "lifted seg.",
          "global SCP", "lifted SCP"]
N_MAX = 50_000
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "thingi10k_pointcloud"
VIS_MODES = ["checkerboard", "conformal distortion"]


# -- Rotation ----------------------------------------------------------------
def _rotation_matrix(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
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
        axis = np.array([float(p) for p in parts[1:]])
        return _rotation_matrix(angle, axis)
    raise ValueError(f"Invalid --rotate spec: {spec!r}")


# -- Data loading ------------------------------------------------------------
def _available() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.npz"))


def _load(name: str, *, rotate: np.ndarray | None = None):
    path = DATA_DIR / f"{name}.npz"
    pc = load_pointcloud(path)
    pts = np.asarray(pc.points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(pc.normals, dtype=float).reshape(-1, 3)
    valid = (np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
             & (np.linalg.norm(nrm, axis=1) > 1e-8))
    pts, nrm = pts[valid], normalize_vectors(nrm[valid])
    if len(pts) > N_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N_MAX, replace=False)
        pts, nrm = pts[idx], nrm[idx]
    if rotate is not None:
        pts = pts @ rotate.T
        nrm = nrm @ rotate.T
    return pts, nrm


# -- Mass matrix estimation --------------------------------------------------
def _estimate_mass(pts: np.ndarray, W: sparse.spmatrix,
                   d_manifold: int = 2) -> np.ndarray:
    N = len(pts)
    coo = W.tocoo()
    mask = coo.row != coo.col
    rows, cols = coo.row[mask], coo.col[mask]
    if rows.size == 0:
        return np.ones(N, dtype=float)
    dx = pts[rows] - pts[cols]
    dist = np.linalg.norm(dx, axis=1)
    r_k = np.zeros(N, dtype=float)
    np.maximum.at(r_k, rows, dist)
    k_per = np.bincount(rows, minlength=N).astype(float)
    k_per = np.clip(k_per, 1.0, None)
    V_d = np.pi ** (d_manifold / 2.0) / math_gamma(d_manifold / 2.0 + 1.0)
    mass = V_d * (r_k ** d_manifold) / k_per
    zero = mass <= 0.0
    if zero.any():
        pos = mass[~zero]
        mass[zero] = float(np.median(pos)) if pos.size > 0 else 1.0
    return mass


# -- Spectral Conformal Parameterization (single component) ------------------
def _scp_single(L: sparse.spmatrix, mass: np.ndarray) -> np.ndarray:
    N = L.shape[0]
    if N < 4:
        return np.zeros((N, 2))

    M = sparse.diags(mass, format="csr")
    k_eig = min(3, N - 1)
    DENSE_THRESHOLD = 2000

    if N <= DENSE_THRESHOLD or k_eig >= N - 1:
        Ld = L.toarray() if sparse.issparse(L) else np.asarray(L)
        Md = M.toarray() if sparse.issparse(M) else np.asarray(M)
        from scipy.linalg import eigh as dense_eigh
        evals, evecs = dense_eigh(Ld, Md, subset_by_index=[0, k_eig - 1])
    else:
        ncv = min(max(4 * k_eig + 1, 40), N - 1)
        try:
            evals, evecs = spla.eigsh(
                L, k=k_eig, M=M, sigma=-1e-5, which="LM", ncv=ncv)
        except spla.ArpackNoConvergence as exc:
            if exc.eigenvalues.size >= k_eig:
                evals, evecs = exc.eigenvalues, exc.eigenvectors
            else:
                from scipy.linalg import eigh as dense_eigh
                evals, evecs = dense_eigh(
                    L.toarray(), M.toarray(),
                    subset_by_index=[0, k_eig - 1])

    order = np.argsort(evals)
    evecs = evecs[:, order]

    if evecs.shape[1] >= 3:
        uv = evecs[:, 1:3].copy()
    elif evecs.shape[1] == 2:
        uv = np.column_stack([evecs[:, 1], np.zeros(N)])
    else:
        return np.zeros((N, 2))

    uv_range = uv.max(axis=0) - uv.min(axis=0)
    scale = uv_range.max()
    if scale > 1e-15:
        uv *= 1.0 / scale
    return uv


def _scp_global(pts: np.ndarray, *, k: int) -> np.ndarray:
    L, W, _ = pointcloud_laplacian(
        pts, k=k, h="local", normalized=False, return_parts=True)
    return _scp_single(L, _estimate_mass(pts, W))


def _scp_per_segment(pts, labels, n_clusters, *, k):
    N = len(pts)
    uv = np.zeros((N, 2))
    for c in range(n_clusters):
        mask = labels == c
        n_c = int(mask.sum())
        if n_c < 4:
            continue
        pts_c = pts[mask]
        k_c = min(k, n_c - 1)
        L_c, W_c, _ = pointcloud_laplacian(
            pts_c, k=k_c, h="local", normalized=False, return_parts=True)
        uv[mask] = _scp_single(L_c, _estimate_mass(pts_c, W_c))
    return uv


# -- UV colouring / segments --------------------------------------------------
def _uv_to_checker(uv: np.ndarray, cell: float) -> np.ndarray:
    u = np.nan_to_num(uv[:, 0], nan=0.0)
    v = np.nan_to_num(uv[:, 1], nan=0.0)
    cu, cv = u / max(cell, 1e-15), v / max(cell, 1e-15)
    check = ((np.floor(cu).astype(int) + np.floor(cv).astype(int)) % 2)
    blue = np.array([0.40, 0.45, 0.85])
    white = np.array([0.95, 0.95, 0.98])
    return np.where(check[:, None], blue, white)


def _uv_extent(uv: np.ndarray) -> float:
    extents = []
    for col in range(2):
        vals = uv[:, col]
        vals = vals[np.isfinite(vals) & (vals != 0.0)]
        if vals.size < 4:
            continue
        q10, q90 = np.percentile(vals, [10, 90])
        extents.append(q90 - q10)
    return float(max(extents)) if extents else 1.0


def _segment_colors(labels: np.ndarray) -> np.ndarray:
    N = len(labels)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    phi = (1 + np.sqrt(5)) / 2
    colors = np.full((N, 3), 0.7)
    for c in range(n_clusters):
        mask = labels == c
        colors[mask] = hsv_to_rgb(np.array([(c * phi) % 1.0, 0.65, 0.85]))
    colors[labels < 0] = [0.7, 0.7, 0.7]
    return colors


# -- Distortion metrics -------------------------------------------------------
def _estimate_jacobian_svs(pts, nrm, uv, *, k=12, lam=1e-6,
                            labels=None):
    from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
    N = len(pts)
    tree = cKDTree(pts)
    k_query = k if labels is None else min(3 * k, N - 1)
    _, nn_idx = tree.query(pts, k=min(k_query + 1, N))
    nn_idx = nn_idx[:, 1:]
    frames = _normals_to_tangent_frames(nrm)
    sigma1 = np.zeros(N)
    sigma2 = np.zeros(N)
    for i in range(N):
        nbrs = nn_idx[i]
        nbrs = nbrs[nbrs < N]
        if labels is not None:
            nbrs = nbrs[labels[nbrs] == labels[i]]
        if len(nbrs) < 2:
            continue
        dx = pts[nbrs] - pts[i]
        dt = dx @ frames[i]
        duv = uv[nbrs] - uv[i]
        S = dt.T @ dt
        S += lam * np.trace(S) / 2.0 * np.eye(2)
        try:
            Jt = np.linalg.solve(S, dt.T @ duv)
        except np.linalg.LinAlgError:
            continue
        svs = np.sort(np.linalg.svd(Jt, compute_uv=False))[::-1]
        sigma1[i] = svs[0]
        sigma2[i] = svs[1]
    return sigma1, sigma2


def _distortion_stats(sigma1, sigma2):
    eps = 1e-12
    s1, s2 = np.clip(sigma1, eps, None), np.clip(sigma2, eps, None)
    return {
        "conformal": s1 / s2 + s2 / s1,
        "area": s1 * s2,
        "dirichlet": s1 ** 2 + s2 ** 2,
    }


def _print_distortion_summary(label, stats):
    for key in ("conformal", "dirichlet"):
        vals = stats[key]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            print(f"  {label} {key}: no valid data")
            continue
        med = float(np.median(vals))
        p95 = float(np.percentile(vals, 95))
        print(f"  {label} {key:>10s}:  median={med:.4f}  p95={p95:.4f}")
    area = stats["area"]
    area = area[np.isfinite(area) & (area > 0)]
    if area.size > 0:
        cv = float(np.std(area) / np.mean(area))
        print(f"  {label}       area:  CV={cv:.4f}")


# -- Helpers for DBSCAN clustering on a Laplacian ----------------------------
def _cluster(L, n_components, *, dbscan_eps, dbscan_min_samples):
    labels, evals, _, _ = spectral_clustering_from_laplacian(
        L, n_components,
        cluster_method="dbscan",
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples)
    n_found = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels < 0).sum())
    counts = np.bincount(labels[labels >= 0], minlength=n_found)
    return labels, n_found, n_noise, counts


# -- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Surface parameterization: segmentation comparison + SCP")
    parser.add_argument("--pointcloud", "-p", type=str, default=None)
    parser.add_argument("--n-clusters", "-n", type=int, default=15,
                        help="Spectral embedding dimension for DBSCAN")
    parser.add_argument("--dbscan-eps", type=float, default=0.1)
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num-levels", type=int, default=1,
                        help="Number of blow-up levels (1 = level-1, 2 = level-2, ...)")
    parser.add_argument("--kernel", type=str, default="product",
                        choices=["self_tuning", "gaussian", "product"])
    parser.add_argument("--sigma-x", type=float, default=0.5)
    parser.add_argument("--sigma-u", type=float, default=0.5)
    parser.add_argument("--n-cells", type=float, default=8.0)
    parser.add_argument("--rotate", "-r", type=str, default=None)
    args = parser.parse_args()

    available = _available()
    if args.pointcloud is None:
        print("Available point clouds:")
        for name in available:
            print(f"  {name}")
        print("\nUsage: python thingi10k.py -p <name>")
        return

    if args.pointcloud not in available:
        print(f"Unknown: {args.pointcloud!r}. Available: {', '.join(available)}")
        return

    R = _parse_rotation(args.rotate) if args.rotate else None
    pts, nrm = _load(args.pointcloud, rotate=R)
    N = len(pts)
    k = args.k
    n_comp = args.n_clusters
    n_cells = args.n_cells
    dbscan_kw = dict(dbscan_eps=args.dbscan_eps,
                     dbscan_min_samples=args.dbscan_min_samples)

    print(f"Loaded {N} points from {args.pointcloud}")

    # ==================================================================
    # Segmentation comparison
    # ==================================================================

    # (a) Euclidean
    print("Euclidean spectral segmentation ...", end="", flush=True)
    L_euc = pointcloud_laplacian(pts, k=k, h="local", normalized=True)
    lab_euc, n_euc, noise_euc, cnt_euc = _cluster(
        L_euc, n_comp, **dbscan_kw)
    print(f" {n_euc} clusters"
          f"{f', noise={noise_euc}' if noise_euc else ''}")

    # (b) Bilateral
    print("Bilateral spectral segmentation ...", end="", flush=True)
    L_bil = bilateral_pointcloud_laplacian(
        pts, nrm, k=k, sigma_x=args.sigma_x, sigma_n=args.sigma_u,
        normalized=True)
    lab_bil, n_bil, noise_bil, cnt_bil = _cluster(
        L_bil, n_comp, **dbscan_kw)
    print(f" {n_bil} clusters"
          f"{f', noise={noise_bil}' if noise_bil else ''}")

    # (c) Lifted
    num_levels = args.num_levels
    print(f"Building level-{num_levels} blow-up ...", end="", flush=True)
    tangent_frames = BlownUpSample.from_normals(pts, nrm).dualize().basis
    level = BlowUpLevel.from_point_tangents(pts, tangent_frames)
    for _ in range(num_levels):
        level = level.lift(k=k, alpha=args.alpha, lam=1e-3)
    print(f" D={level.D}")

    print("Lifted spectral segmentation ...", end="", flush=True)
    L_lift, _, _ = lifted_laplacian(
        level, kernel=args.kernel, k=k, h="local", normalized=True,
        sigma_x=args.sigma_x, sigma_u=args.sigma_u)
    lab_lift, n_lift, noise_lift, cnt_lift = _cluster(
        L_lift, n_comp, **dbscan_kw)
    print(f" {n_lift} clusters"
          f"{f', noise={noise_lift}' if noise_lift else ''}")

    seg_euc = _segment_colors(lab_euc)
    seg_bil = _segment_colors(lab_bil)
    seg_lift = _segment_colors(lab_lift)

    # ==================================================================
    # Parameterization comparison
    # ==================================================================

    # (a) Global SCP — no segmentation
    print("Global SCP (baseline) ...", end="", flush=True)
    uv_global = _scp_global(pts, k=k)
    print(" done")

    # (b) Per-segment SCP using lifted segmentation
    print("Lifted-seg + SCP ...", end="", flush=True)
    uv_lift = _scp_per_segment(pts, lab_lift, n_lift, k=k)
    print(" done")

    # ==================================================================
    # Distortion metrics
    # ==================================================================
    print("Computing distortion metrics ...", end="", flush=True)
    s1_g, s2_g = _estimate_jacobian_svs(pts, nrm, uv_global, k=min(k, 12))
    s1_l, s2_l = _estimate_jacobian_svs(pts, nrm, uv_lift, k=min(k, 12),
                                         labels=lab_lift)
    stats_global = _distortion_stats(s1_g, s2_g)
    stats_lift = _distortion_stats(s1_l, s2_l)
    print(" done")
    _print_distortion_summary("Global     ", stats_global)
    _print_distortion_summary("Lifted-seg ", stats_lift)

    # ==================================================================
    # Polyscope
    # ==================================================================
    print(f"Checker cells: {n_cells:.1f}")

    bbox_x = pts[:, 0].max() - pts[:, 0].min()
    spacing = bbox_x * 1.4

    panel_pts = {}
    for i, name in enumerate(PANELS):
        panel_pts[name] = pts + np.array([i * spacing, 0.0, 0.0])

    state = {
        "vis_mode": 0,
        "n_cells_global": n_cells,
        "n_cells_lift": n_cells,
    }

    conf_global = stats_global["conformal"]
    conf_lift = stats_lift["conformal"]
    valid_conf = np.concatenate([
        conf_global[conf_global > 0], conf_lift[conf_lift > 0]])
    conf_vmax = float(np.percentile(valid_conf, 95)) if valid_conf.size else 10.0

    extent_global = _uv_extent(uv_global)
    extent_lift = _uv_extent(uv_lift)

    def _apply_vis():
        vis = VIS_MODES[state["vis_mode"]]

        if vis == "conformal distortion":
            for name, conf in [("global SCP", conf_global),
                               ("lifted SCP", conf_lift)]:
                ps.get_point_cloud(name).add_scalar_quantity(
                    "conformal", conf, enabled=True,
                    cmap="reds", vminmax=(2.0, conf_vmax))
            return

        cell_g = extent_global / max(state["n_cells_global"], 1e-12)
        cell_l = extent_lift / max(state["n_cells_lift"], 1e-12)
        ps.get_point_cloud("global SCP").add_color_quantity(
            "uv", _uv_to_checker(uv_global, cell_g), enabled=True)
        ps.get_point_cloud("lifted SCP").add_color_quantity(
            "uv", _uv_to_checker(uv_lift, cell_l), enabled=True)

    def callback():
        psim.TextUnformatted(
            f"{args.pointcloud}  |  {N} pts  |  alpha={args.alpha}")
        psim.TextUnformatted(
            f"Segments:  Euc={n_euc}  Bil={n_bil}  Lift={n_lift}")
        psim.Separator()

        changed_mode, new_mode = psim.Combo(
            "Visualisation", state["vis_mode"], VIS_MODES)
        if changed_mode:
            state["vis_mode"] = new_mode
            _apply_vis()

        g_changed, new_g = psim.SliderFloat(
            "Global cells", state["n_cells_global"],
            v_min=2.0, v_max=30.0)
        l_changed, new_l = psim.SliderFloat(
            "Lifted cells", state["n_cells_lift"],
            v_min=2.0, v_max=30.0)
        if g_changed:
            state["n_cells_global"] = new_g
        if l_changed:
            state["n_cells_lift"] = new_l
        if g_changed or l_changed:
            _apply_vis()

        psim.Separator()
        psim.TextUnformatted("Conformal distortion (lower = better, 2 = perfect)")
        med_g = float(np.median(conf_global[conf_global > 0]))
        med_l = float(np.median(conf_lift[conf_lift > 0]))
        psim.TextUnformatted(f"  Global:     {med_g:.3f}")
        psim.TextUnformatted(f"  Lifted-seg: {med_l:.3f}")

        psim.Separator()
        if psim.Button("Screenshot (.png)"):
            fname = f"scp_{args.pointcloud}.png"
            ps.screenshot(fname, transparent_bg=False)
            print(f"Saved {fname}")

    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    # (1) Original
    c = ps.register_point_cloud("original", panel_pts["original"], radius=0.001)
    c.set_color((0.75, 0.75, 0.75))

    # (2-4) Segmentations
    for name, colors in [("Euclidean seg.", seg_euc),
                         ("bilateral seg.", seg_bil),
                         ("lifted seg.", seg_lift)]:
        c = ps.register_point_cloud(name, panel_pts[name], radius=0.001)
        c.add_color_quantity("clusters", colors, enabled=True)

    # (5-6) Parameterizations
    ps.register_point_cloud("global SCP", panel_pts["global SCP"], radius=0.001)
    ps.register_point_cloud("lifted SCP", panel_pts["lifted SCP"], radius=0.001)

    _apply_vis()

    print(f"Panels: {' | '.join(PANELS)}")
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
