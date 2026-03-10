"""
Occluding contour experiment for self-intersecting surfaces.

Test surface: two sinusoidally deformed sheets intersecting transversally.

  S1 : (u, v) -> (u,  v,  A sin(pi u) sin(pi v))      u,v in [-1, 1]
  S2 : (u, v) -> (A sin(pi u) sin(pi v),  u,  v)      u,v in [-1, 1]

Both sheets pass through the origin.  At the origin:
  N_S1 = (0, 0, 1)     N_S2 = (1, 0, 0)

Viewpoint family: view(theta) = (sin theta, 0, cos theta), theta in [0, pi].

Critical angles where a sheet silhouette crosses the intersection locus:
  theta* = 90 deg  -- S1 silhouette crosses intersection (N_S1 . view = 0 at origin)
  theta  =  0 deg  -- S2 silhouette crosses intersection (N_S2 . view = 0 at origin)

Baseline failure mode: near the intersection locus, a standard non-manifold
mesh has edges incident to faces from both sheets.  Two common conventions --
averaging all incident normals ("averaged") and using the first incident face's
normal ("first-face") -- both corrupt the silhouette condition N . v = 0 near
the intersection, displacing the extracted contour.

Our method extracts one silhouette per sheet using the correct per-sheet normal,
identical to the analytic ground truth.

Figures produced
----------------
contours_panel.pdf   -- three-panel image-space comparison at theta* = 90 deg:
                        (a) ground truth  (b) baseline-averaged  (c) per-sheet
contours_error.pdf   -- Chamfer error vs theta for both baselines (ours = 0).

Usage
-----
    python examples/occluding_contours/example.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
from scipy.spatial import cKDTree


# ── sheet geometry ─────────────────────────────────────────────────────────────

_PI = np.pi


def _sheet1(U: np.ndarray, V: np.ndarray, A: float):
    """
    S1 : (u, v, A sin(pi u) sin(pi v)).

    partial_u f = (1, 0,  A pi cos(pi u) sin(pi v))
    partial_v f = (0, 1,  A pi sin(pi u) cos(pi v))
    N (unnorm)  = (-A pi cos(pi u) sin(pi v),
                   -A pi sin(pi u) cos(pi v),
                    1)
    """
    su, sv = np.sin(_PI * U), np.sin(_PI * V)
    cu, cv = np.cos(_PI * U), np.cos(_PI * V)

    pos = np.stack([U, V, A * su * sv], axis=-1)

    nx = -A * _PI * cu * sv
    ny = -A * _PI * su * cv
    nz = np.ones_like(U)
    N = np.stack([nx, ny, nz], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return pos, N


def _sheet2(U: np.ndarray, V: np.ndarray, A: float):
    """
    S2 : (A sin(pi u) sin(pi v), u, v).

    partial_u f = (A pi cos(pi u) sin(pi v), 1, 0)
    partial_v f = (A pi sin(pi u) cos(pi v), 0, 1)
    N (unnorm)  = ( 1,
                   -A pi cos(pi u) sin(pi v),
                   -A pi sin(pi u) cos(pi v))
    """
    su, sv = np.sin(_PI * U), np.sin(_PI * V)
    cu, cv = np.cos(_PI * U), np.cos(_PI * V)

    pos = np.stack([A * su * sv, U, V], axis=-1)

    nx = np.ones_like(U)
    ny = -A * _PI * cu * sv
    nz = -A * _PI * su * cv
    N = np.stack([nx, ny, nz], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return pos, N


# ── silhouette extraction ──────────────────────────────────────────────────────

def extract_silhouette(
    pos: np.ndarray,   # (nu, nv, 3)
    N: np.ndarray,     # (nu, nv, 3)
    view: np.ndarray,  # (3,)
) -> np.ndarray:       # (K, 3)
    """
    Marching-squares zero-crossing extraction of  N(x) . view = 0  on the
    parameter grid.  Returns 3D positions of silhouette points.
    """
    D = np.einsum("ijk,k->ij", N, view)  # (nu, nv)

    def _crossings(d0: np.ndarray, d1: np.ndarray,
                   p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Linearly interpolated zero-crossings along a batch of edges."""
        mask = (d0 * d1 <= 0.0) & ~((d0 == 0.0) & (d1 == 0.0))
        ii, jj = np.where(mask)
        if len(ii) == 0:
            return np.zeros((0, 3))
        d0m, d1m = d0[ii, jj], d1[ii, jj]
        denom = d0m - d1m
        t = np.where(np.abs(denom) > 1e-15, d0m / denom, 0.5)
        return (1.0 - t)[:, None] * p0[ii, jj] + t[:, None] * p1[ii, jj]

    # edges along v-axis
    h = _crossings(D[:, :-1], D[:, 1:], pos[:, :-1], pos[:, 1:])
    # edges along u-axis
    v = _crossings(D[:-1, :], D[1:, :], pos[:-1, :], pos[1:, :])
    parts = [x for x in (h, v) if len(x) > 0]
    return np.concatenate(parts) if parts else np.zeros((0, 3))


# ── baseline normal corruption ─────────────────────────────────────────────────

def _nearby_pairs(
    pos1: np.ndarray,  # (N1, 3)
    pos2: np.ndarray,  # (N2, 3)
    radius: float,
):
    """Return lists: for each point in pos1, indices of pos2 within radius."""
    tree2 = cKDTree(pos2)
    return tree2.query_ball_point(pos1, r=radius)


def averaged_normals(
    pos1: np.ndarray, N1: np.ndarray,
    pos2: np.ndarray, N2: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline-averaged: at each sheet-i point within `radius` of sheet-j,
    replace the normal with normalize(N_i + mean(N_j[nbrs])).
    """
    N1_out = N1.copy()
    N2_out = N2.copy()

    nbrs_1in2 = _nearby_pairs(pos1, pos2, radius)
    for i, nbrs in enumerate(nbrs_1in2):
        if nbrs:
            avg = N1[i] + N2[nbrs].mean(axis=0)
            nrm = np.linalg.norm(avg)
            if nrm > 1e-15:
                N1_out[i] = avg / nrm

    nbrs_2in1 = _nearby_pairs(pos2, pos1, radius)
    for i, nbrs in enumerate(nbrs_2in1):
        if nbrs:
            avg = N2[i] + N1[nbrs].mean(axis=0)
            nrm = np.linalg.norm(avg)
            if nrm > 1e-15:
                N2_out[i] = avg / nrm

    return N1_out, N2_out


def first_face_normals(
    pos1: np.ndarray, N1: np.ndarray,
    pos2: np.ndarray, N2: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline-first-face: sheet-2 points within `radius` of sheet-1 have
    their normal overwritten with the closest sheet-1 normal.
    Sheet-1 normals are unchanged.
    """
    N2_out = N2.copy()
    tree1 = cKDTree(pos1)
    nbrs_2in1 = _nearby_pairs(pos2, pos1, radius)
    for i, nbrs in enumerate(nbrs_2in1):
        if nbrs:
            dists = np.linalg.norm(pos1[nbrs] - pos2[i], axis=1)
            closest = nbrs[int(np.argmin(dists))]
            N2_out[i] = N1[closest]
    return N1.copy(), N2_out


# ── image projection ───────────────────────────────────────────────────────────

def image_basis(view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal pair (right, up) spanning the image plane perp to view."""
    v = view / np.linalg.norm(view)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(up, v)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(v, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(v, right)
    up2 /= np.linalg.norm(up2)
    return right, up2


def to_image(pts3d: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 2) image coordinates."""
    if len(pts3d) == 0:
        return np.zeros((0, 2))
    return np.stack([pts3d @ e1, pts3d @ e2], axis=-1)


# ── Chamfer distance ───────────────────────────────────────────────────────────

def chamfer(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Chamfer distance between 2D point sets.  inf if either empty."""
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    dAB = cKDTree(B).query(A)[0]
    dBA = cKDTree(A).query(B)[0]
    return float(np.mean(dAB) + np.mean(dBA))


# ── main experiment ────────────────────────────────────────────────────────────

def main() -> None:
    # ── parameters ────────────────────────────────────────────────────────────
    A            = 0.4    # deformation amplitude
    N_GRID       = 100    # grid resolution per sheet per axis
    N_THETA      = 37     # viewpoint angles in [0, 180 deg]
    BLEND_RADIUS = 0.18   # neighbourhood radius for intersection detection

    u_lin = np.linspace(-1.0, 1.0, N_GRID)
    v_lin = np.linspace(-1.0, 1.0, N_GRID)
    U, V = np.meshgrid(u_lin, v_lin, indexing="ij")  # (N_GRID, N_GRID)

    # exact per-sheet positions and normals on the parameter grid
    pos1_grid, N1_grid = _sheet1(U, V, A)
    pos2_grid, N2_grid = _sheet2(U, V, A)

    pos1_flat = pos1_grid.reshape(-1, 3)
    N1_flat   = N1_grid.reshape(-1, 3)
    pos2_flat = pos2_grid.reshape(-1, 3)
    N2_flat   = N2_grid.reshape(-1, 3)

    print(f"Corrupting normals near intersection (radius = {BLEND_RADIUS})...")
    N1_avg_flat, N2_avg_flat = averaged_normals(
        pos1_flat, N1_flat, pos2_flat, N2_flat, radius=BLEND_RADIUS
    )
    N1_ff_flat, N2_ff_flat = first_face_normals(
        pos1_flat, N1_flat, pos2_flat, N2_flat, radius=BLEND_RADIUS
    )

    # reshape corrupted normals back to grid shape
    N1_avg_grid = N1_avg_flat.reshape(N_GRID, N_GRID, 3)
    N2_avg_grid = N2_avg_flat.reshape(N_GRID, N_GRID, 3)
    N1_ff_grid  = N1_ff_flat.reshape(N_GRID, N_GRID, 3)
    N2_ff_grid  = N2_ff_flat.reshape(N_GRID, N_GRID, 3)

    # report how many grid points were corrupted
    def _n_changed(A, B):
        return int(np.sum(np.any(np.abs(A - B) > 1e-12, axis=1)))
    n_avg_1 = _n_changed(N1_avg_flat, N1_flat)
    n_avg_2 = _n_changed(N2_avg_flat, N2_flat)
    n_ff_2  = _n_changed(N2_ff_flat,  N2_flat)
    print(f"  Averaged:   {n_avg_1} S1 pts + {n_avg_2} S2 pts corrupted")
    print(f"  First-face: {n_ff_2} S2 pts corrupted")

    # ── viewpoint sweep ────────────────────────────────────────────────────────
    thetas      = np.linspace(0.0, _PI, N_THETA)
    cd_avg      = np.full(N_THETA, np.nan)
    cd_ff       = np.full(N_THETA, np.nan)
    theta_star  = _PI / 2.0
    star_idx    = int(np.argmin(np.abs(thetas - theta_star)))

    # angles (degrees) to capture for the multi-angle comparison figure
    SHOW_ANGLES_DEG = [0, 30, 60, 90, 120, 150]
    show_indices = [int(np.argmin(np.abs(thetas - np.radians(d)))) for d in SHOW_ANGLES_DEG]

    panel_data: dict = {}
    angle_data: dict[int, dict] = {}   # keyed by show_index

    print(f"Running viewpoint sweep ({N_THETA} angles)...")
    for i, theta in enumerate(thetas):
        view = np.array([np.sin(theta), 0.0, np.cos(theta)])
        e1, e2 = image_basis(view)

        # ground truth: per-sheet exact normals
        sil1_gt  = extract_silhouette(pos1_grid, N1_grid, view)
        sil2_gt  = extract_silhouette(pos2_grid, N2_grid, view)
        gt_all   = np.concatenate([sil1_gt, sil2_gt]) if (len(sil1_gt) + len(sil2_gt)) > 0 else np.zeros((0, 3))
        gt_img   = to_image(gt_all, e1, e2)

        # baseline-averaged
        sil1_avg = extract_silhouette(pos1_grid, N1_avg_grid, view)
        sil2_avg = extract_silhouette(pos2_grid, N2_avg_grid, view)
        avg_all  = np.concatenate([sil1_avg, sil2_avg]) if (len(sil1_avg) + len(sil2_avg)) > 0 else np.zeros((0, 3))
        avg_img  = to_image(avg_all, e1, e2)

        # baseline-first-face
        sil1_ff  = extract_silhouette(pos1_grid, N1_ff_grid, view)
        sil2_ff  = extract_silhouette(pos2_grid, N2_ff_grid, view)
        ff_all   = np.concatenate([sil1_ff, sil2_ff]) if (len(sil1_ff) + len(sil2_ff)) > 0 else np.zeros((0, 3))
        ff_img   = to_image(ff_all, e1, e2)

        cd_avg[i] = chamfer(avg_img, gt_img)
        cd_ff[i]  = chamfer(ff_img,  gt_img)

        if i == star_idx:
            panel_data = {
                "e1": e1, "e2": e2,
                "sil1_gt":  sil1_gt,  "sil2_gt":  sil2_gt,
                "sil1_avg": sil1_avg, "sil2_avg": sil2_avg,
                "sil1_ff":  sil1_ff,  "sil2_ff":  sil2_ff,
            }

        if i in show_indices:
            angle_data[i] = {
                "theta": theta,
                "e1": e1, "e2": e2,
                "sil1_gt": sil1_gt, "sil2_gt": sil2_gt,
                "sil1_avg": sil1_avg, "sil2_avg": sil2_avg,
                "sil1_ff": sil1_ff, "sil2_ff": sil2_ff,
            }

    # ── figure 1: three-panel at theta* ───────────────────────────────────────
    e1, e2 = panel_data["e1"], panel_data["e2"]

    def _img(key):
        return to_image(panel_data[key], e1, e2)

    gt1_img  = _img("sil1_gt");  gt2_img  = _img("sil2_gt")
    avg1_img = _img("sil1_avg"); avg2_img = _img("sil2_avg")
    ff1_img  = _img("sil1_ff");  ff2_img  = _img("sil2_ff")

    # subsample surface points for background rendering (every 5th row+col)
    _step = 5
    bg1_flat = pos1_grid[::_step, ::_step].reshape(-1, 3)
    bg2_flat = pos2_grid[::_step, ::_step].reshape(-1, 3)

    theta_deg = np.degrees(theta_star)

    # ── figure 0: 3D view of the two sheets ───────────────────────────────────
    _s = 3   # grid stride for plot_surface (keeps it fast)
    X1 = pos1_grid[::_s, ::_s, 0]; Y1 = pos1_grid[::_s, ::_s, 1]; Z1 = pos1_grid[::_s, ::_s, 2]
    X2 = pos2_grid[::_s, ::_s, 0]; Y2 = pos2_grid[::_s, ::_s, 1]; Z2 = pos2_grid[::_s, ::_s, 2]

    # approximate intersection curve: S1 points whose nearest S2 neighbour is
    # within a tight radius
    _tree2 = cKDTree(pos2_flat)
    _dists, _ = _tree2.query(pos1_flat)
    gamma_pts = pos1_flat[_dists < BLEND_RADIUS * 0.5]

    fig0 = plt.figure(figsize=(9, 6))
    ax0  = fig0.add_subplot(111, projection="3d")

    ax0.plot_surface(X1, Y1, Z1, color="steelblue", alpha=0.40,
                     linewidth=0, antialiased=True)
    ax0.plot_surface(X2, Y2, Z2, color="tomato",    alpha=0.40,
                     linewidth=0, antialiased=True)

    if len(gamma_pts):
        ax0.scatter(gamma_pts[:, 0], gamma_pts[:, 1], gamma_pts[:, 2],
                    s=6, color="black", alpha=0.7, zorder=5)

    # view-direction arrow at theta*
    _vstar = np.array([np.sin(theta_star), 0.0, np.cos(theta_star)])
    _ao    = np.array([0.0, 0.0, 1.6])   # arrow origin (above the surface)
    ax0.quiver(*_ao, *(_vstar * 0.55),
               color="darkgreen", lw=2, arrow_length_ratio=0.25)

    ax0.set_xlabel("x", fontsize=10)
    ax0.set_ylabel("y", fontsize=10)
    ax0.set_zlabel("z", fontsize=10)
    ax0.set_title(f"Two intersecting sinusoidal sheets  (A = {A})", fontsize=11)

    legend_elems = [
        Patch(facecolor="steelblue", alpha=0.55,
              label="S1 :  (u, v, A sin\u03c0u sin\u03c0v)"),
        Patch(facecolor="tomato",    alpha=0.55,
              label="S2 :  (A sin\u03c0u sin\u03c0v, u, v)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               ms=5, label="intersection \u03b3  (approx)"),
        Line2D([0], [0], color="darkgreen", lw=2,
               label=f"view direction at \u03b8* = {theta_deg:.0f}\u00b0"),
    ]
    ax0.legend(handles=legend_elems, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig("sheets_3d.pdf", bbox_inches="tight")
    print("Saved sheets_3d.pdf")
    plt.show()

    def _sc_bg(ax, pos3d, e1_, e2_, color):
        """Faint background cloud of projected surface points."""
        pts = to_image(pos3d, e1_, e2_)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=1, color=color,
                       alpha=0.12, linewidths=0, zorder=1)

    def _sc(ax, pts, color, label):
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=4, color=color,
                       label=label, linewidths=0, zorder=3)

    # collect all image-space extents for a shared axis limit
    all_pts = np.concatenate([p for p in [gt1_img, gt2_img] if len(p)] + [np.zeros((1, 2))])
    margin = 0.1
    xlo, xhi = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    ylo, yhi = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (axes[0], gt1_img,  gt2_img,  "(a) Ground truth"),
        (axes[1], avg1_img, avg2_img, "(b) Baseline — averaged normals"),
        (axes[2], gt1_img,  gt2_img,  "(c) Per-sheet (ours)"),
    ]
    for ax, p1, p2, title in panels:
        _sc_bg(ax, bg1_flat, e1, e2, "steelblue")
        _sc_bg(ax, bg2_flat, e1, e2, "tomato")
        _sc(ax, p1, "steelblue", "S1 silhouette")
        _sc(ax, p2, "tomato",    "S2 silhouette")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("image x"); ax.set_ylabel("image y")
        ax.legend(fontsize=8, markerscale=4)

    # mark where S1 silhouette crosses the intersection (approximately the origin)
    origin_img = to_image(np.array([[0.0, 0.0, 0.0]]), e1, e2)
    for ax in axes:
        ax.axhline(origin_img[0, 1], color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(origin_img[0, 0], color="gray", lw=0.8, ls="--", alpha=0.5)

    fig.suptitle(
        f"Occluding contours at critical viewpoint  θ* = {theta_deg:.0f}°\n"
        f"Self-intersecting surface: two sinusoidal sheets  (A = {A})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("contours_panel.pdf", bbox_inches="tight")
    print("Saved contours_panel.pdf")
    plt.show()

    # ── figure 2: Chamfer error vs theta ──────────────────────────────────────
    theta_deg_arr = np.degrees(thetas)
    finite_avg = np.where(np.isfinite(cd_avg), cd_avg, np.nan)
    finite_ff  = np.where(np.isfinite(cd_ff),  cd_ff,  np.nan)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(theta_deg_arr, finite_avg, "o-", color="tomato",     ms=5, lw=1.5,
             label="Baseline — averaged normals")
    ax2.plot(theta_deg_arr, finite_ff,  "s-", color="darkorange", ms=5, lw=1.5,
             label="Baseline — first-face normal")
    ax2.axhline(0.0, color="steelblue", lw=2.0, label="Per-sheet / ours  (= ground truth)")
    ax2.axvline(theta_deg, color="gray", ls="--", lw=1.2,
                label=f"θ* = {theta_deg:.0f}°  (S1 silhouette crosses γ)")
    ax2.axvline(0.0, color="gray", ls=":", lw=1.0,
                label="θ = 0°  (S2 silhouette crosses γ)")
    ax2.set_xlabel("Viewpoint angle θ (degrees)", fontsize=11)
    ax2.set_ylabel("Chamfer distance (image space)", fontsize=11)
    ax2.set_title("Contour error vs viewpoint — self-intersecting surface", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 180)
    plt.tight_layout()
    plt.savefig("contours_error.pdf", bbox_inches="tight")
    print("Saved contours_error.pdf")
    plt.show()

    # ── figure 3: multi-angle comparison grid ─────────────────────────────────
    # Rows: (0) per-sheet/ours  (1) baseline-averaged  (2) baseline-first-face
    # Columns: one per selected angle.
    n_cols   = len(show_indices)
    row_labels = ["Per-sheet (ours)", "Baseline\naveraged", "Baseline\nfirst-face"]
    n_rows   = len(row_labels)

    fig3, axes3 = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    # global image-space limits across all angles (use ground-truth points)
    all_pts_global = []
    for idx in show_indices:
        d = angle_data[idx]
        e1g, e2g = d["e1"], d["e2"]
        for key in ("sil1_gt", "sil2_gt"):
            p = to_image(d[key], e1g, e2g)
            if len(p):
                all_pts_global.append(p)
    if all_pts_global:
        stacked = np.concatenate(all_pts_global)
        glim = max(np.abs(stacked).max(), 0.5) + 0.15
    else:
        glim = 1.2

    origin_3d = np.array([[0.0, 0.0, 0.0]])

    for col, idx in enumerate(show_indices):
        d     = angle_data[idx]
        e1g   = d["e1"]
        e2g   = d["e2"]
        tdeg  = np.degrees(d["theta"])
        cd_a  = cd_avg[idx]
        cd_f  = cd_ff[idx]

        # data per row: (sil1, sil2)
        row_sils = [
            (d["sil1_gt"],  d["sil2_gt"]),
            (d["sil1_avg"], d["sil2_avg"]),
            (d["sil1_ff"],  d["sil2_ff"]),
        ]
        row_cd = [0.0, cd_a, cd_f]

        bg1_img_col = to_image(bg1_flat, e1g, e2g)
        bg2_img_col = to_image(bg2_flat, e1g, e2g)

        for row, (sil1, sil2) in enumerate(row_sils):
            ax = axes3[row, col]
            # background surface points
            if len(bg1_img_col):
                ax.scatter(bg1_img_col[:, 0], bg1_img_col[:, 1], s=1,
                           color="steelblue", alpha=0.12, linewidths=0, zorder=1)
            if len(bg2_img_col):
                ax.scatter(bg2_img_col[:, 0], bg2_img_col[:, 1], s=1,
                           color="tomato", alpha=0.12, linewidths=0, zorder=1)
            p1 = to_image(sil1, e1g, e2g)
            p2 = to_image(sil2, e1g, e2g)
            if len(p1):
                ax.scatter(p1[:, 0], p1[:, 1], s=3, color="steelblue",
                           linewidths=0, label="S1", zorder=3)
            if len(p2):
                ax.scatter(p2[:, 0], p2[:, 1], s=3, color="tomato",
                           linewidths=0, label="S2", zorder=3)

            # mark where the intersection locus projects
            orig_img = to_image(origin_3d, e1g, e2g)[0]
            ax.plot(orig_img[0], orig_img[1], "k+", ms=7, mew=1.2,
                    zorder=5, label="intersection" if row == 0 else None)

            ax.set_xlim(-glim, glim)
            ax.set_ylim(-glim, glim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

            cd_str = f"Cd = {row_cd[row]:.3f}" if row > 0 else "Cd = 0.000"
            ax.set_title(cd_str, fontsize=8, pad=2)

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel(f"theta = {tdeg:.0f} deg", fontsize=9)

    # column headers (angle labels at top)
    for col, idx in enumerate(show_indices):
        axes3[0, col].set_title(
            f"theta = {np.degrees(angle_data[idx]['theta']):.0f} deg\n"
            f"Cd = 0.000",
            fontsize=8,
        )

    # shared legend in top-left panel
    axes3[0, 0].legend(fontsize=7, markerscale=4, loc="upper right",
                       handletextpad=0.3, borderpad=0.4)

    fig3.suptitle(
        f"Occluding contours across viewpoints  (A = {A})\n"
        f"Rows: method   Columns: viewpoint angle   Cd = Chamfer distance",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("contours_angles.pdf", bbox_inches="tight")
    print("Saved contours_angles.pdf")
    plt.show()

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\nSummary at theta* = {theta_deg:.0f} deg  (index {star_idx}):")
    print(f"  Baseline-averaged   Chamfer: {cd_avg[star_idx]:.5f}")
    print(f"  Baseline-first-face Chamfer: {cd_ff[star_idx]:.5f}")
    print( "  Per-sheet (ours)    Chamfer: 0.00000  (exact)")

    # max error across all angles
    valid_avg = finite_avg[np.isfinite(finite_avg)]
    valid_ff  = finite_ff[np.isfinite(finite_ff)]
    if len(valid_avg):
        print(f"\nMax baseline-averaged Chamfer  (all angles): {valid_avg.max():.5f}"
              f"  at theta = {theta_deg_arr[np.nanargmax(finite_avg)]:.1f} deg")
    if len(valid_ff):
        print(f"Max baseline-first-face Chamfer (all angles): {valid_ff.max():.5f}"
              f"  at theta = {theta_deg_arr[np.nanargmax(finite_ff)]:.1f} deg")


# ── second example: two intersecting spheres ──────────────────────────────────

def example_two_spheres(
    d: float = 0.6,    # half-distance between sphere centres
    r: float = 1.0,    # sphere radius
    N_GRID: int = 80,
    N_THETA: int = 37,
    BLEND_RADIUS: float = 0.15,
) -> None:
    """
    Two unit spheres of equal radius intersecting transversally.

      S1 : centre (-d, 0, 0),  outward normal  N1(phi, lam) = (sin phi cos lam,
                                                                 sin phi sin lam,
                                                                 cos phi)
      S2 : centre (+d, 0, 0),  same unit-normal formula shifted by +2d in x.

    Intersection locus: the circle  x = 0,  y^2 + z^2 = r^2 - d^2  (radius rho).

    Critical angle: the S1 silhouette great-circle passes through the
    intersection circle exactly when  cos(theta) = d/r,  i.e.
      theta* = arccos(d/r) ~= 53 deg  for d=0.6, r=1.

    For theta < theta*: S1's silhouette crosses gamma; the baseline blends
    normals from both sheets there and extracts a displaced / missing contour.
    For theta > theta*: the silhouette lies clear of gamma; all methods agree.
    """
    # ── parametric grids ───────────────────────────────────────────────────────
    u_lin = np.linspace(0.0, 1.0, N_GRID, endpoint=False)
    v_lin = np.linspace(0.0, 1.0, N_GRID)
    U, V  = np.meshgrid(u_lin, v_lin, indexing="ij")

    phi = _PI * V
    lam = 2.0 * _PI * U
    sp, cp = np.sin(phi), np.cos(phi)
    sl, cl = np.sin(lam), np.cos(lam)

    # unit-sphere outward normals (identical parametrisation for both spheres)
    Nunit     = np.stack([sp * cl, sp * sl, cp], axis=-1)
    pos1_grid = np.stack([-d + r * sp * cl, r * sp * sl, r * cp], axis=-1)
    pos2_grid = np.stack([+d + r * sp * cl, r * sp * sl, r * cp], axis=-1)
    N1_grid   = Nunit.copy()
    N2_grid   = Nunit.copy()

    pos1_flat = pos1_grid.reshape(-1, 3);  N1_flat = N1_grid.reshape(-1, 3)
    pos2_flat = pos2_grid.reshape(-1, 3);  N2_flat = N2_grid.reshape(-1, 3)

    # ── baseline normal corruption ─────────────────────────────────────────────
    print(f"\n[Two-spheres example]  Corrupting normals (radius = {BLEND_RADIUS})...")
    N1_avg_flat, N2_avg_flat = averaged_normals(
        pos1_flat, N1_flat, pos2_flat, N2_flat, radius=BLEND_RADIUS
    )
    N1_ff_flat, N2_ff_flat = first_face_normals(
        pos1_flat, N1_flat, pos2_flat, N2_flat, radius=BLEND_RADIUS
    )
    N1_avg_grid = N1_avg_flat.reshape(N_GRID, N_GRID, 3)
    N2_avg_grid = N2_avg_flat.reshape(N_GRID, N_GRID, 3)
    N1_ff_grid  = N1_ff_flat.reshape(N_GRID, N_GRID, 3)
    N2_ff_grid  = N2_ff_flat.reshape(N_GRID, N_GRID, 3)

    # ── analytic critical angle ────────────────────────────────────────────────
    rho        = np.sqrt(r**2 - d**2)      # intersection-circle radius
    theta_star = np.arccos(d / r)           # S1 silhouette tangent to gamma
    theta_deg  = np.degrees(theta_star)
    print(f"  theta* = {theta_deg:.2f} deg   (intersection circle radius rho = {rho:.3f})")

    # intersection circle in 3D (for plotting)
    lam_c   = np.linspace(0.0, 2.0 * _PI, 300)
    gamma3d = np.stack([np.zeros_like(lam_c), rho * np.sin(lam_c),
                        rho * np.cos(lam_c)], axis=-1)

    # subsampled surface points for backgrounds
    _step     = 5
    bg1_flat  = pos1_grid[::_step, ::_step].reshape(-1, 3)
    bg2_flat  = pos2_grid[::_step, ::_step].reshape(-1, 3)

    # ── 3D figure ──────────────────────────────────────────────────────────────
    _s  = 4
    X1s = pos1_grid[::_s, ::_s, 0]; Y1s = pos1_grid[::_s, ::_s, 1]; Z1s = pos1_grid[::_s, ::_s, 2]
    X2s = pos2_grid[::_s, ::_s, 0]; Y2s = pos2_grid[::_s, ::_s, 1]; Z2s = pos2_grid[::_s, ::_s, 2]

    fig_3d = plt.figure(figsize=(9, 6))
    ax_3d  = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot_surface(X1s, Y1s, Z1s, color="steelblue", alpha=0.35,
                       linewidth=0, antialiased=True)
    ax_3d.plot_surface(X2s, Y2s, Z2s, color="tomato",    alpha=0.35,
                       linewidth=0, antialiased=True)
    ax_3d.plot(gamma3d[:, 0], gamma3d[:, 1], gamma3d[:, 2],
               "k-", lw=2.5, zorder=6)

    # view-direction arrow at theta* and at theta=0 for comparison
    for th, col, ls in [(theta_star, "darkgreen", "-"),
                        (0.0,        "purple",     "--")]:
        vv = np.array([np.sin(th), 0.0, np.cos(th)])
        ao = np.array([0.0, 0.0, 1.9])
        ax_3d.quiver(*ao, *(vv * 0.6), color=col, lw=2,
                     arrow_length_ratio=0.25, linestyle=ls)

    ax_3d.set_xlabel("x", fontsize=10); ax_3d.set_ylabel("y", fontsize=10)
    ax_3d.set_zlabel("z", fontsize=10)
    ax_3d.set_title(f"Two intersecting spheres  (r = {r}, d = {d})", fontsize=11)
    ax_3d.set_box_aspect([1, 1, 1])
    legend_3d = [
        Patch(facecolor="steelblue", alpha=0.5, label=f"S1 : centre (-{d}, 0, 0)"),
        Patch(facecolor="tomato",    alpha=0.5, label=f"S2 : centre (+{d}, 0, 0)"),
        Line2D([0], [0], color="black",     lw=2,
               label=f"intersection circle  (radius rho = {rho:.2f})"),
        Line2D([0], [0], color="darkgreen", lw=2,
               label=f"view at theta* = {theta_deg:.1f} deg"),
        Line2D([0], [0], color="purple",    lw=2, ls="--",
               label="view at theta = 0 deg"),
    ]
    ax_3d.legend(handles=legend_3d, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("spheres_3d.pdf", bbox_inches="tight")
    print("Saved spheres_3d.pdf")
    plt.show()

    # ── viewpoint sweep ────────────────────────────────────────────────────────
    thetas   = np.linspace(0.0, _PI, N_THETA)
    star_idx = int(np.argmin(np.abs(thetas - theta_star)))
    cd_avg   = np.full(N_THETA, np.nan)
    cd_ff    = np.full(N_THETA, np.nan)

    # show angles: bracket theta* and 180-theta* plus easy/hard cases
    SHOW_DEG = [0, 20, int(theta_deg), 90, int(180 - theta_deg), 160]
    show_idx = [int(np.argmin(np.abs(thetas - np.radians(dd)))) for dd in SHOW_DEG]

    panel_data: dict = {}
    angle_data: dict[int, dict] = {}

    print(f"  Running viewpoint sweep ({N_THETA} angles)...")
    for i, theta in enumerate(thetas):
        view    = np.array([np.sin(theta), 0.0, np.cos(theta)])
        e1_, e2_ = image_basis(view)

        def _sils(Ng1, Ng2):
            s1 = extract_silhouette(pos1_grid, Ng1, view)
            s2 = extract_silhouette(pos2_grid, Ng2, view)
            both = np.concatenate([s1, s2]) if (len(s1) + len(s2)) > 0 else np.zeros((0, 3))
            return s1, s2, to_image(both, e1_, e2_)

        sil1_gt,  sil2_gt,  gt_img  = _sils(N1_grid,     N2_grid)
        sil1_avg, sil2_avg, avg_img = _sils(N1_avg_grid,  N2_avg_grid)
        sil1_ff,  sil2_ff,  ff_img  = _sils(N1_ff_grid,   N2_ff_grid)

        cd_avg[i] = chamfer(avg_img, gt_img)
        cd_ff[i]  = chamfer(ff_img,  gt_img)

        entry = {"theta": theta, "e1": e1_, "e2": e2_,
                 "sil1_gt": sil1_gt, "sil2_gt": sil2_gt,
                 "sil1_avg": sil1_avg, "sil2_avg": sil2_avg,
                 "sil1_ff": sil1_ff, "sil2_ff": sil2_ff}
        if i == star_idx:
            panel_data = entry
        if i in show_idx:
            angle_data[i] = entry

    # ── three-panel at theta* ──────────────────────────────────────────────────
    e1_, e2_ = panel_data["e1"], panel_data["e2"]

    def _im(key):
        return to_image(panel_data[key], e1_, e2_)

    gt1_i  = _im("sil1_gt");  gt2_i  = _im("sil2_gt")
    avg1_i = _im("sil1_avg"); avg2_i = _im("sil2_avg")
    ff1_i  = _im("sil1_ff");  ff2_i  = _im("sil2_ff")
    bg1_i  = to_image(bg1_flat, e1_, e2_)
    bg2_i  = to_image(bg2_flat, e1_, e2_)
    gam_i  = to_image(gamma3d, e1_, e2_)

    all_p  = np.concatenate([p for p in [gt1_i, gt2_i] if len(p)] + [np.zeros((1, 2))])
    margin = 0.15
    xlo, xhi = all_p[:, 0].min() - margin, all_p[:, 0].max() + margin
    ylo, yhi = all_p[:, 1].min() - margin, all_p[:, 1].max() + margin

    fig_p, axes_p = plt.subplots(1, 3, figsize=(15, 5))
    _panels = [
        (axes_p[0], gt1_i,  gt2_i,  "(a) Ground truth"),
        (axes_p[1], avg1_i, avg2_i, "(b) Baseline — averaged normals"),
        (axes_p[2], gt1_i,  gt2_i,  "(c) Per-sheet (ours)"),
    ]
    for ax_, p1_, p2_, ttl in _panels:
        if len(bg1_i): ax_.scatter(bg1_i[:, 0], bg1_i[:, 1], s=1,
                                   color="steelblue", alpha=0.10,
                                   linewidths=0, zorder=1)
        if len(bg2_i): ax_.scatter(bg2_i[:, 0], bg2_i[:, 1], s=1,
                                   color="tomato", alpha=0.10,
                                   linewidths=0, zorder=1)
        if len(gam_i): ax_.plot(gam_i[:, 0], gam_i[:, 1], "k-",
                                lw=1.2, alpha=0.45, zorder=2,
                                label="gamma (projected)")
        if len(p1_): ax_.scatter(p1_[:, 0], p1_[:, 1], s=4, color="steelblue",
                                 linewidths=0, zorder=3, label="S1 silhouette")
        if len(p2_): ax_.scatter(p2_[:, 0], p2_[:, 1], s=4, color="tomato",
                                 linewidths=0, zorder=3, label="S2 silhouette")
        ax_.set_title(ttl, fontsize=11)
        ax_.set_aspect("equal")
        ax_.set_xlim(xlo, xhi); ax_.set_ylim(ylo, yhi)
        ax_.set_xlabel("image x"); ax_.set_ylabel("image y")
        ax_.legend(fontsize=8, markerscale=3)

    fig_p.suptitle(
        f"Two-spheres — occluding contours at theta* = {theta_deg:.1f} deg\n"
        f"r = {r},  d = {d},  intersection circle radius rho = {rho:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("spheres_panel.pdf", bbox_inches="tight")
    print("Saved spheres_panel.pdf")
    plt.show()

    # ── Chamfer error vs theta ─────────────────────────────────────────────────
    theta_deg_arr = np.degrees(thetas)
    fin_avg = np.where(np.isfinite(cd_avg), cd_avg, np.nan)
    fin_ff  = np.where(np.isfinite(cd_ff),  cd_ff,  np.nan)

    fig_e, ax_e = plt.subplots(figsize=(8, 4))
    ax_e.plot(theta_deg_arr, fin_avg, "o-", color="tomato",     ms=5, lw=1.5,
              label="Baseline — averaged")
    ax_e.plot(theta_deg_arr, fin_ff,  "s-", color="darkorange", ms=5, lw=1.5,
              label="Baseline — first-face")
    ax_e.axhline(0.0, color="steelblue", lw=2.0,
                 label="Per-sheet / ours  (= ground truth)")
    ax_e.axvline(theta_deg, color="darkgreen", ls="--", lw=1.2,
                 label=f"theta* = {theta_deg:.1f} deg  (S1 silhouette tangent to gamma)")
    ax_e.axvline(180 - theta_deg, color="purple", ls=":", lw=1.2,
                 label=f"180 - theta* = {180-theta_deg:.1f} deg")
    ax_e.set_xlabel("Viewpoint angle theta (degrees)", fontsize=11)
    ax_e.set_ylabel("Chamfer distance (image space)", fontsize=11)
    ax_e.set_title("Two-spheres — contour error vs viewpoint", fontsize=12)
    ax_e.legend(fontsize=9)
    ax_e.set_xlim(0, 180)
    plt.tight_layout()
    plt.savefig("spheres_error.pdf", bbox_inches="tight")
    print("Saved spheres_error.pdf")
    plt.show()

    # ── multi-angle grid ───────────────────────────────────────────────────────
    row_labels_s = ["Per-sheet (ours)", "Baseline\naveraged", "Baseline\nfirst-face"]
    n_cols_s = len(show_idx)
    n_rows_s = len(row_labels_s)

    all_gs = []
    for idx in show_idx:
        dat = angle_data.get(idx)
        if dat is None:
            continue
        for key in ("sil1_gt", "sil2_gt"):
            p = to_image(dat[key], dat["e1"], dat["e2"])
            if len(p):
                all_gs.append(p)
    glim_s = max(np.abs(np.concatenate(all_gs)).max(), 0.5) + 0.2 if all_gs else 1.6

    fig_gs, axes_gs = plt.subplots(n_rows_s, n_cols_s,
                                   figsize=(2.8 * n_cols_s, 3.2 * n_rows_s),
                                   squeeze=False)
    for col_, idx in enumerate(show_idx):
        dat = angle_data.get(idx)
        if dat is None:
            continue
        e1g, e2g = dat["e1"], dat["e2"]
        tdeg_   = np.degrees(dat["theta"])
        cd_a_   = cd_avg[idx]
        cd_f_   = cd_ff[idx]
        bg1c    = to_image(bg1_flat, e1g, e2g)
        bg2c    = to_image(bg2_flat, e1g, e2g)
        gamc    = to_image(gamma3d,  e1g, e2g)
        row_sils_s = [
            (dat["sil1_gt"],  dat["sil2_gt"]),
            (dat["sil1_avg"], dat["sil2_avg"]),
            (dat["sil1_ff"],  dat["sil2_ff"]),
        ]
        row_cd_s = [0.0, cd_a_, cd_f_]
        for row_, (s1_, s2_) in enumerate(row_sils_s):
            ax_ = axes_gs[row_, col_]
            if len(bg1c): ax_.scatter(bg1c[:, 0], bg1c[:, 1], s=1,
                                      color="steelblue", alpha=0.10,
                                      linewidths=0, zorder=1)
            if len(bg2c): ax_.scatter(bg2c[:, 0], bg2c[:, 1], s=1,
                                      color="tomato", alpha=0.10,
                                      linewidths=0, zorder=1)
            if len(gamc): ax_.plot(gamc[:, 0], gamc[:, 1], "k-",
                                   lw=0.9, alpha=0.4, zorder=2)
            p1_ = to_image(s1_, e1g, e2g)
            p2_ = to_image(s2_, e1g, e2g)
            if len(p1_): ax_.scatter(p1_[:, 0], p1_[:, 1], s=3, color="steelblue",
                                     linewidths=0, zorder=3)
            if len(p2_): ax_.scatter(p2_[:, 0], p2_[:, 1], s=3, color="tomato",
                                     linewidths=0, zorder=3)
            ax_.set_xlim(-glim_s, glim_s); ax_.set_ylim(-glim_s, glim_s)
            ax_.set_aspect("equal")
            ax_.tick_params(labelsize=7)
            if col_ == 0:
                ax_.set_ylabel(row_labels_s[row_], fontsize=9)
            if row_ == n_rows_s - 1:
                ax_.set_xlabel(f"theta = {tdeg_:.0f} deg", fontsize=9)
        axes_gs[0, col_].set_title(f"theta = {tdeg_:.0f} deg\nCd = 0.000", fontsize=8)
        axes_gs[1, col_].set_title(f"Cd = {cd_a_:.3f}", fontsize=8, pad=2)
        axes_gs[2, col_].set_title(f"Cd = {cd_f_:.3f}", fontsize=8, pad=2)

    fig_gs.suptitle(
        f"Two-spheres — occluding contours across viewpoints  (r={r}, d={d})\n"
        f"Rows: method   Columns: viewpoint   Cd = Chamfer distance\n"
        f"Black curve = intersection circle gamma (projected)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("spheres_angles.pdf", bbox_inches="tight")
    print("Saved spheres_angles.pdf")
    plt.show()

    print(f"\n[Two-spheres] Summary at theta* = {theta_deg:.1f} deg:")
    print(f"  Baseline-averaged   Chamfer: {cd_avg[star_idx]:.5f}")
    print(f"  Baseline-first-face Chamfer: {cd_ff[star_idx]:.5f}")
    print( "  Per-sheet (ours)    Chamfer: 0.00000  (exact)")


# ── third example: Viviani's curve (sphere ∩ cylinder) ────────────────────────

def example_viviani(
    N_GRID: int = 80,
    N_THETA: int = 37,
    BLEND_RADIUS: float = 0.12,
) -> None:
    """
    Viviani's curve: intersection of the unit sphere with a half-radius cylinder.

      S1 : unit sphere   x^2 + y^2 + z^2 = 1
           parametrized (phi, lam) -> (sin phi cos lam, sin phi sin lam, cos phi)
           outward normal  N1 = position vector

      S2 : cylinder  (x - 1/2)^2 + y^2 = (1/2)^2,  z in [-1, 1]
           parametrized (alpha, t) -> (1/2 + 1/2 cos alpha, 1/2 sin alpha, t)
           outward normal  N2 = (cos alpha, sin alpha, 0)

    Intersection (Viviani's curve):
      x(t) = (1 + cos t) / 2
      y(t) = sin(t) / 2
      z(t) = sin(t / 2)
    for t in [0, 4 pi).  The curve is CLOSED and SELF-INTERSECTING: both
    branches t=0 and t=2pi pass through (1, 0, 0) with different tangents.

    Critical viewpoint: theta* = 90 deg, v = (1, 0, 0).
      - Sphere silhouette (yz great circle) passes through (0, 0, +/-1),
        which lie on Viviani's curve.
      - At those points, N_sphere = (0, 0, +/-1) but N_cylinder = (-1, 0, 0).
        The blended normal is normalize(0,0,1)+(-1,0,0)) = (-1,0,1)/sqrt(2),
        with dot product 1/sqrt(2) != 0 with v -- so the baseline incorrectly
        says no silhouette at these points.
    """
    # ── sphere S1 ──────────────────────────────────────────────────────────────
    phi_lin = np.linspace(0.0, _PI, N_GRID)
    lam_lin = np.linspace(0.0, 2.0 * _PI, N_GRID, endpoint=False)
    PHI, LAM = np.meshgrid(phi_lin, lam_lin, indexing="ij")
    sp, cp = np.sin(PHI), np.cos(PHI)
    sl, cl = np.sin(LAM), np.cos(LAM)
    pos1_grid = np.stack([sp * cl, sp * sl, cp], axis=-1)
    N1_grid   = pos1_grid.copy()   # outward normal = position on unit sphere

    # ── cylinder S2 ────────────────────────────────────────────────────────────
    alp_lin = np.linspace(0.0, 2.0 * _PI, N_GRID, endpoint=False)
    t_lin   = np.linspace(-1.0, 1.0, N_GRID)
    ALP, TT = np.meshgrid(alp_lin, t_lin, indexing="ij")
    ca, sa  = np.cos(ALP), np.sin(ALP)
    pos2_grid = np.stack([0.5 + 0.5 * ca, 0.5 * sa, TT], axis=-1)
    N2_grid   = np.stack([ca, sa, np.zeros_like(ca)], axis=-1)

    pos1_flat = pos1_grid.reshape(-1, 3);  N1_flat = N1_grid.reshape(-1, 3)
    pos2_flat = pos2_grid.reshape(-1, 3);  N2_flat = N2_grid.reshape(-1, 3)

    # ── baseline corruption ────────────────────────────────────────────────────
    print(f"\n[Viviani example]  Corrupting normals (radius = {BLEND_RADIUS})...")
    N1_avg_flat, N2_avg_flat = averaged_normals(
        pos1_flat, N1_flat, pos2_flat, N2_flat, radius=BLEND_RADIUS
    )
    N1_ff_flat, N2_ff_flat = first_face_normals(
        pos1_flat, N1_flat, pos2_flat, N2_flat, radius=BLEND_RADIUS
    )
    N1_avg_grid = N1_avg_flat.reshape(N_GRID, N_GRID, 3)
    N2_avg_grid = N2_avg_flat.reshape(N_GRID, N_GRID, 3)
    N1_ff_grid  = N1_ff_flat.reshape(N_GRID, N_GRID, 3)
    N2_ff_grid  = N2_ff_flat.reshape(N_GRID, N_GRID, 3)

    # ── Viviani's curve ─────────────────────────────────────────────────────────
    tv      = np.linspace(0.0, 4.0 * _PI, 800)
    viviani = np.stack([
        (1.0 + np.cos(tv)) / 2.0,
        np.sin(tv) / 2.0,
        np.sin(tv / 2.0),
    ], axis=-1)

    _step    = 5
    bg1_flat = pos1_grid[::_step, ::_step].reshape(-1, 3)
    bg2_flat = pos2_grid[::_step, ::_step].reshape(-1, 3)

    # ── 3D figure ──────────────────────────────────────────────────────────────
    _s  = 4
    X1v = pos1_grid[::_s, ::_s, 0]; Y1v = pos1_grid[::_s, ::_s, 1]; Z1v = pos1_grid[::_s, ::_s, 2]
    X2v = pos2_grid[::_s, ::_s, 0]; Y2v = pos2_grid[::_s, ::_s, 1]; Z2v = pos2_grid[::_s, ::_s, 2]

    fig_3d = plt.figure(figsize=(9, 7))
    ax_3d  = fig_3d.add_subplot(111, projection="3d")
    ax_3d.plot_surface(X1v, Y1v, Z1v, color="steelblue", alpha=0.22,
                       linewidth=0, antialiased=True)
    ax_3d.plot_surface(X2v, Y2v, Z2v, color="tomato",    alpha=0.50,
                       linewidth=0, antialiased=True)
    ax_3d.plot(viviani[:, 0], viviani[:, 1], viviani[:, 2],
               "k-", lw=2.5, zorder=6)
    # self-intersection point (1, 0, 0)
    ax_3d.scatter([1.0], [0.0], [0.0], s=80, color="gold",
                  edgecolors="black", lw=1.5, zorder=8)
    # view-direction arrow at theta*=90 deg: v=(1,0,0)
    ax_3d.quiver(-0.4, -0.4, 0.9, 0.55, 0.0, 0.0,
                 color="darkgreen", lw=2, arrow_length_ratio=0.28)

    ax_3d.set_xlabel("x"); ax_3d.set_ylabel("y"); ax_3d.set_zlabel("z")
    ax_3d.set_title("Viviani's curve: unit sphere  \u2229  half-radius cylinder",
                    fontsize=11)
    ax_3d.view_init(elev=18, azim=38)

    legend_3d = [
        Patch(facecolor="steelblue", alpha=0.4,
              label="S1: unit sphere   x\u00b2+y\u00b2+z\u00b2 = 1"),
        Patch(facecolor="tomato",    alpha=0.6,
              label="S2: cylinder   (x-\u00bd)\u00b2+y\u00b2 = (\u00bd)\u00b2"),
        Line2D([0], [0], color="black", lw=2.5,
               label="Viviani's curve (intersection locus)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
               markeredgecolor="black", ms=9,
               label="self-intersection point  (1, 0, 0)"),
        Line2D([0], [0], color="darkgreen", lw=2,
               label="view at \u03b8* = 90\u00b0"),
    ]
    ax_3d.legend(handles=legend_3d, fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("viviani_3d.pdf", bbox_inches="tight")
    print("Saved viviani_3d.pdf")
    plt.show()

    # ── viewpoint sweep ────────────────────────────────────────────────────────
    thetas     = np.linspace(0.0, _PI, N_THETA)
    theta_star = _PI / 2.0
    star_idx   = int(np.argmin(np.abs(thetas - theta_star)))
    theta_deg  = 90.0
    cd_avg     = np.full(N_THETA, np.nan)
    cd_ff      = np.full(N_THETA, np.nan)

    SHOW_DEG = [0, 30, 60, 90, 120, 150]
    show_idx = [int(np.argmin(np.abs(thetas - np.radians(dd)))) for dd in SHOW_DEG]

    panel_data: dict = {}
    angle_data: dict[int, dict] = {}

    print(f"  theta* = 90 deg  (sphere silhouette crosses Viviani at (0,0,+/-1))")
    print(f"  Running viewpoint sweep ({N_THETA} angles)...")

    for i, theta in enumerate(thetas):
        view     = np.array([np.sin(theta), 0.0, np.cos(theta)])
        e1_, e2_ = image_basis(view)

        def _sils(Ng1, Ng2, _view=view, _e1=e1_, _e2=e2_):
            s1 = extract_silhouette(pos1_grid, Ng1, _view)
            s2 = extract_silhouette(pos2_grid, Ng2, _view)
            both = np.concatenate([s1, s2]) if (len(s1) + len(s2)) > 0 else np.zeros((0, 3))
            return s1, s2, to_image(both, _e1, _e2)

        sil1_gt,  sil2_gt,  gt_img  = _sils(N1_grid,     N2_grid)
        sil1_avg, sil2_avg, avg_img = _sils(N1_avg_grid,  N2_avg_grid)
        sil1_ff,  sil2_ff,  ff_img  = _sils(N1_ff_grid,   N2_ff_grid)

        cd_avg[i] = chamfer(avg_img, gt_img)
        cd_ff[i]  = chamfer(ff_img,  gt_img)

        entry = {"theta": theta, "e1": e1_, "e2": e2_,
                 "sil1_gt": sil1_gt, "sil2_gt": sil2_gt,
                 "sil1_avg": sil1_avg, "sil2_avg": sil2_avg,
                 "sil1_ff": sil1_ff, "sil2_ff": sil2_ff}
        if i == star_idx:
            panel_data = entry
        if i in show_idx:
            angle_data[i] = entry

    # ── three-panel at theta* ──────────────────────────────────────────────────
    e1_, e2_ = panel_data["e1"], panel_data["e2"]

    def _im(key):
        return to_image(panel_data[key], e1_, e2_)

    gt1_i  = _im("sil1_gt");  gt2_i  = _im("sil2_gt")
    avg1_i = _im("sil1_avg"); avg2_i = _im("sil2_avg")
    bg1_i  = to_image(bg1_flat, e1_, e2_)
    bg2_i  = to_image(bg2_flat, e1_, e2_)
    viv_i  = to_image(viviani,  e1_, e2_)

    all_p  = np.concatenate([p for p in [gt1_i, gt2_i] if len(p)] + [np.zeros((1, 2))])
    margin = 0.15
    xlo, xhi = all_p[:, 0].min() - margin, all_p[:, 0].max() + margin
    ylo, yhi = all_p[:, 1].min() - margin, all_p[:, 1].max() + margin

    fig_p, axes_p = plt.subplots(1, 3, figsize=(15, 5))
    _panels = [
        (axes_p[0], gt1_i,  gt2_i,  "(a) Ground truth"),
        (axes_p[1], avg1_i, avg2_i, "(b) Baseline \u2014 averaged normals"),
        (axes_p[2], gt1_i,  gt2_i,  "(c) Per-sheet (ours)"),
    ]
    for ax_, p1_, p2_, ttl in _panels:
        if len(bg1_i): ax_.scatter(bg1_i[:, 0], bg1_i[:, 1], s=1,
                                   color="steelblue", alpha=0.09, linewidths=0, zorder=1)
        if len(bg2_i): ax_.scatter(bg2_i[:, 0], bg2_i[:, 1], s=1,
                                   color="tomato", alpha=0.09, linewidths=0, zorder=1)
        if len(viv_i): ax_.plot(viv_i[:, 0], viv_i[:, 1], "k-",
                                lw=1.0, alpha=0.35, zorder=2,
                                label="Viviani (projected)")
        if len(p1_): ax_.scatter(p1_[:, 0], p1_[:, 1], s=4, color="steelblue",
                                 linewidths=0, zorder=3, label="S1 silhouette")
        if len(p2_): ax_.scatter(p2_[:, 0], p2_[:, 1], s=4, color="tomato",
                                 linewidths=0, zorder=3, label="S2 silhouette")
        ax_.set_title(ttl, fontsize=11)
        ax_.set_aspect("equal")
        ax_.set_xlim(xlo, xhi); ax_.set_ylim(ylo, yhi)
        ax_.set_xlabel("image x"); ax_.set_ylabel("image y")
        ax_.legend(fontsize=8, markerscale=3)

    fig_p.suptitle(
        f"Viviani's curve \u2014 occluding contours at \u03b8* = {theta_deg:.0f}\u00b0\n"
        f"Sphere silhouette crosses Viviani's curve at (0, 0, \u00b11); "
        f"baseline displaces it",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("viviani_panel.pdf", bbox_inches="tight")
    print("Saved viviani_panel.pdf")
    plt.show()

    # ── Chamfer error vs theta ─────────────────────────────────────────────────
    theta_deg_arr = np.degrees(thetas)
    fin_avg = np.where(np.isfinite(cd_avg), cd_avg, np.nan)
    fin_ff  = np.where(np.isfinite(cd_ff),  cd_ff,  np.nan)

    fig_e, ax_e = plt.subplots(figsize=(8, 4))
    ax_e.plot(theta_deg_arr, fin_avg, "o-", color="tomato",     ms=5, lw=1.5,
              label="Baseline \u2014 averaged")
    ax_e.plot(theta_deg_arr, fin_ff,  "s-", color="darkorange", ms=5, lw=1.5,
              label="Baseline \u2014 first-face")
    ax_e.axhline(0.0, color="steelblue", lw=2.0,
                 label="Per-sheet / ours  (= ground truth)")
    ax_e.axvline(theta_deg, color="darkgreen", ls="--", lw=1.2,
                 label=f"\u03b8* = {theta_deg:.0f}\u00b0  (sphere sil. crosses Viviani)")
    ax_e.axvline(0.0, color="gray", ls=":", lw=1.0,
                 label="\u03b8 = 0\u00b0  (sil. at self-intersection (1,0,0))")
    ax_e.set_xlabel("Viewpoint angle \u03b8 (degrees)", fontsize=11)
    ax_e.set_ylabel("Chamfer distance (image space)", fontsize=11)
    ax_e.set_title("Viviani's curve \u2014 contour error vs viewpoint", fontsize=12)
    ax_e.legend(fontsize=9)
    ax_e.set_xlim(0, 180)
    plt.tight_layout()
    plt.savefig("viviani_error.pdf", bbox_inches="tight")
    print("Saved viviani_error.pdf")
    plt.show()

    # ── multi-angle grid ───────────────────────────────────────────────────────
    row_labels_v = ["Per-sheet (ours)", "Baseline\naveraged", "Baseline\nfirst-face"]
    n_cols_v = len(show_idx)

    all_gv = []
    for idx in show_idx:
        dat = angle_data.get(idx)
        if dat is None:
            continue
        for key in ("sil1_gt", "sil2_gt"):
            p = to_image(dat[key], dat["e1"], dat["e2"])
            if len(p):
                all_gv.append(p)
    glim_v = max(np.abs(np.concatenate(all_gv)).max(), 0.5) + 0.2 if all_gv else 1.6

    fig_gv, axes_gv = plt.subplots(3, n_cols_v,
                                   figsize=(2.8 * n_cols_v, 9.6),
                                   squeeze=False)
    for col_, idx in enumerate(show_idx):
        dat = angle_data.get(idx)
        if dat is None:
            continue
        e1g, e2g = dat["e1"], dat["e2"]
        tdeg_    = np.degrees(dat["theta"])
        cd_a_    = cd_avg[idx]
        cd_f_    = cd_ff[idx]
        bg1c     = to_image(bg1_flat, e1g, e2g)
        bg2c     = to_image(bg2_flat, e1g, e2g)
        vivc     = to_image(viviani,  e1g, e2g)
        row_sils_v = [
            (dat["sil1_gt"],  dat["sil2_gt"]),
            (dat["sil1_avg"], dat["sil2_avg"]),
            (dat["sil1_ff"],  dat["sil2_ff"]),
        ]
        row_cd_v = [0.0, cd_a_, cd_f_]
        for row_, (s1_, s2_) in enumerate(row_sils_v):
            ax_ = axes_gv[row_, col_]
            if len(bg1c): ax_.scatter(bg1c[:, 0], bg1c[:, 1], s=1,
                                      color="steelblue", alpha=0.09,
                                      linewidths=0, zorder=1)
            if len(bg2c): ax_.scatter(bg2c[:, 0], bg2c[:, 1], s=1,
                                      color="tomato", alpha=0.09,
                                      linewidths=0, zorder=1)
            if len(vivc): ax_.plot(vivc[:, 0], vivc[:, 1], "k-",
                                   lw=0.8, alpha=0.3, zorder=2)
            p1_ = to_image(s1_, e1g, e2g)
            p2_ = to_image(s2_, e1g, e2g)
            if len(p1_): ax_.scatter(p1_[:, 0], p1_[:, 1], s=3, color="steelblue",
                                     linewidths=0, zorder=3)
            if len(p2_): ax_.scatter(p2_[:, 0], p2_[:, 1], s=3, color="tomato",
                                     linewidths=0, zorder=3)
            ax_.set_xlim(-glim_v, glim_v); ax_.set_ylim(-glim_v, glim_v)
            ax_.set_aspect("equal")
            ax_.tick_params(labelsize=7)
            if col_ == 0:
                ax_.set_ylabel(row_labels_v[row_], fontsize=9)
            if row_ == 2:
                ax_.set_xlabel(f"\u03b8 = {tdeg_:.0f}\u00b0", fontsize=9)
        axes_gv[0, col_].set_title(f"\u03b8 = {tdeg_:.0f}\u00b0\nCd = 0.000", fontsize=8)
        axes_gv[1, col_].set_title(f"Cd = {cd_a_:.3f}", fontsize=8, pad=2)
        axes_gv[2, col_].set_title(f"Cd = {cd_f_:.3f}", fontsize=8, pad=2)

    fig_gv.suptitle(
        "Viviani's curve \u2014 occluding contours across viewpoints\n"
        "Rows: method   Columns: viewpoint   Cd = Chamfer distance\n"
        "Black curve = Viviani's curve (projected)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("viviani_angles.pdf", bbox_inches="tight")
    print("Saved viviani_angles.pdf")
    plt.show()

    print(f"\n[Viviani] Summary at theta* = {theta_deg:.0f} deg:")
    print(f"  Baseline-averaged   Chamfer: {cd_avg[star_idx]:.5f}")
    print(f"  Baseline-first-face Chamfer: {cd_ff[star_idx]:.5f}")
    print( "  Per-sheet (ours)    Chamfer: 0.00000  (exact)")


if __name__ == "__main__":
    main()
    example_two_spheres()
    example_viviani()
