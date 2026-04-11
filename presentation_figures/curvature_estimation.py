"""
Curvature Estimation: Classical Local Quadric vs Lifted Blow-Up
----------------------------------------------------------------
For self-intersecting and non-manifold shapes, compare:

  Classical : local quadric fit in Euclidean k-NN  (z = ax^2 + bxy + cy^2;
              H = a+c, K = 4ac - b^2).  Near self-intersections the
              Euclidean k-NN mixes points from different sheets, corrupting
              the estimate.

  Lifted    : second fundamental form from the level-1 tangent blow-up.
              k-NN in the Chordal-Sasaki metric (position + projector)
              stays on the same sheet, giving correct per-sheet curvature.

Figure per example -- 2 rows x 2 cols (or 2x3 with ground truth):
  (0,0) Classical mean curvature H      (0,1) Lifted mean curvature H
  (1,0) Classical Gaussian curvature K  (1,1) Lifted Gaussian curvature K

Toy surfaces  : whitney_umbrella, cube_surface, klein_bottle, cone,
                cylinders_tangent, plane_paraboloid_tangent, tangent_spheres
Thingi10k     : klein_bottle_one, klein_bottle_two, icicles
Curves (2D)   : figure8, tangent_parabola_line
Curves (3D)   : helix, figure8_space, trefoil_knot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from tangent_blowups.testsupport import (
    RandomSurface, UniformCurve, sample,
    whitney_umbrella, cube_surface, klein_bottle, cone,
    cylinders_tangent, plane_paraboloid_tangent, tangent_spheres,
    figure8, tangent_parabola_line,
    helix, figure8_space, trefoil_knot,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, extract_level1
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K      = 40      # k-NN for both methods
LAM    = 1e-3    # ridge regularisation for lifted curvature
ALPHA  = 1.0  # Chordal-Sasaki weight
N_TOY  = 25000
N_REAL = 25000
ELEV, AZIM = 25, 45
CMAP   = "RdBu_r"
PT_SIZE = 4.0

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "thingi10k_pointcloud"


# ---------------------------------------------------------------------------
# Classical curvature: local quadric fit
# ---------------------------------------------------------------------------

def _tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal tangent vectors perpendicular to unit normal n."""
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)  # already unit
    return t1, t2


def classical_curvature(pts: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean (H), Gaussian (K), and total curvature via local quadric fitting.

    In each point's local frame, fits z = a*x^2 + b*x*y + c*y^2, then:
        H          = a + c             (mean curvature, sum of principal curvatures)
        K          = 4ac - b^2         (Gaussian curvature)
        total_curv = ||h||_F           (Frobenius norm of shape operator [[a,b/2],[b/2,c]])
                   = sqrt(a^2 + b^2/2 + c^2)

    total_curv is sign-invariant (unlike H which depends on normal orientation).
    Uses Euclidean k-NN, so at self-intersections neighbors from both
    sheets contaminate the fit.
    """
    N = len(pts)
    H  = np.full(N, np.nan)
    Kg = np.full(N, np.nan)
    Tc = np.full(N, np.nan)

    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=K + 1)
    nn_idx = nn_idx[:, 1:]  # exclude self

    for i in range(N):
        ni = normals[i]
        nrm = np.linalg.norm(ni)
        if not np.isfinite(ni).all() or nrm < 1e-8:
            continue
        ni = ni / nrm
        t1, t2 = _tangent_basis(ni)

        nbs = pts[nn_idx[i]] - pts[i]   # (k, 3)
        x = nbs @ t1
        y = nbs @ t2
        z = nbs @ ni

        A = np.column_stack([x**2, x * y, y**2])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b, c = coeffs
        H[i]  = a + c
        Kg[i] = 4.0 * a * c - b**2
        Tc[i] = float(np.sqrt(a**2 + 0.5 * b**2 + c**2))

    return H, Kg, Tc


# ---------------------------------------------------------------------------
# Lifted curvature: level-1 blow-up
# ---------------------------------------------------------------------------

def lifted_curvature(
    pts: np.ndarray,
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Mean (H), Gaussian (K), and total curvature from the level-1 tangent blow-up.

    total_curvature = ||h||_F = sqrt(kappa1^2 + kappa2^2) — sign-invariant.
    H sign depends on QR-determined normal orientation and may flip across
    the surface; total_curvature and K are orientation-independent.
    Returns H (N,), K (N,) or None, total_curv (N,).
    """
    level0 = BlowUpLevel.from_point_tangents(pts, frames)
    level1 = level0.lift(k=K, alpha=ALPHA, lam=LAM)
    inv = extract_level1(level1)
    H  = inv.mean_curvature[:, 0]   # sign depends on normal orientation
    Kg = inv.gaussian_curvature     # (N,) or None — orientation-invariant
    Tc = inv.total_curvature[:, 0]  # ||h||_F — orientation-invariant
    return H, Kg, Tc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _sample_toy(surface_fn, *, u_bounds, v_bounds):
    """Sample a parametric surface; return (pts, frames, normals, params)."""
    surf = surface_fn()
    rng  = np.random.default_rng(42)
    s = sample(surf,
               RandomSurface(n=N_TOY, u_bounds=u_bounds, v_bounds=v_bounds, rng=rng),
               with_tangents=True, with_normals=True)
    pts    = np.asarray(s.points,   dtype=float)
    frames = np.asarray(s.tangents, dtype=float)
    nrm    = np.asarray(s.normals,  dtype=float)
    u_p    = np.asarray(s.params[0], dtype=float).ravel()
    v_p    = np.asarray(s.params[1], dtype=float).ravel()

    valid = (np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
             & (np.linalg.norm(nrm, axis=1) > 1e-8))
    pts, frames, nrm = pts[valid], frames[valid], nrm[valid]
    params = (u_p[valid], v_p[valid])
    return pts, frames, normalize_vectors(nrm), params


# ---------------------------------------------------------------------------
# Ground truth curvature for toy examples
# ---------------------------------------------------------------------------

def _cone_gt(params, scale_r: float = 1.0, scale_z: float = 1.0):
    """
    Analytical curvature for the default cone parametrisation.

    For cone(scale_r, scale_z):
        H_true = scale_z / (2 * scale_r * u * sqrt(scale_r^2 + scale_z^2))
        K_true = 0  (cone is developable)
    where u is the radial parameter (u > 0).
    """
    u = params[0]
    denom = 2.0 * scale_r * u * np.sqrt(scale_r**2 + scale_z**2)
    H_true = scale_z / denom
    K_true = np.zeros_like(u)
    return H_true, K_true


def _cube_gt(params, edge_margin: float = 0.18):
    """
    Analytical curvature for cube_surface.

    Each face is flat: H = 0, K = 0.
    Points near face edges (within edge_margin fraction of face extent)
    are masked to NaN -- they neighbour another face and curvature
    is not well-defined.
    """
    u, v = params
    s_local = u - np.floor(u)   # position within face [0, 1)
    interior = ((s_local > edge_margin) & (s_local < 1.0 - edge_margin) &
                (v       > edge_margin) & (v       < 1.0 - edge_margin))
    H_true = np.where(interior, 0.0, np.nan)
    K_true = np.where(interior, 0.0, np.nan)
    return H_true, K_true


# ---------------------------------------------------------------------------
# Thingi10k loading
# ---------------------------------------------------------------------------

def _load_thingi(name: str):
    """Load thingi10k point cloud; return (pts, frames, normals)."""
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    pc  = load_pointcloud(path)
    pts = np.asarray(pc.points,  dtype=float).reshape(-1, 3)
    nrm = np.asarray(pc.normals, dtype=float).reshape(-1, 3)

    valid = (np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
             & (np.linalg.norm(nrm, axis=1) > 1e-8))
    pts, nrm = pts[valid], normalize_vectors(nrm[valid])

    if len(pts) > N_REAL:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N_REAL, replace=False)
        pts, nrm = pts[idx], nrm[idx]

    frames = BlowUpLevel.from_normals(pts, nrm).frame
    return pts, frames, nrm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _clean_ax3d(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=ELEV, azim=AZIM)


def _equal_aspect(ax, pts):
    lims = np.array([[pts[:, i].min(), pts[:, i].max()] for i in range(3)])
    r = 0.5 * (lims[:, 1] - lims[:, 0]).max()
    c = lims.mean(axis=1)
    for i, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter([c[i] - r, c[i] + r])


def _scatter_curv(ax, pts, vals, title: str, pct_clip: float = 2.0):
    """Scatter plot with symmetric, percentile-clipped color scale."""
    finite = vals[np.isfinite(vals)]
    if finite.size:
        lo, hi = np.percentile(finite, [pct_clip, 100 - pct_clip])
        lim = max(abs(lo), abs(hi), 1e-12)
    else:
        lim = 1.0
    plot_vals = np.where(np.isfinite(vals), vals, 0.0)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=plot_vals, s=PT_SIZE, cmap=CMAP,
               vmin=-lim, vmax=lim, linewidths=0, rasterized=True)
    _clean_ax3d(ax)
    _equal_aspect(ax, pts)
    ax.set_title(title, fontsize=9, pad=3)


def _scatter_pos(ax, pts, vals, title: str, pct_clip: float = 2.0):
    """Scatter plot for non-negative data (sequential colormap, percentile-clipped)."""
    finite = vals[np.isfinite(vals)]
    vmax = float(np.percentile(finite, 100 - pct_clip)) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    plot_vals = np.where(np.isfinite(vals), vals, 0.0)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=plot_vals, s=PT_SIZE, cmap="viridis",
               vmin=0.0, vmax=vmax, linewidths=0, rasterized=True)
    _clean_ax3d(ax)
    _equal_aspect(ax, pts)
    ax.set_title(title, fontsize=9, pad=3)


def _scatter_err(ax, pts, err, title: str):
    """Scatter plot of absolute error -- sequential colormap, clipped at 95th pct."""
    finite = err[np.isfinite(err)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    plot_vals = np.where(np.isfinite(err), err, 0.0)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=plot_vals, s=PT_SIZE, cmap="YlOrRd",
               vmin=0.0, vmax=vmax, linewidths=0, rasterized=True)
    _clean_ax3d(ax)
    _equal_aspect(ax, pts)
    ax.set_title(title, fontsize=9, pad=3)


def _make_figure(pts, H_cl, K_cl, Tc_cl, H_li, K_li, Tc_li, title: str, fname: str,
                 H_true=None, K_true=None):
    has_gt = H_true is not None
    n_cols = 3 if has_gt else 2
    fig, axes = plt.subplots(3, n_cols, figsize=(5.0 * n_cols, 12.5),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")

    # --- row 0: H ---
    t_cl, t_li = "Classical H (mean curvature)", "Lifted H (mean curvature)"

    _scatter_curv(axes[0, 0], pts, H_cl, t_cl)
    _scatter_curv(axes[0, 1], pts, H_li, t_li)
    if has_gt:
        err_H_cl = np.abs(H_cl - H_true)
        err_H_li = np.abs(H_li - H_true)
        _scatter_err(axes[0, 2], pts,
                     np.where(np.isfinite(H_true), err_H_cl, np.nan),
                     "|H error|  Classical (orange) / Lifted (blue below)")
        axes[0, 2].scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=np.where(np.isfinite(H_true), err_H_li, 0.0),
                           s=PT_SIZE, cmap="Blues",
                           vmin=0.0, vmax=max(float(np.nanpercentile(err_H_li, 95)), 1e-12),
                           linewidths=0, rasterized=True, alpha=0.55)

    # --- row 1: K ---
    t_cl_k, t_li_k = "Classical K (Gaussian curvature)", "Lifted K (Gaussian curvature)"

    _scatter_curv(axes[1, 0], pts, K_cl, t_cl_k)
    if K_li is not None:
        _scatter_curv(axes[1, 1], pts, K_li, t_li_k)
    else:
        axes[1, 1].set_visible(False)

    if has_gt and K_true is not None and K_li is not None:
        err_K_cl = np.abs(K_cl - K_true)
        err_K_li = np.abs(K_li - K_true)
        _scatter_err(axes[1, 2], pts,
                     np.where(np.isfinite(K_true), err_K_cl, np.nan),
                     "|K error|  Classical (orange) / Lifted (blue below)")
        axes[1, 2].scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=np.where(np.isfinite(K_true), err_K_li, 0.0),
                           s=PT_SIZE, cmap="Blues",
                           vmin=0.0, vmax=max(float(np.nanpercentile(err_K_li, 95)), 1e-12),
                           linewidths=0, rasterized=True, alpha=0.55)
    elif has_gt:
        axes[1, 2].set_visible(False)

    # --- row 2: total curvature ||h||_F (orientation-invariant) ---
    _scatter_pos(axes[2, 0], pts, Tc_cl, "Classical ||h||_F (total curvature)")
    _scatter_pos(axes[2, 1], pts, Tc_li, "Lifted ||h||_F (total curvature)")
    if has_gt:
        axes[2, 2].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout(pad=1.0)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"  Saved {fname}")
    plt.close(fig)


def _run(pts, frames, normals, title: str, fname: str, params=None, gt_fn=None):
    print("  Classical curvature...", end="", flush=True)
    H_cl, K_cl, Tc_cl = classical_curvature(pts, normals)
    print(f" done  (H range [{np.nanmin(H_cl):.3g}, {np.nanmax(H_cl):.3g}])")

    print("  Lifted curvature (level-1)...", end="", flush=True)
    H_li, K_li, Tc_li = lifted_curvature(pts, frames)
    print(f" done  (H range [{np.nanmin(H_li):.3g}, {np.nanmax(H_li):.3g}])")

    H_true = K_true = None
    if gt_fn is not None and params is not None:
        H_true, K_true = gt_fn(params)

    _make_figure(pts, H_cl, K_cl, Tc_cl, H_li, K_li, Tc_li, title, fname, H_true, K_true)


# ---------------------------------------------------------------------------
# Run -- Toy surface examples
# ---------------------------------------------------------------------------

TOYS = [
    # (surface_fn, sample_bounds, title, fname, gt_fn or None)
    (whitney_umbrella, dict(u_bounds=(-1.5, 1.5), v_bounds=(-1.5, 1.5)),
     "Whitney Umbrella (pinch point)", "curvature_whitney_umbrella.png", None),

    (cube_surface, dict(u_bounds=(0.0, 6.0), v_bounds=(0.0, 1.0)),
     "Cube Surface (sharp edges)", "curvature_cube.png", _cube_gt),

    (klein_bottle, dict(u_bounds=(0.0, 2 * np.pi), v_bounds=(0.0, 2 * np.pi)),
     "Klein Bottle toy (self-intersecting)", "curvature_klein_bottle_toy.png", None),

    (cone, dict(u_bounds=(0.05, 2.0), v_bounds=(0.0, 2 * np.pi)),
     "Cone (apex singularity)", "curvature_cone.png", _cone_gt),

    # tangentially intersecting surfaces
    (cylinders_tangent, dict(u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0)),
     "Cylinders (tangential intersection)", "curvature_cylinders_tangent.png", None),

    (plane_paraboloid_tangent, dict(u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0)),
     "Plane + Paraboloid (tangential intersection)", "curvature_plane_paraboloid_tangent.png", None),

    (tangent_spheres, dict(u_bounds=(0.0, 2.0), v_bounds=(0.05, 0.95)),
     "Tangent Spheres", "curvature_tangent_spheres.png", None),
]

for surface_fn, bounds, title, fname, gt_fn in TOYS:
    print(f"\n=== {title} ===")
    pts, frames, normals, params = _sample_toy(surface_fn, **bounds)
    print(f"  {len(pts)} valid points")
    _run(pts, frames, normals, title, fname, params=params, gt_fn=gt_fn)

# ---------------------------------------------------------------------------
# Run -- Thingi10k examples
# ---------------------------------------------------------------------------

REAL = [
    ("klein_bottle_one", "Klein Bottle 1 (thingi10k)", "curvature_thingi_klein1.png"),
    ("klein_bottle_two", "Klein Bottle 2 (thingi10k)", "curvature_thingi_klein2.png"),
    ("icicles",          "Icicles (thingi10k)",         "curvature_thingi_icicles.png"),
]

for name, title, fname in REAL:
    print(f"\n=== {title} ===")
    try:
        pts, frames, normals = _load_thingi(name)
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        continue
    print(f"  {len(pts)} valid points")
    _run(pts, frames, normals, title, fname)

print("\nAll done (surfaces).")

# ===========================================================================
# 2D curves and space curves
# ===========================================================================

# ---------------------------------------------------------------------------
# Classical curvature for curves: local quadratic fit
# ---------------------------------------------------------------------------

def _normal_basis(tangent: np.ndarray) -> np.ndarray:
    """
    Returns an orthonormal basis for the normal space of a unit tangent vector.

    For a 2D tangent (2,): returns a (2,1) array -- one normal vector.
    For a 3D tangent (3,): returns a (3,2) array -- two normal vectors.
    """
    n = len(tangent)
    if n == 2:
        return np.array([[-tangent[1], tangent[0]]]).T  # (2,1)
    # 3D: two vectors perpendicular to tangent
    ref = np.array([1., 0., 0.]) if abs(tangent[0]) < 0.9 else np.array([0., 1., 0.])
    b1 = np.cross(tangent, ref); b1 /= np.linalg.norm(b1)
    b2 = np.cross(tangent, b1)
    return np.column_stack([b1, b2])  # (3,2)


def classical_curvature_curve(pts: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """
    Scalar curvature of a curve via local quadratic fit in the normal plane.

    For each point, projects k neighbours into the local (tangent, normals) frame,
    fits normal_displacement = A * tangential_displacement^2,
    then kappa = 2 * ||A||.

    Works for both 2D (n=2) and 3D (n=3) curves.
    """
    N = len(pts)
    kappa = np.full(N, np.nan)
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=K + 1)
    nn_idx = nn_idx[:, 1:]

    for i in range(N):
        ti = tangents[i]
        Ni = _normal_basis(ti)          # (n, n_comp)  n_comp = 1 or 2
        nbs = pts[nn_idx[i]] - pts[i]  # (k, n)
        x = nbs @ ti                   # (k,)  tangential displacement
        Z = nbs @ Ni                   # (k, n_comp)  normal displacements

        A_col = (x**2)[:, None]        # (k, 1)
        try:
            a, _, _, _ = np.linalg.lstsq(A_col, Z, rcond=None)  # (1, n_comp)
        except np.linalg.LinAlgError:
            continue
        kappa[i] = 2.0 * float(np.linalg.norm(a))

    return kappa


# ---------------------------------------------------------------------------
# Ground truth curvature for curves
# ---------------------------------------------------------------------------

def _figure8_gt(params) -> np.ndarray:
    """Analytical signed curvature of figure8 (Lemniscate of Gerono, scale=1).
    x = sin(t),  y = sin(t)*cos(t) = sin(2t)/2
    kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    """
    t   = np.asarray(params[0], dtype=float)
    xp  =  np.cos(t)
    yp  =  np.cos(2 * t)
    xpp = -np.sin(t)
    ypp = -2 * np.sin(2 * t)
    speed2 = xp**2 + yp**2
    kappa = (xp * ypp - yp * xpp) / np.where(speed2 > 1e-12, speed2**1.5, np.nan)
    return kappa


def _line_part_gt(params, t_split: float = 1.0) -> np.ndarray:
    """kappa = 0 for the line part (t >= t_split) of tangent_parabola_line; NaN elsewhere."""
    t = np.asarray(params[0], dtype=float)
    return np.where(t >= t_split, 0.0, np.nan)


def _helix_gt(params, radius: float = 1.0, pitch: float = 0.25) -> np.ndarray:
    """Constant curvature of a helix: kappa = radius / (radius^2 + pitch^2)."""
    t = np.asarray(params[0], dtype=float)
    return np.full_like(t, radius / (radius**2 + pitch**2))


# ---------------------------------------------------------------------------
# Plotting helpers for curves
# ---------------------------------------------------------------------------

def _scatter_curve(ax, pts, vals, title: str, is_3d: bool, pct_clip: float = 2.0,
                   view: tuple | None = None):
    finite = vals[np.isfinite(vals)]
    if finite.size:
        lo, hi = np.percentile(finite, [pct_clip, 100 - pct_clip])
        lim = max(abs(lo), abs(hi), 1e-12)
    else:
        lim = 1.0
    c = np.where(np.isfinite(vals), vals, 0.0)
    if is_3d:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=c, s=PT_SIZE, cmap=CMAP, vmin=-lim, vmax=lim,
                   linewidths=0, rasterized=True)
        _clean_ax3d(ax)
        if view is not None:
            ax.view_init(elev=view[0], azim=view[1])
        _equal_aspect(ax, pts)
    else:
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=c, s=PT_SIZE * 1.5, cmap=CMAP, vmin=-lim, vmax=lim,
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, pad=3)


def _scatter_curve_err(ax, pts, err, title: str, is_3d: bool,
                       err_overlay=None, view: tuple | None = None):
    finite = err[np.isfinite(err)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    c = np.where(np.isfinite(err), err, 0.0)
    kw = dict(s=PT_SIZE * (1 if is_3d else 1.5), linewidths=0, rasterized=True)
    if is_3d:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, cmap="YlOrRd",
                   vmin=0.0, vmax=vmax, **kw)
        if err_overlay is not None:
            vmax2 = max(float(np.nanpercentile(err_overlay, 95)), 1e-12)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=np.where(np.isfinite(err_overlay), err_overlay, 0.0),
                       cmap="Blues", vmin=0.0, vmax=vmax2, alpha=0.55, **kw)
        _clean_ax3d(ax)
        if view is not None:
            ax.view_init(elev=view[0], azim=view[1])
        _equal_aspect(ax, pts)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], c=c, cmap="YlOrRd",
                   vmin=0.0, vmax=vmax, **kw)
        if err_overlay is not None:
            vmax2 = max(float(np.nanpercentile(err_overlay, 95)), 1e-12)
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=np.where(np.isfinite(err_overlay), err_overlay, 0.0),
                       cmap="Blues", vmin=0.0, vmax=vmax2, alpha=0.55, **kw)
        ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, pad=3)


def _make_figure_curve(pts, k_cl, k_li, title: str, fname: str,
                       k_true=None, view: tuple | None = None):
    is_3d = pts.shape[1] == 3
    has_gt = k_true is not None
    n_cols = 3 if has_gt else 2
    proj = {"projection": "3d"} if is_3d else {}
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.0),
                             subplot_kw=proj)
    fig.patch.set_facecolor("white")

    t_cl, t_li = "Classical kappa", "Lifted kappa"
    if has_gt:
        err_cl = np.abs(k_cl - k_true)
        err_li = np.abs(k_li - k_true)

    _scatter_curve(axes[0], pts, k_cl, t_cl, is_3d, view=view)
    _scatter_curve(axes[1], pts, k_li, t_li, is_3d, view=view)
    if has_gt:
        _scatter_curve_err(
            axes[2], pts,
            np.where(np.isfinite(k_true), err_cl, np.nan),
            "|kappa error|  Classical (orange) / Lifted (blue)",
            is_3d,
            err_overlay=np.where(np.isfinite(k_true), err_li, np.nan),
            view=view,
        )

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout(pad=1.0)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"  Saved {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sampling and run
# ---------------------------------------------------------------------------

def _sample_curve(curve_fn, t_min: float, t_max: float, n: int = 3000):
    """Sample a 2D or 3D parametric curve; return (pts, tangents, params)."""
    s = sample(curve_fn(),
               UniformCurve(n=n, t_min=t_min, t_max=t_max, endpoint=False),
               with_tangents=True, with_normals=False)
    pts  = np.asarray(s.points,   dtype=float)
    tans = np.asarray(s.tangents, dtype=float)
    t    = np.asarray(s.params, dtype=float).ravel()
    valid = np.isfinite(pts).all(1) & np.isfinite(tans).all(1)
    return pts[valid], tans[valid], (t[valid],)


def _run_curve(pts, tans, title: str, fname: str, params=None, gt_fn=None,
               view: tuple | None = None):
    """Estimate curvature classically and via lifted blow-up, then plot."""
    frames = tans[:, :, np.newaxis]   # (N, n, 1) for d=1

    print("  Classical kappa...", end="", flush=True)
    k_cl = classical_curvature_curve(pts, tans)
    print(" done")

    print("  Lifted kappa (level-1)...", end="", flush=True)
    level0 = BlowUpLevel.from_point_tangents(pts, frames)
    level1 = level0.lift(k=K, alpha=ALPHA, lam=LAM)
    inv    = extract_level1(level1)
    # scalar curvature = ||mean_curvature_vector||
    k_li = np.linalg.norm(inv.mean_curvature, axis=1)
    print(" done")

    k_true = gt_fn(params) if (gt_fn is not None and params is not None) else None

    _make_figure_curve(pts, k_cl, k_li, title, fname, k_true, view=view)


# ---------------------------------------------------------------------------
# 2D examples
# ---------------------------------------------------------------------------
N_CURVE = 4000

print("\n=== 2D: Figure-8 (transverse self-intersection) ===")
pts2d, tans2d, params2d = _sample_curve(figure8, 0.0, 2 * np.pi, n=N_CURVE)
_run_curve(pts2d, tans2d,
           "Figure-8 (2D transverse crossing)",
           "curvature_figure8_2d.png",
           params=params2d, gt_fn=_figure8_gt)

print("\n=== 2D: Tangent parabola + line (tangential intersection) ===")
pts2d, tans2d, params2d = _sample_curve(
    lambda: tangent_parabola_line(curvature=1.0), 0.0, 2.0, n=N_CURVE)
_run_curve(pts2d, tans2d,
           "Tangent parabola + line (2D tangential, line part gt: kappa=0)",
           "curvature_tangent_parabola_line_2d.png",
           params=params2d, gt_fn=_line_part_gt)

# ---------------------------------------------------------------------------
# Space curve examples
# ---------------------------------------------------------------------------

print("\n=== Space curve: Helix (smooth, constant curvature) ===")
pts3d, tans3d, params3d = _sample_curve(helix, 0.0, 4 * np.pi, n=N_CURVE)
_run_curve(pts3d, tans3d,
           "Helix (3D, smooth -- accuracy baseline)",
           "curvature_helix_3d.png",
           params=params3d, gt_fn=_helix_gt)

print("\n=== Space curve: Figure-8 (3D self-intersecting) ===")
pts3d, tans3d, params3d = _sample_curve(figure8_space, 0.0, 2 * np.pi, n=N_CURVE)
_run_curve(pts3d, tans3d,
           "Figure-8 space curve (3D transverse self-intersection)",
           "curvature_figure8_space_3d.png",
           view=(50, 20))

print("\n=== Space curve: Trefoil knot (smooth, non-self-intersecting) ===")
pts3d, tans3d, params3d = _sample_curve(trefoil_knot, 0.0, 2 * np.pi, n=N_CURVE)
_run_curve(pts3d, tans3d,
           "Trefoil knot (3D smooth knot)",
           "curvature_trefoil_3d.png",
           view=(35, 60))

print("\nAll done.")
