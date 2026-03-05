"""
Frenet-Serret Recovery via Iterated Nash Blow-Up
-------------------------------------------------
Demonstrates full Frenet-Serret data recovery from a point cloud using two
iterations of the tangent blow-up construction.

For a space curve (d=1, n=3):

  Level 1 : shape operator B^(1) in R^{2x1}
            -> curvature kappa  and Frenet normal n

  Level 2 : gradient of B^(1) along the curve
            -> torsion tau  and  d kappa/ds

Two test curves with analytical ground truth:

  1. Circular helix  (radius r, pitch p):
       kappa = r / (r^2 + p^2)   [constant]
       tau   = p / (r^2 + p^2)   [constant]

  2. Trefoil knot:
       kappa and tau vary along the curve; ground truth computed by
       numerical finite differences on the parametric formula.

Figures
-------
Figure 1 - Helix: estimated vs ground-truth kappa and tau along arc length,
           plus d(kappa)/ds (should be ~0 for a helix).
Figure 2 - Trefoil: same layout but with varying kappa and tau.
Figure 3 - 3D visualizations of both curves coloured by estimated kappa.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.testsupport import UniformCurve, sample
from tangent_blowups.testsupport.examples_3D import helix, trefoil_knot
from tangent_blowups.geometry.iterated_grassmann import iterated_blowup
from tangent_blowups.geometry.frenet_serret import extract_frenet_serret

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_HELIX   = 400    # sample points on the helix   (2 full turns)
N_TREFOIL = 600    # sample points on the trefoil  (1 full turn)

HELIX_RADIUS = 1.0
HELIX_PITCH  = 0.30   # z per radian

K_BLOWUP  = 15     # k-NN for blow-up regression
ALPHA     = 1.0    # Chordal-Sasaki weight
LAM       = 1e-3   # ridge regularisation


# ---------------------------------------------------------------------------
# Ground-truth utilities
# ---------------------------------------------------------------------------

def _arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative chord length as a proxy for arc length."""
    diffs = np.diff(points, axis=0)
    seg   = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _numerical_frenet(pos_func, t_arr: np.ndarray, eps: float = 1e-5):
    """
    Numerical Frenet-Serret quantities via finite differences on the
    parametric formula.

    Returns kappa, tau, dkappa_ds as arrays aligned with t_arr.
    Uses central differences; boundary points fall back to forward/backward.
    """
    N = len(t_arr)
    # Velocity and acceleration by central differences
    r0 = pos_func(t_arr)           # (N, 3)  positions
    rp = pos_func(t_arr + eps)     # (N, 3)
    rm = pos_func(t_arr - eps)     # (N, 3)

    r1 = (rp - rm) / (2 * eps)    # r'  velocity   (N, 3)
    r2 = (rp - 2*r0 + rm) / eps**2  # r'' acceleration (N, 3)

    rp2 = pos_func(t_arr + 2*eps)
    rm2 = pos_func(t_arr - 2*eps)
    r3 = (rp2 - 2*rp + 2*rm - rm2) / (2 * eps**3)   # r''' jerk (N, 3)

    speed = np.linalg.norm(r1, axis=1, keepdims=True)   # (N, 1)
    speed = np.where(speed > 1e-14, speed, 1e-14)

    cross_12 = np.cross(r1, r2)   # r' x r''  (N, 3)
    cross_norm = np.linalg.norm(cross_12, axis=1)       # (N,)
    speed_scalar = speed[:, 0]

    # kappa = ||r' x r''|| / ||r'||^3
    kappa = cross_norm / speed_scalar**3

    # tau = (r' x r'') . r''' / ||r' x r''||^2
    dot_123 = np.einsum("ni,ni->n", cross_12, r3)       # (N,)
    cross_norm_sq = cross_norm**2
    tau = np.where(cross_norm_sq > 1e-28,
                   dot_123 / cross_norm_sq,
                   0.0)

    # d kappa/ds: numerical derivative via finite differences in t, then
    # convert to arc-length parameterisation: d kappa/ds = (d kappa/dt) / speed
    kappa_p = np.gradient(kappa, t_arr)
    dkappa_ds = kappa_p / speed_scalar

    return kappa, tau, dkappa_ds


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _set_equal3d(ax):
    lims  = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ctr   = lims.mean(axis=1)
    r     = max(0.5 * float((lims[:, 1] - lims[:, 0]).max()), 1e-3)
    ax.set_xlim3d(ctr[0]-r, ctr[0]+r)
    ax.set_ylim3d(ctr[1]-r, ctr[1]+r)
    ax.set_zlim3d(ctr[2]-r, ctr[2]+r)


def plot_comparison(
    ax,
    s:        np.ndarray,   # arc length (N,)
    est:      np.ndarray,   # estimated scalar field (N,)
    gt:       np.ndarray,   # ground-truth scalar field (N,)
    gt_label: str,
    ylabel:   str,
    title:    str,
    gt_const: float | None = None,
) -> None:
    """Plot estimated vs ground-truth scalar along arc length."""
    # clip outliers for display
    p2, p98 = np.percentile(est[np.isfinite(est)], [2, 98])
    est_clipped = np.clip(est, p2 - 0.5*abs(p2), p98 + 0.5*abs(p98))

    ax.plot(s, est_clipped, lw=1.2, alpha=0.7, label="estimated", color="steelblue")
    if gt_const is not None:
        ax.axhline(gt_const, color="tomato", lw=1.5, ls="--",
                   label=f"ground truth ({gt_label}={gt_const:.4f})")
    else:
        ax.plot(s, gt, lw=1.5, ls="--", color="tomato", label="ground truth")
    ax.set_xlabel("arc length  s")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Pipeline: sample -> blow-up -> Frenet-Serret extraction
# ---------------------------------------------------------------------------

def run_pipeline(curve_func, n_points, t_min, t_max):
    """Sample, run 2-level blow-up, return (points, t_params, fs_inv)."""
    curve    = curve_func()
    strategy = UniformCurve(n=n_points, t_min=t_min, t_max=t_max, endpoint=False)
    s        = sample(curve, strategy, with_tangents=True, with_normals=False)

    points   = np.asarray(s.points,   dtype=float)   # (N, 3)
    tangents = np.asarray(s.tangents, dtype=float)   # (N, 3)
    t_params = np.asarray(s.params,   dtype=float)   # (N,)

    # Drop any degenerate frames
    valid = (np.all(np.isfinite(points), axis=1) &
             np.all(np.isfinite(tangents), axis=1))
    points, tangents, t_params = points[valid], tangents[valid], t_params[valid]

    levels = iterated_blowup(points, tangents, num_levels=1,
                             k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l0, l1 = levels

    fs = extract_frenet_serret(l0, l1, k=K_BLOWUP, lam=LAM)
    return points, t_params, fs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # 1.  Helix
    # -----------------------------------------------------------------------
    r, p = HELIX_RADIUS, HELIX_PITCH
    c2   = r**2 + p**2
    kappa_gt_helix = r  / c2       # exact, constant
    tau_gt_helix   = p  / c2       # exact, constant

    print("=== Helix ===")
    print(f"  Ground truth: kappa = {kappa_gt_helix:.4f},  tau = {tau_gt_helix:.4f}")

    def _helix_func():
        return helix(radius=r, pitch=p)

    t_max_helix = 4.0 * np.pi    # 2 full turns
    pts_h, t_h, fs_h = run_pipeline(
        lambda: helix(radius=r, pitch=p),
        N_HELIX, 0.0, t_max_helix
    )
    s_h = _arc_lengths(pts_h)
    speed_h = np.sqrt(r**2 + p**2)   # constant for helix

    tau_h_est = np.abs(fs_h.torsion)    # |tau|; sign depends on QR orientation
    print(f"  Estimated kappa: mean={fs_h.curvature.mean():.4f}  "
          f"std={fs_h.curvature.std():.4f}")
    print(f"  Estimated |tau|: mean={tau_h_est.mean():.4f}    "
          f"std={tau_h_est.std():.4f}")
    print(f"  Estimated dkappa/ds: mean={fs_h.dkappa_ds.mean():.4f}  "
          f"(expected ~0)")

    # -----------------------------------------------------------------------
    # 2.  Trefoil knot (varying kappa, tau)
    # -----------------------------------------------------------------------
    print("\n=== Trefoil knot ===")

    pts_t, t_t, fs_t = run_pipeline(
        lambda: trefoil_knot(scale=1.0),
        N_TREFOIL, 0.0, 2.0*np.pi
    )
    s_t = _arc_lengths(pts_t)

    # Ground truth via finite differences on the parametric formula
    def _trefoil_pos(t):
        t = np.asarray(t, dtype=float)
        return np.stack([
            np.sin(t) + 2.0*np.sin(2*t),
            np.cos(t) - 2.0*np.cos(2*t),
            -np.sin(3*t),
        ], axis=-1)

    kappa_gt_t, tau_gt_t, dkappa_gt_t = _numerical_frenet(_trefoil_pos, t_t)

    tau_t_est = np.abs(fs_t.torsion)   # compare magnitudes
    tau_gt_t_abs = np.abs(tau_gt_t)
    print(f"  Estimated kappa: mean={fs_t.curvature.mean():.4f}  "
          f"gt mean={kappa_gt_t.mean():.4f}")
    print(f"  Estimated |tau|: mean={tau_t_est.mean():.4f}    "
          f"gt mean={tau_gt_t_abs.mean():.4f}")

    # -----------------------------------------------------------------------
    # Figure 1 – Helix
    # -----------------------------------------------------------------------
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    plot_comparison(
        axes[0], s_h, fs_h.curvature, None, "kappa",
        ylabel="kappa",
        title=f"Helix: curvature kappa\n(gt = {kappa_gt_helix:.4f})",
        gt_const=kappa_gt_helix,
    )
    plot_comparison(
        axes[1], s_h, tau_h_est, None, "tau",
        ylabel="|tau|",
        title=f"Helix: torsion |tau|\n(gt = {tau_gt_helix:.4f})",
        gt_const=tau_gt_helix,
    )
    # d kappa/ds should be ~0 for constant-curvature helix
    axes[2].plot(s_h, fs_h.dkappa_ds, lw=1.2, alpha=0.7,
                 color="steelblue", label="estimated d kappa/ds")
    axes[2].axhline(0, color="tomato", lw=1.5, ls="--", label="ground truth (0)")
    axes[2].set_xlabel("arc length  s")
    axes[2].set_ylabel("d kappa / ds")
    axes[2].set_title("Helix: curvature gradient\n(expected ~0)", fontsize=9)
    axes[2].legend(fontsize=8)

    fig1.suptitle(
        f"Helix (r={r}, pitch={p}): Frenet-Serret recovery via iterated blow-up\n"
        "Level 1 -> kappa, n, b     Level 2 -> tau, d(kappa)/ds",
        fontsize=10,
    )
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 2 – Trefoil knot
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

    plot_comparison(
        axes2[0], s_t, fs_t.curvature, kappa_gt_t, "kappa",
        ylabel="kappa",
        title="Trefoil knot: curvature kappa",
    )
    plot_comparison(
        axes2[1], s_t, tau_t_est, tau_gt_t_abs, "|tau|",
        ylabel="|tau|",
        title="Trefoil knot: torsion |tau|",
    )
    plot_comparison(
        axes2[2], s_t, fs_t.dkappa_ds, dkappa_gt_t, "d kappa/ds",
        ylabel="d kappa / ds",
        title="Trefoil knot: curvature gradient",
    )

    fig2.suptitle(
        "Trefoil knot: Frenet-Serret recovery via iterated blow-up\n"
        "estimated (solid) vs numerical ground truth (dashed)",
        fontsize=10,
    )
    plt.tight_layout()

    # -----------------------------------------------------------------------
    # Figure 3 – 3D views coloured by curvature
    # -----------------------------------------------------------------------
    fig3 = plt.figure(figsize=(14, 6))

    ax_h = fig3.add_subplot(1, 2, 1, projection="3d")
    sc_h = ax_h.scatter(
        pts_h[:, 0], pts_h[:, 1], pts_h[:, 2],
        c=fs_h.curvature, s=12, cmap="plasma",
        vmin=0.0, vmax=float(np.percentile(fs_h.curvature, 98)),
    )
    ax_h.set_title(
        f"Helix: point cloud coloured by kappa\n"
        f"(gt = {kappa_gt_helix:.4f}, estimated mean = {fs_h.curvature.mean():.4f})",
        fontsize=9,
    )
    ax_h.set_xlabel("x"); ax_h.set_ylabel("y"); ax_h.set_zlabel("z")
    _set_equal3d(ax_h)
    plt.colorbar(sc_h, ax=ax_h, fraction=0.03, pad=0.08, label="kappa")

    ax_t = fig3.add_subplot(1, 2, 2, projection="3d")
    sc_t = ax_t.scatter(
        pts_t[:, 0], pts_t[:, 1], pts_t[:, 2],
        c=fs_t.curvature, s=12, cmap="plasma",
        vmin=0.0, vmax=float(np.percentile(fs_t.curvature, 98)),
    )
    ax_t.set_title(
        "Trefoil knot: point cloud coloured by kappa",
        fontsize=9,
    )
    ax_t.set_xlabel("x"); ax_t.set_ylabel("y"); ax_t.set_zlabel("z")
    _set_equal3d(ax_t)
    plt.colorbar(sc_t, ax=ax_t, fraction=0.03, pad=0.08, label="kappa")

    fig3.suptitle(
        "Frenet-Serret via iterated Nash blow-up: 3D curvature maps",
        fontsize=11,
    )
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
