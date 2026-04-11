"""
SGP figure: cross-sheet lower bound on the figure-8 curve.

Illustrates the separation guarantee of the lifted metric:

    d_M(p~, q~) >= sqrt(alpha/2) * (||P_x0^1 - P_x0^2||_F - C_kappa * eps_0)
                 -> sqrt(alpha) * delta   as   eps_0 -> 0

Same-sheet pairs cluster near the diagonal d_M ~ ||p-q||, while cross-sheet
pairs are bounded away from zero by the tangent-space mismatch at the crossing.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformCurve, figure8, sample

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset":  "dejavuserif",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.20,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "axes.linewidth":    0.6,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "lines.linewidth":   1.6,
    "font.size":         9,
})

# ── Colors (derived from RdBu_r) ──────────────────────────────────────────────
_cmap   = plt.get_cmap("RdBu_r")
SHEET1_C = _cmap(0.85)   # blue  — sheet 1 / same-sheet reference line
SHEET2_C = _cmap(0.15)   # red   — sheet 2
SAME_C   = _cmap(0.75)   # light blue — same-sheet pairs
CROSS_C  = _cmap(0.10)   # deep red   — cross-sheet pairs
BOUND_C  = _cmap(0.50)   # neutral    — lower bound line

# ── Parameters ────────────────────────────────────────────────────────────────
ALPHA  = 1.0
N_PTS  = 1200
SCALE  = 2.0
SEED   = 7
RNG    = np.random.default_rng(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Analytic helpers for the figure-8:  gamma(t) = (sin t, sin(2t)/2)
# ═══════════════════════════════════════════════════════════════════════════════

def _gamma(t):
    return np.stack([np.sin(t), np.sin(2 * t) / 2], axis=-1)

def _unit_tangent(t):
    tx, ty = np.cos(t), np.cos(2 * t)
    nrm = np.hypot(tx, ty)[:, None]
    return np.stack([tx, ty], axis=-1) / nrm

def _projector(t):
    u = _unit_tangent(t)
    return np.einsum("...i,...j->...ij", u, u)

def _curvature(t):
    xp, yp = np.cos(t), np.cos(2 * t)
    xpp, ypp = -np.sin(t), -2 * np.sin(2 * t)
    return (xp * ypp - yp * xpp) / (xp**2 + yp**2)**1.5


# Analytic constants at the crossing x_0 = (0, 0)
# Tangent at t=0:  (1, 1)/sqrt(2)
# Tangent at t=pi: (-1, 1)/sqrt(2)   => 90 deg apart => delta = 1
_t_dense = np.linspace(1e-6, 2 * np.pi - 1e-6, 50_000)
C_KAPPA  = float(np.sqrt(2) * np.max(np.abs(_curvature(_t_dense))))
DELTA    = 1.0
PF_X0    = np.sqrt(2) * DELTA       # ||P_x0^1 - P_x0^2||_F
FLOOR    = np.sqrt(ALPHA) * DELTA   # asymptotic lower bound


# ═══════════════════════════════════════════════════════════════════════════════
# Sample the curve and compute pairwise lifted distances
# ═══════════════════════════════════════════════════════════════════════════════

# Arms near the crossing: t near 0 (sheet 1) and t near pi (sheet 2)
N_ARM  = 500
t1 = np.linspace(0.01, 0.75, N_ARM)
t2 = np.pi + t1

pts1, pts2 = _gamma(t1), _gamma(t2)
P1, P2     = _projector(t1), _projector(t2)
d1 = np.linalg.norm(pts1, axis=1)
d2 = np.linalg.norm(pts2, axis=1)

# Cross-sheet pairs within a neighbourhood of the crossing
EPS = 0.42
m1, m2 = d1 < EPS, d2 < EPS
i1, i2 = np.where(m1)[0], np.where(m2)[0]
ii_c, jj_c = np.repeat(i1, len(i2)), np.tile(i2, len(i1))
if len(ii_c) > 5000:
    sel = RNG.choice(len(ii_c), 5000, replace=False)
    ii_c, jj_c = ii_c[sel], jj_c[sel]

xc   = np.linalg.norm(pts1[ii_c] - pts2[jj_c], axis=1)
Pfc  = np.linalg.norm((P1[ii_c] - P2[jj_c]).reshape(len(ii_c), -1), axis=1)
dMc  = np.sqrt(xc**2 + ALPHA / 2 * Pfc**2)
eps0 = np.maximum(d1[ii_c], d2[jj_c])
lb   = np.sqrt(ALPHA / 2) * np.maximum(0.0, PF_X0 - C_KAPPA * eps0)

# Same-sheet pairs on sheet 1
ii_s, jj_s = np.triu_indices(len(i1), k=1)
ia, ib = i1[ii_s], i1[jj_s]
if len(ia) > 3000:
    sel = RNG.choice(len(ia), 3000, replace=False)
    ia, ib = ia[sel], ib[sel]
xs  = np.linalg.norm(pts1[ia] - pts1[ib], axis=1)
dMs = np.sqrt(xs**2 + ALPHA / 2
              * np.linalg.norm((P1[ia] - P1[ib]).reshape(len(ia), -1), axis=1)**2)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(*, labels: bool = True) -> plt.Figure:
    """
    Build the three-panel figure.

    Parameters
    ----------
    labels : bool
        If True, draw legends, inline annotations, and percentage text.
        If False, keep axes, tick labels, titles, and axis labels but
        omit legends and extra annotations for a cleaner presentation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.16, top=0.82,
                        wspace=0.38)

    # ── (a) Figure-8 geometry ────────────────────────────────────────────────
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)

    t_full = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    curve_pts = _gamma(t_full)
    ax.plot(curve_pts[:, 0], curve_pts[:, 1], color=SHEET1_C, lw=2.0, zorder=2)

    # Crossing point
    ax.scatter([0], [0], s=50, color=CROSS_C, zorder=8)

    # Tangent lines at x_0
    T_a = np.array([1.0, 1.0]) / np.sqrt(2)
    T_b = np.array([-1.0, 1.0]) / np.sqrt(2)
    L = 0.32
    for T, c, lbl in [(T_a, SHEET1_C, "sheet 1"), (T_b, SHEET2_C, "sheet 2")]:
        ax.plot([-L * T[0], L * T[0]], [-L * T[1], L * T[1]],
                color=c, lw=1.5, ls="--", alpha=0.85, zorder=5,
                label=lbl if labels else None)

    # Angle arc
    arc = mpatches.Arc((0, 0), 0.22, 0.22, angle=0,
                        theta1=45, theta2=135, color="0.45", lw=1.2)
    ax.add_patch(arc)

    # eps_0 neighbourhood
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(EPS * np.cos(th), EPS * np.sin(th),
            color="0.55", lw=0.8, ls=":", alpha=0.6)

    if labels:
        ax.text(0.05, -0.10, r"$x_0$", fontsize=10, fontweight="bold",
                va="top")
        ax.text(0.0, 0.15, r"$\delta = 1$", fontsize=8.5, ha="center",
                va="bottom", color="0.40")
        ax.text(EPS + 0.03, 0.01, r"$\varepsilon_0$", fontsize=8.5,
                color="0.50", va="center")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.8,
                  handlelength=1.2)

    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-0.72, 0.72)
    ax.set_title(r"(a) Figure-8 with crossing", fontsize=10, pad=6)

    # ── (b) d_M vs ||p-q|| ──────────────────────────────────────────────────
    ax = axes[1]

    ax.scatter(xs, dMs, s=2, c=[SAME_C], alpha=0.35, linewidths=0,
               rasterized=True, label="same-sheet" if labels else None)
    ax.scatter(xc, dMc, s=2.5, c=[CROSS_C], alpha=0.45, linewidths=0,
               rasterized=True, label="cross-sheet" if labels else None)

    xmax = max(xs.max(), xc.max()) * 1.08
    xr = np.array([0.0, xmax])
    ax.plot(xr, xr, color=SHEET1_C, lw=1.2, ls="--", alpha=0.6,
            label=r"$d_M = \|p - q\|$" if labels else None)
    ax.axhline(FLOOR, color=BOUND_C, lw=1.8, ls="--",
               label=(rf"floor $= \sqrt{{\alpha}}\,\delta = {FLOOR:.2f}$"
                      if labels else None))

    ax.set_xlim(0, xmax); ax.set_ylim(0, max(dMs.max(), dMc.max()) * 1.08)
    ax.set_xlabel(r"Euclidean distance $\|p - q\|$")
    ax.set_ylabel(r"Lifted distance $d_M$")
    ax.set_title(r"(b) Same-sheet vs. cross-sheet distances",
                 fontsize=10, pad=6)
    if labels:
        ax.legend(loc="upper left", framealpha=0.8, handlelength=1.2)

    # ── (c) d_M vs eps_0 with tightening lower bound ────────────────────────
    ax = axes[2]

    ax.scatter(eps0, dMc, s=2.5, c=[CROSS_C], alpha=0.40, linewidths=0,
               rasterized=True, label="cross-sheet" if labels else None)

    er = np.linspace(0, eps0.max() * 1.05, 300)
    lb_curve = np.sqrt(ALPHA / 2) * np.maximum(0.0, PF_X0 - C_KAPPA * er)
    ax.plot(er, lb_curve, color=BOUND_C, lw=2.0, ls="--",
            label=((r"$\sqrt{\alpha/2}"
                    r"\,(\|P^1_{x_0}-P^2_{x_0}\|_F"
                    r" - C_\kappa\,\varepsilon_0)$")
                   if labels else None))

    ybot = min(dMc.min(), lb_curve[lb_curve > 0].min()) * 0.85
    ax.set_ylim(bottom=max(0, ybot))
    ax.set_xlabel(
        r"$\varepsilon_0 = \max(\|p - x_0\|,\, \|q - x_0\|)$")
    ax.set_ylabel(r"Lifted distance $d_M$")
    ax.set_title(r"(c) Lower bound tightens as $\varepsilon_0 \to 0$",
                 fontsize=10, pad=6)

    if labels:
        frac = float((dMc >= lb - 1e-8).mean())
        ax.text(0.96, 0.07, rf"${100 * frac:.1f}\%$ satisfy bound",
                transform=ax.transAxes, ha="right", fontsize=8,
                color="#16a34a" if frac > 0.999 else CROSS_C)
        ax.legend(loc="upper right", framealpha=0.8, handlelength=1.2)

    fig.suptitle(
        r"Cross-sheet separation bound on the figure-8"
        r"    ($\alpha = 1$,  $\delta = \sin 90° = 1$)",
        fontsize=11, y=0.96,
    )

    return fig


# ── Save both versions ────────────────────────────────────────────────────────
fig_labeled = make_figure(labels=True)
fig_labeled.savefig("fig8_cross_sheet_bound.pdf", bbox_inches="tight", dpi=300)
print("Saved: fig8_cross_sheet_bound.pdf")

fig_clean = make_figure(labels=False)
fig_clean.savefig("fig8_cross_sheet_bound_clean.pdf", bbox_inches="tight",
                  dpi=300)
print("Saved: fig8_cross_sheet_bound_clean.pdf")

plt.show()
