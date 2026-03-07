"""
bounds_test.py
--------------
Empirical verification of the cross-sheet lower bound:

    d_M(p~, q~)  >=  sqrt(alpha/2) * (||P0^1 - P0^2||_F - C_kappa * eps0)
                 ->  sqrt(alpha) * delta   as   eps0 -> 0

where  d_M^2 = ||p-q||^2 + alpha/2 * ||P_p - P_q||_F^2,
       C_kappa = Lipschitz const of  p -> P_p  (curvature-controlled),
       delta   = chordal dist between tangent spaces at crossing,
       eps0    = max(||p-x0||, ||q-x0||).

Two configurations:
  A. Figure-8 curve (d=1, n=2): analytic C_kappa = sqrt(2)*max|kappa|
  B. Two planes in R^3 at 45 deg: C_kappa = 0 (flat), delta = sin(45 deg)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Georgia", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset":  "dejavuserif",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.6,
})

ALPHA = 1.0
RNG   = np.random.default_rng(0)

ARM1_C  = "#2563eb"   # blue       — sheet 1 / same-sheet
ARM2_C  = "#c2410c"   # orange-red — sheet 2
SAME_C  = "#93c5fd"   # light blue — same-sheet pairs
CROSS_C = "#dc2626"   # red        — cross-sheet pairs
FLOOR_C = "#7c3aed"   # purple     — floor / bound


# ════════════════════════════════════════════════════════════════════════════════
# A.  FIGURE-8   gamma(t) = (sin t,  sin(2t)/2)
# ════════════════════════════════════════════════════════════════════════════════
#
# Tangent:    gamma'(t) = (cos t,  cos(2t))
# At t=0:    gamma=(0,0), unit tangent=(1,1)/sqrt(2)
# At t=pi:   gamma=(0,0), unit tangent=(-1,1)/sqrt(2)  [90 deg apart -> delta=1]
# C_kappa  = sqrt(2)*max|kappa(t)|   (||dP/ds||_F = sqrt(2)*|kappa| for d=1)

def _f8_gamma(t):
    return np.stack([np.sin(t), np.sin(2*t)/2], axis=-1)

def _f8_unit_tan(t):
    tx = np.cos(t);  ty = np.cos(2*t)
    return np.stack([tx, ty], axis=-1) / np.hypot(tx, ty)[:, None]

def _f8_projector(t):
    u = _f8_unit_tan(t)
    return np.einsum("...i,...j->...ij", u, u)

def _f8_kappa(t):
    xp  = np.cos(t);   yp  = np.cos(2*t)
    xpp = -np.sin(t);  ypp = -2*np.sin(2*t)
    return (xp*ypp - yp*xpp) / (xp**2 + yp**2)**1.5

_t_dense  = np.linspace(1e-6, 2*np.pi - 1e-6, 50_000)
F8_CKAPPA = float(np.sqrt(2) * np.max(np.abs(_f8_kappa(_t_dense))))
F8_DELTA  = 1.0              # exact: tangent angle = 90 deg, delta = sin(90) = 1
F8_PF_X0  = np.sqrt(2)      # ||P0^1 - P0^2||_F = sqrt(2)*delta  (exact)
F8_FLOOR  = np.sqrt(ALPHA)  # sqrt(alpha)*delta

# -- Sample arms near x0 = (0,0) -----------------------------------------------
N_F8   = 500
EPS_F8 = 0.42
t1 = np.linspace(0.01, 0.75, N_F8)   # arm 1: near t=0
t2 = np.pi + t1                        # arm 2: near t=pi

pts1 = _f8_gamma(t1);  pts2 = _f8_gamma(t2)
P1   = _f8_projector(t1);  P2 = _f8_projector(t2)
d1   = np.linalg.norm(pts1, axis=1)
d2   = np.linalg.norm(pts2, axis=1)

# Cross-arm pairs within EPS_F8 of x0
m1 = d1 < EPS_F8;  m2 = d2 < EPS_F8
i1 = np.where(m1)[0];  i2 = np.where(m2)[0]
ii, jj = np.repeat(i1, len(i2)), np.tile(i2, len(i1))
if len(ii) > 5000:
    sel = RNG.choice(len(ii), 5000, replace=False);  ii, jj = ii[sel], jj[sel]

f8_xc   = np.linalg.norm(pts1[ii] - pts2[jj], axis=1)
f8_Pfc  = np.linalg.norm((P1[ii] - P2[jj]).reshape(len(ii), -1), axis=1)
f8_dMc  = np.sqrt(f8_xc**2 + ALPHA/2 * f8_Pfc**2)
f8_eps0 = np.maximum(d1[ii], d2[jj])
f8_lb   = np.sqrt(ALPHA/2) * np.maximum(0., F8_PF_X0 - F8_CKAPPA * f8_eps0)
f8_frac = float((f8_dMc >= f8_lb - 1e-8).mean())

# Same-arm pairs on arm 1
ii_s, jj_s = np.triu_indices(len(i1), k=1)
ia, ib = i1[ii_s], i1[jj_s]
if len(ia) > 3000:
    sel = RNG.choice(len(ia), 3000, replace=False);  ia, ib = ia[sel], ib[sel]
f8_xs  = np.linalg.norm(pts1[ia] - pts1[ib], axis=1)
f8_dMs = np.sqrt(f8_xs**2 + ALPHA/2 *
                 np.linalg.norm((P1[ia]-P1[ib]).reshape(len(ia),-1), axis=1)**2)

print(f"Figure-8:  C_kappa={F8_CKAPPA:.4f}, floor={F8_FLOOR:.4f}, "
      f"cross pairs={len(f8_xc)}, frac_ok={f8_frac:.4f}")

# Same-sheet pairs over the FULL curve (for upper bound panel)
t_ub   = np.linspace(0, 2*np.pi, 350, endpoint=False)
pts_ub = _f8_gamma(t_ub)
P_ub   = _f8_projector(t_ub)
ii_ub, jj_ub = np.triu_indices(len(t_ub), k=1)
if len(ii_ub) > 5000:
    sel = RNG.choice(len(ii_ub), 5000, replace=False)
    ii_ub, jj_ub = ii_ub[sel], jj_ub[sel]
f8_x_ub  = np.linalg.norm(pts_ub[ii_ub] - pts_ub[jj_ub], axis=1)
f8_Pf_ub = np.linalg.norm((P_ub[ii_ub]-P_ub[jj_ub]).reshape(len(ii_ub),-1), axis=1)
f8_dM_ub = np.sqrt(f8_x_ub**2 + ALPHA/2 * f8_Pf_ub**2)
F8_UPPER = float(np.sqrt(1.0 + ALPHA/2 * F8_CKAPPA**2))


# ════════════════════════════════════════════════════════════════════════════════
# B.  INTERSECTING PLANES  (two 2D planes in R^3, dihedral angle THETA)
# ════════════════════════════════════════════════════════════════════════════════
#
# Plane 1: z = 0         (xy-plane),  basis {e1, e2}
# Plane 2: z = x*tan(T)  (tilted by T about y-axis),  basis {v1, e2}
# Intersection: the y-axis  (x = 0)
# Distance to intersection: eps = |x|
#
# C_kappa = 0  (projector is constant on each flat plane — no curvature)
# delta   = sin(T)   (principal angle between the two 2D tangent planes)
# ||P1-P2||_F = sqrt(2)*sin(T)  (exact, constant everywhere)

THETA     = np.pi / 4           # 45 degrees
PL_DELTA  = float(np.sin(THETA))
PL_PF_X0  = float(np.sqrt(2) * PL_DELTA)
PL_FLOOR  = float(np.sqrt(ALPHA) * PL_DELTA)
PL_CKAPPA = 0.0

e1 = np.array([1., 0., 0.]);  e2 = np.array([0., 1., 0.])
v1 = np.array([np.cos(THETA), 0., np.sin(THETA)])
P1_pl = np.outer(e1, e1) + np.outer(e2, e2)   # diag(1,1,0)
P2_pl = np.outer(v1, v1) + np.outer(e2, e2)
assert abs(np.linalg.norm(P1_pl - P2_pl, "fro") - PL_PF_X0) < 1e-12

# -- Sample grid on each plane -------------------------------------------------
N_PL   = 38
EPS_PL = 0.42
x_pl = np.linspace(-0.65, 0.65, N_PL)
y_pl = np.linspace(-0.65, 0.65, N_PL)
XX, YY = np.meshgrid(x_pl, y_pl)
pts1_pl = np.stack([XX.ravel(), YY.ravel(), np.zeros(N_PL**2)], axis=1)
pts2_pl = np.stack([XX.ravel(), YY.ravel(), XX.ravel() * np.tan(THETA)], axis=1)

mask_pl1 = np.abs(pts1_pl[:, 0]) < EPS_PL
mask_pl2 = np.abs(pts2_pl[:, 0]) < EPS_PL
ip1 = np.where(mask_pl1)[0];  ip2 = np.where(mask_pl2)[0]

# Cross-plane pairs
ii_pl = np.repeat(ip1, len(ip2));  jj_pl = np.tile(ip2, len(ip1))
if len(ii_pl) > 5000:
    sel = RNG.choice(len(ii_pl), 5000, replace=False)
    ii_pl, jj_pl = ii_pl[sel], jj_pl[sel]

pl_xc   = np.linalg.norm(pts1_pl[ii_pl] - pts2_pl[jj_pl], axis=1)
pl_dMc  = np.sqrt(pl_xc**2 + ALPHA/2 * PL_PF_X0**2)   # Pf constant
pl_eps0 = np.maximum(np.abs(pts1_pl[ii_pl, 0]), np.abs(pts2_pl[jj_pl, 0]))
pl_lb   = float(np.sqrt(ALPHA/2) * PL_PF_X0)           # constant (C_kappa=0)
pl_frac = float((pl_dMc >= pl_lb - 1e-8).mean())

# Same-plane pairs on plane 1
ii_ps, jj_ps = np.triu_indices(len(ip1), k=1)
ia_pl, ib_pl = ip1[ii_ps], ip1[jj_ps]
if len(ia_pl) > 3000:
    sel = RNG.choice(len(ia_pl), 3000, replace=False)
    ia_pl, ib_pl = ia_pl[sel], ib_pl[sel]
pl_xs  = np.linalg.norm(pts1_pl[ia_pl] - pts1_pl[ib_pl], axis=1)
pl_dMs = pl_xs.copy()   # same plane: ||P_p-P_q||_F = 0, so d_M = ||p-q||

print(f"Planes:    C_kappa=0, delta={PL_DELTA:.4f}, floor={PL_FLOOR:.4f}, "
      f"cross pairs={len(pl_xc)}, frac_ok={pl_frac:.4f}")

# Same-sheet pairs over the FULL plane 1 grid (for upper bound panel)
# C_kappa=0 -> d_M = ||p-q|| exactly -> ratio = 1 for every pair
ii_pub, jj_pub = np.triu_indices(N_PL**2, k=1)
if len(ii_pub) > 5000:
    sel = RNG.choice(len(ii_pub), 5000, replace=False)
    ii_pub, jj_pub = ii_pub[sel], jj_pub[sel]
pl_x_ub  = np.linalg.norm(pts1_pl[ii_pub] - pts1_pl[jj_pub], axis=1)
pl_dM_ub = pl_x_ub.copy()   # ||P_p-P_q||_F = 0 -> d_M = ||p-q||
PL_UPPER = float(np.sqrt(1.0 + ALPHA/2 * PL_CKAPPA**2))   # = 1.0 exactly


# ════════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════════

def _style(ax, xlabel="", ylabel="", title="", legend_loc="upper left"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, framealpha=0.7, handlelength=1.4)


# ── Panel *1: d_M vs ||p-q|| — same-sheet band + cross-sheet floor ────────────
def _panel_main(ax, xs, dMs, xc, dMc, upper, floor, legend_loc="upper left"):
    ax.scatter(xs,  dMs, s=2,   c=SAME_C,  alpha=0.30, linewidths=0,
               rasterized=True, label="same-sheet pairs")
    ax.scatter(xc,  dMc, s=2.5, c=CROSS_C, alpha=0.45, linewidths=0,
               rasterized=True, label="cross-sheet pairs")

    xmax = max(xs.max(), xc.max()) * 1.08
    xr   = np.array([0., xmax])
    ax.plot(xr, xr, color=ARM1_C, lw=1.5, ls="--",
            label=r"$d_M = \|p-q\|$  (lower)")
    if upper > 1.001:          # skip if degenerate (flat case)
        ax.plot(xr, upper * xr, color="#64748b", lw=1.5, ls="-.",
                label=f"$d_M = {upper:.3f}\\,\\|p-q\\|$  (upper)")
    ax.axhline(floor, color=FLOOR_C, lw=1.8, ls="--",
               label=rf"floor $= \sqrt{{\alpha}}\,\delta = {floor:.3f}$")

    ymax = max(dMs.max(), dMc.max()) * 1.08
    ax.set_xlim(0, xmax);  ax.set_ylim(0, ymax)
    _style(ax,
           xlabel=r"Euclidean distance $\|p - q\|$",
           ylabel=r"Lifted distance $d_M$",
           title=r"$\|p-q\| \leq d_M \leq C\,\|p-q\|$ (same)"
                 "\nand  $d_M \\geq$ floor (cross-sheet)",
           legend_loc=legend_loc)


# ── Panel A0: Figure-8 geometry ───────────────────────────────────────────────
def _panel_f8_geom(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)

    t_full = np.linspace(0, 2*np.pi, 600, endpoint=False)
    pts    = _f8_gamma(t_full)
    ax.plot(pts[:, 0], pts[:, 1], color=ARM1_C, lw=2.0)

    # Crossing
    ax.scatter([0], [0], s=90, color="black", zorder=8)
    ax.text(0.04, -0.09, r"$x_0$", fontsize=10, fontweight="bold", va="top")

    # Tangent lines at x0
    T_a1 = np.array([ 1., 1.]) / np.sqrt(2)
    T_a2 = np.array([-1., 1.]) / np.sqrt(2)
    L = 0.30
    for T, c in [(T_a1, ARM1_C), (T_a2, ARM2_C)]:
        ax.plot([-L*T[0], L*T[0]], [-L*T[1], L*T[1]],
                color=c, lw=1.4, ls="--", alpha=0.8, zorder=5)

    # Arc + delta label
    arc = matplotlib.patches.Arc((0, 0), 0.20, 0.20, angle=0,
                                  theta1=45, theta2=135, color="gray", lw=1.3)
    ax.add_patch(arc)
    ax.text(0, 0.14, r"$\delta = 1$", fontsize=9, ha="center",
            va="bottom", color="gray")

    # eps0 circle
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(EPS_F8*np.cos(th), EPS_F8*np.sin(th),
            color="gray", lw=0.9, ls=":", alpha=0.55)
    ax.text(EPS_F8 + 0.03, 0.02, r"$\varepsilon_0$",
            fontsize=9, color="gray", va="center")

    ax.text(0.03, 0.97,
            rf"$\alpha={ALPHA},\quad\delta={F8_DELTA:.0f}$" + "\n"
            rf"$C_\kappa = \sqrt{{2}}\max|\kappa| = {F8_CKAPPA:.2f}$" + "\n"
            rf"floor $= \sqrt{{\alpha}}\,\delta = {F8_FLOOR:.3f}$",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f4ff",
                      edgecolor="#c7d2fe", alpha=0.9))

    ax.set_xlim(-1.05, 1.05);  ax.set_ylim(-0.72, 0.72)
    ax.set_title("Figure-8: self-crossing at $x_0$", pad=5)


# ── Panel B0: Planes geometry (edge-on, x-z view) ────────────────────────────
def _panel_planes_geom(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)

    L = 0.80
    ax.plot([-L,  L], [0, 0], color=ARM1_C, lw=2.5,
            label=r"plane 1  ($z = 0$)")
    ax.plot([-L*np.cos(THETA),  L*np.cos(THETA)],
            [-L*np.sin(THETA),  L*np.sin(THETA)],
            color=ARM2_C, lw=2.5,
            label=rf"plane 2  ($z = x\tan\theta$)")

    # Intersection point (origin = projection of y-axis)
    ax.scatter([0], [0], s=90, color="black", zorder=8)
    ax.text(0.03, -0.07, "intersection\n(y-axis)", fontsize=7.5,
            va="top", color="gray")

    # Angle arc + label
    arc = matplotlib.patches.Arc((0, 0), 0.28, 0.28, angle=0,
                                  theta1=0, theta2=np.degrees(THETA),
                                  color="gray", lw=1.3)
    ax.add_patch(arc)
    mid = THETA / 2
    ax.text(0.18*np.cos(mid), 0.18*np.sin(mid),
            rf"$\theta = {int(np.degrees(THETA))}°$",
            fontsize=9, ha="center", va="center", color="gray")

    # eps0 arrow
    ax.annotate("", xy=(0.50, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.0))
    ax.text(0.25, -0.07, r"$\varepsilon_0 = |x|$",
            fontsize=8.5, ha="center", color="gray")

    ax.text(0.03, 0.97,
            rf"$\alpha={ALPHA},\quad\delta = \sin\theta = {PL_DELTA:.3f}$" + "\n"
            rf"$C_\kappa = 0$  (flat planes)" + "\n"
            rf"floor $= \sqrt{{\alpha}}\,\delta = {PL_FLOOR:.3f}$",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7ed",
                      edgecolor="#fed7aa", alpha=0.9))

    ax.set_xlim(-0.95, 0.95);  ax.set_ylim(-0.50, 0.60)
    ax.set_title(rf"Two planes at $\theta = {int(np.degrees(THETA))}°$"
                 "\n(edge-on view, looking along y-axis)", pad=5)
    ax.legend(loc="upper right", fontsize=8)


# ── Panel *2: d_M vs eps0 with lower bound ────────────────────────────────────
def _panel_vs_eps(ax, eps0, dMc, C_kappa, pf_x0, frac):
    ax.scatter(eps0, dMc, s=2.5, c=CROSS_C, alpha=0.35, linewidths=0,
               rasterized=True, label="cross-sheet pairs")

    er = np.linspace(0, eps0.max() * 1.05, 300)
    lb = np.sqrt(ALPHA/2) * np.maximum(0., pf_x0 - C_kappa * er)
    ax.plot(er, lb, color=FLOOR_C, lw=2.0, ls="--",
            label=r"$\sqrt{\frac{\alpha}{2}}\,(\|P^1_{x_0}-P^2_{x_0}\|_F"
                  r" - C_\kappa\,\varepsilon_0)$")

    ax.text(0.97, 0.06, rf"${100*frac:.1f}\%$ satisfy bound",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color="#16a34a" if frac > 0.999 else CROSS_C)
    ybot = min(dMc.min(), lb[lb > 0].min() if (lb > 0).any() else dMc.min()) * 0.85
    ax.set_ylim(bottom=max(0, ybot))
    _style(ax,
           xlabel="Distance to crossing $\\varepsilon_0 = \\max(\\|p-x_0\\|, \\|q-x_0\\|)$",
           ylabel="Lifted distance $d_M$",
           title="Cross-sheet: lower bound tightens as $\\varepsilon_0 \\to 0$",
           legend_loc="upper right")


# ════════════════════════════════════════════════════════════════════════════════
# Layout: 2 rows x 3 cols
#   col 0: geometry
#   col 1: d_M vs ||p-q||  (same-sheet band + cross-sheet floor)
#   col 2: cross-sheet lower bound  (d_M vs eps0)
# ════════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
plt.subplots_adjust(top=0.88, bottom=0.09, hspace=0.58, wspace=0.35)

# Row 0: Figure-8
_panel_f8_geom(axes[0, 0])
_panel_main(axes[0, 1], f8_xs, f8_dMs, f8_xc, f8_dMc, F8_UPPER, F8_FLOOR)
_panel_vs_eps(axes[0, 2], f8_eps0, f8_dMc, F8_CKAPPA, F8_PF_X0, f8_frac)

# Row 1: Intersecting planes
_panel_planes_geom(axes[1, 0])
_panel_main(axes[1, 1], pl_xs, pl_dMs, pl_xc, pl_dMc, PL_UPPER, PL_FLOOR,
            legend_loc="upper right")
_panel_vs_eps(axes[1, 2], pl_eps0, pl_dMc, PL_CKAPPA, PL_PF_X0, pl_frac)

# Row labels
fig.text(0.005, 0.70, "Figure-8", rotation=90, va="center",
         fontsize=11, fontweight="bold", color="#1e3a5f")
fig.text(0.005, 0.26, "Planes", rotation=90, va="center",
         fontsize=11, fontweight="bold", color="#7c2d12")

fig.suptitle(
    rf"Lifted metric bounds  ($\alpha = {ALPHA}$)" + "\n"
    r"Same-sheet: $\|p-q\| \leq d_M \leq \sqrt{1+\frac{\alpha}{2}C_\kappa^2}\,\|p-q\|$"
    r"     Cross-sheet: $d_M \geq \sqrt{\frac{\alpha}{2}}"
    r"\,(\|P^1_{x_0}-P^2_{x_0}\|_F - C_\kappa\,\varepsilon_0) \to \sqrt{\alpha}\,\delta$",
    fontsize=11, y=0.97,
)

out = "bounds_test.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
