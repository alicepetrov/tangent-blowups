"""
Affinity Matrices: Euclidean vs Lifted (Chordal-Sasaki) Metric
---------------------------------------------------------------
A Gaussian affinity W_ij = exp(-d(i,j)^2 / h^2) is built from two metrics:

  Euclidean:  d^2 = ||x_i - x_j||^2
  Lifted:     d^2 = ||x_i - x_j||^2  +  alpha * ||P_i - P_j||_F^2 / 2

where P_i = U_i U_i^T is the tangent projector at x_i.

At a self-intersection, spatial coordinates collide (x_i = x_j) but the
tangent projectors differ (P_i != P_j).  The lifted metric adds a penalty
proportional to how differently oriented the two branches are.

Two examples with the same bandwidth h (same base kernel scale):
  Figure-8    -- transverse crossing:  tangent directions are ~90 deg apart
                 => lifted metric drives cross-branch affinity to zero
  Tangent circles -- tangential meeting:  both branches share the SAME
                 tangent direction at the contact point
                 => lifted metric is identical to Euclidean there
                 (motivating the level-1 iterated blow-up)

Points are sorted by branch so that same-branch entries form the two
diagonal blocks; cross-branch entries appear in the off-diagonal blocks.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import (
    figure8,
    tangent_circles,
    UniformCurve,
    sample,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N      = 120     # total sample points per example
ALPHA  = 5.0    # Grassmannian weight  alpha  in the lifted metric
H      = 0.5    # Gaussian bandwidth h  (same for Euclidean and lifted)
PT_SZ  = 8.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _affinity(D: np.ndarray, h: float) -> np.ndarray:
    """Gaussian affinity matrix from distance matrix, with zeroed diagonal."""
    W = np.exp(-(D ** 2) / (h ** 2))
    np.fill_diagonal(W, 0.0)
    return W


def _build(curve, n_pts, *, t_max=2.0 * np.pi, arm_split=np.pi):
    """
    Sample a 2D closed curve, return spatial points, tangent-lifted sample,
    parameter array, and arm label (0 or 1).
    """
    strat = UniformCurve(n=n_pts, t_min=0.0, t_max=t_max, endpoint=False)
    s     = sample(curve, strat, with_tangents=True)

    pts  = np.asarray(s.points,   dtype=float)
    tans = normalize_vectors(np.asarray(s.tangents, dtype=float))
    t    = np.linspace(0.0, t_max, n_pts, endpoint=False)
    arm  = (t >= arm_split).astype(int)

    bup = BlownUpSample.from_tangents(pts, tans)
    return pts, tans, t, arm, bup


def _sort_perm(arm: np.ndarray) -> np.ndarray:
    """Permutation that places arm-0 first, arm-1 second (stable)."""
    return np.argsort(arm, kind="stable")


def _block_annotation(ax, n: int, n0: int, color: str = "lime", lw: float = 1.5):
    """
    Draw a rectangle around the off-diagonal blocks indicating the crossing
    region in a block-ordered N x N affinity matrix.
    """
    n1 = n - n0
    # Top-right cross block
    rect = mpatches.Rectangle(
        (n0 - 0.5, -0.5), n1, n0,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    # Bottom-left cross block
    rect2 = mpatches.Rectangle(
        (-0.5, n0 - 0.5), n0, n1,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect2)


def _divider_line(ax, n0: int, color: str = "cyan", lw: float = 1.2):
    """Draw lines between the two blocks."""
    ax.axhline(n0 - 0.5, color=color, lw=lw)
    ax.axvline(n0 - 0.5, color=color, lw=lw)


def _aff_imshow(ax, W: np.ndarray, title: str, n0: int,
                block_color: str = "lime"):
    n = W.shape[0]
    im = ax.imshow(W, cmap="hot_r", vmin=0.0, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    _divider_line(ax, n0, color=block_color)
    _block_annotation(ax, n, n0, color=block_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    # label blocks
    ax.text(n0 // 2, n0 // 2, "branch 1",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.text(n0 + (n - n0) // 2, n0 + (n - n0) // 2, "branch 2",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    return im


# ---------------------------------------------------------------------------
# Crossing-lines builder (no testsupport needed — geometry is trivial)
# ---------------------------------------------------------------------------

def _build_cross_lines(n_pts: int, scale: float = 1.5):
    """
    Two straight lines crossing at the origin at right angles.

    Branch 0: horizontal  y = 0,  tangent = (1, 0),  P = [[1,0],[0,0]]
    Branch 1: vertical    x = 0,  tangent = (0, 1),  P = [[0,0],[0,1]]

    Chordal distance at the crossing = 1 (maximum, since tangents are
    perpendicular).  The lifted metric completely blocks cross-branch
    affinity.  The Euclidean metric connects both branches near the origin.
    """
    half = n_pts // 2

    coords_h = np.linspace(-scale, scale, half)
    pts_h  = np.column_stack([coords_h, np.zeros(half)])
    tans_h = np.tile([1.0, 0.0], (half, 1))
    arm_h  = np.zeros(half, dtype=int)

    coords_v = np.linspace(-scale, scale, half)
    pts_v  = np.column_stack([np.zeros(half), coords_v])
    tans_v = np.tile([0.0, 1.0], (half, 1))
    arm_v  = np.ones(half, dtype=int)

    pts  = np.vstack([pts_h,  pts_v])
    tans = np.vstack([tans_h, tans_v])
    arm  = np.concatenate([arm_h, arm_v])

    bup = BlownUpSample.from_tangents(pts, tans)
    return pts, tans, arm, bup


# ---------------------------------------------------------------------------
# Build examples
# ---------------------------------------------------------------------------

print("Sampling figure-8 ...")
pts_f8, tans_f8, t_f8, arm_f8, bup_f8 = _build(
    figure8(scale=1.5), N, t_max=2.0 * np.pi, arm_split=np.pi
)
D_euclid_f8 = bup_f8.distance_matrix(alpha=0.0)
D_lifted_f8 = bup_f8.distance_matrix(alpha=ALPHA)
W_euclid_f8 = _affinity(D_euclid_f8, H)
W_lifted_f8 = _affinity(D_lifted_f8, H)

perm_f8 = _sort_perm(arm_f8)
n0_f8   = int(np.sum(arm_f8 == 0))
W_e_f8_s = W_euclid_f8[np.ix_(perm_f8, perm_f8)]
W_l_f8_s = W_lifted_f8[np.ix_(perm_f8, perm_f8)]
pts_f8_s  = pts_f8[perm_f8]
arm_f8_s  = arm_f8[perm_f8]

print("Building crossing lines ...")
pts_cl, tans_cl, arm_cl, bup_cl = _build_cross_lines(N, scale=1.5)
D_euclid_cl = bup_cl.distance_matrix(alpha=0.0)
D_lifted_cl = bup_cl.distance_matrix(alpha=ALPHA)
W_euclid_cl = _affinity(D_euclid_cl, H)
W_lifted_cl = _affinity(D_lifted_cl, H)

perm_cl = _sort_perm(arm_cl)
n0_cl   = int(np.sum(arm_cl == 0))
W_e_cl_s = W_euclid_cl[np.ix_(perm_cl, perm_cl)]
W_l_cl_s = W_lifted_cl[np.ix_(perm_cl, perm_cl)]
pts_cl_s  = pts_cl[perm_cl]
arm_cl_s  = arm_cl[perm_cl]

print("Sampling tangent circles ...")
pts_tc, tans_tc, t_tc, arm_tc, bup_tc = _build(
    tangent_circles(radius=1.0), N, t_max=2.0, arm_split=1.0
)
D_euclid_tc = bup_tc.distance_matrix(alpha=0.0)
D_lifted_tc = bup_tc.distance_matrix(alpha=ALPHA)
W_euclid_tc = _affinity(D_euclid_tc, H)
W_lifted_tc = _affinity(D_lifted_tc, H)

perm_tc = _sort_perm(arm_tc)
n0_tc   = int(np.sum(arm_tc == 0))
W_e_tc_s = W_euclid_tc[np.ix_(perm_tc, perm_tc)]
W_l_tc_s = W_lifted_tc[np.ix_(perm_tc, perm_tc)]
pts_tc_s  = pts_tc[perm_tc]
arm_tc_s  = arm_tc[perm_tc]

# Chordal distance at the crossing / contact point (diagnostic)
P_f8 = bup_f8.projectors    # (N, 2, 2)
# crossing arms are near-index 0 (t=0) and near-index N//2 (t=pi)
i0 = 0          # arm-1 near t=0
i1 = N // 2     # arm-2 near t=pi (first arm-2 index)
d_chord_f8 = float(np.linalg.norm(P_f8[i0] - P_f8[i1], "fro")) / np.sqrt(2)
print(f"  Figure-8:  chordal distance at crossing = {d_chord_f8:.3f}  (max = 1)")

P_tc = bup_tc.projectors
# touching point: arm-1 index 0 (t=0) and arm-2 index N//2 (t=1)
i0_tc = 0
i1_tc = N // 2
d_chord_tc = float(np.linalg.norm(P_tc[i0_tc] - P_tc[i1_tc], "fro")) / np.sqrt(2)
print(f"  Tangent circles: chordal distance at contact = {d_chord_tc:.3f}  (0 = same tangent)")

# Cross-branch lifted affinities at crossing (diagnostic)
w_euclid_cross_f8 = float(W_euclid_f8[i0, i1])
w_lifted_cross_f8 = float(W_lifted_f8[i0, i1])
w_euclid_cross_tc = float(W_euclid_tc[i0_tc, i1_tc])
w_lifted_cross_tc = float(W_lifted_tc[i0_tc, i1_tc])
print(f"  Figure-8 at crossing:       W_euclid = {w_euclid_cross_f8:.3f},  W_lifted = {w_lifted_cross_f8:.3f}")
print(f"  Tangent circles at contact: W_euclid = {w_euclid_cross_tc:.3f},  W_lifted = {w_lifted_cross_tc:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.42, wspace=0.22,
    left=0.05, right=0.97, top=0.91, bottom=0.04,
)

# ── Row 0: Figure-8 ────────────────────────────────────────────────────────

ax_pc0 = fig.add_subplot(gs[0, 0])
for b, c, lbl in [(0, "C0", "branch 1"), (1, "C1", "branch 2")]:
    mask = arm_f8 == b
    ax_pc0.scatter(pts_f8[mask, 0], pts_f8[mask, 1],
                   s=PT_SZ, color=c, label=lbl, linewidths=0)
ax_pc0.scatter([0], [0], s=100, marker="*", c="k", zorder=10, label="crossing")
ax_pc0.set_aspect("equal")
ax_pc0.set_xticks([]); ax_pc0.set_yticks([])
ax_pc0.legend(fontsize=7, loc="upper right")
ax_pc0.set_title("Figure-8  (transverse crossing)\npoint cloud", fontsize=9)

ax_e0 = fig.add_subplot(gs[0, 1])
im_e0 = _aff_imshow(ax_e0, W_e_f8_s,
                    "Euclidean affinity\ncross-branch entries non-zero at crossing", n0_f8)

ax_l0 = fig.add_subplot(gs[0, 2])
im_l0 = _aff_imshow(ax_l0, W_l_f8_s,
                    f"Lifted affinity  (alpha={ALPHA})\ncross-branch = 0  (tangents ~90 deg apart)", n0_f8)

# ── Row 1: Crossing lines ──────────────────────────────────────────────────

ax_pc1 = fig.add_subplot(gs[1, 0])
for b, c, lbl in [(0, "C0", "horizontal"), (1, "C1", "vertical")]:
    mask = arm_cl == b
    ax_pc1.scatter(pts_cl[mask, 0], pts_cl[mask, 1],
                   s=PT_SZ, color=c, label=lbl, linewidths=0)
ax_pc1.scatter([0], [0], s=100, marker="*", c="k", zorder=10, label="crossing")
ax_pc1.set_aspect("equal")
ax_pc1.set_xticks([]); ax_pc1.set_yticks([])
ax_pc1.legend(fontsize=7, loc="upper right")
ax_pc1.set_title("Two crossing lines  (orthogonal, transverse)\npoint cloud", fontsize=9)

ax_e1 = fig.add_subplot(gs[1, 1])
im_e1 = _aff_imshow(ax_e1, W_e_cl_s,
                    "Euclidean affinity\nboth lines connected near origin", n0_cl)

ax_l1 = fig.add_subplot(gs[1, 2])
im_l1 = _aff_imshow(ax_l1, W_l_cl_s,
                    f"Lifted affinity  (alpha={ALPHA})\nclean block diagonal  (P_horiz != P_vert everywhere)", n0_cl)

# ── Row 2: Tangent circles ─────────────────────────────────────────────────

ax_pc2 = fig.add_subplot(gs[2, 0])
for b, c, lbl in [(0, "C0", "circle 1"), (1, "C1", "circle 2")]:
    mask = arm_tc == b
    ax_pc2.scatter(pts_tc[mask, 0], pts_tc[mask, 1],
                   s=PT_SZ, color=c, label=lbl, linewidths=0)
ax_pc2.scatter([0], [0], s=100, marker="*", c="k", zorder=10, label="contact")
ax_pc2.set_aspect("equal")
ax_pc2.set_xticks([]); ax_pc2.set_yticks([])
ax_pc2.legend(fontsize=7, loc="upper right")
ax_pc2.set_title("Tangent circles  (tangential contact)\npoint cloud", fontsize=9)

ax_e2 = fig.add_subplot(gs[2, 1])
im_e2 = _aff_imshow(ax_e2, W_e_tc_s,
                    "Euclidean affinity\ncross-branch entry at contact point", n0_tc)

ax_l2 = fig.add_subplot(gs[2, 2])
im_l2 = _aff_imshow(ax_l2, W_l_tc_s,
                    f"Lifted affinity  (alpha={ALPHA})\ncontact persists  (same tangent => P_i = P_j)", n0_tc)

# ── Shared colorbar ─────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.975, 0.04, 0.012, 0.85])
fig.colorbar(im_l0, cax=cbar_ax, label="affinity W")

# ── Title ───────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Affinity matrices:  Euclidean  vs  Lifted (Chordal-Sasaki, alpha={ALPHA})\n"
    "$W_{ij}^{\\rm euc} = \\exp\\!\\left(-\\|x_i - x_j\\|^2 / h^2\\right)$"
    "      "
    "$W_{ij}^{\\rm lift} = \\exp\\!\\left(-(\\|x_i - x_j\\|^2 + \\alpha\\,\\|P_i - P_j\\|_F^2/2) / h^2\\right)$"
    f"\n$h = {H}$.  Green box = cross-branch block.  "
    "Rows 0-1: transverse crossings — lift separates cleanly.  "
    "Row 2: tangential contact — lift cannot resolve (motivates level-1 blow-up).",
    fontsize=9.5, y=0.98,
)

fname = "affinity_matrix.png"
plt.savefig(fname, bbox_inches="tight", dpi=150)
print(f"Saved {fname}")
plt.close(fig)
print("Done.")
