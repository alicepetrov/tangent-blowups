"""
Lifted Kernels on 2D Curves: Transverse vs Tangential Intersections
--------------------------------------------------------------------
Demonstrates the two lifted kernel families — product and self-tuning
(Zelnik-Manor) — on two canonical two-component curve examples.
The key comparison is between levels: which blow-up level first provides
enough geometric information for the graph Laplacian to metrically separate
the two components?

Examples
--------
A. **Figure-8** (transverse self-intersection at origin):
      Two lobes cross at right angles.  Their tangent directions at the
      crossing differ (angle ≈ 90°), so the first blow-up (level 1) encodes
      the angular separation in its Chordal-Sasaki metric.

      Level 0, self-tuning : position only → crossing couples both lobes,
                              Fiedler vector is a smooth interpolation.
      Level 0, product      : position k-NN, but angular factor downweights
                              cross-tangent edges → partial separation.
      Level 1, self-tuning  : Chordal-Sasaki distance encodes tangent plane;
                              cross-lobe pairs are far → clean separation.

B. **Line + Parabola** (tangential intersection at origin):
      Both components share position AND tangent direction at the contact
      point, so NEITHER level-0 nor level-1 metrics can separate them there.
      The curvature differs (κ_parabola = 2, κ_line = 0), which is encoded
      only at level 2.

      Level 0, self-tuning : merged (same position)
      Level 1, self-tuning : merged (same tangent)
      Level 2, self-tuning : separated (different curvature in metric)
      Level 2, product     : same, independent spatial + angular bandwidths

Figures
-------
Fig 1 — Transverse (figure-8): Fiedler vector & eigenvalue spectrum,
        three kernels / levels.
Fig 2 — Tangential (line+parabola) level-by-level: levels 0, 1, 2.
Fig 3 — Kernel comparison at level 2 (tangential): self-tuning vs product.

In each top row the **Fiedler vector** (first non-trivial eigenvector of the
graph Laplacian, coloured on the original 2D curve) shows whether the two
components are spectrally separated.  In each bottom row the first eight
eigenvalues show the size of the **Fiedler gap** λ₁ - λ₀.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from tangent_blowups.testsupport import (
    UniformCurve,
    figure8,
    sample,
    tangent_parabola_line,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, iterated_blowup
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.solvers.eigen import laplacian_spectrum


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_FIG8   = 1000    # sample count for figure-8
N_TANG   = 1200    # sample count for line + parabola (600 per component)
CURVATURE = 1.0    # parabola y = CURVATURE * x^2  →  κ = 2*CURVATURE at origin

K_BLOWUP  = 10     # k-NN for curvature regression inside iterated_blowup
ALPHA     = 1.0    # Chordal-Sasaki weighting factor
LAM       = 1e-3   # ridge regularisation for curvature regression

K_KERNEL  = 12     # k-NN for affinity construction
N_EIG     = 8      # eigenvalues to show in spectrum subplots
JITTER    = 0.01   # spatial noise
SEED      = 42


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _build_figure8() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample figure-8 (Lemniscate of Gerono).

    Returns:
        pts    (N, 2) spatial coordinates
        tans   (N, 2) unit tangent vectors
        gt     (N,)   0 = right lobe (t < π),  1 = left lobe (t ≥ π)
    """
    curve = figure8(scale=2.0)
    strat = UniformCurve(n=N_FIG8, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    s = sample(curve, strat, with_tangents=True, with_normals=False)

    pts  = np.asarray(s.points,   dtype=float)
    tans = np.asarray(s.tangents, dtype=float)
    t_p  = np.asarray(s.params,   dtype=float)

    rng = np.random.default_rng(SEED)
    pts = pts + rng.normal(scale=JITTER, size=pts.shape)

    nrm = np.linalg.norm(tans, axis=1, keepdims=True).clip(min=1e-10)
    tans = tans / nrm
    gt = (t_p >= np.pi).astype(int)
    return pts, tans, gt


def _build_line_parabola() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample line + parabola tangential union.

    Returns:
        pts    (N, 2) spatial coordinates
        tans   (N, 2) unit tangent vectors
        gt     (N,)   0 = parabola (t < 1),  1 = line (t ≥ 1)
    """
    curve = tangent_parabola_line(curvature=CURVATURE)
    strat = UniformCurve(n=N_TANG, t_min=0.0, t_max=2.0, endpoint=False)
    s = sample(curve, strat, with_tangents=True, with_normals=False)

    pts  = np.asarray(s.points,   dtype=float)
    rng = np.random.default_rng(SEED)
    pts = pts + rng.normal(scale=JITTER, size=pts.shape)
    tans = np.asarray(s.tangents, dtype=float)
    t_p  = np.asarray(s.params,   dtype=float)
    gt   = (t_p >= 1.0).astype(int)
    return pts, tans, gt


# ---------------------------------------------------------------------------
# Auto-bandwidth helpers
# ---------------------------------------------------------------------------

def _median_embed_dist(level: BlowUpLevel) -> float:
    """Median k-NN distance in level.embedded (for the Gaussian kernel sigma)."""
    tree = cKDTree(level.embedded)
    d, _ = tree.query(level.embedded, k=K_KERNEL + 1)
    return float(np.median(d[:, 1:]))


def _median_spatial_dist(level: BlowUpLevel) -> float:
    """Median k-NN distance in the original spatial coordinates."""
    pos = level.embedded[:, :level.n_orig]
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=K_KERNEL + 1)
    return float(np.median(d[:, 1:]))


def _median_proj_dist(level: BlowUpLevel) -> float:
    """Median ||P_i - P_j||_F over k-NN pairs via the chordal identity."""
    tree = cKDTree(level.embedded)
    _, idx = tree.query(level.embedded, k=K_KERNEL + 1)
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(level.N), K_KERNEL)
    cols = idx.ravel()
    U = level.frame                                    # (N, D, d)
    M = np.einsum("eka,ekb->eab", U[rows], U[cols])   # (E, d, d)
    inner_sq = np.einsum("eab,eab->e", M, M)
    dist = np.sqrt(np.maximum(2.0 * level.d - 2.0 * inner_sq, 0.0))
    pos = dist[dist > 0]
    return float(np.median(pos)) if pos.size else 1.0


# ---------------------------------------------------------------------------
# Kernel / Laplacian construction
# ---------------------------------------------------------------------------

def _build_L(level: BlowUpLevel, kernel: str) -> tuple:
    """
    Build a lifted Laplacian (L, W, D) with automatically selected bandwidths.

    Bandwidth selection:
      self_tuning : h = "local"  (Zelnik-Manor pointwise)
      product     : sigma_x = median k-NN distance in original positions,
                    sigma_u = median Frobenius projector dist over k-NN pairs
    """
    if kernel == "lifted":
        return lifted_laplacian(level, kernel="lifted", k=K_KERNEL, h="local")
    if kernel == "product":
        sigma_x = _median_spatial_dist(level)
        sigma_u = _median_proj_dist(level)
        return lifted_laplacian(
            level, kernel="product", k=K_KERNEL,
            sigma_x=sigma_x, sigma_u=sigma_u,
        )
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _spectrum(L) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (evals, fiedler_vec):
      evals       first N_EIG eigenvalues (including λ₀ ≈ 0)
      fiedler_vec eigenvector for λ₁  (first non-trivial mode)
    """
    evals, evecs = laplacian_spectrum(L, k=N_EIG, drop_first=False,
                                      return_eigenvalues=True)
    fiedler_vec = evecs[:, min(1, evecs.shape[1] - 1)]
    return evals, fiedler_vec


def _cross_frac(W, gt: np.ndarray) -> float:
    """Fraction of total affinity weight on cross-component edges."""
    A = np.where(gt == 0)[0]
    B = np.where(gt == 1)[0]
    W_AB = W[A, :][:, B]
    W_BA = W[B, :][:, A]
    cross  = float(W_AB.sum() + W_BA.sum())
    total  = float(W.sum())
    return cross / (total + 1e-12)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_fiedler(ax, pts, fvec, title, *, cf: float, fgap: float):
    """
    Scatter plot coloured by the Fiedler vector value.

    A bimodal colour pattern (clean red / blue split) indicates spectral
    separation of the two components.
    """
    vabs = float(np.percentile(np.abs(fvec), 99)) + 1e-9
    ax.scatter(pts[:, 0], pts[:, 1], c=fvec,
               cmap="RdBu_r", vmin=-vabs, vmax=vabs,
               s=4, rasterized=True, linewidths=0)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel(
        f"cross-edge fraction: {cf:.1%}   |   Fiedler gap: {fgap:.4f}",
        fontsize=7,
    )


def _plot_spectrum(ax, evals, title):
    """Bar chart of the first N_EIG eigenvalues; highlights the Fiedler gap."""
    xs = np.arange(len(evals))
    ax.bar(xs, evals, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=6)
    ax.set_ylabel("λ", fontsize=7)
    if len(evals) >= 2:
        gap = float(evals[1] - evals[0])
        clr = "firebrick" if gap > 0.02 else "gray"
        ax.annotate(
            f"gap\n{gap:.4f}",
            xy=(0.5, (evals[0] + evals[1]) / 2),
            xytext=(1.6, max(evals) * 0.7),
            fontsize=6, color=clr,
            arrowprops=dict(arrowstyle="->", color=clr, lw=0.8),
        )


def _make_figure(cases, pts, suptitle, figsize=(13, 7)):
    """
    Build a 2-row figure:  top = Fiedler vectors,  bottom = eigenvalue spectra.

    cases: list of (title, evals, fiedler_vec, cross_frac) tuples.
    """
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=figsize,
                             gridspec_kw={"height_ratios": [3, 1.5]})
    if n == 1:
        axes = axes[:, np.newaxis]

    for j, (title, evals, fvec, cf) in enumerate(cases):
        fgap = float(evals[1] - evals[0]) if len(evals) >= 2 else 0.0
        _plot_fiedler(axes[0, j], pts, fvec, title, cf=cf, fgap=fgap)
        _plot_spectrum(axes[1, j], evals, f"Spectrum — {title.split(chr(10))[0]}")

    fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ===================================================================
    # A. Transverse intersection — Figure-8
    # ===================================================================
    print("=" * 60)
    print("A. Figure-8  (transverse, crossing angle ~90 deg)")
    print("=" * 60)

    pts8, tan8, gt8 = _build_figure8()
    print(f"   Sampled N={len(pts8)}  "
          f"({(gt8==0).sum()} right lobe, {(gt8==1).sum()} left lobe)")

    levels8 = iterated_blowup(pts8, tan8, num_levels=1,
                              k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l8_0, l8_1 = levels8
    print(f"   Embedding dims: {l8_0.D} -> {l8_1.D}")

    print("   Building kernels...")

    # (A) Level 0, self-tuning: k-NN in position space only.
    #     The crossing point has k-NN neighbours from BOTH lobes
    #     → affinity couples the two branches.
    L8_A, W8_A, _ = _build_L(l8_0, "lifted")
    ev8_A, fv8_A  = _spectrum(L8_A)
    cf8_A         = _cross_frac(W8_A, gt8)
    print(f"   (A) Level 0 self-tuning :  cross frac = {cf8_A:.2%},  "
          f"Fiedler gap = {ev8_A[1]-ev8_A[0]:.5f}")

    # (B) Level 0, product: same position k-NN, but edge weights multiplied by
    #     exp(-||P_i - P_j||²_F / σ_u²).  At the crossing the two passages have
    #     perpendicular tangents (||ΔP||_F = √2) so the angular factor kills
    #     the cross-lobe edges while leaving same-lobe edges intact.
    L8_B, W8_B, _ = _build_L(l8_0, "product")
    ev8_B, fv8_B  = _spectrum(L8_B)
    cf8_B         = _cross_frac(W8_B, gt8)
    print(f"   (B) Level 0 product      :  cross frac = {cf8_B:.2%},  "
          f"Fiedler gap = {ev8_B[1]-ev8_B[0]:.5f}")

    # (C) Level 1, self-tuning: k-NN in the Chordal-Sasaki metric.
    #     d²_1 = ||x_i - x_j||² + (α/2)||P_i - P_j||²_F.
    #     At the crossing, same-position but different tangent
    #     → large d²_1 → no cross-lobe k-NN edges at all.
    L8_C, W8_C, _ = _build_L(l8_1, "lifted")
    ev8_C, fv8_C  = _spectrum(L8_C)
    cf8_C         = _cross_frac(W8_C, gt8)
    print(f"   (C) Level 1 self-tuning :  cross frac = {cf8_C:.2%},  "
          f"Fiedler gap = {ev8_C[1]-ev8_C[0]:.5f}")

    # ===================================================================
    # B. Tangential intersection — Line + Parabola
    # ===================================================================
    print()
    print("=" * 60)
    print(f"B. Line + Parabola  (tangential, kappa_parabola = {2*CURVATURE:.1f}, kappa_line = 0)")
    print("=" * 60)

    pts_lp, tan_lp, gt_lp = _build_line_parabola()
    print(f"   Sampled N={len(pts_lp)}  "
          f"({(gt_lp==0).sum()} parabola, {(gt_lp==1).sum()} line)")

    levels_lp = iterated_blowup(pts_lp, tan_lp, num_levels=2,
                                k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l0, l1, l2 = levels_lp
    print(f"   Embedding dims: {l0.D} -> {l1.D} -> {l2.D}")

    print("   Building kernels...")

    # (D) Level 0, self-tuning: position only.
    #     Near the origin both curves have the same (x, y) → merged.
    L_D, W_D, _ = _build_L(l0, "lifted")
    ev_D, fv_D  = _spectrum(L_D)
    cf_D        = _cross_frac(W_D, gt_lp)
    print(f"   (D) Level 0 self-tuning :  cross frac = {cf_D:.2%},  "
          f"Fiedler gap = {ev_D[1]-ev_D[0]:.5f}")

    # (E) Level 1, self-tuning: position + tangent plane.
    #     Both curves share the horizontal tangent at the origin
    #     → level-1 Chordal-Sasaki still merges them there.
    L_E, W_E, _ = _build_L(l1, "lifted")
    ev_E, fv_E  = _spectrum(L_E)
    cf_E        = _cross_frac(W_E, gt_lp)
    print(f"   (E) Level 1 self-tuning :  cross frac = {cf_E:.2%},  "
          f"Fiedler gap = {ev_E[1]-ev_E[0]:.5f}")

    # (F) Level 2, self-tuning: position + tangent + curvature.
    #     κ_parabola ≠ κ_line → level-2 Chordal-Sasaki separates them.
    L_F, W_F, _ = _build_L(l2, "lifted")
    ev_F, fv_F  = _spectrum(L_F)
    cf_F        = _cross_frac(W_F, gt_lp)
    print(f"   (F) Level 2 self-tuning :  cross frac = {cf_F:.2%},  "
          f"Fiedler gap = {ev_F[1]-ev_F[0]:.5f}")

    # (G) Level 2, product: k-NN in level2.embedded, but separate spatial (σ_x)
    #     and angular (σ_u) bandwidths for the level-2 tangent projectors.
    L_G, W_G, _ = _build_L(l2, "product")
    ev_G, fv_G  = _spectrum(L_G)
    cf_G        = _cross_frac(W_G, gt_lp)
    print(f"   (G) Level 2 product     :  cross frac = {cf_G:.2%},  "
          f"Fiedler gap = {ev_G[1]-ev_G[0]:.5f}")

    # ===================================================================
    # Figure 1: Transverse — kernel comparison
    # ===================================================================
    cases_fig1 = [
        (
            "Level 0 — self-tuning\n(position only)\ncross-lobe k-NN at crossing",
            ev8_A, fv8_A, cf8_A,
        ),
        (
            "Level 0 — product\n(position × angular weight)\nangular factor ↓ cross-lobe edges",
            ev8_B, fv8_B, cf8_B,
        ),
        (
            "Level 1 — self-tuning\n(Chordal-Sasaki: pos + tangent)\nno cross-lobe k-NN at all",
            ev8_C, fv8_C, cf8_C,
        ),
    ]
    _make_figure(
        cases_fig1,
        pts8,
        (
            "Figure 1 — Transverse Intersection: Figure-8 Curve\n"
            "Two lobes cross at ≈90°.  Tangent direction at the crossing\n"
            "differs between lobes — tangent-aware kernels resolve them at level 0 or 1."
        ),
        figsize=(13, 7),
    )

    # ===================================================================
    # Figure 2: Tangential — level-by-level
    # ===================================================================
    cases_fig2 = [
        (
            "Level 0 — self-tuning\n(position only)\nmerged: shared (x, y)",
            ev_D, fv_D, cf_D,
        ),
        (
            "Level 1 — self-tuning\n(pos + tangent)\nmerged: shared tangent at origin",
            ev_E, fv_E, cf_E,
        ),
        (
            "Level 2 — self-tuning\n(pos + tan + curvature)\nSEPARATED: κ_par≠κ_line",
            ev_F, fv_F, cf_F,
        ),
    ]
    _make_figure(
        cases_fig2,
        pts_lp,
        (
            f"Figure 2 — Tangential Intersection: Line + Parabola  "
            f"(κ_parabola = {2*CURVATURE:.1f},  κ_line = 0)\n"
            "Position AND tangent coincide at the origin — curvature is the\n"
            "distinguishing invariant.  Only the level-2 embedding encodes it."
        ),
        figsize=(13, 7),
    )

    # ===================================================================
    # Figure 3: Kernel comparison at level 2 (tangential case)
    # ===================================================================
    cases_fig3 = [
        (
            "Level 2 — self-tuning\n(adaptive h_i·h_j bandwidth)\nZelnik-Manor & Perona 2004",
            ev_F, fv_F, cf_F,
        ),
        (
            "Level 2 — product\n(σ_x spatial, σ_u angular)\nindependent bandwidths, PD kernel",
            ev_G, fv_G, cf_G,
        ),
    ]
    _make_figure(
        cases_fig3,
        pts_lp,
        (
            "Figure 3 — Kernel Comparison at Level 2: Line + Parabola\n"
            "All three kernel families separate the components once curvature\n"
            "is encoded (level 2).  Differences are in bandwidth adaptivity."
        ),
        figsize=(13, 7),
    )

    plt.show()


if __name__ == "__main__":
    main()
