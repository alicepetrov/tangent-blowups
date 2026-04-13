"""
2D Tangential Intersection: Spectral Clustering via Self-Tuning Kernel Laplacian
----------------------------------------------------------------------------------
Demonstrates spectral clustering on a tangentially intersecting pair of planar
curves (line + parabola) using the self-tuning kernel Laplacian built on the
iterated blow-up embedding.

Key result
----------
  Level 1  (position + tangent encoded in Chordal-Sasaki metric):
      FAILS to separate -- both curves share the same tangent direction at the
      contact point, so the level-1 Chordal-Sasaki distance is still small there.

  Level 2  (position + tangent + curvature encoded in Chordal-Sasaki metric):
      SUCCEEDS -- curvature differs (kappa_parabola = 2*c != 0, kappa_line = 0),
      so the level-2 embedding pulls the two components apart near the contact.

The figure is laid out as a 3-row grid:
  Row 0 (exact tangents):         GT | Level 1 | Level 2
  Row 1 (PCA tangents, naive):    GT | Level 1 | Level 2
  Row 2 (PCA tangents, branch-split): GT | singular pts | Level 2

Branch-split approach
---------------------
Although the tangential intersection is degenerate in position space (both curves
share the same tangent at the contact), the first blow-up RESOLVES the singularity:

  In level-1 Chordal-Sasaki space the level-1 curve-tangent directions are:
    line:     d/ds Phi_1 = (1, 0, 0,  0,            0,           0)
    parabola: d/ds Phi_1 = (1, 0, 0, sqrt(a/2)*2k, sqrt(a/2)*2k, 0)

  where a = alpha, k = kappa = 2*CURVATURE.  With alpha=30, kappa=6, the angle
  between these is ~89 degrees -- a near-TRANSVERSE intersection in level-1 space.

We exploit this:
  1. Build level-1 from PCA tangents (standard).
  2. For each point, run 2-component PCA on level-1 Chordal-Sasaki neighbourhoods.
     Points near the contact have bimodal neighbourhoods (eigenvalue ratio ~1);
     points away from the contact have unimodal neighbourhoods (ratio ~0).
  3. Duplicate singular points: copy A gets the top-PCA-direction as its level-1
     frame, copy B gets the second-PCA-direction.  The two copies now have
     DIFFERENT projectors and hence different Phi_2 embeddings.
  4. Lift the augmented level-1 to level-2.  The two copies map to separate
     locations; spectral clustering cleanly separates the components.


# NOTE: This is a dev script, I'm just tryinig to see what works

"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tangent_blowups.testsupport import UniformCurve, tangent_parabola_line, sample
from tangent_blowups.geometry.iterated_grassmann import iterated_blowup, BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.clustering import spectral_embedding
from sklearn.cluster import KMeans
from tangent_blowups.pointcloud import estimate_tangents_pca


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N          = 5000       # total samples (600 per component)
CURVATURE  = 1.0        # parabola: y = CURVATURE * x^2  =>  kappa = 2*CURVATURE at origin
JITTER     = 0.01      # spatial noise
SEED       = 42

K_BLOWUP   = 30         # k-NN for curvature regression inside lift()
K_PCA      = 30         # k-NN for local PCA tangent estimation
ALPHA  = 1.0  # Chordal-Sasaki scaling
LAM        = 1e-3       # ridge regularisation for curvature regression

K_KERNEL   = 20         # k-NN for affinity construction
N_CLUSTERS = 2

K_BRANCH      = 320      # k-NN for branch detection (position-space)
                        # needs to span past the mixing zone (~sqrt(JITTER/kappa) arc-length)
BRANCH_THRESH = 0.2     # eigenvalue ratio lambda_2/lambda_1 of sum(P_j) for singularity

N_LEVELS      = 2       # number of blow-up levels to compute (2 or 3)

N_SMOOTH      = 3       # iterations of projector smoothing in level-1 space


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _build_line_parabola() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample line + parabola tangential union.

    Returns:
        pts  (N, 2)  spatial coordinates
        tans (N, 2)  unit tangent vectors (exact)
        gt   (N,)    0 = parabola (t < 1),  1 = line (t >= 1)
    """
    curve = tangent_parabola_line(curvature=CURVATURE)
    strat = UniformCurve(n=N, t_min=0.0, t_max=2.0, endpoint=False)
    s = sample(curve, strat, with_tangents=True, with_normals=False)

    pts  = np.asarray(s.points,   dtype=float)
    rng  = np.random.default_rng(SEED)
    pts  = pts + rng.normal(scale=JITTER, size=pts.shape)
    tans = np.asarray(s.tangents, dtype=float)
    t_p  = np.asarray(s.params,   dtype=float)
    gt   = (t_p >= 1.0).astype(int)
    return pts, tans, gt


# ---------------------------------------------------------------------------
# Tangent quality metric
# ---------------------------------------------------------------------------

def _mean_alignment(tans_est: np.ndarray, tans_gt: np.ndarray) -> float:
    """
    Mean |cos theta| between estimated and ground-truth tangents.

    Uses absolute value to handle the sign ambiguity of PCA eigenvectors.
    Returns 1.0 for perfect alignment, 0.0 for orthogonal.
    """
    dots = np.einsum("ni,ni->n", tans_est, tans_gt)
    return float(np.mean(np.abs(dots)))


# ---------------------------------------------------------------------------
# Accuracy (best assignment over the 2 possible label permutations)
# ---------------------------------------------------------------------------

def _accuracy(labels: np.ndarray, gt: np.ndarray) -> float:
    acc0 = np.mean(labels == gt)
    acc1 = np.mean(labels == (1 - gt))
    return float(max(acc0, acc1))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _cluster_cmap(n_clusters: int):
    return plt.get_cmap("tab10", n_clusters)


def _plot_clusters(
    ax,
    pts: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    acc: float | None = None,
    fgap: float | None = None,
) -> None:
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    cmap = _cluster_cmap(max(n_clusters, 1))
    ax.scatter(
        pts[:, 0], pts[:, 1],
        c=labels, cmap=cmap,
        vmin=-0.5, vmax=max(n_clusters - 0.5, 0.5),
        s=8, rasterized=True, linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    parts = []
    if fgap is not None:
        parts.append(f"Fiedler gap = {fgap:.4f}")
    if acc is not None:
        parts.append(f"accuracy = {acc:.1%}")
    if parts:
        ax.set_xlabel("  |  ".join(parts), fontsize=7)


def _plot_tangents(
    ax,
    pts: np.ndarray,
    tans: np.ndarray,
    gt: np.ndarray,
    title: str,
    *,
    stride: int = 25,
) -> None:
    """Quiver plot of tangent vectors on the point cloud, colored by GT label."""
    tans2d = tans[:, :, 0] if tans.ndim == 3 else tans
    cmap = _cluster_cmap(2)
    ax.scatter(
        pts[:, 0], pts[:, 1], c=gt, cmap=cmap,
        vmin=-0.5, vmax=1.5, s=2, rasterized=True, linewidths=0, alpha=0.35,
    )
    idx = np.arange(0, len(pts), stride)
    colors = [cmap(int(gt[i])) for i in idx]
    extent = max(pts[:, 0].max() - pts[:, 0].min(),
                 pts[:, 1].max() - pts[:, 1].min())
    scale = 20.0 / extent   # each unit tangent draws as ~1/20 of data extent
    ax.quiver(
        pts[idx, 0], pts[idx, 1],
        tans2d[idx, 0], tans2d[idx, 1],
        color=colors, angles="xy", scale_units="xy", scale=scale,
        width=0.003, headwidth=3, headlength=4, alpha=0.9,
    )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _plot_singular(ax, pts: np.ndarray, singular_mask: np.ndarray, title: str) -> None:
    """Visualise which points were flagged as singular (bimodal in level-1 space)."""
    ax.scatter(
        pts[~singular_mask, 0], pts[~singular_mask, 1],
        c="steelblue", s=6, rasterized=True, linewidths=0, label="regular",
    )
    ax.scatter(
        pts[singular_mask, 0], pts[singular_mask, 1],
        c="orange", s=14, rasterized=True, linewidths=0, label="singular",
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    n_sing = int(singular_mask.sum())
    ax.set_xlabel(f"{n_sing}/{len(pts)} singular pts detected", fontsize=7)
    ax.legend(fontsize=6, loc="upper right", markerscale=1.5)


def _print_cluster_sizes(label: str, labels: np.ndarray) -> None:
    n_clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=n_clusters)
    print(f"  {label} cluster sizes: {counts.tolist()}")


# ---------------------------------------------------------------------------
# Standard two-level pipeline
# ---------------------------------------------------------------------------

LevelResult = tuple[np.ndarray, float, float]   # (labels, gap, acc)


def _run_levels(
    pts: np.ndarray,
    tans: np.ndarray,
    gt: np.ndarray,
    label: str,
    *,
    n_levels: int = N_LEVELS,
) -> list[LevelResult]:
    """Cluster at each blow-up level 1..n_levels; return list of (labels, gap, acc)."""
    levels = iterated_blowup(pts, tans, num_levels=n_levels,
                             k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    results: list[LevelResult] = []
    for lvl_num, lvl in enumerate(levels[1:], 1):
        L, _, _ = lifted_laplacian(lvl, kernel="product", self_tuning=True, k=K_KERNEL, normalized=True)
        evals, emb = spectral_embedding(L, n_components=N_CLUSTERS)
        labels = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(emb)
        gap = float(evals[1] - evals[0]) if len(evals) >= 2 else 0.0
        acc = _accuracy(labels, gt)
        print(f"  {label} L{lvl_num}: gap={gap:.5f}  acc={acc:.1%}")
        _print_cluster_sizes(f"{label} L{lvl_num}", labels)
        results.append((labels, gap, acc))
    return results


# ---------------------------------------------------------------------------
# Branch-split pipeline
# ---------------------------------------------------------------------------

def _branch_split_level2(
    pts: np.ndarray,
    tans: np.ndarray,
    gt: np.ndarray,
    *,
    k_branch: int,
    branch_thresh: float,
    n_levels: int = N_LEVELS,
) -> tuple[list[LevelResult], np.ndarray]:
    """
    Level-2 and level-3 clustering using branch-aware singular-point doubling.

    Algorithm
    ---------
    1. Build level-0 and level-1 from the supplied (PCA) tangents.
    2. Detect "singular" points via the POSITION-SPACE k-NN.  For each point
       sum the level-1 projectors of its position-space neighbours:
           M_i = sum_{j in pos-kNN(i)} P_j^{level-1}
       M_i is rank-1 when all neighbours are on one branch (one dominant
       tangent direction) and rank-2 when neighbours span two branches.
       We flag points where lambda_2(M_i) / lambda_1(M_i) > branch_thresh.
    3. Duplicate each singular point.  The original copy receives the top
       eigenvector of M_i as its level-1 frame; the duplicate receives the
       second eigenvector.  Both copies have the same level-1 embedded
       coordinates but different projectors P = u u^T, so their level-2
       embeddings (Phi_2 = (Phi_1, sqrt(alpha/2)*vec(P))) differ by a
       large amount when alpha is large.
    4. Lift the augmented level-1 to level-2 and run spectral clustering.
    5. Map labels back to the original N points (duplicate copies are dropped).

    Why position-space k-NN (not level-1 k-NN)?
    With large alpha the level-1 Chordal-Sasaki distance already separates the
    branches (parabola points are ~2*sqrt(alpha)*kappa*eps farther than line
    points at the same arc-length eps).  So level-1 neighbourhoods are already
    pure and yield M_i ~ rank-1 everywhere -- the bimodal signal is invisible.
    Position-space k-NN deliberately mixes the branches near the contact, so
    M_i = sum P_j genuinely has two large eigenvalues there.

    Returns
    -------
    results       : list of (labels, gap, acc) for levels 2..n_levels
    singular_mask : (N,) boolean, True at detected singular points
    """
    N = len(pts)

    # ------------------------------------------------------------------
    # Level-0 -> Level-1 (standard)
    # ------------------------------------------------------------------
    l0 = BlowUpLevel.from_point_tangents(pts, tans)   # tans: (N,2) or (N,2,1)
    l1 = l0.lift(k=K_BLOWUP, alpha=ALPHA, lam=LAM)

    # ------------------------------------------------------------------
    # Branch detection: sum of level-1 projectors in position-space k-NN
    # ------------------------------------------------------------------
    # For each point i, collect the level-1 projectors P_j of its
    # POSITION-SPACE neighbours and form M_i = sum_j P_j^{level-1}.
    # When both branches pass through the neighbourhood, M_i has two
    # comparable eigenvalues (one per branch tangent direction).
    # When only one branch is present, M_i is approximately rank-1.
    tree_pos = cKDTree(pts)
    k_eff = min(k_branch + 1, N)
    _, idx = tree_pos.query(pts, k=k_eff)
    idx = idx[:, 1:]  # exclude self  -> (N, k_branch)

    P_all = l1.projectors  # (N, D1, D1)

    singular_mask = np.zeros(N, dtype=bool)
    branch_A = np.empty((N, l1.D), dtype=float)
    branch_B = np.empty((N, l1.D), dtype=float)

    for i in range(N):
        M    = P_all[idx[i]].sum(axis=0)              # (D1, D1)
        vals, vecs = np.linalg.eigh(M)                # ascending eigenvalues
        lam1, lam2 = vals[-1], vals[-2]
        if lam1 > 0 and lam2 / lam1 > branch_thresh:
            singular_mask[i] = True
            branch_A[i] = vecs[:, -1]   # dominant tangent direction  (branch A)
            branch_B[i] = vecs[:, -2]   # second tangent direction     (branch B)

    n_singular = int(singular_mask.sum())
    print(f"  Branch-split: {n_singular}/{N} singular points detected in level-1")
    singular_idx = np.where(singular_mask)[0]

    # ------------------------------------------------------------------
    # Augment level-1: one extra copy per singular point
    # ------------------------------------------------------------------
    aug_embedded = np.vstack(
        [l1.embedded, l1.embedded[singular_idx]]
    )                                                         # (N_aug, D1)
    aug_frame = np.concatenate(
        [l1.frame.copy(), l1.frame[singular_idx].copy()], axis=0
    )                                                         # (N_aug, D1, d1)

    # Assign per-branch tangent directions in level-1 space
    for i in singular_idx:
        v = branch_A[i].copy()
        v /= max(np.linalg.norm(v), 1e-12)
        aug_frame[i, :, 0] = v                   # original copy: branch A

    for k_s, i in enumerate(singular_idx):
        v = branch_B[i].copy()
        v /= max(np.linalg.norm(v), 1e-12)
        aug_frame[N + k_s, :, 0] = v             # duplicate copy: branch B

    # ------------------------------------------------------------------
    # Iteratively lift augmented level-1 -> level-2 -> ... -> n_levels
    # ------------------------------------------------------------------
    # The two copies of each singular point differ only in their frame
    # (projector P = u u^T).  In the level-2 Chordal-Sasaki embedding
    #   Phi_2 = (Phi_1,  sqrt(alpha/2) * vec(P_1))
    # they land at different locations because P_1 differs.  The k-NN
    # built in Phi_2 space therefore sees clean single-branch neighbourhoods
    # for each copy, giving accurate curvature estimates per branch.
    aug_lvl = BlowUpLevel(
        embedded=aug_embedded, frame=aug_frame,
        level=1, n_orig=l1.n_orig,
    )
    results: list[LevelResult] = []
    for lvl_num in range(2, n_levels + 1):
        aug_lvl = aug_lvl.lift(k=K_BLOWUP, alpha=ALPHA, lam=LAM)
        L, _, _ = lifted_laplacian(aug_lvl, kernel="product", self_tuning=True, k=K_KERNEL, normalized=True)
        evals, emb = spectral_embedding(L, n_components=N_CLUSTERS)
        labels_aug = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(emb)
        gap = float(evals[1] - evals[0]) if len(evals) >= 2 else 0.0
        labels = labels_aug[:N].copy()
        acc = _accuracy(labels, gt)
        print(f"  Branch-split L{lvl_num}: gap={gap:.5f}  acc={acc:.1%}")
        _print_cluster_sizes(f"Branch-split L{lvl_num}", labels)
        results.append((labels, gap, acc))

    return results, singular_mask

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building line + parabola samples...")
    pts, tans_exact, gt = _build_line_parabola()
    n_par = int((gt == 0).sum())
    n_lin = int((gt == 1).sum())
    print(f"  N={len(pts)},  parabola: {n_par},  line: {n_lin}")

    # ------------------------------------------------------------------
    # PCA tangent estimation
    # ------------------------------------------------------------------
    print(f"Estimating tangents via local PCA (k={K_PCA})...")
    tans_pca = estimate_tangents_pca(pts, k=K_PCA, tangent_dim=1)
    align = _mean_alignment(tans_pca, tans_exact)
    print(f"  Mean |cos theta| (PCA vs exact): {align:.4f}")

    # ------------------------------------------------------------------
    # Row 0: exact tangents
    # ------------------------------------------------------------------
    print("\nExact tangents:")
    res_exact = _run_levels(pts, tans_exact, gt, "Exact", n_levels=N_LEVELS)

    # ------------------------------------------------------------------
    # Row 1: PCA tangents (naive)
    # ------------------------------------------------------------------
    print("\nPCA-estimated tangents (naive):")
    res_pca = _run_levels(pts, tans_pca, gt, "PCA", n_levels=N_LEVELS)

    # ------------------------------------------------------------------
    # Row 2: PCA tangents + branch-split
    # ------------------------------------------------------------------
    print("\nPCA-estimated tangents + branch-split:")
    res_bs, singular_mask = _branch_split_level2(
        pts, tans_pca, gt,
        k_branch=K_BRANCH, branch_thresh=BRANCH_THRESH, n_levels=N_LEVELS,
    )


    # ------------------------------------------------------------------
    # Plot: 4 x (N_LEVELS+1) grid
    #   Row 0 (exact):        tangents | L1 | L2 | [L3]
    #   Row 1 (PCA naive):    tangents | L1 | L2 | [L3]
    #   Row 2 (branch-split): GT       | singular pts | L2 | [L3]
    # ------------------------------------------------------------------
    n_cols = N_LEVELS + 1
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 21))

    _plot_tangents(axes[0, 0], pts, tans_exact, gt, "Exact tangents\n(colored by GT)")
    for j, (labels, gap, acc) in enumerate(res_exact, 1):
        suffix = "\nFAILS: shared tangent at contact" if j == 1 else ""
        _plot_clusters(axes[0, j], pts, labels,
                       f"Exact tans -- Level {j}{suffix}",
                       acc=acc, fgap=gap)

    # Row 1: PCA (naive)
    _plot_tangents(axes[1, 0], pts, tans_pca, gt,
                   f"PCA tangents  (k={K_PCA})\n(colored by GT)")
    for j, (labels, gap, acc) in enumerate(res_pca, 1):
        if j == 1:
            suffix = "\nFAILS: shared tangent at contact"
        elif j == 2:
            suffix = "\nmixed neighbourhoods"
        else:
            suffix = ""
        _plot_clusters(axes[1, j], pts, labels,
                       f"PCA tans -- Level {j}{suffix}",
                       acc=acc, fgap=gap)

    # Row 2: branch-split
    _plot_clusters(axes[2, 0], pts, gt, "Ground truth\n(parabola vs line)")
    _plot_singular(axes[2, 1], pts, singular_mask,
                   f"Branch detection in level-1 space\n"
                   f"(eigenvalue ratio > {BRANCH_THRESH})\n"
                   f"orange = bimodal = near contact")
    for j, (labels, gap, acc) in enumerate(res_bs, 2):
        suffix = "\nsingular pts doubled" if j == 2 else ""
        _plot_clusters(axes[2, j], pts, labels,
                       f"Branch-split -- Level {j}{suffix}",
                       acc=acc, fgap=gap)

    # Row labels
    row_labels = [
        "Exact tangents",
        f"PCA tangents  (k={K_PCA},  mean |cos|={align:.3f})",
        f"PCA + branch-split  (k_branch={K_BRANCH})",
    ]
    for row, row_label in enumerate(row_labels):
        axes[row, 0].set_ylabel(row_label, fontsize=9, labelpad=8)

    kappa = 2 * CURVATURE
    fig.suptitle(
        f"Tangential Intersection: Line + Parabola  (kappa = {kappa:.1f},  alpha = {ALPHA})\n"
        f"Level-1 blow-up resolves tangential -> transverse (angle ~"
        f"{int(round(np.degrees(np.arccos(1.0 / np.sqrt(1 + 4 * ALPHA * kappa**2)))))}deg"
        f" between branch tangents in level-1 space)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
