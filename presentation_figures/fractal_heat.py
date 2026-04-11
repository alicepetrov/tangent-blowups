"""
fractal_heat.py
---------------
Three branching 2D trees placed side by side so their branches
intersect transversely.  Compare heat diffusion computed with:
  (a) standard Euclidean graph Laplacian  (alpha=0)
  (b) level-1 blow-up Laplacian — curvature-aware lifted metric

Key insight:
  - At Y-junctions curvature is similar on both child branches,
    so the level-1 kernel connects them -> heat spreads into both forks.
  - At X-crossings between different trees curvature and tangent
    directions differ, so the level-1 kernel blocks the connection
    -> heat goes straight through without leaking.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.embedded import KDTree

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.pointcloud.geodesic_heat import lifted_heat_method

# -- Parameters ----------------------------------------------------------------
N_PER_SEG    = 25          # points per branch segment (sparse for heat spread)
DEPTH        = 3           # recursion depth per tree
ANGLE_SPREAD = np.pi / 6   # half-angle spread at each fork (~30 deg)
LENGTH_SCALE = 0.5         # child length = parent * scale
ROOT_LENGTH  = 0.4          # length of trunk

N_TREES      = 3            # number of trees in the forest
SEED         = 7            # random seed for reproducible layout

K            = 5           # k-NN for Laplacian
T_SCALE      = 800000.0     # heat time scale multiplier (spread heat far)

# Product kernel bandwidths (set automatically from data if None)
SIGMA_U      = 0.3          # angular bandwidth — tight to block X-crossings

SAVE_PATH    = "fractal_heat.png"


# -- Tree construction ---------------------------------------------------------

def _segment_pts(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Linearly spaced points along a segment (excludes endpoint)."""
    t = np.linspace(0.0, 1.0, n + 1)[:-1]
    return start[None, :] + t[:, None] * (end - start)[None, :]


def _smooth_tangents(pts, tans, radius, group_ids, iterations=3):
    """Spatially smooth tangent vectors within each group only."""
    tree = KDTree(pts)
    result = tans.copy()
    for _ in range(iterations):
        new = np.zeros_like(result)
        neighbors = tree.query_ball_point(pts, radius)
        for i, nbrs in enumerate(neighbors):
            nbrs = np.array(nbrs)
            same = nbrs[group_ids[nbrs] == group_ids[i]]
            if len(same) == 0:
                new[i] = result[i]
                continue
            ti = result[i]
            nbr_tans = result[same]
            signs = np.sign(nbr_tans @ ti)
            signs[signs == 0] = 1.0
            aligned = nbr_tans * signs[:, None]
            new[i] = aligned.mean(axis=0)
        norms = np.linalg.norm(new, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        result = new / norms
    return result


def _build_single_tree(
    root: np.ndarray,
    trunk_angle: float,
    depth: int,
    angle_spread: float,
    length_scale: float,
    root_length: float,
    n_per_seg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one binary tree and return (pts, tans, levels)."""
    pts_list, tans_list, levels_list = [], [], []

    def grow(start, angle, length, level):
        direction = np.array([np.sin(angle), np.cos(angle)])
        end = start + length * direction
        seg = _segment_pts(start, end, n_per_seg)
        tan = np.tile(direction, (n_per_seg, 1))
        pts_list.append(seg)
        tans_list.append(tan)
        levels_list.append(np.full(n_per_seg, level, dtype=int))
        if level < depth:
            grow(end, angle - angle_spread, length * length_scale, level + 1)
            grow(end, angle + angle_spread, length * length_scale, level + 1)

    grow(root, trunk_angle, root_length, 0)
    return (
        np.concatenate(pts_list),
        np.concatenate(tans_list),
        np.concatenate(levels_list),
    )


def build_forest(
    *,
    n_trees: int = N_TREES,
    depth: int = DEPTH,
    angle_spread: float = ANGLE_SPREAD,
    length_scale: float = LENGTH_SCALE,
    root_length: float = ROOT_LENGTH,
    n_per_seg: int = N_PER_SEG,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Place *n_trees* binary trees with varied positions, orientations,
    and branching parameters so their canopies overlap with transverse
    crossings and each tree looks distinct.
    """
    rng = np.random.default_rng(seed)
    spacing = root_length * 0.55

    all_pts, all_tans, all_levels, all_ids = [], [], [], []
    seg_lengths = []

    for i in range(n_trees):
        x = (i - (n_trees - 1) / 2) * spacing + rng.uniform(-0.15, 0.15)
        y = rng.uniform(-0.1, 0.1)
        root_i = np.array([x, y])
        angle_i = rng.uniform(-np.pi / 5, np.pi / 5)
        spread_i = angle_spread * rng.uniform(0.8, 1.3)
        scale_i = length_scale * rng.uniform(0.9, 1.1)
        length_i = root_length * rng.uniform(0.85, 1.15)

        pts_i, tans_i, lvl_i = _build_single_tree(
            root_i, angle_i, depth,
            spread_i, scale_i, length_i, n_per_seg,
        )
        all_pts.append(pts_i)
        all_tans.append(tans_i)
        all_levels.append(lvl_i)
        all_ids.append(np.full(len(pts_i), i, dtype=int))
        for lev in range(depth + 1):
            seg_lengths.append(length_i * scale_i ** lev)

    pts = np.concatenate(all_pts)
    tans = np.concatenate(all_tans)
    levels = np.concatenate(all_levels)
    tree_ids = np.concatenate(all_ids)

    avg_seg_spacing = np.mean(seg_lengths) / n_per_seg
    tans = _smooth_tangents(pts, tans, radius=avg_seg_spacing * 5,
                            group_ids=tree_ids, iterations=3)

    return pts, tans, levels, tree_ids


# -- Heat diffusion on a blow-up level ----------------------------------------

def _heat_diffuse(L, W, pts, source_index, t_scale):
    """Solve one implicit heat step: (I + t L) u = delta."""
    N = L.shape[0]
    W_csr = W.tocsr()
    coo = W_csr.tocoo()
    mask = coo.row != coo.col
    rows, cols = coo.row[mask], coo.col[mask]
    if rows.size > 0:
        d = np.linalg.norm(pts[rows] - pts[cols], axis=1)
        d = d[d > 0]
        t = t_scale * float(np.median(d) ** 2) if d.size else t_scale
    else:
        t = t_scale

    rhs = np.zeros(N)
    src_idx = np.atleast_1d(source_index)
    rhs[src_idx] = 1.0
    A = sparse.eye(N, format="csr") + float(t) * L
    u = spla.spsolve(A, rhs)
    return u


# -- Visualization helpers -----------------------------------------------------

def _plot_heat(ax, pts, u, title, norm=None):
    """Plot heat diffusion values with log-scale coloring."""
    u_plot = u.copy()
    u_plot[~np.isfinite(u_plot)] = 0.0
    u_plot = np.clip(u_plot, 0, None)
    sc = ax.scatter(
        pts[:, 0], pts[:, 1],
        c=u_plot, cmap="magma",
        norm=norm,
        s=14, linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return sc


# -- Main ----------------------------------------------------------------------

def main():
    pts, tans, levels, tree_ids = build_forest()
    tans = normalize_vectors(tans)
    N = pts.shape[0]
    n_trees = tree_ids.max() + 1
    print(f"Forest: {N} points, {n_trees} trees, depth={DEPTH}")

    # Source on the middle tree in the overlap zone
    middle_tree = n_trees // 2
    cand_mask = (tree_ids == middle_tree) & (levels == 2)
    if not cand_mask.any():
        cand_mask = (tree_ids == middle_tree) & (levels == 1)
    cand_indices = np.where(cand_mask)[0]
    _tree = KDTree(pts)
    _, _idxs = _tree.query(pts[cand_indices], k=K + 1)
    cross_counts = np.zeros(len(cand_indices), dtype=int)
    for ci, _i in enumerate(cand_indices):
        for j in _idxs[ci, 1:]:
            if tree_ids[j] != middle_tree:
                cross_counts[ci] += 1
    src = cand_indices[cross_counts.argmax()]
    src_tree = tree_ids[src]
    print(f"Source index={src}, position=({pts[src, 0]:.3f}, {pts[src, 1]:.3f}), "
          f"tree={src_tree}, level={levels[src]}")

    # --- Euclidean heat (alpha=0, level-0 only) ---
    print("Running heat method (Euclidean, alpha=0)...")
    bup = BlowUpLevel.from_point_tangents(pts, tans)
    _, u_euc, _, _ = lifted_heat_method(
        bup, source_index=src, k=K, h="local", alpha=0.0, t_scale=T_SCALE,
        return_intermediate=True,
    )

    # --- Lifted heat via product kernel on level-0 blow-up ---
    # The product kernel separates spatial and angular bandwidths:
    #   W_ij = exp(-||x_i - x_j||^2 / sigma_x^2) * exp(-||P_i - P_j||_F^2 / sigma_u^2)
    # sigma_x controls spatial reach, sigma_u controls tangent sensitivity.
    # Tight sigma_u blocks connections at crossings where tangents differ.
    print("Computing level-0 blow-up...")
    level0 = BlowUpLevel.from_point_tangents(pts, tans)

    # Estimate spatial bandwidth from the data
    _dists, _ = KDTree(pts).query(pts, k=K + 1)
    median_spatial = float(np.median(_dists[:, -1]))
    sigma_x = median_spatial * 2.0   # spatial: connect along branches
    sigma_u = SIGMA_U                # angular: tight to block X-crossings

    print(f"Building product-kernel Laplacian (sigma_x={sigma_x:.4f}, sigma_u={sigma_u:.2f})...")
    L1, W1, _ = lifted_laplacian(
        level0, kernel="product", k=K, sigma_x=sigma_x, sigma_u=sigma_u,
    )

    print("Running heat diffusion on product-kernel Laplacian...")
    u_lift = _heat_diffuse(L1, W1, pts, src, T_SCALE)

    # Measure leaking
    other_mask = tree_ids != src_tree
    leak_euc = u_euc[other_mask].sum() / u_euc.sum() if other_mask.any() else 0.0
    leak_lift = u_lift[other_mask].sum() / u_lift.sum() if other_mask.any() else 0.0
    print(f"Heat leaked to other trees: Euclidean={leak_euc:.1%}, Product={leak_lift:.1%}")

    # -- Figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 0: forest colored by tree identity
    ax = axes[0]
    tree_cmap = plt.cm.Set1
    tree_colors = tree_cmap(tree_ids / max(n_trees - 1, 1))
    ax.scatter(pts[:, 0], pts[:, 1], c=tree_colors, s=10, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title("Network (colored by tree)", fontsize=12)
    ax.axis("off")

    # Log normalization spanning ~3 decades
    umax = max(u_euc.max(), u_lift.max())
    norm = mcolors.LogNorm(vmin=umax * 1e-3, vmax=umax, clip=True)

    # Panel 1: Euclidean heat
    _plot_heat(axes[1], pts, u_euc, "Euclidean heat (alpha=0)", norm=norm)
    axes[1].text(0.02, 0.02, f"Cross-leak: {leak_euc:.1%}",
                 transform=axes[1].transAxes, fontsize=10,
                 color="white", bbox=dict(facecolor="black", alpha=0.6))

    # Panel 2: Level-1 blow-up heat
    sc2 = _plot_heat(axes[2], pts, u_lift,
                     "Product kernel heat", norm=norm)
    axes[2].text(0.02, 0.02, f"Cross-leak: {leak_lift:.1%}",
                 transform=axes[2].transAxes, fontsize=10,
                 color="white", bbox=dict(facecolor="black", alpha=0.6))

    fig.colorbar(sc2, ax=axes[2], fraction=0.04, pad=0.02, label="heat u")

    fig.suptitle(
        "Heat diffusion on three overlapping tree networks\n"
        "Product kernel: spreads through Y-junctions, blocks leaking at X-crossings",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {SAVE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
