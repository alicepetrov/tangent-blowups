"""
Lifted Kernels on 3D Surface Point Clouds (Thingi10k)
------------------------------------------------------
Demonstrates the two lifted kernel families on real 3D surface point clouds.

For 3D surfaces (d=2 tangent plane, n_comp=1 normal) the kernel hierarchy is:

  Level 0, self-tuning  : k-NN purely in (x, y, z) position space
  Level 0, product      : position k-NN, edge weight multiplied by
                            exp(-||P_i - P_j||_F^2 / sigma_u^2)
                          where P_i is the 3x3 tangent-plane projector.
                          Points with different normal directions are
                          downweighted even when spatially close.
  Level 1, self-tuning  : k-NN in Chordal-Sasaki metric
                            d^2 = ||x_i - x_j||^2 + (alpha/2)||P_i - P_j||_F^2
                          Normal information baked into the neighbourhood graph.
  Level 1, product      : Chordal-Sasaki k-NN, separate sigma_x / sigma_u
                          for the current level-1 tangent projectors (encodes
                          curvature directions).

Effects on real geometry
------------------------
Near a sharp crease two surface patches are spatially close but have very
different normal directions.  The positional kernel couples them strongly;
the normal-aware kernels (level-0 product, level-1) suppress the connection.
This reshapes the low-frequency spectral modes: eigenvectors align with
geometric patches rather than spatial proximity.

Figures
-------
Fig 1 — Fiedler vector (1st non-trivial Laplacian eigenvector) colored on the
        3D surface for three kernels: level-0 flat, level-0 product, level-1.
        Bottom row: eigenvalue spectrum for each.

Fig 2 — First 6 non-trivial eigenvectors for the level-1 self-tuning kernel,
        showing the spectral modes of the lifted Laplacian.

Fig 3 — Eigenvalue spectrum comparison across all five kernel / level
        combinations side by side.

Usage
-----
    python thingi10k.py
    python thingi10k.py --pointcloud icicles
    python thingi10k.py --pointcloud propeller --max-points 5000
    python thingi10k.py --pointcloud block_E   --elev 20 --azim 45

Available point clouds (data/thingi10k_pointcloud/):
    block_E, icicles, klein_bottle_one, klein_bottle_two,
    propeller, rock_sculpture, snowflake  (default)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, iterated_blowup
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.pointcloud.laplacian import laplacian_spectrum


# ---------------------------------------------------------------------------
# Parameters (overridable via CLI)
# ---------------------------------------------------------------------------

K_BLOWUP  = 15     # k-NN for curvature regression inside iterated_blowup
ALPHA  = 1.0  # Chordal-Sasaki weighting factor
LAM       = 1e-3   # ridge regularisation

K_KERNEL  = 16     # k-NN for affinity construction
N_EIG     = 8      # eigenvalues shown in spectrum plots
N_MODES   = 6      # non-trivial eigenvectors shown in Fig 2


# ---------------------------------------------------------------------------
# Data I/O helpers  (shared with other thingi10k examples)
# ---------------------------------------------------------------------------

def _pc_root() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "thingi10k_pointcloud"


def _resolve_path(name: str | Path | None) -> Path:
    if name is None:
        return _pc_root() / "snowflake.npz"
    p = Path(name)
    if p.exists():
        return p
    if p.suffix == "":
        p = p.with_suffix(".npz")
    return _pc_root() / p


def load_cloud(
    path: str | Path | None,
    *,
    max_points: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a thingi10k point cloud and return (points, normals).

    Args:
        path:       file name (with or without .npz) or full path.
                    Bare names are resolved under data/thingi10k_pointcloud/.
        max_points: if given, randomly subsample to this many points.
        seed:       random seed for subsampling.

    Returns:
        points  (N, 3) float64
        normals (N, 3) float64, unit-length
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Point cloud not found: {resolved}")

    s = load_pointcloud(resolved)
    if s.normals is None:
        raise ValueError(f"{resolved.stem}: point cloud has no normals.")

    pts = np.asarray(s.points,  dtype=float)
    nrm = np.asarray(s.normals, dtype=float)

    # Flatten in case of (N, 1, 3) shape
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if nrm.ndim > 2:
        nrm = nrm.reshape(-1, nrm.shape[-1])

    # Drop degenerate points
    valid = (np.isfinite(pts).all(axis=1) & np.isfinite(nrm).all(axis=1)
             & (np.linalg.norm(nrm, axis=1) > 1e-8))
    pts = pts[valid]
    nrm = normalize_vectors(nrm[valid])

    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        nrm = nrm[idx]

    dropped = int(np.sum(~valid))
    if dropped:
        print(f"   Dropped {dropped} degenerate points.")
    return pts, nrm


# ---------------------------------------------------------------------------
# Geometry: tangent frame from normals
# ---------------------------------------------------------------------------

def tangent_frame_from_normals(normals: np.ndarray) -> np.ndarray:
    """
    Build orthonormal 2D tangent frames from surface unit normals.

    For each normal n_i, finds two orthonormal vectors u_i, v_i such that
    {u_i, v_i, n_i} is a right-handed ONB.  The tangent frame is constructed
    via cross products, vectorised over all N points.

    Args:
        normals: (N, 3) unit-length normal vectors.

    Returns:
        frames: (N, 3, 2) orthonormal tangent frames.
    """
    N = len(normals)
    # Reference vector for Gram-Schmidt; swap to (0,1,0) where n ~ (±1,0,0)
    ref = np.broadcast_to(np.array([1.0, 0.0, 0.0]), (N, 3)).copy()
    parallel = np.abs(normals[:, 0]) > 0.9
    ref[parallel] = [0.0, 1.0, 0.0]

    u = np.cross(normals, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True).clip(min=1e-10)

    v = np.cross(normals, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-10)

    return np.stack([u, v], axis=2)   # (N, 3, 2)


# ---------------------------------------------------------------------------
# Auto-bandwidth helpers  (same logic as 2D example)
# ---------------------------------------------------------------------------

def _median_embed_dist(level: BlowUpLevel) -> float:
    tree = cKDTree(level.embedded)
    d, _ = tree.query(level.embedded, k=K_KERNEL + 1)
    return float(np.median(d[:, 1:]))


def _median_spatial_dist(level: BlowUpLevel) -> float:
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
    """Build (L, W, D) with auto bandwidths."""
    if kernel == "self_tuning":
        return lifted_laplacian(level, kernel="self_tuning", k=K_KERNEL, h="local")
    if kernel == "product":
        sigma_x = _median_spatial_dist(level)
        sigma_u = _median_proj_dist(level)
        return lifted_laplacian(
            level, kernel="product", k=K_KERNEL,
            sigma_x=sigma_x, sigma_u=sigma_u,
        )
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _spectrum(L, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (evals, evecs):
      evals  first N_EIG eigenvalues (incl. trivial zero)
      evecs  (N, n_modes) first n_modes non-trivial eigenvectors
    """
    k_total = max(N_EIG, n_modes + 1)  # +1 to include trivial mode
    evals_all, evecs_all = laplacian_spectrum(
        L, k=k_total, drop_first=False, return_eigenvalues=True,
    )
    # Fiedler vector = index 1, next modes = 2, 3, ...
    evecs = evecs_all[:, 1:n_modes + 1]
    return evals_all[:N_EIG], evecs


# ---------------------------------------------------------------------------
# 3D plotting helpers
# ---------------------------------------------------------------------------

def _set_equal_aspect_3d(ax):
    """Force equal axis scales for a 3D axes."""
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = lims.mean(axis=1)
    r = 0.5 * np.max(np.abs(lims[:, 1] - lims[:, 0]))
    ax.set_xlim3d(center[0] - r, center[0] + r)
    ax.set_ylim3d(center[1] - r, center[1] + r)
    ax.set_zlim3d(center[2] - r, center[2] + r)


def _scatter3d(ax, pts, vals, title, *, elev=30, azim=45, s=4, cmap="RdBu_r"):
    vabs = float(np.percentile(np.abs(vals), 99)) + 1e-9
    sc = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=vals, cmap=cmap, vmin=-vabs, vmax=vabs,
        s=s, rasterized=True, depthshade=False, linewidths=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    _set_equal_aspect_3d(ax)
    return sc


def _spectrum_bar(ax, evals, title):
    xs = np.arange(len(evals))
    ax.bar(xs, evals, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=6)
    ax.set_ylabel("lambda", fontsize=7)
    if len(evals) >= 2:
        gap = float(evals[1] - evals[0])
        ax.text(0.97, 0.95, f"gap={gap:.4f}",
                ha="right", va="top", transform=ax.transAxes, fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lifted kernel Laplacians on a Thingi10k surface point cloud."
    )
    parser.add_argument(
        "--pointcloud", type=str, default=None,
        help=(
            "Point cloud name (e.g. 'icicles') or full path to .npz file. "
            "Resolved under data/thingi10k_pointcloud/ if not an absolute path. "
            "Default: snowflake."
        ),
    )
    parser.add_argument(
        "--max-points", type=int, default=None,
        help="Randomly subsample to at most this many points (useful for large meshes).",
    )
    parser.add_argument(
        "--elev", type=float, default=25.0,
        help="3D view elevation angle in degrees (default: 25).",
    )
    parser.add_argument(
        "--azim", type=float, default=45.0,
        help="3D view azimuth angle in degrees (default: 45).",
    )
    args = parser.parse_args()

    elev = args.elev
    azim = args.azim

    # ---------------------------------------------------------------
    # Load point cloud
    # ---------------------------------------------------------------
    resolved = _resolve_path(args.pointcloud)
    label = resolved.stem
    print(f"Loading '{label}' from {resolved} ...")
    pts, nrm = load_cloud(resolved, max_points=args.max_points)
    N = len(pts)
    print(f"   N = {N} points")

    # ---------------------------------------------------------------
    # Build tangent frame and iterated blow-up (levels 0 and 1)
    # ---------------------------------------------------------------
    print("Building tangent frames from normals ...")
    frames = tangent_frame_from_normals(nrm)   # (N, 3, 2)

    print(f"Running iterated blow-up (k={K_BLOWUP}, alpha={ALPHA}) ...")
    levels = iterated_blowup(pts, frames, num_levels=1,
                             k=K_BLOWUP, alpha=ALPHA, lam=LAM)
    l0, l1 = levels
    print(f"   Embedding dims: {l0.D} -> {l1.D}")
    print(f"   n_orig = {l0.n_orig}  (first {l0.n_orig} coords are spatial)")

    # ---------------------------------------------------------------
    # Build five kernels
    # ---------------------------------------------------------------
    print("Building kernels ...")

    # (A) Level 0, self-tuning: purely positional
    print("   (A) Level 0, self-tuning ...")
    L_A, _, _ = _build_L(l0, "self_tuning")
    ev_A, modes_A = _spectrum(L_A, N_MODES)
    print(f"       Fiedler gap = {ev_A[1]-ev_A[0]:.5f}")

    # (B) Level 0, product: same k-NN, but angular weight downweights
    #     pairs with different normal directions (different tangent planes)
    print("   (B) Level 0, product ...")
    L_B, _, _ = _build_L(l0, "product")
    ev_B, modes_B = _spectrum(L_B, N_MODES)
    print(f"       Fiedler gap = {ev_B[1]-ev_B[0]:.5f}")

    # (C) Level 1, self-tuning: k-NN in Chordal-Sasaki (pos + tangent plane)
    print("   (C) Level 1, self-tuning ...")
    L_C, _, _ = _build_L(l1, "self_tuning")
    ev_C, modes_C = _spectrum(L_C, N_MODES)
    print(f"       Fiedler gap = {ev_C[1]-ev_C[0]:.5f}")

    # (D) Level 1, product: separate spatial + angular bandwidths using
    #     level-1 tangent projectors (which encode curvature direction)
    print("   (D) Level 1, product ...")
    L_D, _, _ = _build_L(l1, "product")
    ev_D, _ = _spectrum(L_D, 1)   # only need eigenvalues for spectrum comparison
    print(f"       Fiedler gap = {ev_D[1]-ev_D[0]:.5f}")

    # ---------------------------------------------------------------
    # Figure 1: Fiedler vector + eigenvalue spectrum
    #   3 cols: (A) Level 0 flat,  (B) Level 0 product,  (C) Level 1 self-tuning
    # ---------------------------------------------------------------
    fig1 = plt.figure(figsize=(13, 8))
    gs1  = fig1.add_gridspec(2, 3, height_ratios=[3, 1.4], hspace=0.35, wspace=0.1)

    panels1 = [
        (modes_A[:, 0], ev_A,
         "Level 0 — self-tuning\n(position only)"),
        (modes_B[:, 0], ev_B,
         "Level 0 — product\n(pos. x tangent-plane weight)"),
        (modes_C[:, 0], ev_C,
         "Level 1 — self-tuning\n(Chordal-Sasaki: pos + tangent)"),
    ]

    for j, (fvec, evals, title) in enumerate(panels1):
        ax3d = fig1.add_subplot(gs1[0, j], projection="3d")
        ax_s = fig1.add_subplot(gs1[1, j])
        _scatter3d(ax3d, pts, fvec, title, elev=elev, azim=azim)
        _spectrum_bar(ax_s, evals, f"Spectrum — {title.split(chr(10))[0]}")

    fig1.suptitle(
        f"Fig 1 — {label}: Fiedler Vector (1st non-trivial Laplacian eigenvector)\n"
        "Normal-aware kernels reshape spectral modes along geometric boundaries "
        "rather than spatial proximity alone.",
        fontsize=9, y=1.01,
    )

    # ---------------------------------------------------------------
    # Figure 2: First N_MODES eigenvectors for the level-1 self-tuning kernel
    # ---------------------------------------------------------------
    ncols = 3
    nrows = (N_MODES + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(
        nrows, ncols, figsize=(13, 4.5 * nrows),
        subplot_kw={"projection": "3d"},
    )
    axes2_flat = np.asarray(axes2).ravel()

    for m in range(N_MODES):
        ax = axes2_flat[m]
        _scatter3d(ax, pts, modes_C[:, m],
                   f"Mode {m+1}  (lambda_{m+1} = {ev_C[m+1]:.4f})",
                   elev=elev, azim=azim)

    # Hide any extra axes
    for m in range(N_MODES, len(axes2_flat)):
        axes2_flat[m].set_visible(False)

    fig2.suptitle(
        f"Fig 2 — {label}: First {N_MODES} Non-Trivial Eigenvectors\n"
        "Level-1 self-tuning kernel (Chordal-Sasaki metric).  "
        "Low modes capture large-scale geometric variation.",
        fontsize=9,
    )
    fig2.tight_layout()

    # ---------------------------------------------------------------
    # Figure 3: Eigenvalue spectrum comparison — all five kernels
    # ---------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 4, figsize=(12, 3.5), sharey=False)

    spec_cases = [
        (ev_A, "L0 self-tuning\n(position)"),
        (ev_B, "L0 product\n(pos x angular)"),
        (ev_C, "L1 self-tuning\n(Chordal-Sasaki)"),
        (ev_D, "L1 product\n(L1 pos x angular)"),
    ]

    for ax, (evals, title) in zip(axes3, spec_cases):
        _spectrum_bar(ax, evals, title)

    fig3.suptitle(
        f"Fig 3 — {label}: Eigenvalue Spectrum Comparison\n"
        "Incorporating normal/tangent information into the kernel shifts "
        "the spectral gap and changes the number of near-zero modes.",
        fontsize=9,
    )
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
