"""
Lifted Laplacian Eigenvectors on Point Clouds
----------------------------------------------
For each example (toy 3D surfaces + thingi10k), save two figures:

  eigenspectrum_{name}_level1.png
  eigenspectrum_{name}_level2.png

Figure layout:
  rows    : eigenvector index 1..EIG_K  (drop_first=True skips the trivial one)
  col 0   : Euclidean Laplacian (reference, same in both figures)
  col 1.. : lifted self-tuning Laplacian at each alpha in ALPHAS

Points are coloured by eigenvector value (RdBu_r, max-abs normalised).
Vertical dotted lines in the title show the detected eigengap for each column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.testsupport import (
    RandomSurface, sample,
    plane_cross, cylinders_tangent, plane_paraboloid_tangent,
)
from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.pointcloud.laplacian import (
    pointcloud_laplacian, laplacian_spectrum,
)
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.solvers.linalg import normalize_vectors

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ALPHAS  = [1.0, 5.0, 10.0, 20.0, 30.0]
K       = 20
LAM     = 1e-3
EIG_K   = 5       # eigenvectors to show per Laplacian (after drop_first)
N_TOY   = 6000
N_REAL  = 12000
ELEV, AZIM = 25, 45
CMAP    = "RdBu_r"
PT_SIZE = 3.0

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "thingi10k_pointcloud"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _level0_from_surface(surface_fn):
    surf = surface_fn()
    rng  = np.random.default_rng(42)
    s = sample(surf,
               RandomSurface(n=N_TOY, u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0),
                             rng=rng),
               with_tangents=True, with_normals=False)
    pts    = np.asarray(s.points,   dtype=float)
    frames = np.asarray(s.tangents, dtype=float)
    return pts, BlowUpLevel.from_point_tangents(pts, frames)


def _level0_from_thingi(name: str):
    path = DATA_DIR / f"{name}.npz"
    pc   = load_pointcloud(path)
    pts  = np.asarray(pc.points,  dtype=float).reshape(-1, 3)
    nrm  = np.asarray(pc.normals, dtype=float).reshape(-1, 3)

    valid = np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
    valid &= np.linalg.norm(nrm, axis=1) > 1e-8
    pts, nrm = pts[valid], normalize_vectors(nrm[valid])

    if len(pts) > N_REAL:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N_REAL, replace=False)
        pts, nrm = pts[idx], nrm[idx]

    frames = BlownUpSample.from_normals(pts, nrm).dualize().basis
    return pts, BlowUpLevel.from_point_tangents(pts, frames)


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------

def _eigenvecs(L, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (evals, evecs) for the bottom k non-trivial eigenpairs."""
    evals, evecs = laplacian_spectrum(
        L, k=k, drop_first=True, return_eigenvalues=True,
    )
    return evals, evecs   # evecs: (N, k)


def _eigengap(evals: np.ndarray) -> int:
    """0-based index of the largest gap between consecutive eigenvalues."""
    if evals.size < 2:
        return 0
    return int(np.argmax(np.diff(evals)))


# ---------------------------------------------------------------------------
# Figure builder
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


def _make_figure(
    pts: np.ndarray,
    evals_list: list[np.ndarray],   # len = 1 + len(ALPHAS)
    evecs_list: list[np.ndarray],   # same
    col_labels: list[str],          # same length
    title: str,
    fname: str,
) -> None:
    """
    Grid of 3D scatter plots coloured by eigenvector values.

    rows = EIG_K eigenvectors
    cols = Euclidean reference + one per alpha
    """
    n_rows = EIG_K
    n_cols = len(evecs_list)

    cell_w, cell_h = 3.8, 3.5
    fig = plt.figure(figsize=(cell_w * n_cols, cell_h * n_rows))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, y=1.01)

    for col, (evals, evecs, col_lbl) in enumerate(
        zip(evals_list, evecs_list, col_labels)
    ):
        # per-column shared colour scale (max-abs of all eigenvectors in column)
        vmax = float(np.max(np.abs(evecs))) if evecs.size else 1.0
        vmin = -vmax

        gap = _eigengap(evals)   # 0-based; gap after eigenvector gap

        for row in range(n_rows):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1,
                                 projection="3d")
            vals = evecs[:, row]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                            c=vals, s=PT_SIZE, cmap=CMAP,
                            vmin=vmin, vmax=vmax,
                            linewidths=0, rasterized=True)
            _clean_ax3d(ax)
            _equal_aspect(ax, pts)

            lam = evals[row] if row < len(evals) else float("nan")
            row_title = f"eig {row+1}  (lambda={lam:.3g})"
            # top row: also show column label
            if row == 0:
                row_title = f"{col_lbl}\n{row_title}"
            # mark eigengap boundary row
            if row == gap:
                row_title += "  <-- gap"
            ax.set_title(row_title, fontsize=7.5, pad=3)

    plt.tight_layout(pad=0.8)
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    print(f"  Saved {fname}")
    plt.close(fig)


def _run_example(pts: np.ndarray, level0: BlowUpLevel,
                 name: str, title: str) -> None:
    """Compute Laplacians and save level-1 and level-2 figures."""
    print("  Euclidean...", end="", flush=True)
    L_euc = pointcloud_laplacian(pts, k=K, h="local", normalized=True)
    evals_euc, evecs_euc = _eigenvecs(L_euc, EIG_K)
    print(" done")

    evals_l1, evecs_l1 = [], []
    evals_l2, evecs_l2 = [], []

    for alpha in ALPHAS:
        print(f"  alpha={alpha}: level 1...", end="", flush=True)
        l1 = level0.lift(k=K, alpha=alpha, lam=LAM)
        L1, _, _ = lifted_laplacian(l1, kernel="self_tuning", k=K,
                                    h="local", normalized=True)
        ev1, ew1 = _eigenvecs(L1, EIG_K)
        evals_l1.append(ev1); evecs_l1.append(ew1)

        print(" level 2...", end="", flush=True)
        l2 = l1.lift(k=K, alpha=alpha, lam=LAM)
        L2, _, _ = lifted_laplacian(l2, kernel="self_tuning", k=K,
                                    h="local", normalized=True)
        ev2, ew2 = _eigenvecs(L2, EIG_K)
        evals_l2.append(ev2); evecs_l2.append(ew2)
        print(" done")

    col_labels = ["Euclidean"] + [f"alpha={a}" for a in ALPHAS]

    _make_figure(
        pts,
        [evals_euc] + evals_l1,
        [evecs_euc] + evecs_l1,
        col_labels,
        f"{title} — Level 1",
        f"eigenspectrum_{name}_level1.png",
    )
    _make_figure(
        pts,
        [evals_euc] + evals_l2,
        [evecs_euc] + evecs_l2,
        col_labels,
        f"{title} — Level 2",
        f"eigenspectrum_{name}_level2.png",
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

TOYS = [
    (plane_cross,              "plane_cross",    "Transverse (plane cross)"),
    (cylinders_tangent,        "cylinders",      "Tangential (cylinders)"),
    (plane_paraboloid_tangent, "paraboloid",     "Tangential (plane + paraboloid)"),
]

REAL = [
    ("snowflake", "Snowflake"),
    ("propeller", "Propeller"),
    ("icicles",   "Icicles"),
]

for surface_fn, name, title in TOYS:
    print(f"\n=== {title} ===")
    pts, level0 = _level0_from_surface(surface_fn)
    _run_example(pts, level0, name, title)

for name, title in REAL:
    print(f"\n=== {title} ===")
    pts, level0 = _level0_from_thingi(name)
    _run_example(pts, level0, name, title)

print("\nAll done.")
