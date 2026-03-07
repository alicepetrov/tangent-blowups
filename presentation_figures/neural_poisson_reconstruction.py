"""
Neural Poisson Reconstruction — Presentation Figures
=====================================================

Side-by-side comparison of input point clouds and their neural Poisson
reconstructions for:
  (a) Figure-8 curve in 2D (d=1, n=2)
  (b) Whitney umbrella surface in 3D (d=2, n=3)

Each example produces a 1x3 panel:
  [input samples] | [reconstruction overlay] | [training loss]

The reconstructed points are coloured by a Grassmannian coordinate from the
blow-up embedding to produce visually informative plots.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.reconstruction.implicit.neural_poisson import (
    NeuralPoissonReconstructor,
    PoissonConfig,
)
from tangent_blowups.testsupport import (
    figure8, UniformCurve, sample,
    whitney_umbrella, RandomSurface,
)
from tangent_blowups.solvers.linalg import normalize_vectors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Helpers
# ============================================================================

def _clean_ax_2d(ax):
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=7)


def _clean_ax_3d(ax, elev=25, azim=-60):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)


def _equal_lims_3d(ax, pts):
    lims = np.array([[pts[:, i].min(), pts[:, i].max()] for i in range(3)])
    r = 0.5 * (lims[:, 1] - lims[:, 0]).max() * 1.05
    c = lims.mean(axis=1)
    for i, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter([c[i] - r, c[i] + r])


def _style_loss_ax(ax):
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Total loss", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ============================================================================
# Figure-8 (2D curve, d=1, n=2)
# ============================================================================

def make_figure8():
    print("=== Figure-8 (2D) ===")
    n_samples = 1000
    curve = figure8(scale=2.0)
    strategy = UniformCurve(n=n_samples, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    s = sample(curve, strategy, with_tangents=True)

    points = np.asarray(s.points, dtype=np.float64)
    tangents = normalize_vectors(np.asarray(s.tangents, dtype=np.float64))
    t_param = np.asarray(s.params, dtype=np.float64).ravel()

    rng = np.random.default_rng(42)
    points_noisy = points + rng.normal(scale=0.03, size=points.shape)

    # Lift
    level0 = BlowUpLevel.from_point_tangents(points_noisy, tangents)
    level1 = level0.lift(k=16, alpha=1.0)

    # Train
    config = PoissonConfig(
        hidden_dim=128,
        n_layers=3,
        lr=1e-4,
        n_epochs=3000,
        lambda_fit=10.0,
        lambda_align=10.0,
        lambda_screen=10.0,
        lambda_ortho=5.0,
        lambda_smooth=1.0,
        off_manifold_std=0.2,
        off_manifold_refresh=200,
        device=DEVICE,
        print_every=500,
    )
    recon = NeuralPoissonReconstructor(level1, config=config)
    print(f"Training F: R^{recon.D} -> R^{recon.codim} ...")
    losses = recon.fit()

    # Reconstruct
    result = recon.reconstruct(
        n_search=50,
        search_std=0.15,
        n_iters=30,
        lr=0.5,
        residual_threshold=0.005,
    )
    print(f"Recovered {len(result.spatial)} points, mean residual {result.residuals.mean():.6f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("white")

    cmap_input = "twilight"
    cmap_recon = "plasma"

    # (a) Input
    ax = axes[0]
    ax.set_title("Input samples", fontsize=10, fontweight="bold")
    sc = ax.scatter(
        points_noisy[:, 0], points_noisy[:, 1],
        c=t_param, cmap=cmap_input, s=6, alpha=0.8, edgecolors="none",
    )
    ax.quiver(
        points_noisy[::15, 0], points_noisy[::15, 1],
        tangents[::15, 0], tangents[::15, 1],
        scale=30, color="0.5", alpha=0.3, width=0.003,
    )
    _clean_ax_2d(ax)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04, label="t")

    # (b) Reconstruction — colour by pseudo arc-length
    ax = axes[1]
    ax.set_title("Reconstruction", fontsize=10, fontweight="bold")
    # Ground truth in light grey
    ax.scatter(
        points[:, 0], points[:, 1],
        s=1.5, c="0.85", zorder=1,
    )
    # Reconstructed points coloured by first Grassmannian coordinate
    grass_coord = result.reconstructed[:, level1.n_orig]
    sc2 = ax.scatter(
        result.spatial[:, 0], result.spatial[:, 1],
        c=grass_coord, cmap=cmap_recon, s=3, alpha=0.7, edgecolors="none", zorder=2,
    )
    _clean_ax_2d(ax)
    plt.colorbar(sc2, ax=ax, fraction=0.04, pad=0.04, label="Grassmann coord")

    # (c) Loss
    ax = axes[2]
    ax.set_title("Training loss", fontsize=10, fontweight="bold")
    ax.semilogy(losses, color="#2c7fb8", linewidth=1.0)
    _style_loss_ax(ax)

    fig.suptitle(
        f"Neural Poisson Reconstruction  --  figure-8  (D={level1.D}, codim={recon.codim})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("neural_poisson_fig8.png", bbox_inches="tight", dpi=200)
    print("Saved neural_poisson_fig8.png")
    return fig


# ============================================================================
# Whitney Umbrella (3D surface, d=2, n=3)
# ============================================================================

def make_whitney():
    print("\n=== Whitney Umbrella (3D) ===")
    n_samples = 1000
    surface = whitney_umbrella(scale=1.0)
    strategy = RandomSurface(
        n=n_samples,
        u_bounds=(-1.5, 1.5),
        v_bounds=(-1.5, 1.5),
        rng=np.random.default_rng(42),
    )
    s = sample(surface, strategy, with_tangents=True)

    points = np.asarray(s.points, dtype=np.float64)
    tangents = np.asarray(s.tangents, dtype=np.float64)

    rng = np.random.default_rng(123)
    points_noisy = points + rng.normal(scale=0.01, size=points.shape)

    # Lift
    level0 = BlowUpLevel.from_point_tangents(points_noisy, tangents)
    level1 = level0.lift(k=16, alpha=1.0)

    # Train
    config = PoissonConfig(
        hidden_dim=256,
        n_layers=4,
        lr=1e-4,
        n_epochs=5000,
        lambda_fit=10.0,
        lambda_align=10.0,
        lambda_screen=10.0,
        lambda_ortho=5.0,
        lambda_smooth=1.0,
        off_manifold_std=0.15,
        off_manifold_refresh=200,
        batch_size=512,
        device=DEVICE,
        print_every=500,
    )
    recon = NeuralPoissonReconstructor(level1, config=config)
    print(f"Training F: R^{recon.D} -> R^{recon.codim} ...")
    losses = recon.fit()

    # Reconstruct
    result = recon.reconstruct(
        n_search=30,
        search_std=0.12,
        n_iters=30,
        lr=0.5,
        residual_threshold=0.005,
    )
    print(f"Recovered {len(result.spatial)} points, mean residual {result.residuals.mean():.6f}")

    # --- Plot ---
    fig = plt.figure(figsize=(16, 5.5))
    fig.patch.set_facecolor("white")

    cmap_surf = "viridis"

    # (a) Input — colour by z
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.set_title("Input samples", fontsize=10, fontweight="bold")
    sc = ax.scatter(
        points_noisy[:, 0], points_noisy[:, 1], points_noisy[:, 2],
        c=points_noisy[:, 2], cmap=cmap_surf, s=6, alpha=0.8, edgecolors="none",
    )
    _clean_ax_3d(ax)
    _equal_lims_3d(ax, points_noisy)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.08, label="z", shrink=0.7)

    # (b) Reconstruction — colour recovered points by z coordinate
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.set_title("Reconstruction", fontsize=10, fontweight="bold")
    # faint ground truth
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=1, c="0.85", alpha=0.3, zorder=1,
    )
    # reconstructed, coloured by first Grassmannian coordinate
    grass_coord = result.reconstructed[:, level1.n_orig]
    sc2 = ax.scatter(
        result.spatial[:, 0], result.spatial[:, 1], result.spatial[:, 2],
        c=grass_coord, cmap="magma", s=3, alpha=0.6, edgecolors="none",
        zorder=2,
    )
    _clean_ax_3d(ax)
    _equal_lims_3d(ax, np.vstack([points, result.spatial]))
    fig.colorbar(sc2, ax=ax, fraction=0.04, pad=0.08, label="Grassmann coord", shrink=0.7)

    # (c) Loss
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Training loss", fontsize=10, fontweight="bold")
    ax.semilogy(losses, color="#d95f02", linewidth=1.0)
    _style_loss_ax(ax)

    fig.suptitle(
        f"Neural Poisson Reconstruction  --  Whitney umbrella  (D={level1.D}, d={level1.d}, codim={recon.codim})",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("neural_poisson_whitney.png", bbox_inches="tight", dpi=200)
    print("Saved neural_poisson_whitney.png")
    return fig


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    make_figure8()
    make_whitney()
    plt.show()
