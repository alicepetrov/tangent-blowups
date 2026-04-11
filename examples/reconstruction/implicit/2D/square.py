"""
Neural Poisson reconstruction of a square in 2D.

The square is a piecewise-linear curve with four transverse corners.  At each
corner the tangent direction changes by 90 degrees, so the corners are genuine
singularities: the limiting tangent from each incoming edge is different.

Pipeline:
  1. Sample the square with piecewise-constant tangent vectors.
  2. Lift to the level-1 blow-up space R^6 via Chordal-Sasaki embedding.
     (A 1D curve in R^2 becomes a 1D curve in R^6, codim = 5.)
  3. Train a neural implicit F: R^6 -> R^5 whose zero-set is the lifted curve.
  4. Reconstruct by projecting a search cloud onto F^{-1}(0).

The four edges have axis-aligned tangents (+x, +y, -x, -y), so the level-1
embedding naturally clusters points by edge.  The corners appear as sharp
transitions between clusters; the network must bridge these with a connected
zero-set while keeping the edges straight and orthogonal.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.reconstruction.implicit.neural_poisson import (
    NeuralPoissonReconstructor,
    PoissonConfig,
)
from tangent_blowups.testsupport import square, UniformCurve, sample
from tangent_blowups.solvers.linalg import normalize_vectors

# ── 1.  Generate data ────────────────────────────────────────────────
n_samples = 1000
curve = square(side=2.0)
# t in [0, 4): each unit interval is one edge; corners land at t=0,1,2,3
strategy = UniformCurve(n=n_samples, t_min=0.0, t_max=4.0, endpoint=False)
s = sample(curve, strategy, with_tangents=True)

points = np.asarray(s.points, dtype=np.float64)
tangents = normalize_vectors(np.asarray(s.tangents, dtype=np.float64))

rng = np.random.default_rng(42)
points_noisy = points + rng.normal(scale=0.02, size=points.shape)

print(f"Sampled {n_samples} points on a unit square in R^2")
print(f"  points  : {points_noisy.shape}  (noise std=0.02)")
print(f"  tangents: {tangents.shape}  (piecewise constant, 4 directions)")

# ── 2.  Lift to blow-up space ────────────────────────────────────────
level0 = BlowUpLevel.from_point_tangents(points_noisy, tangents)
level1 = level0.lift(k=20, alpha=1.0)

print(f"\nBlow-up level 1:")
print(f"  Ambient dim D = {level1.D}  (was {level0.D})")
print(f"  Intrinsic d   = {level1.d}")
print(f"  Codim         = {level1.D - level1.d}")

# ── 3.  Train implicit network ───────────────────────────────────────
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
    device="cuda" if torch.cuda.is_available() else "cpu",
    print_every=500,
)

recon = NeuralPoissonReconstructor(level1, config=config)
print(f"\nTraining F: R^{recon.D} -> R^{recon.codim} ...")
losses = recon.fit()

# ── 4.  Reconstruct ──────────────────────────────────────────────────
print("\nReconstructing (projecting search cloud to zero-set)...")
result = recon.reconstruct(
    n_search=50,
    search_std=0.15,
    n_iters=30,
    lr=0.5,
    residual_threshold=0.005,
)

print(f"  Recovered {len(result.spatial)} points")
print(f"  Mean residual ||F||: {result.residuals.mean():.6f}")

# ── 5.  Visualize ────────────────────────────────────────────────────

# Lifted tangent x-component p[2] = u_x^2 distinguishes horizontal (+x, -x)
# from vertical (+y, -y) edges, giving a natural 4-colour scheme.
# We approximate the dominant tangent direction at each reconstructed point
# from the spatial component of the level-1 embedding (first 2 coords).
recon_lifted = result.reconstructed          # (M, 6)
# projector diagonal: P[0,0] = ux^2, P[1,1] = uy^2 (in vec form: indices 0, 3)
ux2 = recon_lifted[:, 2]   # component u_x^2 of projector vec

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# (a) Input data
ax = axes[0]
ax.set_title("Input (noisy samples)")
ax.scatter(points_noisy[:, 0], points_noisy[:, 1], s=4, c="steelblue", alpha=0.6)
ax.quiver(
    points_noisy[::20, 0], points_noisy[::20, 1],
    tangents[::20, 0], tangents[::20, 1],
    scale=30, color="gray", alpha=0.5, width=0.003,
)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# (b) Ground truth vs reconstruction
ax = axes[1]
ax.set_title("Reconstructed curve")
ax.scatter(points[:, 0], points[:, 1], s=2, c="black", alpha=0.15, label="ground truth")
ax.scatter(result.spatial[:, 0], result.spatial[:, 1], s=2, c="red", alpha=0.5, label="reconstructed")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# (c) Colour by u_x^2 — reveals the edge structure in blow-up space:
#     horizontal edges have ux^2 near 1, vertical edges near 0
ax = axes[2]
ax.set_title("Reconstruction coloured by $u_x^2$\n(horizontal=1, vertical=0)")
sc = ax.scatter(
    result.spatial[:, 0], result.spatial[:, 1],
    c=ux2, cmap="coolwarm", s=3, alpha=0.8, vmin=0, vmax=1,
)
plt.colorbar(sc, ax=ax, label="$u_x^2$")
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# (d) Training loss
ax = axes[3]
ax.set_title("Training loss")
ax.semilogy(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Total loss")
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Neural Poisson Reconstruction  (square, D={level1.D}, codim={recon.codim})",
    fontsize=13,
)
plt.tight_layout()
plt.show()
