"""
Neural Poisson reconstruction of a trefoil knot in 3D.

This is the natural test case for the paper "Poisson Manifold Reconstruction
-- Beyond Co-dimension One": a curve in R^3 has co-dimension 2, so standard
Poisson surface reconstruction (which handles codim-1) cannot be applied
directly.

Pipeline:
  1. Sample the trefoil knot with unit tangent vectors.
  2. Lift to the level-1 blow-up space R^12 via Chordal-Sasaki embedding.
     (A 1D curve in R^3 becomes a 1D curve in R^12, with codim = 11.)
  3. Train a neural implicit F: R^12 -> R^11 whose zero-set is the lifted
     curve.
  4. Reconstruct by projecting a search cloud onto F^{-1}(0) and extracting
     the first 3 spatial coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.reconstruction.implicit.neural_poisson import (
    NeuralPoissonReconstructor,
    PoissonConfig,
)
from tangent_blowups.testsupport import trefoil_knot, UniformCurve, sample
from tangent_blowups.solvers.linalg import normalize_vectors

# ── 1.  Generate data ────────────────────────────────────────────────
n_samples = 2000
curve = trefoil_knot(scale=1.0, z_scale=1.0)
strategy = UniformCurve(n=n_samples, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
s = sample(curve, strategy, with_tangents=True)

points = np.asarray(s.points, dtype=np.float64)
tangents = normalize_vectors(np.asarray(s.tangents, dtype=np.float64))

# Optional: add noise
rng = np.random.default_rng(42)
points_noisy = points + rng.normal(scale=0.02, size=points.shape)
tangents = normalize_vectors(tangents + rng.normal(scale=0.05, size=tangents.shape))

print(f"Sampled {n_samples} points on a trefoil knot in R^3")
print(f"  points  : {points_noisy.shape}")
print(f"  tangents: {tangents.shape}")

# ── 2.  Lift to blow-up space ────────────────────────────────────────
level0 = BlowUpLevel.from_point_tangents(points_noisy, tangents)
level1 = level0.lift(k=16, alpha=1.0)

print(f"\nBlow-up level 1:")
print(f"  Ambient dim D = {level1.D}  (was {level0.D})")
print(f"  Intrinsic d   = {level1.d}")
print(f"  Codim         = {level1.D - level1.d}  (number of implicit functions)")

# ── 3.  Train implicit network ───────────────────────────────────────
config = PoissonConfig(
    hidden_dim=256,
    n_layers=4,
    lr=1e-3,
    n_epochs=3000,
    lambda_fit=10.0,
    lambda_align=10.0,
    lambda_screen=10.0,
    lambda_ortho=5.0,
    lambda_smooth=1.0,
    off_manifold_std=0.15,
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
    search_std=0.12,
    n_iters=30,
    lr=0.5,
    residual_threshold=0.005,
)

print(f"  Recovered {len(result.spatial)} points")
print(f"  Mean residual ||F||: {result.residuals.mean():.6f}")

# ── 5.  Visualize ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5))

# (a) Input data
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_title("Input (noisy samples)")
ax.scatter(
    points_noisy[:, 0], points_noisy[:, 1], points_noisy[:, 2],
    s=3, c="steelblue", alpha=0.6,
)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

# (b) Reconstruction
ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_title("Reconstructed curve")
ax.scatter(
    points[:, 0], points[:, 1], points[:, 2],
    s=1, c="black", alpha=0.15, label="ground truth",
)
ax.scatter(
    result.spatial[:, 0], result.spatial[:, 1], result.spatial[:, 2],
    s=2, c="red", alpha=0.5, label="reconstructed",
)
ax.legend(fontsize=8)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

# (c) Training loss
ax = fig.add_subplot(1, 3, 3)
ax.set_title("Training loss")
ax.semilogy(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Total loss")
ax.grid(True, alpha=0.3)

plt.suptitle(
    f"Neural Poisson Reconstruction  (trefoil, D={level1.D}, codim={recon.codim})",
    fontsize=13,
)
plt.tight_layout()
plt.show()
