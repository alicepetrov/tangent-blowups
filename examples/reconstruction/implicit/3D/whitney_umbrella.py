"""
Neural Poisson reconstruction of a Whitney umbrella in 3D.

The Whitney umbrella is a surface (d=2) in R^3, so it has co-dimension 1 in
the ambient space.  However, the pinch-point singularity makes classical
surface reconstruction difficult.

In the blow-up framework, the surface lifts to a 2-dimensional manifold in
R^12 with co-dimension 10:

Pipeline:
  1. Sample the Whitney umbrella with orthonormal 2-frames.
  2. Lift to the level-1 blow-up space R^12 via Chordal-Sasaki embedding.
     (A 2D surface in R^3 becomes a 2D manifold in R^12, codim = 10.)
  3. Train a neural implicit F: R^12 -> R^10 whose zero-set is the lifted
     surface.
  4. Reconstruct by projecting a search cloud onto F^{-1}(0) and extracting
     the first 3 spatial coordinates.

Note: d=2 means the fit loss involves C(12,2) = 66 Pluecker coordinates,
each requiring a 10x10 determinant.  Training is slower than for curves.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.reconstruction.implicit.neural_poisson import (
    NeuralPoissonReconstructor,
    PoissonConfig,
)
from tangent_blowups.testsupport import whitney_umbrella, RandomSurface, sample

# -- 1.  Generate data --------------------------------------------------------
n_samples = 1000
surface = whitney_umbrella(scale=1.0)

# Avoid the pinch-point singularity at (u=0, v=0) by keeping |v| >= 0.2
strategy = RandomSurface(
    n=n_samples,
    u_bounds=(-1.5, 1.5),
    v_bounds=(0.2, 1.5),
    rng=np.random.default_rng(42),
)
s = sample(surface, strategy, with_tangents=True)

points = np.asarray(s.points, dtype=np.float64)       # (N, 3)
tangents = np.asarray(s.tangents, dtype=np.float64)    # (N, 3, 2)

# Add noise
rng = np.random.default_rng(123)
points_noisy = points + rng.normal(scale=0.02, size=points.shape)

# Perturb each frame column and re-orthonormalize via QR
tangents_noisy = tangents + rng.normal(scale=0.05, size=tangents.shape)
Q, _ = np.linalg.qr(tangents_noisy)
tangents_noisy = Q[:, :, :2]

print(f"Sampled {n_samples} points on a Whitney umbrella in R^3")
print(f"  points  : {points_noisy.shape}")
print(f"  tangents: {tangents_noisy.shape}")

# -- 2.  Lift to blow-up space ------------------------------------------------
level0 = BlowUpLevel.from_point_tangents(points_noisy, tangents_noisy)
level1 = level0.lift(k=16, alpha=1.0)

print(f"\nBlow-up level 1:")
print(f"  Ambient dim D = {level1.D}  (was {level0.D})")
print(f"  Intrinsic d   = {level1.d}")
print(f"  Codim         = {level1.D - level1.d}  (number of implicit functions)")

# -- 3.  Train implicit network -----------------------------------------------
config = PoissonConfig(
    hidden_dim=256,
    n_layers=4,
    lr=1e-3,
    n_epochs=3000,
    lambda_fit=10.0,
    lambda_align=10.0,
    lambda_screen=10.0,
    lambda_ortho=5.0,
    off_manifold_std=0.15,
    off_manifold_refresh=200,
    batch_size=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
    print_every=500,
)

recon = NeuralPoissonReconstructor(level1, config=config)
print(f"\nTraining F: R^{recon.D} -> R^{recon.codim} ...")
losses = recon.fit()

# -- 4.  Reconstruct ----------------------------------------------------------
print("\nReconstructing (projecting search cloud to zero-set)...")
result = recon.reconstruct(
    n_search=30,
    search_std=0.12,
    n_iters=30,
    lr=0.5,
    residual_threshold=0.005,
)

print(f"  Recovered {len(result.spatial)} points")
print(f"  Mean residual ||F||: {result.residuals.mean():.6f}")

# -- 5.  Visualize ------------------------------------------------------------
fig = plt.figure(figsize=(16, 5))

# (a) Input data
ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.set_title("Input (noisy samples)")
sc = ax.scatter(
    points_noisy[:, 0], points_noisy[:, 1], points_noisy[:, 2],
    s=4, c=points_noisy[:, 2], cmap="viridis", alpha=0.7,
)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

# (b) Reconstruction
ax = fig.add_subplot(1, 3, 2, projection="3d")
ax.set_title("Reconstructed surface")
ax.scatter(
    points[:, 0], points[:, 1], points[:, 2],
    s=1, c="black", alpha=0.15, label="ground truth",
)
ax.scatter(
    result.spatial[:, 0], result.spatial[:, 1], result.spatial[:, 2],
    s=2, c="red", alpha=0.4, label="reconstructed",
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
    f"Neural Poisson Reconstruction  (Whitney umbrella, D={level1.D}, d={level1.d}, codim={recon.codim})",
    fontsize=13,
)
plt.tight_layout()
plt.show()
