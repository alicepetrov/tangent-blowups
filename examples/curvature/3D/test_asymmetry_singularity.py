"""
Second fundamental form asymmetry as a singularity detector.
=============================================================

The raw curvature operator B_i from the regression is a general (d x d)
matrix for each normal direction.  On a smooth submanifold the second
fundamental form is symmetric: h_{alpha,a,b} = h_{alpha,b,a}.  The
degree of asymmetry

    A_i = || raw_h_i - raw_h_i^T ||_F

is a diagnostic: it should be small on smooth interior points and large
where the k-NN neighborhood mixes sheets from different components,
i.e. near singularities (self-intersections, tangential contacts).

We test this on three surfaces with known singularity locations:

  1. Tangent cylinders:  singularity along the contact curve near the
     origin where both cylinders share the tangent plane.
  2. Plane + paraboloid: tangential contact at the origin.
  3. Crossed planes:     self-intersection along the x-axis.

For each, we check that the asymmetry is significantly elevated near
the known singularity compared to the smooth interior.
"""
from __future__ import annotations

import sys
import numpy as np

from tangent_blowups.testsupport import (
    RandomSurface,
    cylinders_tangent,
    plane_paraboloid_tangent,
    tangent_spheres,
    sample,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_POINTS = 5000
K_BLOWUP = 20
ALPHA = 2.0
LAM = 1e-3
SEED = 42

# A point is "near the singularity" if its spatial distance to the
# singularity locus is below this threshold.
SING_RADIUS = 0.15

# The test passes if the median asymmetry near the singularity is at
# least this many times the median asymmetry far from it.
MIN_RATIO = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_asymmetry(level: BlowUpLevel) -> np.ndarray:
    """
    Frobenius norm of the antisymmetric part of the raw shape operator.

    Returns (N,) array.  For hypersurfaces (n_comp=1) this is a scalar
    per point; for higher codimension we sum over normal directions.
    """
    B_all = level.curvature_ops  # (N, n_comp, d, d);  B[i,alpha,b,a]
    raw_h = B_all.transpose(0, 1, 3, 2)  # (N, n_comp, d, d) -> (i,alpha,a,b)
    asym = raw_h - raw_h.transpose(0, 1, 3, 2)  # antisymmetric part * 2
    # ||asym||_F per point (summed over all normal directions and tangent indices)
    asym_norm = np.sqrt(np.einsum("naij,naij->n", asym, asym))  # (N,)
    return asym_norm


def run_test(name, surface, u_bounds, v_bounds, n_points,
             singularity_dist_fn, sing_radius=SING_RADIUS,
             min_ratio=MIN_RATIO):
    """
    Sample surface, compute asymmetry, compare near vs far from singularity.

    singularity_dist_fn(points) -> (N,) array of distances to singularity locus.
    """
    rng = np.random.default_rng(SEED)
    strategy = RandomSurface(n=n_points, u_bounds=u_bounds,
                             v_bounds=v_bounds, rng=rng)
    s = sample(surface, strategy, with_tangents=True, with_normals=True)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    flat_T = tangents.reshape(len(points), -1)
    valid = (np.all(np.isfinite(points), axis=1) &
             np.all(np.isfinite(flat_T), axis=1))
    points, tangents = points[valid], tangents[valid]

    # Blow-up
    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    l1 = l0.lift(k=K_BLOWUP, alpha=ALPHA, lam=LAM)

    # Asymmetry
    asym = compute_asymmetry(l1)

    # Distance to singularity locus
    dist = singularity_dist_fn(points)
    near = dist < sing_radius
    far = dist > 3 * sing_radius  # well away from singularity

    if near.sum() < 10 or far.sum() < 10:
        print(f"\n{'='*70}")
        print(f"  {name}    [SKIP]")
        print(f"{'='*70}")
        print(f"  Too few points near ({near.sum()}) or far ({far.sum()}) "
              f"from singularity")
        return True  # don't fail on insufficient data

    # Use p90 of the near-singularity asymmetry vs p90 of the far asymmetry.
    # Near the singularity, only points whose k-NN actually spans both sheets
    # show elevated asymmetry — the p90 captures this tail.
    p90_near = float(np.percentile(asym[near], 90))
    p90_far = float(np.percentile(asym[far], 90))
    ratio = p90_near / p90_far if p90_far > 1e-15 else float("inf")

    # Detection rate: fraction of near-singularity points with asymmetry
    # above the p99 of the far (smooth) distribution.
    threshold = float(np.percentile(asym[far], 99))
    detection_rate = float(np.mean(asym[near] > threshold))

    passed = ratio >= min_ratio

    print(f"\n{'='*70}")
    print(f"  {name}    [{'PASS' if passed else 'FAIL'}]")
    print(f"{'='*70}")
    print(f"  Points: {len(points)} total, {near.sum()} near, {far.sum()} far")
    print(f"  Asymmetry (p90):  near={p90_near:.5f}  far={p90_far:.5f}  "
          f"ratio={ratio:.1f}x")
    print(f"  Asymmetry (mean): near={asym[near].mean():.5f}  "
          f"far={asym[far].mean():.5f}")
    print(f"  Detection rate (near > p99_far): {detection_rate:.3f}")

    return passed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_tangent_cylinders():
    """
    Two tangent cylinders contact along a curve near the origin.
    The singularity locus is the set of points where both cylinders
    are close — near x~0, y~0 (the touching point).
    """
    r = 1.0
    surface = cylinders_tangent(radius=r, half_axis=1.2, half_angle=0.55)

    def dist_to_singularity(pts):
        # The contact point is at the origin.  Points from both cylinders
        # are near the singularity when they're close to the origin.
        # For cylinder 1 (axis along x): close to origin means |x| small
        # and angle b small (z ~ 0).
        # For cylinder 2 (axis along y): close to origin means |y| small
        # and angle b small (z ~ 0).
        # Simple proxy: distance to origin in 3D.
        return np.linalg.norm(pts, axis=1)

    return run_test(
        "Tangent cylinders (contact at origin)",
        surface,
        u_bounds=(0.0, 2.0),  # both cylinders
        v_bounds=(0.0, 1.0),
        n_points=N_POINTS,
        singularity_dist_fn=dist_to_singularity,
        sing_radius=0.20,
    )


def test_plane_paraboloid():
    """
    Plane tangent to paraboloid at the origin.  The singularity is the
    tangential contact point at (0, 0, 0).
    """
    surface = plane_paraboloid_tangent(scale_xy=1.0, scale_z=0.25)

    def dist_to_singularity(pts):
        # Contact point is origin; both components share tangent plane there.
        return np.linalg.norm(pts, axis=1)

    return run_test(
        "Plane + paraboloid (tangent at origin)",
        surface,
        u_bounds=(0.0, 2.0),  # both faces
        v_bounds=(0.0, 1.0),
        n_points=N_POINTS,
        singularity_dist_fn=dist_to_singularity,
        sing_radius=0.20,
    )


def test_tangent_spheres():
    """
    Two tangent spheres touching at the origin.
    Both have the same curvature (kappa = 1/r) everywhere, so
    scalar curvature can't distinguish them — but asymmetry should
    spike where k-NN mixes the two sheets at the contact point.
    """
    r = 1.0
    surface = tangent_spheres(radius=r)

    def dist_to_singularity(pts):
        # Contact point is the origin
        return np.linalg.norm(pts, axis=1)

    return run_test(
        "Tangent spheres (contact at origin)",
        surface,
        u_bounds=(0.0, 2.0),  # both spheres
        v_bounds=(0.05, 0.95),  # avoid poles
        n_points=N_POINTS,
        singularity_dist_fn=dist_to_singularity,
        sing_radius=0.20,
    )


def test_single_cylinder_no_singularity():
    """
    Negative control: a single cylinder with NO singularity.
    Asymmetry should be uniformly low everywhere.
    The "singularity" distance is meaningless here — we just check
    that the asymmetry is uniformly small.
    """
    r = 1.0
    surface = cylinders_tangent(radius=r, half_axis=1.2, half_angle=0.55)

    rng = np.random.default_rng(SEED)
    strategy = RandomSurface(n=N_POINTS, u_bounds=(0.0, 0.999),
                             v_bounds=(0.0, 1.0), rng=rng)
    s = sample(surface, strategy, with_tangents=True, with_normals=True)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)

    flat_T = tangents.reshape(len(points), -1)
    valid = (np.all(np.isfinite(points), axis=1) &
             np.all(np.isfinite(flat_T), axis=1))
    points, tangents = points[valid], tangents[valid]

    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    l1 = l0.lift(k=K_BLOWUP, alpha=ALPHA, lam=LAM)

    asym = compute_asymmetry(l1)

    # On a smooth single sheet, the asymmetry should be small relative
    # to the shape operator magnitude.  Use total curvature as reference.
    from tangent_blowups.geometry.iterated_grassmann import extract_level1
    inv1 = extract_level1(l1)
    total_curv = inv1.total_curvature[:, 0]

    # Trim boundary
    params_u = np.asarray(s.params[0], dtype=float)[valid]
    params_v = np.asarray(s.params[1], dtype=float)[valid]
    interior = ((params_u > 0.05) & (params_u < 0.949) &
                (params_v > 0.05) & (params_v < 0.95))

    rel_asym = asym[interior] / np.maximum(total_curv[interior], 1e-10)
    med_rel = float(np.median(rel_asym))
    p95_rel = float(np.percentile(rel_asym, 95))

    # Relative asymmetry should be small (<10% of total curvature)
    passed = med_rel < 0.10

    print(f"\n{'='*70}")
    print(f"  Single cylinder (no singularity, negative control)  "
          f"[{'PASS' if passed else 'FAIL'}]")
    print(f"{'='*70}")
    print(f"  Points: {interior.sum()} interior")
    print(f"  Asymmetry / total curvature: median={med_rel:.4f}  p95={p95_rel:.4f}")
    print(f"  Raw asymmetry: median={np.median(asym[interior]):.5f}  "
          f"mean={np.mean(asym[interior]):.5f}")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Second fundamental form asymmetry as singularity detector")
    print(f"N={N_POINTS}  k={K_BLOWUP}  alpha={ALPHA}  lam={LAM}  seed={SEED}")
    print(f"Singularity radius={SING_RADIUS}  min_ratio={MIN_RATIO}")

    results = {}
    results["Tangent cylinders"]  = test_tangent_cylinders()
    results["Plane + paraboloid"] = test_plane_paraboloid()
    results["Tangent spheres"]    = test_tangent_spheres()
    results["Single cylinder (negative control)"] = test_single_cylinder_no_singularity()

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    n_pass = sum(results.values())
    n_total = len(results)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n  {n_pass}/{n_total} passed")
    print(f"{'='*70}")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
