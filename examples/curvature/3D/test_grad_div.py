"""
Discrete gradient and divergence: accuracy and consistency tests.
=================================================================

Tests the lifted_gradient and lifted_divergence operators against:

  1. Linear function on a cylinder: grad(f) should be exact.
  2. Quadratic function on a sphere: grad(f) compared to analytic.
  3. div(grad f) vs -Lf consistency: the composed operator should
     approximate the graph Laplacian.
  4. Divergence theorem: integral of div(X) over the surface should
     equal zero for a tangent vector field with no boundary flux.

Each test prints error statistics and a PASS/FAIL verdict.
"""
from __future__ import annotations

import sys
import numpy as np
from scipy import sparse

from tangent_blowups.testsupport import (
    RandomSurface,
    cylinders_tangent,
    plane_paraboloid_tangent,
    tangent_spheres,
    sample,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import (
    lifted_affinity,
    lifted_gradient,
    lifted_divergence,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_POINTS = 3000
K_AFFINITY = 20
K_BLOWUP = 20
ALPHA = 2.0
LAM = 1e-3
SEED = 42

GRAD_LINEAR_TOL = 0.05     # 5% median relative error for exact linear grad
GRAD_QUAD_TOL = 0.10        # 10% median relative error for quadratic grad
DIV_GRAD_CORR_TOL = 0.90    # Pearson correlation between div(grad f) and -Lf
DIV_INTEGRAL_TOL = 0.10     # integral of div(X) relative to mean|div|


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _correlation(a, b):
    """Pearson correlation between two vectors."""
    a = a - a.mean()
    b = b - b.mean()
    num = a @ b
    den = np.sqrt((a @ a) * (b @ b))
    return num / den if den > 1e-15 else 0.0


def _trim_boundary(params_u, params_v, u_bounds, v_bounds, margin):
    """Boolean mask for interior points."""
    u_lo, u_hi = u_bounds
    v_lo, v_hi = v_bounds
    return ((params_u > u_lo + margin) &
            (params_u < u_hi - margin) &
            (params_v > v_lo + margin) &
            (params_v < v_hi - margin))


def _sample_surface(surface, u_bounds, v_bounds, n_points, seed):
    """Sample a surface and return (points, tangents, params_u, params_v)."""
    rng = np.random.default_rng(seed)
    strategy = RandomSurface(n=n_points, u_bounds=u_bounds,
                             v_bounds=v_bounds, rng=rng)
    s = sample(surface, strategy, with_tangents=True, with_normals=True)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)
    params_u = np.asarray(s.params[0], dtype=float)
    params_v = np.asarray(s.params[1], dtype=float)

    flat_T = tangents.reshape(len(points), -1)
    valid = (np.all(np.isfinite(points), axis=1) &
             np.all(np.isfinite(flat_T), axis=1))
    return (points[valid], tangents[valid],
            params_u[valid], params_v[valid])


# ---------------------------------------------------------------------------
# Test 1: Linear function on a cylinder
# ---------------------------------------------------------------------------

def test_gradient_linear():
    """
    On a cylinder (axis along x), f(x,y,z) = x is a linear function
    whose surface gradient is the unit vector along the x-axis
    (which is tangent to the cylinder).

    At level 0 the blow-up embeds positions in R^3; grad(f) should
    recover the x-direction exactly for a linear f.
    """
    r = 1.0
    surface = cylinders_tangent(radius=r, half_axis=1.5, half_angle=0.5)
    points, tangents, pu, pv = _sample_surface(
        surface, (0.0, 0.999), (0.0, 1.0), N_POINTS, SEED)

    interior = _trim_boundary(pu, pv, (0.0, 0.999), (0.0, 1.0), 0.05)

    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    W = lifted_affinity(l0, k=K_AFFINITY)

    # f = x-coordinate
    f = points[:, 0].copy()

    grad_f = lifted_gradient(l0, f, W, lam=LAM)

    # Ground truth: surface gradient of f=x on the cylinder (axis along x)
    # is the projection of (1,0,0) onto the tangent plane.
    # For cylinder 1: tangent frame is e1=(1,0,0), e2=(0,cos b,-sin b).
    # So proj of (1,0,0) = (1,0,0), and grad(x) = (1,0,0) everywhere.
    # But this is in the AMBIENT 3D space; at level 0 D=3.
    gt_grad = np.zeros_like(grad_f)
    gt_grad[:, 0] = 1.0
    # Project ground truth into each tangent plane to be fair
    for i in range(len(points)):
        Ui = l0.frame[i]  # (3, 2)
        gt_grad[i] = Ui @ (Ui.T @ gt_grad[i])

    mask = interior & np.all(np.isfinite(grad_f), axis=1)
    err = np.linalg.norm(grad_f[mask] - gt_grad[mask], axis=1)
    gt_norm = np.linalg.norm(gt_grad[mask], axis=1)
    rel_err = err / np.where(gt_norm > 1e-10, gt_norm, 1.0)

    med_rel = float(np.median(rel_err))
    p95_rel = float(np.percentile(rel_err, 95))
    passed = med_rel < GRAD_LINEAR_TOL

    print(f"\n{'='*70}")
    print(f"  Test 1: Gradient of linear f=x on cylinder    [{'PASS' if passed else 'FAIL'}]")
    print(f"{'='*70}")
    print(f"  Points: {mask.sum()} interior")
    print(f"  grad(f) rel error: median={med_rel:.4f}  p95={p95_rel:.4f}")
    return passed


# ---------------------------------------------------------------------------
# Test 2: Quadratic function on a sphere
# ---------------------------------------------------------------------------

def test_gradient_quadratic_sphere():
    """
    On a unit sphere, f(x,y,z) = z = cos(phi).  The surface gradient
    is the projection of (0,0,1) onto the tangent plane, which has
    magnitude sin(phi) and points in the -phi direction.

    |grad f| = sin(phi) = sqrt(1 - z^2).
    """
    r = 1.0
    surface = tangent_spheres(radius=r)
    points, tangents, pu, pv = _sample_surface(
        surface, (0.0, 0.999), (0.1, 0.9), N_POINTS, SEED)

    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    W = lifted_affinity(l0, k=K_AFFINITY)

    # Shift points to center of sphere 1 (center at (-r, 0, 0))
    centers = points.copy()
    centers[:, 0] += r  # now relative to sphere center

    # f = z-coordinate (height on sphere)
    f = points[:, 2].copy()

    grad_f = lifted_gradient(l0, f, W, lam=LAM)

    # Ground truth: grad(z) on sphere = projection of e_z onto tangent plane
    # = e_z - (e_z . n) n, where n = (x-cx, y, z)/r = centers/r
    normals = centers / r
    ez = np.array([0.0, 0.0, 1.0])
    gt_grad = ez[np.newaxis, :] - np.sum(ez * normals, axis=1, keepdims=True) * normals

    # Compare magnitudes (sign/direction in the frame can vary)
    est_norm = np.linalg.norm(grad_f[:, :3], axis=1)  # first 3 coords = spatial
    gt_norm = np.linalg.norm(gt_grad, axis=1)

    # Skip poles where |grad f| ~ 0
    mask = gt_norm > 0.1
    rel_err = np.abs(est_norm[mask] - gt_norm[mask]) / gt_norm[mask]

    med_rel = float(np.median(rel_err))
    p95_rel = float(np.percentile(rel_err, 95))
    passed = med_rel < GRAD_QUAD_TOL

    print(f"\n{'='*70}")
    print(f"  Test 2: Gradient of f=z on sphere (magnitude)  [{'PASS' if passed else 'FAIL'}]")
    print(f"{'='*70}")
    print(f"  Points: {mask.sum()} (excluding poles)")
    print(f"  |grad f| rel error: median={med_rel:.4f}  p95={p95_rel:.4f}")
    return passed


# ---------------------------------------------------------------------------
# Test 3: div(grad f) approximates the analytic Laplacian
# ---------------------------------------------------------------------------

def test_div_grad_analytic():
    """
    On a flat plane, div(grad f) for f = x^2 * y should approximate the
    true Laplace-Beltrami: Delta(x^2 y) = 2y.

    We test the Pearson correlation between div(grad f) and the true
    Delta f = 2y.  This is a stronger test than comparing to the graph
    Laplacian L = D - W, which itself is an unnormalised approximation.
    """
    surface = plane_paraboloid_tangent(scale_xy=1.0, scale_z=0.25)
    points, tangents, pu, pv = _sample_surface(
        surface, (0.0, 0.999), (0.0, 1.0), N_POINTS, SEED)

    interior = _trim_boundary(pu, pv, (0.0, 0.999), (0.0, 1.0), 0.1)

    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    W = lifted_affinity(l0, k=K_AFFINITY)

    # f = x^2 * y  =>  Delta f = 2y  (spatially varying)
    f = points[:, 0]**2 * points[:, 1]
    Delta_true = 2.0 * points[:, 1]

    grad_f = lifted_gradient(l0, f, W, lam=LAM)
    div_grad_f = lifted_divergence(l0, grad_f, W, lam=LAM)

    mask = interior
    corr = _correlation(div_grad_f[mask], Delta_true[mask])

    passed = corr > DIV_GRAD_CORR_TOL

    print(f"\n{'='*70}")
    print(f"  Test 3: div(grad(x^2 y)) vs true Laplacian 2y  [{'PASS' if passed else 'FAIL'}]")
    print(f"{'='*70}")
    print(f"  Points: {mask.sum()} interior")
    print(f"  Pearson correlation with true Delta f: {corr:.4f}")
    return passed


# ---------------------------------------------------------------------------
# Test 4: Divergence theorem (integral of div = 0 for closed surface)
# ---------------------------------------------------------------------------

def test_divergence_integral():
    """
    For a tangent vector field X on a closed surface (sphere), the integral
    of div(X) over the surface should be zero (divergence theorem with no
    boundary).

    We use X = grad(z) on the sphere, which is a smooth tangent field.
    The Laplacian of z on the unit sphere is -2z (eigenfunction), so
    div(grad z) = -2z, whose integral over the sphere is zero by symmetry.
    """
    r = 1.0
    surface = tangent_spheres(radius=r)
    # Sample a full sphere (avoid poles slightly)
    points, tangents, pu, pv = _sample_surface(
        surface, (0.0, 0.999), (0.05, 0.95), N_POINTS, SEED)

    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    W = lifted_affinity(l0, k=K_AFFINITY)

    # f = z coordinate
    f = points[:, 2].copy()

    grad_f = lifted_gradient(l0, f, W, lam=LAM)
    div_grad_f = lifted_divergence(l0, grad_f, W, lam=LAM)

    # The integral (sum) of div(grad f) should be approximately zero
    integral = float(np.sum(div_grad_f))
    mean_abs = float(np.mean(np.abs(div_grad_f)))
    relative = abs(integral) / (len(points) * mean_abs) if mean_abs > 1e-15 else 0.0

    passed = relative < DIV_INTEGRAL_TOL

    print(f"\n{'='*70}")
    print(f"  Test 4: Divergence theorem (integral ~ 0)      [{'PASS' if passed else 'FAIL'}]")
    print(f"{'='*70}")
    print(f"  Points: {len(points)}")
    print(f"  sum(div(grad z)):    {integral:.4f}")
    print(f"  mean|div(grad z)|:   {mean_abs:.4f}")
    print(f"  |integral|/(N*mean): {relative:.6f}")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Discrete gradient and divergence: accuracy and consistency tests")
    print(f"N={N_POINTS}  k_affinity={K_AFFINITY}  lam={LAM}  seed={SEED}")

    results = {}
    results["Gradient (linear, cylinder)"] = test_gradient_linear()
    results["Gradient (quadratic, sphere)"] = test_gradient_quadratic_sphere()
    results["div(grad f) vs analytic"]     = test_div_grad_analytic()
    results["Divergence theorem"]           = test_divergence_integral()

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
