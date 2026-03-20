"""
Curvature estimation accuracy on analytic surfaces with known ground truth.
============================================================================

Tests the iterated blow-up curvature extraction against exact principal
curvatures on four families of surfaces:

  1. Cylinder  (H = 1/(2r), K = 0)
  2. Sphere    (H = 1/r,    K = 1/r^2,  umbilic)
  3. Paraboloid z = c(x^2+y^2)  (variable H, K; known closed form)
  4. Monkey saddle z = c(u^3 - 3uv^2)  (H = 0 at origin, K < 0)

For each surface the script:
  - Samples N points with analytic tangent frames
  - Runs a single-level blow-up
  - Extracts Level-1 invariants (H, K, principal curvatures)
  - Compares to closed-form ground truth
  - Prints per-surface error statistics and PASS/FAIL summary
"""
from __future__ import annotations

import sys
import numpy as np

from tangent_blowups.testsupport import (
    RandomSurface,
    cylinders_tangent,
    plane_paraboloid_tangent,
    tangent_spheres,
    monkey_saddle,
    sample,
)
from tangent_blowups.geometry.iterated_grassmann import (
    BlowUpLevel,
    extract_level1,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_POINTS   = 4000     # points per surface
K_BLOWUP   = 20       # k-NN for curvature regression
ALPHA      = 2.0      # Chordal-Sasaki weight
LAM        = 1e-3     # ridge regularisation
SEED       = 42

# Tolerance for PASS/FAIL (median relative error for H, absolute for K)
H_REL_TOL  = 0.05     # 5% median relative error on |H|
K_ABS_TOL  = 0.02     # absolute tolerance on K (useful when K ~ 0)
K_REL_TOL  = 0.10     # 10% median relative error on K when |K| is large


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _paraboloid_curvatures(u: np.ndarray, v: np.ndarray, c: float):
    """
    Exact principal curvatures for z = c(x^2 + y^2).

    For the isotropic paraboloid the second fundamental form is diagonal
    with equal eigenvalues at every point (umbilic surface):

        kappa_1 = kappa_2 = 2c / (1 + 4c^2 r^2)^{3/2}

    where r^2 = x^2 + y^2.

    Actually more carefully: the surface z = c(x^2 + y^2) has
        f_x = 2cx,  f_y = 2cy,  f_xx = f_yy = 2c,  f_xy = 0

    First fundamental form:  E = 1 + 4c^2 x^2,  F = 4c^2 xy,  G = 1 + 4c^2 y^2
    Second fundamental form:  e = 2c/sqrt(1+4c^2 r^2),  f = 0,
                               g = 2c/sqrt(1+4c^2 r^2)
    Wait -- that's only true for the normal curvature in the (x,y) directions
    when rotated to principal axes.

    For z = c(x^2 + y^2), by rotational symmetry kappa_1 = kappa_2 everywhere.
    The principal curvature at radius r from the apex is:

        kappa(r) = 2c / (1 + 4c^2 r^2)^{3/2}     ... meridional
        kappa(r) = 2c / sqrt(1 + 4c^2 r^2)        ... azimuthal

    Wait, they are NOT equal for a paraboloid of revolution except at r=0.
    Let me be precise. For z = c*r^2 (r = sqrt(x^2+y^2)):

    Meridional (along radial direction):
        kappa_m = z'' / (1 + z'^2)^{3/2} = 2c / (1 + 4c^2 r^2)^{3/2}

    Azimuthal (perpendicular to radial):
        kappa_a = z' / (r * sqrt(1 + z'^2)) = 2c*r / (r * sqrt(1 + 4c^2 r^2))
                = 2c / sqrt(1 + 4c^2 r^2)

    So: H = (kappa_m + kappa_a) / 2,  K = kappa_m * kappa_a.
    """
    # x, y are the parametric coords scaled by scale_xy
    r2 = u**2 + v**2
    denom = np.sqrt(1.0 + 4.0 * c**2 * r2)
    kappa_m = 2.0 * c / denom**3           # meridional
    kappa_a = 2.0 * c / denom              # azimuthal
    H_gt = (kappa_m + kappa_a) / 2.0
    K_gt = kappa_m * kappa_a
    return H_gt, K_gt, kappa_m, kappa_a


def _monkey_saddle_curvatures(u: np.ndarray, v: np.ndarray,
                               scale_xy: float, scale_z: float):
    """
    Exact curvatures for z = scale_z * (u^3 - 3*u*v^2).

    With x = scale_xy * u, y = scale_xy * v,
    z = scale_z * (u^3 - 3*u*v^2).

    Partial derivatives w.r.t. u, v:
      z_u = scale_z * (3u^2 - 3v^2)
      z_v = scale_z * (-6uv)
      z_uu = scale_z * 6u
      z_uv = scale_z * (-6v)
      z_vv = scale_z * (-6u)

    But the surface is parameterized as r(u,v) = (scale_xy*u, scale_xy*v, z(u,v)).
    So r_u = (scale_xy, 0, z_u) and r_v = (0, scale_xy, z_v).

    First fundamental form:
      E = scale_xy^2 + z_u^2
      F = z_u * z_v
      G = scale_xy^2 + z_v^2

    Second fundamental form (via n . r_ij):
      n = r_u x r_v / |r_u x r_v|
      r_u x r_v = (-scale_xy * z_v, -scale_xy * z_u, scale_xy^2)... wait
      Actually: r_u = (sx, 0, zu), r_v = (0, sx, zv)
      r_u x r_v = (0*zv - sx*zu, ... )
      = (-sx * zv, -sx * (-zu)... let me just compute properly:
      r_u x r_v = | i       j      k     |
                  | sx      0      zu    |
                  | 0       sx     zv    |
      = i(0*zv - zu*sx) - j(sx*zv - zu*0) + k(sx*sx - 0)
      = (-sx*zu, -sx*zv, sx^2)
      Hmm wait: i(0*zv - zu*sx) = -sx*zu... that doesn't look right.
      Cross product: (a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1)
      = (0*zv - zu*sx,  zu*0 - sx*zv,  sx*sx - 0*0)
      = (-zu*sx, -sx*zv, sx^2)

    |n_raw| = sx * sqrt(zu^2 + zv^2 + sx^2)

    Second fundamental form coefficients (Weingarten):
      r_uu = (0, 0, zuu), r_uv = (0, 0, zuv), r_vv = (0, 0, zvv)
      e = n . r_uu = sx^2 * zuu / |n_raw|
      f = n . r_uv = sx^2 * zuv / |n_raw|
      g = n . r_vv = sx^2 * zvv / |n_raw|

    H = (eG - 2fF + gE) / (2(EG - F^2))
    K = (eg - f^2) / (EG - F^2)
    """
    sx = scale_xy
    sz = scale_z

    zu = sz * (3.0 * u**2 - 3.0 * v**2)
    zv = sz * (-6.0 * u * v)
    zuu = sz * 6.0 * u
    zuv = sz * (-6.0 * v)
    zvv = sz * (-6.0 * u)

    E = sx**2 + zu**2
    F = zu * zv
    G = sx**2 + zv**2

    norm_raw = np.sqrt(zu**2 + zv**2 + sx**2)

    e = sx**2 * zuu / norm_raw
    f = sx**2 * zuv / norm_raw
    g = sx**2 * zvv / norm_raw

    det_I = E * G - F**2
    H_gt = (e * G - 2.0 * f * F + g * E) / (2.0 * det_I)
    K_gt = (e * g - f**2) / det_I

    return H_gt, K_gt


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(name, surface, u_bounds, v_bounds, n_points,
             gt_H_fn, gt_K_fn, *,
             boundary_margin=0.0,
             k=K_BLOWUP, alpha=ALPHA, lam=LAM):
    """
    Sample a surface, compute curvature via blow-up, compare to ground truth.

    gt_H_fn(u, v) -> array of H values
    gt_K_fn(u, v) -> array of K values  (or None to skip K check)
    """
    rng = np.random.default_rng(SEED)
    strategy = RandomSurface(n=n_points, u_bounds=u_bounds, v_bounds=v_bounds, rng=rng)
    s = sample(surface, strategy, with_tangents=True, with_normals=True)

    points   = np.asarray(s.points,   dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)
    params_u = np.asarray(s.params[0], dtype=float)
    params_v = np.asarray(s.params[1], dtype=float)

    # Filter degenerate frames
    flat_T = tangents.reshape(len(points), -1)
    valid = (np.all(np.isfinite(points), axis=1) &
             np.all(np.isfinite(flat_T), axis=1))
    points, tangents = points[valid], tangents[valid]
    params_u, params_v = params_u[valid], params_v[valid]

    # Optionally trim boundary points where k-NN is incomplete
    if boundary_margin > 0:
        u_lo, u_hi = u_bounds
        v_lo, v_hi = v_bounds
        interior = ((params_u > u_lo + boundary_margin) &
                    (params_u < u_hi - boundary_margin) &
                    (params_v > v_lo + boundary_margin) &
                    (params_v < v_hi - boundary_margin))
    else:
        interior = np.ones(len(points), dtype=bool)

    # Blow-up level 1
    l0 = BlowUpLevel.from_point_tangents(points, tangents)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam)
    inv1 = extract_level1(l1)

    # Estimated curvatures
    H_est = inv1.mean_curvature[:, 0]        # (N,)  signed H
    K_est = inv1.gaussian_curvature           # (N,)  or None

    # Ground truth
    H_gt = gt_H_fn(params_u, params_v)
    K_gt = gt_K_fn(params_u, params_v) if gt_K_fn is not None else None

    # --- Report H ---
    # Compare |H| because the shape-operator sign depends on the arbitrary
    # choice of normal-frame orientation (complement of tangent frame).
    mask = interior & np.isfinite(H_est) & np.isfinite(H_gt)
    abs_err_H = np.abs(np.abs(H_est[mask]) - np.abs(H_gt[mask]))
    H_gt_abs = np.abs(H_gt[mask])
    # Use relative error where |H| is large, absolute where it's near zero
    large_H = H_gt_abs > 0.01
    if large_H.any():
        rel_err_H = abs_err_H[large_H] / H_gt_abs[large_H]
        med_rel_H = float(np.median(rel_err_H))
        p95_rel_H = float(np.percentile(rel_err_H, 95))
    else:
        med_rel_H = 0.0
        p95_rel_H = 0.0
    med_abs_H = float(np.median(abs_err_H))
    mean_H_gt = float(np.mean(H_gt_abs))

    # --- Report K ---
    if K_gt is not None and K_est is not None:
        K_mask = mask & np.isfinite(K_est)
        abs_err_K = np.abs(K_est[K_mask] - K_gt[K_mask])
        K_gt_abs  = np.abs(K_gt[K_mask])
        med_abs_K = float(np.median(abs_err_K))
        large_K = K_gt_abs > 0.01
        if large_K.any():
            rel_err_K = abs_err_K[large_K] / K_gt_abs[large_K]
            med_rel_K = float(np.median(rel_err_K))
        else:
            med_rel_K = 0.0
    else:
        med_abs_K = None
        med_rel_K = None

    # --- PASS / FAIL ---
    h_pass = med_rel_H < H_REL_TOL if large_H.any() else med_abs_H < K_ABS_TOL
    if K_gt is not None and K_est is not None:
        if large_K.any():
            k_pass = med_rel_K < K_REL_TOL
        else:
            k_pass = med_abs_K < K_ABS_TOL
    else:
        k_pass = True  # no K to check

    status = "PASS" if (h_pass and k_pass) else "FAIL"

    # --- Print ---
    print(f"\n{'='*70}")
    print(f"  {name}    [{status}]")
    print(f"{'='*70}")
    print(f"  Points: {mask.sum()} (of {len(points)} sampled, {interior.sum()} interior)")
    print(f"  Mean |H_gt|: {mean_H_gt:.4f}")
    print(f"  H error:  median_abs={med_abs_H:.5f}  median_rel={med_rel_H:.4f}  p95_rel={p95_rel_H:.4f}")
    if med_abs_K is not None:
        print(f"  K error:  median_abs={med_abs_K:.5f}  median_rel={med_rel_K:.4f}")
    print(f"  H: {'PASS' if h_pass else 'FAIL'}    K: {'PASS' if k_pass else 'FAIL'}")

    return status == "PASS"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_cylinder():
    """Cylinder: H = 1/(2r), K = 0 everywhere."""
    r = 1.0
    surface = cylinders_tangent(radius=r, half_axis=1.2, half_angle=0.55)
    # Only use one cylinder (face 0: u in [0,1))
    H_gt = lambda u, v: np.full_like(u, 1.0 / (2.0 * r))
    K_gt = lambda u, v: np.zeros_like(u)
    return run_test(
        "Cylinder (r=1)",
        surface,
        u_bounds=(0.0, 0.999),  # single cylinder
        v_bounds=(0.0, 1.0),
        n_points=N_POINTS,
        gt_H_fn=H_gt,
        gt_K_fn=K_gt,
        boundary_margin=0.05,
    )


def test_sphere():
    """Sphere: H = 1/r, K = 1/r^2 everywhere (umbilic)."""
    r = 1.0
    surface = tangent_spheres(radius=r)
    # Single sphere: face 0 (u in [0,1))
    # Sign: our H might be negative depending on normal orientation
    # so compare |H|.  But extract_level1 gives signed H from the shape
    # operator; for a sphere with outward normal, H = 1/r > 0.
    # Actually the sign depends on the frame convention, so let's compare |H|.
    def H_gt(u, v):
        return np.full_like(u, 1.0 / r)
    def K_gt(u, v):
        return np.full_like(u, 1.0 / r**2)
    return run_test(
        "Sphere (r=1)",
        surface,
        u_bounds=(0.0, 0.999),  # single sphere
        v_bounds=(0.05, 0.95),  # avoid poles where tangent frame degenerates
        n_points=N_POINTS,
        gt_H_fn=H_gt,
        gt_K_fn=K_gt,
        boundary_margin=0.0,
    )


def test_paraboloid():
    """Paraboloid z = c(x^2 + y^2):  variable H, K with closed form."""
    scale_xy = 1.0
    c = 0.25  # scale_z
    surface = plane_paraboloid_tangent(scale_xy=scale_xy, scale_z=c)

    # Only use face 1 (the paraboloid, u in [1,2))
    # The local parameter s = u - 1 maps to spatial coords:
    #   x = (s - 0.5) * 2 * scale_xy
    #   y = (v - 0.5) * 2 * scale_xy
    def H_gt(u, v):
        s = u - np.floor(u)
        x = (s - 0.5) * 2.0 * scale_xy
        y = (v - 0.5) * 2.0 * scale_xy
        H, K, _, _ = _paraboloid_curvatures(x, y, c)
        return H

    def K_gt(u, v):
        s = u - np.floor(u)
        x = (s - 0.5) * 2.0 * scale_xy
        y = (v - 0.5) * 2.0 * scale_xy
        H, K, _, _ = _paraboloid_curvatures(x, y, c)
        return K

    return run_test(
        "Paraboloid z=0.25(x^2+y^2)",
        surface,
        u_bounds=(1.0, 1.999),  # paraboloid face only
        v_bounds=(0.0, 1.0),
        n_points=N_POINTS,
        gt_H_fn=H_gt,
        gt_K_fn=K_gt,
        boundary_margin=0.05,
    )


def test_monkey_saddle():
    """Monkey saddle z = c(u^3 - 3uv^2): H = 0 at origin, K <= 0."""
    scale_xy = 1.0
    scale_z  = 0.25
    surface  = monkey_saddle(scale_xy=scale_xy, scale_z=scale_z)

    def H_gt(u, v):
        H, K = _monkey_saddle_curvatures(u, v, scale_xy, scale_z)
        return H

    def K_gt(u, v):
        H, K = _monkey_saddle_curvatures(u, v, scale_xy, scale_z)
        return K

    return run_test(
        "Monkey saddle",
        surface,
        u_bounds=(-1.0, 1.0),
        v_bounds=(-1.0, 1.0),
        n_points=N_POINTS,
        gt_H_fn=H_gt,
        gt_K_fn=K_gt,
        boundary_margin=0.15,
    )


def test_plane():
    """Plane: H = 0, K = 0 everywhere (sanity check)."""
    surface = plane_paraboloid_tangent(scale_xy=1.0, scale_z=0.25)
    H_gt = lambda u, v: np.zeros_like(u)
    K_gt = lambda u, v: np.zeros_like(u)
    return run_test(
        "Plane (H=0, K=0)",
        surface,
        u_bounds=(0.0, 0.999),  # plane face only
        v_bounds=(0.0, 1.0),
        n_points=N_POINTS,
        gt_H_fn=H_gt,
        gt_K_fn=K_gt,
        boundary_margin=0.05,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Curvature estimation accuracy: analytic ground truth comparison")
    print(f"N={N_POINTS}  k={K_BLOWUP}  alpha={ALPHA}  lam={LAM}  seed={SEED}")
    print(f"Tolerances: H_rel={H_REL_TOL}  K_rel={K_REL_TOL}  K_abs={K_ABS_TOL}")

    results = {}
    results["Plane"]         = test_plane()
    results["Cylinder"]      = test_cylinder()
    results["Sphere"]        = test_sphere()
    results["Paraboloid"]    = test_paraboloid()
    results["Monkey saddle"] = test_monkey_saddle()

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
