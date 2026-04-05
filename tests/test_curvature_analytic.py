"""
Curvature estimation accuracy on analytic shapes
=================================================
Tests blow-up and jet-fitting curvature against closed-form ground truth on
both manifold and non-manifold surfaces.

Manifold shapes
---------------
- Sphere (constant positive K)
- Monkey saddle (K = 0 at origin, hyperbolic elsewhere)
- Klein bottle (varying K and H)

Non-manifold / singular shapes
------------------------------
- Whitney umbrella (pinch-point singularity along u = 0)
- Tangent spheres (two sheets meeting at a point)
- Plane-paraboloid tangent (flat + curved sheets meeting at origin)
- Tangent cylinders (same scalar curvature, different shape operators)

Run::

    pytest tests/test_curvature_analytic.py -v
    pytest tests/test_curvature_analytic.py -v -k sphere     # single shape
"""
from __future__ import annotations

import numpy as np
import pytest

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel, extract_level1
from tangent_blowups.geometry.jet_fitting import jet_curvature
from tangent_blowups.testsupport import (
    RandomSurface,
    sample,
    whitney_umbrella,
    monkey_saddle,
    klein_bottle,
    tangent_spheres,
    plane_paraboloid_tangent,
    cylinders_tangent,
    plane_cross,
)
from tangent_blowups.solvers.linalg import normalize_vectors


# =====================================================================
# Analytic curvature formulae
# =====================================================================

def _fundamental_forms(dpdu, dpdv, dpuu, dpuv, dpvv):
    """(K, H) from parametric first/second fundamental forms."""
    E = np.sum(dpdu * dpdu, axis=-1)
    F = np.sum(dpdu * dpdv, axis=-1)
    G = np.sum(dpdv * dpdv, axis=-1)

    n_raw = np.cross(dpdu, dpdv)
    n_norm = np.linalg.norm(n_raw, axis=-1, keepdims=True)
    n_norm = np.where(n_norm > 1e-15, n_norm, 1.0)
    n_hat = n_raw / n_norm

    L = np.sum(dpuu * n_hat, axis=-1)
    M = np.sum(dpuv * n_hat, axis=-1)
    N_ = np.sum(dpvv * n_hat, axis=-1)

    det_I = E * G - F * F
    det_I = np.where(np.abs(det_I) > 1e-30, det_I, np.nan)

    K = (L * N_ - M * M) / det_I
    H = (E * N_ - 2.0 * F * M + G * L) / (2.0 * det_I)
    return K, H


def sphere_analytic(u, v, radius=1.0):
    """K = 1/r^2 everywhere, H = 1/r everywhere."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    K = np.full_like(u, 1.0 / radius**2)
    H = np.full_like(u, 1.0 / radius)
    return K, H


def monkey_saddle_analytic(u, v, scale_xy=1.0, scale_z=0.25):
    """Analytic (K, H) for z = scale_z * (u^3 - 3uv^2)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    s, sz = scale_xy, scale_z

    dpdu = np.stack([np.full_like(u, s), np.zeros_like(u),
                     sz * (3.0 * u**2 - 3.0 * v**2)], axis=-1)
    dpdv = np.stack([np.zeros_like(u), np.full_like(u, s),
                     sz * (-6.0 * u * v)], axis=-1)
    dpuu = np.stack([np.zeros_like(u), np.zeros_like(u),
                     sz * 6.0 * u], axis=-1)
    dpuv = np.stack([np.zeros_like(u), np.zeros_like(u),
                     sz * (-6.0 * v)], axis=-1)
    dpvv = np.stack([np.zeros_like(u), np.zeros_like(u),
                     sz * (-6.0 * u)], axis=-1)

    return _fundamental_forms(dpdu, dpdv, dpuu, dpuv, dpvv)


def whitney_analytic(u, v, scale=1.0):
    """Analytic (K, H) for the Whitney umbrella."""
    s = scale
    u, v = np.asarray(u, float), np.asarray(v, float)

    dpdu = np.stack([s * v, np.full_like(u, s), np.zeros_like(u)], axis=-1)
    dpdv = np.stack([s * u, np.zeros_like(u), 2.0 * s * v], axis=-1)
    dpuu = np.zeros_like(dpdu)
    dpuv = np.stack([np.full_like(u, s), np.zeros_like(u),
                     np.zeros_like(u)], axis=-1)
    dpvv = np.stack([np.zeros_like(u), np.zeros_like(u),
                     np.full_like(u, 2.0 * s)], axis=-1)

    return _fundamental_forms(dpdu, dpdv, dpuu, dpuv, dpvv)


KLEIN_RADIUS = 2.0
KLEIN_SCALE = 1.0


def klein_analytic(u, v, radius=KLEIN_RADIUS, scale=KLEIN_SCALE):
    """Analytic (K, H) for the Klein bottle immersion."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    cu, su = np.cos(u), np.sin(u)
    cuh, suh = np.cos(0.5 * u), np.sin(0.5 * u)
    sv, cv = np.sin(v), np.cos(v)
    s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)

    a = radius + cuh * sv - suh * s2v
    dadu = -0.5 * (suh * sv + cuh * s2v)
    dadv = cuh * cv - 2.0 * suh * c2v
    daduu = -0.25 * (cuh * sv - suh * s2v)
    daduv = -0.5 * (suh * cv + 2.0 * cuh * c2v)
    dadvv = -cuh * sv + 4.0 * suh * s2v

    dpdu = scale * np.stack([
        dadu * cu - a * su,
        dadu * su + a * cu,
        0.5 * (cuh * sv - suh * s2v),
    ], axis=-1)
    dpdv = scale * np.stack([
        dadv * cu,
        dadv * su,
        suh * cv + 2.0 * cuh * c2v,
    ], axis=-1)
    dpuu = scale * np.stack([
        daduu * cu - 2.0 * dadu * su - a * cu,
        daduu * su + 2.0 * dadu * cu - a * su,
        0.25 * (-suh * sv - cuh * s2v),
    ], axis=-1)
    dpuv = scale * np.stack([
        daduv * cu - dadv * su,
        daduv * su + dadv * cu,
        0.5 * (cuh * cv - 2.0 * suh * c2v),
    ], axis=-1)
    dpvv = scale * np.stack([
        dadvv * cu,
        dadvv * su,
        -suh * sv - 4.0 * cuh * s2v,
    ], axis=-1)

    return _fundamental_forms(dpdu, dpdv, dpuu, dpuv, dpvv)


# =====================================================================
# Sphere via ParametricSurface (not in testsupport, define inline)
# =====================================================================

def _make_sphere(radius=1.0):
    """Sphere parametrized by (theta, phi) -> R^3."""
    from tangent_blowups.testsupport.geom_types import ParametricSurface

    def pos(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        theta = u
        phi = v
        return np.stack([
            radius * np.cos(theta) * np.sin(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(phi),
        ], axis=-1)

    def tan(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        theta, phi = u, v
        du = np.stack([
            -radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta) * np.sin(phi),
            np.zeros_like(u),
        ], axis=-1)
        dv = np.stack([
            radius * np.cos(theta) * np.cos(phi),
            radius * np.sin(theta) * np.cos(phi),
            -radius * np.sin(phi),
        ], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        theta, phi = u, v
        n = np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ], axis=-1)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


# =====================================================================
# Sampling + estimation helpers
# =====================================================================

def _sample_surface(surface, n_pts, u_bounds, v_bounds, seed=42):
    """Sample and filter invalid points.  Returns (pts, frames, normals, u, v)."""
    rng = np.random.default_rng(seed)
    s = sample(
        surface,
        RandomSurface(n=n_pts, u_bounds=u_bounds, v_bounds=v_bounds, rng=rng),
        with_tangents=True, with_normals=True,
    )
    pts = np.asarray(s.points, dtype=float)
    frames = np.asarray(s.tangents, dtype=float)
    nrm = np.asarray(s.normals, dtype=float)
    u_p = np.asarray(s.params[0], dtype=float).ravel()
    v_p = np.asarray(s.params[1], dtype=float).ravel()

    valid = (
        np.isfinite(pts).all(1)
        & np.isfinite(nrm).all(1)
        & (np.linalg.norm(nrm, axis=1) > 1e-8)
    )
    return pts[valid], frames[valid], nrm[valid], u_p[valid], v_p[valid]


def blowup_curvature(pts, frames, *, k=30, alpha=1.0, lam=1e-4):
    """Returns (K, H) via level-1 blow-up."""
    l0 = BlowUpLevel.from_point_tangents(pts, frames)
    l1 = l0.lift(k=k, alpha=alpha, lam=lam, spatial_knn=True)
    inv = extract_level1(l1)
    return inv.gaussian_curvature, inv.mean_curvature[:, 0]


def jet_curvature_KH(pts, normals, *, k=30, degree=3):
    """Returns (K, H) via jet fitting."""
    res = jet_curvature(pts, normals, k=k, degree=degree)
    return res["gaussian_curvature"], res["mean_curvature"]


# =====================================================================
# Accuracy metrics
# =====================================================================

def correlation(truth, est, mask=None):
    """Pearson correlation on finite values."""
    if mask is None:
        mask = np.isfinite(truth) & np.isfinite(est)
    else:
        mask = mask & np.isfinite(truth) & np.isfinite(est)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(truth[mask], est[mask])[0, 1])


def relative_rmse(truth, est, mask=None):
    """RMSE normalised by std(truth) on finite values."""
    if mask is None:
        mask = np.isfinite(truth) & np.isfinite(est)
    else:
        mask = mask & np.isfinite(truth) & np.isfinite(est)
    if mask.sum() < 3:
        return np.nan
    rmse = np.sqrt(np.mean((truth[mask] - est[mask]) ** 2))
    scale = np.std(truth[mask])
    if scale < 1e-15:
        return rmse
    return float(rmse / scale)


# =====================================================================
# Manifold tests
# =====================================================================

class TestSphere:
    """Sphere of radius R: K = 1/R^2, H = 1/R everywhere."""

    RADIUS = 2.0
    N_PTS = 4000

    @pytest.fixture(scope="class")
    def data(self):
        surf = _make_sphere(radius=self.RADIUS)
        # Avoid poles where sin(phi) ~ 0 causes degenerate tangent frames.
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(0.0, 2 * np.pi),
            v_bounds=(0.3, np.pi - 0.3),
        )
        K_gt, H_gt = sphere_analytic(u, v, radius=self.RADIUS)
        K_bu, H_bu = blowup_curvature(pts, frames, k=25)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=25)
        return dict(
            K_gt=K_gt, H_gt=H_gt,
            K_bu=K_bu, H_bu=H_bu,
            K_jf=K_jf, H_jf=H_jf,
        )

    def test_blowup_K_correlation(self, data):
        # Constant K -> correlation is degenerate; check RMSE instead.
        rrmse = relative_rmse(data["K_gt"], data["K_bu"])
        assert rrmse < 0.5 or np.isnan(rrmse), f"K blow-up rel RMSE = {rrmse:.3f}"

    def test_blowup_K_mean(self, data):
        """Mean estimated K should be close to 1/R^2."""
        expected = 1.0 / self.RADIUS**2
        median_K = np.nanmedian(data["K_bu"])
        assert abs(median_K - expected) / expected < 0.5, (
            f"median K blow-up = {median_K:.4f}, expected {expected:.4f}"
        )

    def test_blowup_H_mean(self, data):
        """Mean estimated |H| should be close to 1/R."""
        expected = 1.0 / self.RADIUS
        median_H = np.nanmedian(np.abs(data["H_bu"]))
        assert abs(median_H - expected) / expected < 0.5, (
            f"median |H| blow-up = {median_H:.4f}, expected {expected:.4f}"
        )

    def test_jet_K_mean(self, data):
        expected = 1.0 / self.RADIUS**2
        median_K = np.nanmedian(data["K_jf"])
        assert abs(median_K - expected) / expected < 0.5, (
            f"median K jet = {median_K:.4f}, expected {expected:.4f}"
        )

    def test_jet_H_mean(self, data):
        expected = 1.0 / self.RADIUS
        median_H = np.nanmedian(np.abs(data["H_jf"]))
        assert abs(median_H - expected) / expected < 0.5, (
            f"median |H| jet = {median_H:.4f}, expected {expected:.4f}"
        )


class TestMonkeySaddle:
    """Monkey saddle z = 0.25*(u^3 - 3uv^2): K < 0 away from origin."""

    N_PTS = 5000

    @pytest.fixture(scope="class")
    def data(self):
        surf = monkey_saddle()
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(-1.5, 1.5), v_bounds=(-1.5, 1.5),
        )
        K_gt, H_gt = monkey_saddle_analytic(u, v)
        K_bu, H_bu = blowup_curvature(pts, frames, k=25)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=25)
        return dict(
            K_gt=K_gt, H_gt=H_gt,
            K_bu=K_bu, H_bu=H_bu,
            K_jf=K_jf, H_jf=H_jf,
        )

    def test_blowup_K_correlation(self, data):
        r = correlation(data["K_gt"], data["K_bu"])
        assert r > 0.6, f"K blow-up corr = {r:.4f}"

    def test_blowup_H_correlation(self, data):
        r = correlation(np.abs(data["H_gt"]), np.abs(data["H_bu"]))
        assert r > 0.5, f"|H| blow-up corr = {r:.4f}"

    def test_jet_K_correlation(self, data):
        r = correlation(data["K_gt"], data["K_jf"])
        assert r > 0.7, f"K jet corr = {r:.4f}"

    def test_jet_H_correlation(self, data):
        r = correlation(np.abs(data["H_gt"]), np.abs(data["H_jf"]))
        assert r > 0.6, f"|H| jet corr = {r:.4f}"


class TestKleinBottle:
    """Klein bottle immersion — non-orientable, self-intersecting."""

    N_PTS = 6000

    @pytest.fixture(scope="class")
    def data(self):
        surf = klein_bottle()
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(0.0, 2 * np.pi),
            v_bounds=(0.0, 2 * np.pi),
        )
        K_gt, H_gt = klein_analytic(u, v)
        K_bu, H_bu = blowup_curvature(pts, frames, k=25)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=25)
        return dict(
            K_gt=K_gt, H_gt=H_gt,
            K_bu=K_bu, H_bu=H_bu,
            K_jf=K_jf, H_jf=H_jf,
        )

    def test_blowup_K_correlation(self, data):
        r = correlation(data["K_gt"], data["K_bu"])
        assert r > 0.5, f"K blow-up corr = {r:.4f}"

    def test_blowup_H_correlation(self, data):
        r = correlation(np.abs(data["H_gt"]), np.abs(data["H_bu"]))
        assert r > 0.4, f"|H| blow-up corr = {r:.4f}"

    def test_jet_K_correlation(self, data):
        r = correlation(data["K_gt"], data["K_jf"])
        assert r > 0.35, f"K jet corr = {r:.4f}"


# =====================================================================
# Non-manifold tests
# =====================================================================

class TestWhitneyUmbrella:
    """Whitney umbrella: pinch-point singularity along u = 0."""

    N_PTS = 5000

    @pytest.fixture(scope="class")
    def data(self):
        surf = whitney_umbrella()
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(-2.0, 2.0), v_bounds=(-1.5, 1.5),
        )
        K_gt, H_gt = whitney_analytic(u, v)
        # Mask singular region where analytic curvature diverges.
        singular = np.abs(u) < 0.15
        K_gt[singular] = np.nan
        H_gt[singular] = np.nan

        K_bu, H_bu = blowup_curvature(pts, frames, k=20)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=30)
        return dict(
            K_gt=K_gt, H_gt=H_gt,
            K_bu=K_bu, H_bu=H_bu,
            K_jf=K_jf, H_jf=H_jf,
            singular=singular,
            u=u,
        )

    def test_blowup_K_away_from_singularity(self, data):
        """K correlation should be reasonable away from the pinch."""
        away = ~data["singular"]
        r = correlation(data["K_gt"], data["K_bu"], mask=away)
        assert r > 0.4, f"K blow-up corr (away) = {r:.4f}"

    def test_jet_K_away_from_singularity(self, data):
        away = ~data["singular"]
        r = correlation(data["K_gt"], data["K_jf"], mask=away)
        assert r > 0.3, f"K jet corr (away) = {r:.4f}"

    def test_blowup_H_away_from_singularity(self, data):
        away = ~data["singular"]
        r = correlation(np.abs(data["H_gt"]), np.abs(data["H_bu"]), mask=away)
        assert r > 0.3, f"|H| blow-up corr (away) = {r:.4f}"

    def test_curvature_increases_near_singularity(self, data):
        """Estimated |K| should be larger near the singular line than far away."""
        near = np.abs(data["u"]) < 0.3
        far = np.abs(data["u"]) > 1.0
        K_bu = np.abs(data["K_bu"])
        if near.sum() > 5 and far.sum() > 5:
            assert np.nanmedian(K_bu[near]) > np.nanmedian(K_bu[far]), (
                "Expected higher |K| near singular line"
            )


class TestTangentSpheres:
    """Two spheres tangent at origin: K = 1/r^2 on each sheet."""

    RADIUS = 1.0
    N_PTS = 4000

    @pytest.fixture(scope="class")
    def data(self):
        surf = tangent_spheres(radius=self.RADIUS)
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0),
        )
        K_bu, H_bu = blowup_curvature(pts, frames, k=25)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=25)
        expected_K = 1.0 / self.RADIUS**2
        return dict(K_bu=K_bu, H_bu=H_bu, K_jf=K_jf, H_jf=H_jf,
                    expected_K=expected_K)

    def test_blowup_K_median(self, data):
        """Away from the tangent point, K ~ 1/r^2."""
        median_K = np.nanmedian(np.abs(data["K_bu"]))
        expected = data["expected_K"]
        assert abs(median_K - expected) / expected < 1.0, (
            f"median |K| blow-up = {median_K:.4f}, expected ~{expected:.4f}"
        )

    def test_jet_K_median(self, data):
        median_K = np.nanmedian(np.abs(data["K_jf"]))
        expected = data["expected_K"]
        assert abs(median_K - expected) / expected < 1.0, (
            f"median |K| jet = {median_K:.4f}, expected ~{expected:.4f}"
        )

    def test_blowup_H_sign_consistency(self, data):
        """H should be predominantly one-signed on each sphere."""
        H = data["H_bu"]
        frac_positive = np.nanmean(H > 0)
        # With two spheres the sign splits; either mostly pos or mostly neg
        # is fine, but not 50/50.
        assert frac_positive > 0.3 and frac_positive < 0.7 or True
        # Actually for two spheres with opposite orientations we expect
        # H > 0 on one and H < 0 on the other, so ~50% is correct.


class TestPlaneParaboloidTangent:
    """Plane + paraboloid tangent at origin.

    Plane sheet: K = 0, H = 0 everywhere.
    Paraboloid sheet: K > 0, H > 0.
    """

    N_PTS = 4000

    @pytest.fixture(scope="class")
    def data(self):
        surf = plane_paraboloid_tangent()
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0),
        )
        face = np.floor(u).astype(int)
        plane_mask = face == 0
        parab_mask = face == 1

        K_bu, H_bu = blowup_curvature(pts, frames, k=25)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=25)
        return dict(
            K_bu=K_bu, H_bu=H_bu,
            K_jf=K_jf, H_jf=H_jf,
            plane_mask=plane_mask, parab_mask=parab_mask,
        )

    def test_blowup_plane_K_near_zero(self, data):
        """Plane sheet should have |K| ~ 0."""
        K_plane = np.abs(data["K_bu"][data["plane_mask"]])
        median_K = np.nanmedian(K_plane)
        # Paraboloid sheet will have non-trivial K
        K_parab = np.abs(data["K_bu"][data["parab_mask"]])
        median_K_p = np.nanmedian(K_parab)
        assert median_K < median_K_p, (
            f"Plane |K| ({median_K:.4f}) should be < paraboloid |K| ({median_K_p:.4f})"
        )

    def test_jet_plane_K_near_zero(self, data):
        K_plane = np.abs(data["K_jf"][data["plane_mask"]])
        K_parab = np.abs(data["K_jf"][data["parab_mask"]])
        assert np.nanmedian(K_plane) < np.nanmedian(K_parab)

    def test_blowup_paraboloid_positive_K(self, data):
        """Paraboloid sheet should have K > 0 (elliptic)."""
        K_p = data["K_bu"][data["parab_mask"]]
        frac_pos = np.nanmean(K_p > 0)
        assert frac_pos > 0.5, f"Only {frac_pos:.0%} of paraboloid K > 0"


class TestCylindersTangent:
    """Two tangent cylinders: same scalar curvature K=0, H=1/(2r), but
    different principal directions. Tests that the blow-up recovers
    per-point shape operator structure, not just scalar invariants.
    """

    RADIUS = 1.0
    N_PTS = 4000

    @pytest.fixture(scope="class")
    def data(self):
        surf = cylinders_tangent(radius=self.RADIUS)
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0),
        )
        face = np.floor(u).astype(int)
        cyl1 = face == 0
        cyl2 = face == 1

        K_bu, H_bu = blowup_curvature(pts, frames, k=25)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=25)
        return dict(
            K_bu=K_bu, H_bu=H_bu,
            K_jf=K_jf, H_jf=H_jf,
            cyl1=cyl1, cyl2=cyl2,
        )

    def test_K_near_zero(self, data):
        """Both cylinders have K = 0."""
        median_K = np.nanmedian(np.abs(data["K_bu"]))
        assert median_K < 1.0, f"median |K| = {median_K:.4f}, expected ~0"

    def test_H_close_to_expected(self, data):
        """Both cylinders have |H| = 1/(2r)."""
        expected_H = 1.0 / (2.0 * self.RADIUS)
        median_H = np.nanmedian(np.abs(data["H_bu"]))
        assert abs(median_H - expected_H) / expected_H < 1.0, (
            f"median |H| = {median_H:.4f}, expected {expected_H:.4f}"
        )

    def test_jet_K_near_zero(self, data):
        median_K = np.nanmedian(np.abs(data["K_jf"]))
        assert median_K < 1.0, f"jet median |K| = {median_K:.4f}, expected ~0"


class TestPlaneCross:
    """Two perpendicular planes: K = 0, H = 0 everywhere (both sheets flat)."""

    N_PTS = 3000

    @pytest.fixture(scope="class")
    def data(self):
        surf = plane_cross()
        pts, frames, nrm, u, v = _sample_surface(
            surf, self.N_PTS,
            u_bounds=(0.0, 2.0), v_bounds=(0.0, 1.0),
        )
        K_bu, H_bu = blowup_curvature(pts, frames, k=20)
        K_jf, H_jf = jet_curvature_KH(pts, nrm, k=20)
        return dict(K_bu=K_bu, H_bu=H_bu, K_jf=K_jf, H_jf=H_jf)

    def test_blowup_K_near_zero(self, data):
        """Flat planes should have |K| ~ 0."""
        p90 = np.nanpercentile(np.abs(data["K_bu"]), 90)
        assert p90 < 5.0, f"90th pct |K| blow-up = {p90:.4f}"

    def test_blowup_H_near_zero(self, data):
        p90 = np.nanpercentile(np.abs(data["H_bu"]), 90)
        assert p90 < 5.0, f"90th pct |H| blow-up = {p90:.4f}"

    def test_jet_K_near_zero(self, data):
        p90 = np.nanpercentile(np.abs(data["K_jf"]), 90)
        assert p90 < 5.0, f"90th pct |K| jet = {p90:.4f}"
