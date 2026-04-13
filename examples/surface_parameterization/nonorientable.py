"""
Spectral parameterization of non-orientable surfaces.

For each non-orientable surface from `tangent_blowups.testsupport.examples_3D`:
    sample(u, v) -> (pts, normals, true uv)
    BlowUpLevel.from_normals(pts, normals).lift(alpha)
    lifted_laplacian(kernel="product", self_tuning=True)
    spectral_embedding(L, n_components=3)
    -> (u, v) := (eigvec_0, eigvec_1)  (the two smallest non-trivial)

Polyscope shows 5 side-by-side panels:
    eigenmap_0 | eigenmap_1 | eigenmap_2 | spectral checker | ground-truth checker

The ground-truth checker uses the analytic (u, v) parametrization so the
viewer can see directly that the lifted eigenmap chart agrees with the
parametric chart everywhere except (interestingly) along the seams induced
by the surface's non-orientability.

Usage:
    python nonorientable.py                       # list available surfaces
    python nonorientable.py -s mobius_band_1
    python nonorientable.py -s boy_surface --alpha 5 --nu 100 --nv 100
"""
from __future__ import annotations

import argparse
from math import pi as _PI

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from tangent_blowups.clustering import spectral_embedding
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import lifted_laplacian
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import examples_3D as _ex3d


# -- Non-orientable surface registry ---------------------------------------
# Each entry: (factory, (u_lo, u_hi), (v_lo, v_hi)).
# Bounds are nudged off coordinate singularities where needed.
_EPS = 1e-3
NON_ORIENTABLE_SURFACES: dict[str, tuple] = {
    "mobius_band_1":         (_ex3d.mobius_band_1,         (-1.0, 1.0),                (0.0, 2 * _PI)),
    "mobius_band_3":         (_ex3d.mobius_band_3,         (-1.0, 1.0),                (0.0, 2 * _PI)),
    "steiner_crosscap":      (_ex3d.steiner_crosscap,      (0.0, 2 * _PI),             (0.0, _PI / 2)),
    "steiner_crosscap_cut":  (_ex3d.steiner_crosscap_cut,  (0.0, _PI),                 (0.0, _PI / 2)),
    "steiner_roman":         (_ex3d.steiner_roman,         (0.0, _PI),                 (0.0, _PI)),
    "busser_decic":          (_ex3d.busser_decic,          (0.0, 2 * _PI),             (-_PI / 2, _PI / 2)),
    "boy_surface":           (_ex3d.boy_surface,           (0.0, 2 * _PI),             (0.0, _PI)),
    "boy_surface_5":         (_ex3d.boy_surface_5,         (0.0, 2 * _PI),             (0.0, _PI)),
    "etruscan_venus":        (_ex3d.etruscan_venus,        (_EPS, 2 * _PI - _EPS),     (_EPS, 2 * _PI - _EPS)),
    "jeener_bonan":          (_ex3d.jeener_bonan,          (0.0, 2 * _PI),             (0.0, 2 * _PI)),
    "jeener_bonan_2cc":      (_ex3d.jeener_bonan_2cc,      (0.0, 2 * _PI),             (0.0, 2 * _PI)),
    "busser_nonorientable":  (_ex3d.busser_nonorientable,  (0.0, 2 * _PI),             (-_PI / 2, _PI / 2)),
    "petit_russian_hat":     (_ex3d.petit_russian_hat,     (0.0, 2 * _PI),             (0.0, _PI)),
    "petit_russian_hat_cut": (_ex3d.petit_russian_hat_cut, (_PI, 2 * _PI),             (0.0, _PI)),
    "banchoff_klein":        (_ex3d.banchoff_klein,        (0.0, 2 * _PI),             (0.0, 2 * _PI)),
    "banchoff_klein_3":      (_ex3d.banchoff_klein_3,      (0.0, 2 * _PI),             (0.0, 2 * _PI)),
}


def _surface_area_element(surf, u, v, h: float = 1e-4) -> np.ndarray:
    """|partial_u P x partial_v P| via central differences on surf.position.

    This is the local area scaling of the parametrization. Sampling (u, v)
    uniformly in domain and rejecting against this gives uniform-density
    samples on the surface itself.
    """
    pu_p = surf.position(u + h, v)
    pu_m = surf.position(u - h, v)
    pv_p = surf.position(u, v + h)
    pv_m = surf.position(u, v - h)
    du = (pu_p - pu_m) / (2.0 * h)
    dv = (pv_p - pv_m) / (2.0 * h)
    return np.linalg.norm(np.cross(du, dv), axis=-1)


def _sample_uv_area_uniform(
    surf, u_lo: float, u_hi: float, v_lo: float, v_hi: float,
    n_target: int, *, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample (u, v) so the resulting 3D points are area-uniform."""
    # Coarse grid to estimate J_max and the average acceptance ratio.
    gu = np.linspace(u_lo, u_hi, 32)
    gv = np.linspace(v_lo, v_hi, 32)
    GU, GV = np.meshgrid(gu, gv, indexing="ij")
    j_grid = _surface_area_element(surf, GU, GV)
    j_max = float(np.nanmax(j_grid))
    if not np.isfinite(j_max) or j_max <= 0:
        j_max = 1.0
    j_max *= 1.05  # safety margin for grid undersampling of true max.

    avg_j = float(np.nanmean(j_grid))
    avg_accept = max(avg_j / j_max, 0.05)
    batch = max(int(n_target / avg_accept) + 1024, 4096)

    u_kept: list[np.ndarray] = []
    v_kept: list[np.ndarray] = []
    kept = 0
    while kept < n_target:
        u_try = rng.uniform(u_lo, u_hi, batch)
        v_try = rng.uniform(v_lo, v_hi, batch)
        j_try = _surface_area_element(surf, u_try, v_try)
        keep = rng.uniform(0.0, j_max, batch) < j_try
        u_kept.append(u_try[keep])
        v_kept.append(v_try[keep])
        kept += int(keep.sum())
    u_all = np.concatenate(u_kept)[:n_target]
    v_all = np.concatenate(v_kept)[:n_target]
    return u_all, v_all


def _sample_surface(
    name: str, nu: int, nv: int,
    *, mode: str = "area", seed: int = 0,
):
    """Sample a non-orientable surface and return (pts, normals, uv_true).

    mode:
      - "grid"   -- regular meshgrid of nu x nv samples (legacy; produces
                    visible isoparametric lines on curved surfaces).
      - "random" -- nu*nv uniform random samples in the (u, v) domain.
                    Removes the grid lines but the 3D density still varies
                    with the surface Jacobian.
      - "area"   -- rejection sampling against the area element so 3D point
                    density is uniform on the surface itself. Default.
    """
    factory, (u_lo, u_hi), (v_lo, v_hi) = NON_ORIENTABLE_SURFACES[name]
    surf = factory()
    n_target = nu * nv

    if mode == "grid":
        u = np.linspace(u_lo, u_hi, nu)
        v = np.linspace(v_lo, v_hi, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        u_flat = U.reshape(-1)
        v_flat = V.reshape(-1)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        u_flat = rng.uniform(u_lo, u_hi, n_target)
        v_flat = rng.uniform(v_lo, v_hi, n_target)
    elif mode == "area":
        rng = np.random.default_rng(seed)
        u_flat, v_flat = _sample_uv_area_uniform(
            surf, u_lo, u_hi, v_lo, v_hi, n_target, rng=rng,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {mode!r}")

    sample = surf.evaluate(u_flat, v_flat, with_tangents=False, with_normals=True)
    pts = np.asarray(sample.points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(sample.normals, dtype=float).reshape(-1, 3)
    uv_true = np.stack([u_flat, v_flat], axis=-1)
    valid = (
        np.isfinite(pts).all(axis=1)
        & np.isfinite(nrm).all(axis=1)
        & (np.linalg.norm(nrm, axis=1) > 1e-8)
    )
    if not np.all(valid):
        print(f"Dropping {int((~valid).sum())} degenerate samples.")
    return pts[valid], normalize_vectors(nrm[valid]), uv_true[valid]


# -- Rotation ---------------------------------------------------------------
def _rotation_matrix(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def _parse_rotation(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "x":
        return _rotation_matrix(90, np.array([0.0, 1.0, 0.0]))
    if spec == "y":
        return _rotation_matrix(-90, np.array([1.0, 0.0, 0.0]))
    parts = spec.split(",")
    if len(parts) == 4:
        return _rotation_matrix(
            float(parts[0]), np.array([float(p) for p in parts[1:]]),
        )
    raise ValueError(f"Invalid --rotate spec: {spec!r}")


# -- Checker overlay --------------------------------------------------------
def _normalize_uv(uv: np.ndarray) -> np.ndarray:
    """Per-axis rescale so each coord spans [-0.5, 0.5]."""
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (uv - lo) / span - 0.5


def _uv_checker(uv: np.ndarray, n_cells: float) -> np.ndarray:
    """RGB checker pattern from per-point (u, v) in [-0.5, 0.5]."""
    cell = 1.0 / max(n_cells, 1e-12)
    cu = np.floor(uv[:, 0] / cell).astype(int)
    cv = np.floor(uv[:, 1] / cell).astype(int)
    check = (cu + cv) % 2
    light = np.array([0.95, 0.95, 0.98])
    dark = np.array([0.30, 0.40, 0.75])
    return np.where(check[:, None], dark, light)


# -- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Spectral parameterization of non-orientable surfaces.",
    )
    parser.add_argument("--surface", "-s", type=str, default=None,
                        choices=sorted(NON_ORIENTABLE_SURFACES.keys()),
                        help="Non-orientable surface name (omit to list).")
    parser.add_argument("--nu", type=int, default=400,
                        help="Grid samples along u (default 80).")
    parser.add_argument("--nv", type=int, default=400,
                        help="Grid samples along v (default 80).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Chordal-Sasaki lift parameter (default 1.0).")
    parser.add_argument("--k", type=int, default=20,
                        help="k-NN neighborhood size (default 20).")
    parser.add_argument("--n-cells", type=float, default=12.0,
                        help="Checker cells per axis (default 12).")
    parser.add_argument("--sampling", type=str, default="area",
                        choices=["grid", "random", "area"],
                        help="(u,v) sampling: regular grid, uniform-random "
                             "in domain, or area-uniform rejection (default).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random/area sampling.")
    parser.add_argument("--rotate", "-r", type=str, default=None)
    args = parser.parse_args()

    if args.surface is None:
        print("Available non-orientable surfaces:")
        for name in sorted(NON_ORIENTABLE_SURFACES):
            print(f"  {name}")
        print("\nUsage: python nonorientable.py -s <name>")
        return

    print(f"Sampling {args.surface} ({args.sampling}, "
          f"target={args.nu * args.nv} points)...")
    pts, nrm, uv_true = _sample_surface(
        args.surface, args.nu, args.nv,
        mode=args.sampling, seed=args.seed,
    )
    print(f"  {len(pts)} points")

    if args.rotate is not None:
        R = _parse_rotation(args.rotate)
        pts = pts @ R.T
        nrm = nrm @ R.T

    print(f"Building level-1 blow-up (alpha={args.alpha})...")
    level = BlowUpLevel.from_normals(pts, nrm).lift(k=args.k, alpha=args.alpha)
    print(f"  embedded dim = {level.D}")

    print("Building self-tuning product Laplacian...")
    L, _, _ = lifted_laplacian(
        level, kernel="product", self_tuning=True,
        k=args.k, normalized=True,
    )
    print(f"  L: shape={L.shape}, nnz={L.nnz}")

    print("Computing spectral embedding (3 smallest non-trivial eigenpairs)...")
    evals, embedding = spectral_embedding(L, n_components=3)
    print("  eigenvalues:", " ".join(f"{e:.5f}" for e in evals))

    # Spectral parameterization: (u, v) := (eigvec_0, eigvec_1).
    uv_spectral = _normalize_uv(embedding[:, :2])
    uv_gt = _normalize_uv(uv_true)

    # --- Polyscope: 5 side-by-side panels ---------------------------------
    # Layout (left -> right):
    #   eigenmap_0 | eigenmap_1 | eigenmap_2 | spectral checker | analytic checker
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    bbox_x = float(pts[:, 0].max() - pts[:, 0].min())
    spacing = max(bbox_x, 1.0) * 1.4
    radius = 0.0035

    eig_panels = [
        ("eigenmap_0", embedding[:, 0]),
        ("eigenmap_1", embedding[:, 1]),
        ("eigenmap_2", embedding[:, 2]),
    ]
    for i, (name, values) in enumerate(eig_panels):
        offset = np.array([i * spacing, 0.0, 0.0])
        cloud = ps.register_point_cloud(name, pts + offset, radius=radius)
        cloud.add_scalar_quantity(name, values, enabled=True, cmap="coolwarm")

    spectral_cloud = ps.register_point_cloud(
        "checker_spectral", pts + np.array([3 * spacing, 0.0, 0.0]), radius=radius,
    )
    analytic_cloud = ps.register_point_cloud(
        "checker_analytic", pts + np.array([4 * spacing, 0.0, 0.0]), radius=radius,
    )

    state = {"n_cells": float(args.n_cells)}

    def _refresh_checkers():
        spectral_cloud.add_color_quantity(
            "checker_spectral", _uv_checker(uv_spectral, state["n_cells"]),
            enabled=True,
        )
        analytic_cloud.add_color_quantity(
            "checker_analytic", _uv_checker(uv_gt, state["n_cells"]),
            enabled=True,
        )

    _refresh_checkers()

    def _callback():
        changed, new_val = psim.SliderFloat(
            "checker cells", state["n_cells"], v_min=2.0, v_max=80.0,
        )
        if changed and new_val > 0:
            state["n_cells"] = float(new_val)
            _refresh_checkers()

    ps.set_user_callback(_callback)
    ps.show()


if __name__ == "__main__":
    main()
