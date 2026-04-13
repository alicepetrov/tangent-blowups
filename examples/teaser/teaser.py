"""
Teaser figure: lifted geometric analysis of a non-orientable surface.

Eight side-by-side panels showing, left to right:

    1. heat_full         -- heat-method geodesic spread from a user-picked
                            source on the full surface
    2. heat_cut          -- same, restricted to the half of the surface with
                            z >= median(z) so the viewer can see inside
    3. eigenmap_0        -- smallest non-trivial eigenvector of the lifted
                            Laplacian (Fiedler mode)
    4. eigenmap_1        -- second non-trivial eigenvector
    5. eigenmap_2        -- third non-trivial eigenvector
    6. gaussian          -- Gaussian curvature K from level-1 blow-up
    7. mean              -- mean curvature magnitude |H| = sqrt(H^2)
    8. total             -- total curvature ||h||_F

All panels share the same underlying level-1 BlowUpLevel and product-kernel
Laplacian; only the colour field changes. The polyscope UI exposes the heat
method's t_scale and diffusion_steps as live sliders so the user can watch
the heat field evolve without rebuilding the Laplacian.

Usage:
    python teaser.py                                # etruscan_venus (default)
    python teaser.py -s boy_surface --alpha 5
    python teaser.py                                # no-arg: run default
"""
from __future__ import annotations

import argparse
from math import pi as _PI

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from tangent_blowups.clustering import spectral_embedding
from tangent_blowups.geometry.iterated_grassmann import (
    BlowUpLevel, extract_level1,
)
from tangent_blowups.geometry.kernels import (
    lifted_laplacian, estimate_product_bandwidths,
)
from tangent_blowups.pointcloud import lifted_heat_method, precompute_heat_method
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import examples_3D as _ex3d


# -- Non-orientable surface registry ----------------------------------------
_EPS = 1e-3
NON_ORIENTABLE_SURFACES: dict[str, tuple] = {
    "mobius_band_1":         (_ex3d.mobius_band_1,         (-1.0, 1.0),            (0.0, 2 * _PI)),
    "mobius_band_3":         (_ex3d.mobius_band_3,         (-1.0, 1.0),            (0.0, 2 * _PI)),
    "steiner_crosscap":      (_ex3d.steiner_crosscap,      (0.0, 2 * _PI),         (0.0, _PI / 2)),
    "steiner_crosscap_cut":  (_ex3d.steiner_crosscap_cut,  (0.0, _PI),             (0.0, _PI / 2)),
    "steiner_roman":         (_ex3d.steiner_roman,         (0.0, _PI),             (0.0, _PI)),
    "busser_decic":          (_ex3d.busser_decic,          (0.0, 2 * _PI),         (-_PI / 2, _PI / 2)),
    "boy_surface":           (_ex3d.boy_surface,           (0.0, 2 * _PI),         (0.0, _PI)),
    "boy_surface_5":         (_ex3d.boy_surface_5,         (0.0, 2 * _PI),         (0.0, _PI)),
    "etruscan_venus":        (_ex3d.etruscan_venus,        (_EPS, 2 * _PI - _EPS), (_EPS, 2 * _PI - _EPS)),
    "jeener_bonan":          (_ex3d.jeener_bonan,          (0.0, 2 * _PI),         (0.0, 2 * _PI)),
    "jeener_bonan_2cc":      (_ex3d.jeener_bonan_2cc,      (0.0, 2 * _PI),         (0.0, 2 * _PI)),
    "busser_nonorientable":  (_ex3d.busser_nonorientable,  (0.0, 2 * _PI),         (-_PI / 2, _PI / 2)),
    "petit_russian_hat":     (_ex3d.petit_russian_hat,     (0.0, 2 * _PI),         (0.0, _PI)),
    "petit_russian_hat_cut": (_ex3d.petit_russian_hat_cut, (_PI, 2 * _PI),         (0.0, _PI)),
    "banchoff_klein":        (_ex3d.banchoff_klein,        (0.0, 2 * _PI),         (0.0, 2 * _PI)),
    "banchoff_klein_3":      (_ex3d.banchoff_klein_3,      (0.0, 2 * _PI),         (0.0, 2 * _PI)),
}


# -- Area-uniform sampling --------------------------------------------------
def _surface_area_element(surf, u, v, h: float = 1e-4) -> np.ndarray:
    """|partial_u P x partial_v P| via central differences on surf.position."""
    pu_p = surf.position(u + h, v); pu_m = surf.position(u - h, v)
    pv_p = surf.position(u, v + h); pv_m = surf.position(u, v - h)
    du = (pu_p - pu_m) / (2.0 * h)
    dv = (pv_p - pv_m) / (2.0 * h)
    return np.linalg.norm(np.cross(du, dv), axis=-1)


def _sample_surface(name: str, n_target: int, seed: int = 0):
    """Area-uniform rejection sampling of a non-orientable surface.

    Returns (pts, normals) both (N, 3), finite, unit-norm normals.
    """
    factory, (u_lo, u_hi), (v_lo, v_hi) = NON_ORIENTABLE_SURFACES[name]
    surf = factory()

    # Coarse pass to estimate J_max and average acceptance.
    gu = np.linspace(u_lo, u_hi, 32)
    gv = np.linspace(v_lo, v_hi, 32)
    GU, GV = np.meshgrid(gu, gv, indexing="ij")
    j_grid = _surface_area_element(surf, GU, GV)
    j_max = float(np.nanmax(j_grid))
    if not np.isfinite(j_max) or j_max <= 0:
        j_max = 1.0
    j_max *= 1.05
    avg_accept = max(float(np.nanmean(j_grid)) / j_max, 0.05)
    batch = max(int(n_target / avg_accept) + 1024, 4096)

    rng = np.random.default_rng(seed)
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
    u_flat = np.concatenate(u_kept)[:n_target]
    v_flat = np.concatenate(v_kept)[:n_target]

    sample = surf.evaluate(u_flat, v_flat, with_tangents=False, with_normals=True)
    pts = np.asarray(sample.points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(sample.normals, dtype=float).reshape(-1, 3)
    valid = (
        np.isfinite(pts).all(axis=1)
        & np.isfinite(nrm).all(axis=1)
        & (np.linalg.norm(nrm, axis=1) > 1e-8)
    )
    return pts[valid], normalize_vectors(nrm[valid])


# -- Heat colouring ---------------------------------------------------------
def _heat_to_colors(u: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    """Log-scale heat field -> RGB. Mirrors alpha_ablation._heat_to_colors."""
    from matplotlib import colormaps
    cmap = colormaps[cmap_name]
    u_abs = np.abs(u)
    u_max = u_abs.max()
    if u_max < 1e-30:
        return np.tile(np.array(cmap(0.0)[:3]), (len(u), 1))
    log_u = np.log10(np.clip(u_abs / u_max, 1e-6, 1.0))
    t = np.clip((log_u + 6.0) / 6.0, 0.0, 1.0)
    return cmap(t)[:, :3]


def _run_heat(level, *, source_index, k, sigma_x, sigma_u,
              t_scale, diffusion_steps, precomputed):
    """Return the scalar heat field u at the given source."""
    _, u, _, _ = lifted_heat_method(
        level, source_index=source_index, k=k,
        kernel="product", sigma_x=sigma_x, sigma_u=sigma_u,
        t_scale=t_scale, diffusion_steps=diffusion_steps,
        return_intermediate=True, _precomputed=precomputed,
    )
    return u


# -- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Teaser figure: heat + eigenmodes + curvature on a "
                    "non-orientable surface.",
    )
    parser.add_argument("--surface", "-s", type=str, default="etruscan_venus",
                        choices=sorted(NON_ORIENTABLE_SURFACES.keys()),
                        help="Non-orientable surface (default etruscan_venus).")
    parser.add_argument("--n-points", "-n", type=int, default=100000,
                        help="Target number of samples (default 8000).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Chordal-Sasaki lift parameter (default 1.0).")
    parser.add_argument("--k", type=int, default=20,
                        help="k-NN neighborhood size (default 20).")
    parser.add_argument("--t-scale", type=float, default=1.0,
                        help="Heat method t scale (Crane et al. default 1).")
    parser.add_argument("--diffusion-steps", type=int, default=1,
                        help="Implicit heat diffusion steps (default 1).")
    parser.add_argument("--source", type=int, default=None,
                        help="Initial source index. Omit to pick in the GUI.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Sampling {args.surface} (area-uniform, target={args.n_points})...")
    pts, nrm = _sample_surface(args.surface, args.n_points, seed=args.seed)
    N = len(pts)
    print(f"  {N} points")

    # --- Cut-in-half: keep the half with z >= median(z) ------------------
    # Purely an index mask so we can reuse the same Laplacian / curvature
    # computed on the full cloud for the cut view.
    z_thresh = float(np.median(pts[:, 2]))
    cut_mask = pts[:, 2] >= z_thresh
    print(f"  cut mask: {int(cut_mask.sum())} / {N} points above z={z_thresh:.3f}")

    # --- Level-1 blow-up + bandwidths ------------------------------------
    print(f"Building level-1 blow-up (alpha={args.alpha})...")
    frames = _normals_to_tangent_frames(nrm)
    l0 = BlowUpLevel.from_point_tangents(pts, frames)
    l1 = l0.lift(k=args.k, alpha=args.alpha)
    sigma_x, sigma_u = estimate_product_bandwidths(l1, args.k)
    print(f"  D={l1.D}  sigma_x={sigma_x:.4f}  sigma_u={[round(s,4) for s in sigma_u]}")

    # --- Shared Laplacian for eigenmodes (self-tuning product) ----------
    print("Building self-tuning product Laplacian for eigenmodes...")
    L_eig, _, _ = lifted_laplacian(
        l1, kernel="product", self_tuning=True, k=args.k, normalized=True,
    )
    print("Computing 3 smallest non-trivial eigenpairs...")
    evals, embedding = spectral_embedding(L_eig, n_components=3)
    print("  eigenvalues:", " ".join(f"{e:.5f}" for e in evals))

    # --- Curvature (Gaussian, mean |H|, total) ----------------------------
    print("Extracting level-1 curvature invariants...")
    inv = extract_level1(l1)
    K = inv.gaussian_curvature
    Tc = inv.total_curvature[:, 0]
    H2 = (Tc ** 2 + 2.0 * K) / 4.0  # H^2 = (||h||^2 + 2K) / 4, orientation-free
    H_abs = np.sqrt(np.clip(H2, 0.0, None))
    # Clamp curvatures to percentile range to keep colormaps readable.
    def _clamp(arr, lo=2.0, hi=98.0):
        a = np.asarray(arr, dtype=float)
        q_lo, q_hi = np.nanpercentile(a, [lo, hi])
        return np.clip(a, q_lo, q_hi)

    K_c = _clamp(K)
    H_c = _clamp(H_abs)
    T_c = _clamp(Tc)

    # --- Precomputed heat-method cache (rebuilt on t_scale change) --------
    # One cache for the full cloud (used for both heat_full and heat_cut
    # since both are views on the same level-1 BlowUpLevel).
    cache: dict[tuple[float, int], dict] = {}

    def _get_cache(t_scale: float, diffusion_steps: int):
        key = (float(t_scale), int(diffusion_steps))
        if key not in cache:
            print(f"  [heat] precomputing (t_scale={t_scale:g}, "
                  f"steps={diffusion_steps})...")
            cache[key] = precompute_heat_method(
                l1, k=args.k, sigma_x=sigma_x, sigma_u=sigma_u,
                t_scale=t_scale,
            )
        return cache[key]

    # --- Polyscope: 8 side-by-side panels --------------------------------
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    bbox_x = float(pts[:, 0].max() - pts[:, 0].min())
    spacing = max(bbox_x, 1.0) * 1.4
    radius = 0.0035

    # Panel spec: (name, kind, data, cmap-or-colors)
    panels = [
        ("heat_full", "heat_full", None, "inferno"),
        ("heat_cut",  "heat_cut",  None, "inferno"),
        ("eigenmap_0", "scalar", embedding[:, 0], "coolwarm"),
        ("eigenmap_1", "scalar", embedding[:, 1], "coolwarm"),
        ("eigenmap_2", "scalar", embedding[:, 2], "coolwarm"),
        ("gaussian",   "scalar", K_c, "coolwarm"),
        ("mean",       "scalar", H_c, "viridis"),
        ("total",      "scalar", T_c, "inferno"),
    ]

    # Register one cloud per panel.  Heat panels start uncoloured until a
    # source is picked; every other panel is coloured once up-front.
    clouds: dict[str, "ps.PointCloud"] = {}
    for i, (name, kind, data, cmap) in enumerate(panels):
        offset = np.array([i * spacing, 0.0, 0.0])
        if kind == "heat_cut":
            cloud_pts = pts[cut_mask] + offset
        else:
            cloud_pts = pts + offset
        cloud = ps.register_point_cloud(name, cloud_pts, radius=radius)
        if kind == "scalar":
            cloud.add_scalar_quantity(
                name, data, enabled=True, cmap=cmap,
            )
        clouds[name] = cloud

    heat_panel_names = {"heat_full", "heat_cut"}

    # --- UI state --------------------------------------------------------
    state = {
        "source_index": int(args.source) if args.source is not None else None,
        "t_scale": float(args.t_scale),
        "diffusion_steps": int(args.diffusion_steps),
        "needs_recompute": args.source is not None,
        "u_full": None,  # cached heat field on the full cloud
    }

    def _place_source_marker(src_idx: int):
        for panel_name in ("heat_full", "heat_cut"):
            tag = f"src_{panel_name}"
            if ps.has_point_cloud(tag):
                ps.remove_point_cloud(tag)
            pt = pts[src_idx:src_idx + 1]
            if panel_name == "heat_cut" and not cut_mask[src_idx]:
                # Source not in the cut half -- skip marker there.
                continue
            i = next(
                i for i, (n, *_rest) in enumerate(panels) if n == panel_name
            )
            offset = np.array([i * spacing, 0.0, 0.0])
            sc = ps.register_point_cloud(tag, pt + offset, radius=0.012)
            sc.set_color((0.0, 1.0, 1.0))

    def _recompute_heat():
        src = state["source_index"]
        if src is None:
            return
        print(f"Computing heat field from source {src} "
              f"(t_scale={state['t_scale']:g}, steps={state['diffusion_steps']})...")
        pre = _get_cache(state["t_scale"], state["diffusion_steps"])
        u_full = _run_heat(
            l1, source_index=src, k=args.k,
            sigma_x=sigma_x, sigma_u=sigma_u,
            t_scale=state["t_scale"],
            diffusion_steps=state["diffusion_steps"],
            precomputed=pre,
        )
        state["u_full"] = u_full
        colors_full = _heat_to_colors(u_full, "inferno")
        clouds["heat_full"].add_color_quantity("heat", colors_full, enabled=True)

        u_cut = u_full[cut_mask]
        colors_cut = _heat_to_colors(u_cut, "inferno")
        clouds["heat_cut"].add_color_quantity("heat", colors_cut, enabled=True)

        _place_source_marker(src)
        print("  done.")

    def callback():
        psim.TextUnformatted(
            f"{args.surface}  |  {N} points  |  alpha={args.alpha:g}",
        )
        psim.TextUnformatted(
            "Ctrl+click heat_full or heat_cut to pick a source.",
        )
        src_str = ("(none -- pick one)" if state["source_index"] is None
                   else str(state["source_index"]))
        psim.TextUnformatted(f"Source: {src_str}")
        psim.Separator()

        pick = ps.get_selection()
        if pick.is_hit and pick.structure_name in heat_panel_names:
            picked = int(pick.local_index)
            # heat_cut indexes into the masked sub-cloud -- map back to full.
            if pick.structure_name == "heat_cut":
                picked = int(np.flatnonzero(cut_mask)[picked])
            psim.TextUnformatted(f"Hovered: {picked}")
            if psim.Button("Set as source"):
                state["source_index"] = picked
                state["needs_recompute"] = True

        changed_t, new_t = psim.InputFloat("t_scale", state["t_scale"])
        if changed_t and new_t > 0:
            state["t_scale"] = float(new_t)
            cache.clear()
            if state["source_index"] is not None:
                state["needs_recompute"] = True

        changed_s, new_s = psim.InputInt(
            "diffusion_steps", state["diffusion_steps"],
        )
        if changed_s and new_s > 0:
            state["diffusion_steps"] = int(new_s)
            if state["source_index"] is not None:
                state["needs_recompute"] = True

        if state["source_index"] is not None and psim.Button("Recompute heat"):
            state["needs_recompute"] = True

        if state["needs_recompute"]:
            state["needs_recompute"] = False
            _recompute_heat()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
