"""
Normal/Point Noise Ablation: Heat Flow on Lifted Point Clouds
--------------------------------------------------------------
Visualizes how lifted heat spread changes as point positions and normals are
corrupted.  The scene is a 3x3 Polyscope grid:

    - point noise increases left to right
    - normal noise increases top to bottom

Usage:
    python normal_noise.py -p 010g_pointcloud --max-points 50000
    python normal_noise.py -p Glykon --rotate y --source 1000
    python normal_noise.py --surface boy_surface --point-noise 0,0.002,0.006
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from matplotlib import colormaps

from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.geometry.kernels import estimate_product_bandwidths
from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud import (
    estimate_normals_pca,
    lifted_heat_method,
    precompute_heat_method,
)
from tangent_blowups.pointcloud.geodesic_heat import _normals_to_tangent_frames
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import examples_3D as _ex3d

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
_HEAT_CMAPS = ["inferno", "magma", "viridis", "plasma", "coolwarm", "hot", "YlOrRd"]
DEFAULT_POINT_NOISE = [0.0, 0.002, 0.006]
DEFAULT_NORMAL_NOISE = [0.0, 0.15, 0.35]

_PI = float(np.pi)
_EPS = 1e-3
NON_ORIENTABLE_SURFACES = {
    "mobius_band_1": (
        _ex3d.mobius_band_1,
        (-1.0, 1.0),
        (0.0, 2 * _PI),
    ),
    "mobius_band_3": (
        _ex3d.mobius_band_3,
        (-1.0, 1.0),
        (0.0, 2 * _PI),
    ),
    "steiner_crosscap": (
        _ex3d.steiner_crosscap,
        (0.0, 2 * _PI),
        (0.0, _PI / 2),
    ),
    "steiner_crosscap_cut": (
        _ex3d.steiner_crosscap_cut,
        (0.0, _PI),
        (0.0, _PI / 2),
    ),
    "steiner_roman": (
        _ex3d.steiner_roman,
        (0.0, _PI),
        (0.0, _PI),
    ),
    "busser_decic": (
        _ex3d.busser_decic,
        (0.0, 2 * _PI),
        (-_PI / 2, _PI / 2),
    ),
    "boy_surface": (
        _ex3d.boy_surface,
        (0.0, 2 * _PI),
        (0.0, _PI),
    ),
    "boy_surface_5": (
        _ex3d.boy_surface_5,
        (0.0, 2 * _PI),
        (0.0, _PI),
    ),
    "etruscan_venus": (
        _ex3d.etruscan_venus,
        (_EPS, 2 * _PI - _EPS),
        (_EPS, 2 * _PI - _EPS),
    ),
    "jeener_bonan": (
        _ex3d.jeener_bonan,
        (0.0, 2 * _PI),
        (0.0, 2 * _PI),
    ),
    "jeener_bonan_2cc": (
        _ex3d.jeener_bonan_2cc,
        (0.0, 2 * _PI),
        (0.0, 2 * _PI),
    ),
    "busser_nonorientable": (
        _ex3d.busser_nonorientable,
        (0.0, 2 * _PI),
        (-_PI / 2, _PI / 2),
    ),
    "petit_russian_hat": (
        _ex3d.petit_russian_hat,
        (0.0, 2 * _PI),
        (0.0, _PI),
    ),
    "petit_russian_hat_cut": (
        _ex3d.petit_russian_hat_cut,
        (_PI, 2 * _PI),
        (0.0, _PI),
    ),
    "banchoff_klein": (
        _ex3d.banchoff_klein,
        (0.0, 2 * _PI),
        (0.0, 2 * _PI),
    ),
    "banchoff_klein_3": (
        _ex3d.banchoff_klein_3,
        (0.0, 2 * _PI),
        (0.0, 2 * _PI),
    ),
}


def _parse_float_list(spec: str, *, name: str, expected: int = 3) -> list[float]:
    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} comma-separated values.")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} values must be non-negative.")
    return values


def _sample_surface(name: str, nu: int, nv: int) -> tuple[np.ndarray, np.ndarray]:
    if name not in NON_ORIENTABLE_SURFACES:
        raise ValueError(
            f"Unknown surface '{name}'. Available: {', '.join(NON_ORIENTABLE_SURFACES)}"
        )
    factory, (u_lo, u_hi), (v_lo, v_hi) = NON_ORIENTABLE_SURFACES[name]
    surf = factory()
    n_target = nu * nv

    def area_element(u_, v_, h=1e-4):
        pu_p = surf.position(u_ + h, v_)
        pu_m = surf.position(u_ - h, v_)
        pv_p = surf.position(u_, v_ + h)
        pv_m = surf.position(u_, v_ - h)
        du = (pu_p - pu_m) / (2.0 * h)
        dv = (pv_p - pv_m) / (2.0 * h)
        return np.linalg.norm(np.cross(du, dv), axis=-1)

    gu = np.linspace(u_lo, u_hi, 32)
    gv = np.linspace(v_lo, v_hi, 32)
    gu_grid, gv_grid = np.meshgrid(gu, gv, indexing="ij")
    j_grid = area_element(gu_grid, gv_grid)
    j_max = float(np.nanmax(j_grid))
    if not np.isfinite(j_max) or j_max <= 0.0:
        j_max = 1.0
    j_max *= 1.05
    avg_accept = max(float(np.nanmean(j_grid)) / j_max, 0.05)
    batch = max(int(n_target / avg_accept) + 1024, 4096)

    rng = np.random.default_rng(0)
    u_kept: list[np.ndarray] = []
    v_kept: list[np.ndarray] = []
    kept = 0
    while kept < n_target:
        u_try = rng.uniform(u_lo, u_hi, batch)
        v_try = rng.uniform(v_lo, v_hi, batch)
        j_try = area_element(u_try, v_try)
        keep = rng.uniform(0.0, j_max, batch) < j_try
        u_kept.append(u_try[keep])
        v_kept.append(v_try[keep])
        kept += int(keep.sum())

    u_flat = np.concatenate(u_kept)[:n_target]
    v_flat = np.concatenate(v_kept)[:n_target]
    sample = surf.evaluate(u_flat, v_flat, with_tangents=False, with_normals=True)
    points = np.asarray(sample.points, dtype=float).reshape(-1, 3)
    normals = np.asarray(sample.normals, dtype=float).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid &= np.linalg.norm(normals, axis=1) > 1e-8
    if not np.all(valid):
        print(f"Dropping {int(np.sum(~valid))} degenerate samples from {name}.")
    return points[valid], normalize_vectors(normals[valid])


def _rotation_matrix(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c = np.cos(np.radians(angle_deg))
    s = np.sin(np.radians(angle_deg))
    k = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * k


def _parse_rotation(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "x":
        return _rotation_matrix(90, np.array([0.0, 1.0, 0.0]))
    if spec == "y":
        return _rotation_matrix(-90, np.array([1.0, 0.0, 0.0]))
    parts = spec.split(",")
    if len(parts) == 4:
        angle = float(parts[0])
        axis = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        return _rotation_matrix(angle, axis)
    raise ValueError(f"Invalid --rotate spec: {spec!r}")


def _available_pointclouds() -> dict[str, Path]:
    result: dict[str, Path] = {}
    if _DATA_ROOT.is_dir():
        for path in _DATA_ROOT.rglob("*.npz"):
            result.setdefault(path.stem, path)
    return dict(sorted(result.items()))


def _resolve_pointcloud(name: str) -> Path:
    path = Path(name)
    if path.suffix == ".npz" and path.exists():
        return path
    available = _available_pointclouds()
    if name in available:
        return available[name]
    raise FileNotFoundError(
        f"Point cloud '{name}' not found. Available: {', '.join(available)}"
    )


def _load_points_and_normals(
    path: Path,
    *,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError(f"Point cloud {path.name} is missing normals.")

    points = np.asarray(sample.points, dtype=float)
    normals = np.asarray(sample.normals, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if normals.ndim > 2:
        normals = normals.reshape(-1, normals.shape[-1])

    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    valid &= np.linalg.norm(normals, axis=1) > normal_eps
    if not np.all(valid):
        print(f"Dropping {int(np.sum(~valid))} invalid samples.")
    return points[valid], normalize_vectors(normals[valid])


def _estimate_normals_lpca(points: np.ndarray, *, k: int) -> np.ndarray:
    normals = estimate_normals_pca(points, k=k)
    return normalize_vectors(np.asarray(normals, dtype=float))


def _heat_to_colors(u: np.ndarray, n_points: int, cmap_name: str = "inferno"):
    cmap = colormaps[cmap_name]
    u_abs = np.abs(u)
    u_max = u_abs.max()
    if u_max < 1e-30:
        return np.tile(np.array(cmap(0.0)[:3]), (n_points, 1))
    log_u = np.log10(np.clip(u_abs / u_max, 1e-6, 1.0))
    t = (log_u + 6.0) / 6.0
    return cmap(np.clip(t, 0.0, 1.0))[:, :3]


def _build_level(points: np.ndarray, normals: np.ndarray, *, alpha: float, k: int):
    frames = _normals_to_tangent_frames(normals)
    level0 = BlowUpLevel.from_point_tangents(points, frames)
    return level0.lift(alpha=alpha, k=k)


def _source_nearest_centroid(points: np.ndarray) -> int:
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    dist2 = np.sum((points - center) ** 2, axis=1)
    return int(np.argmin(dist2))


def _noise_variants(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    point_noise: list[float],
    normal_noise: list[float],
    point_noise_absolute: bool,
    seed: int | None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    variants = {}
    for row, normal_sigma in enumerate(normal_noise):
        for col, point_sigma in enumerate(point_noise):
            sigma_x = point_sigma if point_noise_absolute else point_sigma * bbox_diag
            noisy_points = np.array(points, copy=True)
            noisy_normals = np.array(normals, copy=True)
            if sigma_x > 0.0:
                noisy_points += rng.normal(0.0, sigma_x, size=noisy_points.shape)
            if normal_sigma > 0.0:
                noisy_normals += rng.normal(
                    0.0,
                    normal_sigma,
                    size=noisy_normals.shape,
                )
                noisy_normals = normalize_vectors(noisy_normals)
            variants[(row, col)] = (noisy_points, noisy_normals)
    return variants


def _panel_name(row: int, col: int, point_noise: list[float], normal_noise: list[float]):
    return f"p={point_noise[col]:g}, n={normal_noise[row]:g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3x3 lifted heat spread ablation for point and normal noise."
    )
    parser.add_argument("--pointcloud", "-p", type=str, default=None)
    parser.add_argument(
        "--surface",
        "-s",
        type=str,
        default=None,
        choices=sorted(NON_ORIENTABLE_SURFACES),
        help="Sample a synthetic surface instead of loading a .npz point cloud.",
    )
    parser.add_argument("--surface-nu", type=int, default=220)
    parser.add_argument("--surface-nv", type=int, default=220)
    parser.add_argument(
        "--point-noise",
        type=str,
        default=",".join(str(value) for value in DEFAULT_POINT_NOISE),
        help=(
            "Three comma-separated point-noise levels. By default these are "
            "fractions of the bounding-box diagonal."
        ),
    )
    parser.add_argument(
        "--point-noise-absolute",
        action="store_true",
        help="Interpret --point-noise values as absolute coordinate sigmas.",
    )
    parser.add_argument(
        "--normal-noise",
        type=str,
        default=",".join(str(value) for value in DEFAULT_NORMAL_NOISE),
        help="Three comma-separated Gaussian normal-vector jitter sigmas.",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--estimate-normals-lpca",
        action="store_true",
        help=(
            "Use LPCA-estimated normals instead of loaded/analytic normals. "
            "Normals are estimated once on the initial point cloud before noise."
        ),
    )
    parser.add_argument(
        "--lpca-k",
        type=int,
        default=30,
        help="Neighbors used for LPCA normal estimation.",
    )
    parser.add_argument("--t-scale", type=float, default=1.0)
    parser.add_argument(
        "--bandwidth-mode",
        choices=["baseline", "auto"],
        default="auto",
        help="Use clean-cloud bandwidths for every panel, or auto-estimate per panel.",
    )
    parser.add_argument("--rotate", "-r", type=str, default=None)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--downsample-seed", type=int, default=None)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--source", type=int, default=None)
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.001,
        help="Polyscope point radius for the nine clouds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device for lifted heat method.",
    )
    args = parser.parse_args()

    if args.pointcloud is not None and args.surface is not None:
        parser.error("--pointcloud and --surface are mutually exclusive.")

    if args.pointcloud is None and args.surface is None:
        available = _available_pointclouds()
        print("Available point clouds:")
        if available:
            for name, path in available.items():
                print(f"  {name}  ({path.parent.name})")
        else:
            print("  (none)")
        print("Available synthetic surfaces:")
        for name in sorted(NON_ORIENTABLE_SURFACES):
            print(f"  {name}")
        return

    point_noise = _parse_float_list(args.point_noise, name="--point-noise")
    normal_noise = _parse_float_list(args.normal_noise, name="--normal-noise")

    if args.surface is not None:
        points, normals = _sample_surface(args.surface, args.surface_nu, args.surface_nv)
        print(f"Sampled {len(points)} points from surface '{args.surface}'")
    else:
        path = _resolve_pointcloud(args.pointcloud)
        points, normals = _load_points_and_normals(path)
        print(f"Loaded {len(points)} points from {path.name}")

    if args.max_points and len(points) > args.max_points:
        rng = np.random.default_rng(args.downsample_seed)
        idx = np.sort(rng.choice(len(points), size=args.max_points, replace=False))
        points = points[idx]
        normals = normals[idx]
        print(f"Downsampled to {len(points)} points")

    if args.rotate is not None:
        r = _parse_rotation(args.rotate)
        points = points @ r.T
        normals = normals @ r.T
        print(f"Applied rotation: --rotate {args.rotate}")

    source_index = (
        int(args.source)
        if args.source is not None
        else _source_nearest_centroid(points)
    )
    if source_index < 0:
        source_index += len(points)
    if source_index < 0 or source_index >= len(points):
        raise ValueError("--source is out of range.")

    baseline_normals = normals
    if args.estimate_normals_lpca:
        print(f"Estimating baseline LPCA normals (k={args.lpca_k})...")
        baseline_normals = _estimate_normals_lpca(points, k=args.lpca_k)

    variants = _noise_variants(
        points,
        baseline_normals,
        point_noise=point_noise,
        normal_noise=normal_noise,
        point_noise_absolute=args.point_noise_absolute,
        seed=args.noise_seed,
    )

    clean_level = _build_level(points, baseline_normals, alpha=args.alpha, k=args.k)
    baseline_sigma_x, baseline_sigma_u = estimate_product_bandwidths(
        clean_level,
        args.k,
    )
    print(
        "Baseline bandwidths: "
        f"sigma_x={baseline_sigma_x:.4f}, "
        f"sigma_u={[round(value, 4) for value in baseline_sigma_u]}"
    )

    levels: dict[tuple[int, int], BlowUpLevel] = {}
    bandwidths: dict[tuple[int, int], tuple[float, list[float]]] = {}
    for key, (variant_points, variant_normals) in variants.items():
        row, col = key
        panel = _panel_name(row, col, point_noise, normal_noise)
        level = _build_level(
            variant_points,
            variant_normals,
            alpha=args.alpha,
            k=args.k,
        )
        if args.bandwidth_mode == "auto":
            sigma_x, sigma_u = estimate_product_bandwidths(level, args.k)
        else:
            sigma_x = baseline_sigma_x
            sigma_u = baseline_sigma_u
        levels[key] = level
        bandwidths[key] = (sigma_x, sigma_u)
        print(
            f"  [{panel}] D={level.D}, "
            f"sigma_x={sigma_x:.4f}, "
            f"sigma_u={[round(value, 4) for value in sigma_u]}"
        )

    bbox = points.max(axis=0) - points.min(axis=0)
    x_spacing = max(float(bbox[0]), float(np.max(bbox)) * 0.75) * 1.45
    y_spacing = max(float(bbox[1]), float(np.max(bbox)) * 0.75) * 1.45
    if x_spacing <= 0.0:
        x_spacing = 1.0
    if y_spacing <= 0.0:
        y_spacing = 1.0

    panel_offsets = {
        (row, col): np.array([col * x_spacing, -row * y_spacing, 0.0])
        for row in range(3)
        for col in range(3)
    }

    state = {
        "source_index": source_index,
        "computing": True,
        "heat_cmap_idx": 0,
        "u": {},
        "t_scale": float(args.t_scale),
    }
    cache: dict[tuple[int, int], dict] = {}

    def get_cache(key: tuple[int, int]):
        if key not in cache:
            sigma_x, sigma_u = bandwidths[key]
            cache[key] = precompute_heat_method(
                levels[key],
                k=args.k,
                kernel="product",
                sigma_x=sigma_x,
                sigma_u=sigma_u,
                t_scale=state["t_scale"],
                device=args.device,
            )
        return cache[key]

    def compute_panel(key: tuple[int, int]) -> None:
        row, col = key
        panel = _panel_name(row, col, point_noise, normal_noise)
        sigma_x, sigma_u = bandwidths[key]
        _, u, _, _ = lifted_heat_method(
            levels[key],
            source_index=state["source_index"],
            k=args.k,
            kernel="product",
            sigma_x=sigma_x,
            sigma_u=sigma_u,
            t_scale=state["t_scale"],
            return_intermediate=True,
            device=args.device,
            _precomputed=get_cache(key),
        )
        state["u"][key] = u
        cmap_name = _HEAT_CMAPS[state["heat_cmap_idx"]]
        colors = _heat_to_colors(u, len(points), cmap_name)
        ps.get_point_cloud(panel).add_color_quantity("heat", colors, enabled=True)

        marker_name = f"source {panel}"
        if ps.has_point_cloud(marker_name):
            ps.remove_point_cloud(marker_name)
        variant_points, _ = variants[key]
        marker_point = variant_points[state["source_index"] : state["source_index"] + 1]
        source_cloud = ps.register_point_cloud(
            marker_name,
            marker_point + panel_offsets[key],
            radius=args.point_radius * 8.0,
        )
        source_cloud.set_color((0.0, 1.0, 1.0))

    def compute_all() -> None:
        print(f"Computing heat fields from source {state['source_index']}...")
        for row in range(3):
            for col in range(3):
                key = (row, col)
                print(f"  [{_panel_name(row, col, point_noise, normal_noise)}]")
                compute_panel(key)
        print("Done")

    def recolor_all() -> None:
        cmap_name = _HEAT_CMAPS[state["heat_cmap_idx"]]
        for key, u in state["u"].items():
            row, col = key
            panel = _panel_name(row, col, point_noise, normal_noise)
            colors = _heat_to_colors(u, len(points), cmap_name)
            ps.get_point_cloud(panel).add_color_quantity("heat", colors, enabled=True)

    def callback() -> None:
        psim.TextUnformatted("Point noise increases left to right")
        psim.TextUnformatted("Normal noise increases top to bottom")
        psim.TextUnformatted(f"Source: {state['source_index']}")
        psim.Separator()

        panel_names = {
            _panel_name(row, col, point_noise, normal_noise): (row, col)
            for row in range(3)
            for col in range(3)
        }
        pick = ps.get_selection()
        if pick.is_hit and pick.structure_name in panel_names:
            psim.TextUnformatted(f"Selected point: {pick.local_index}")
            if psim.Button("Use selected source"):
                state["source_index"] = int(pick.local_index)
                state["computing"] = True

        changed_t, new_t = psim.InputFloat("t_scale", state["t_scale"])
        if changed_t and new_t > 0:
            state["t_scale"] = float(new_t)
            cache.clear()
            state["computing"] = True

        if psim.Button("Compute"):
            state["computing"] = True

        if state["u"]:
            changed_cm, new_idx = psim.Combo(
                "Heat colormap",
                state["heat_cmap_idx"],
                _HEAT_CMAPS,
            )
            if changed_cm:
                state["heat_cmap_idx"] = int(new_idx)
                recolor_all()

        if state["computing"]:
            state["computing"] = False
            compute_all()

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    for row in range(3):
        for col in range(3):
            key = (row, col)
            panel = _panel_name(row, col, point_noise, normal_noise)
            variant_points, _ = variants[key]
            cloud = ps.register_point_cloud(
                panel,
                variant_points + panel_offsets[key],
                radius=args.point_radius,
            )
            cloud.set_color((0.7, 0.7, 0.7))

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
