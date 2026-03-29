"""
Pan Pipes Lifted Heat Method Example (Product Kernel)
------------------------------------------------------
Compute lifted heat-method geodesic distances on the pan pipes point cloud
using a fixed-bandwidth product kernel with independent spatial (sigma_x)
and angular (sigma_u) bandwidths.

The product kernel is:

    W_ij = exp(-||x_i - x_j||^2 / sigma_x^2)
         * exp(-||P_i - P_j||_F^2 / sigma_u^2)

- Smaller sigma_x  -> tighter spatial neighborhoods
- Smaller sigma_u  -> more emphasis on angular (normal) separation
- Larger  sigma_u  -> more tolerant of normal differences
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from tangent_blowups.io.load import load_pointcloud
from tangent_blowups.pointcloud import lifted_heat_method
from tangent_blowups.pointcloud.geodesic_heat import (
    _normals_to_tangent_frames,
    _auto_estimate_product_bandwidths,
)
from tangent_blowups.geometry.iterated_grassmann import BlowUpLevel
from tangent_blowups.solvers.linalg import normalize_vectors


def _default_pan_pipes_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "thingi10k_pointcloud" / "pan_pipes.npz"


def load_pan_pipes_points_and_normals(
    path: str | Path | None = None,
    *,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        path = _default_pan_pipes_path()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pan pipes data not found: {path}")

    sample = load_pointcloud(path)
    if sample.normals is None:
        raise ValueError("Pan pipes point cloud is missing normals.")

    points = np.asarray(sample.points, dtype=float)
    normals = np.asarray(sample.normals, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])
    if normals.ndim > 2:
        normals = normals.reshape(-1, normals.shape[-1])

    valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    norms = np.linalg.norm(normals, axis=1)
    valid &= norms > normal_eps

    if not np.all(valid):
        dropped = int(np.sum(~valid))
        print(f"Dropping {dropped} samples with invalid/degenerate normals.")

    points = points[valid]
    normals = normalize_vectors(normals[valid])
    return points, normals


def _build_level(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    alpha: float,
    k: int,
) -> BlowUpLevel:
    """Build a level-1 BlowUpLevel from points and surface normals."""
    frames = _normals_to_tangent_frames(normals)
    level0 = BlowUpLevel.from_point_tangents(points, frames)
    return level0.lift(alpha=alpha, k=k, lam=1e-3)


def _compute_and_display(
    level: BlowUpLevel,
    points: np.ndarray,
    source_index: int,
    state: dict,
    *,
    k: int,
    sigma_x: float | None,
    sigma_u: float | None,
    t_scale: float,
) -> None:
    """Run the heat method and update the polyscope display."""
    # Resolve auto bandwidths so heat method and component detection agree
    sx, su = sigma_x, sigma_u
    if sx is None or su is None:
        auto_sx, auto_su = _auto_estimate_product_bandwidths(level, k)
        if sx is None:
            sx = auto_sx
        if su is None:
            su = auto_su
    print(f"Bandwidths: sigma_x={sx:.4f}, sigma_u={su:.4f}")

    dist, u, _, _ = lifted_heat_method(
        level,
        source_index=source_index,
        k=k,
        kernel="product",
        sigma_x=sx,
        sigma_u=su,
        t_scale=t_scale,
        return_intermediate=True,
    )
    dist = dist - dist[source_index]
    dist = np.maximum(dist, 0.0)

    # Identify reachable points from the heat diffusion: points where
    # the heat u is negligible were never reached by the source.
    u_thresh = np.max(np.abs(u)) * 1e-6
    mask = np.abs(u) > u_thresh
    print(f"Reachable (from heat): {mask.sum()} / {len(points)} points")

    dist_reach = dist[mask]

    # Clamp within-component range for display (Poisson blow-up at weak links)
    if dist_reach.size > 0 and dist_reach.max() > 0:
        p99 = float(np.percentile(dist_reach[dist_reach > 0], 99))
        dist_reach = np.minimum(dist_reach, p99)

    from matplotlib import colormaps
    cmap = colormaps["magma"]
    dmax = float(dist_reach.max()) if dist_reach.size > 0 and dist_reach.max() > 0 else 1.0

    # --- Smooth color view ---
    colors = np.full((len(points), 3), 0.7)  # gray for unreachable
    if dist_reach.size > 0:
        colors[mask] = cmap(dist_reach / dmax)[:, :3]

    # --- Level-set (contour band) colors ---
    # Alternate between colormap color and darkened version for contrast
    n_bands = 15
    band_colors = np.full((len(points), 3), 0.7)
    if dist_reach.size > 0:
        band_width = dmax / n_bands
        bands = np.floor(dist_reach / band_width).astype(int)
        bands = np.clip(bands, 0, n_bands - 1)
        base = cmap(bands / (n_bands - 1))[:, :3]
        # Darken odd bands to create visible contour lines
        darken = np.where(bands % 2 == 1, 0.4, 1.0)[:, np.newaxis]
        band_colors[mask] = base * darken

    # Store both for toggling; display smooth by default
    state["smooth_colors"] = colors
    state["band_colors"] = band_colors
    state["show_bands"] = False

    cloud = ps.get_point_cloud("pan_pipes")
    cloud.add_color_quantity("geodesic", colors, enabled=True)

    # Source marker
    if ps.has_point_cloud("source"):
        ps.remove_point_cloud("source")
    src = ps.register_point_cloud(
        "source", points[source_index : source_index + 1], radius=0.005,
    )
    src.set_color((0.0, 1.0, 1.0))


def main():
    data_path = _default_pan_pipes_path()

    k = 30
    alpha = 1.0
    sigma_x = None   # Auto-estimate from median k-NN spatial distance
    sigma_u = None   # Auto-estimate from median k-NN angular distance
    t_scale = 20.0

    points, normals = load_pan_pipes_points_and_normals(data_path)
    print(f"Loaded {points.shape[0]} points from {data_path}")

    level = _build_level(points, normals, alpha=alpha, k=k)
    print(f"Built level-1 BlowUpLevel: N={level.N}, D={level.D}, d={level.d}")

    # -- Mutable state for the UI callback --
    state = {
        "source_index": None,
        "computing": False,
        "show_bands": False,
        "smooth_colors": None,
        "band_colors": None,
    }

    def callback():
        psim.TextUnformatted("Ctrl+click a point to select source")
        psim.Separator()

        # Check for a picked point
        pick = ps.get_selection()
        have = pick.is_hit and pick.structure_name == "pan_pipes"

        if have:
            picked = pick.local_index
            psim.TextUnformatted(f"Selected point: {picked}  "
                                 f"({points[picked].round(2)})")
        elif state["source_index"] is not None:
            idx = state["source_index"]
            psim.TextUnformatted(f"Source: {idx}  ({points[idx].round(2)})")

        # Button to compute geodesics from picked point
        if have:
            if psim.Button("Compute geodesics from selection"):
                state["source_index"] = pick.local_index
                state["computing"] = True

        # Toggle between smooth and level-set views
        if state["smooth_colors"] is not None:
            changed, new_val = psim.Checkbox("Level sets", state["show_bands"])
            if changed:
                state["show_bands"] = new_val
                cloud = ps.get_point_cloud("pan_pipes")
                c = state["band_colors"] if new_val else state["smooth_colors"]
                cloud.add_color_quantity("geodesic", c, enabled=True)

        if state["computing"]:
            state["computing"] = False
            idx = state["source_index"]
            print(f"Computing geodesics from point {idx}...")
            _compute_and_display(
                level, points, idx, state,
                k=k, sigma_x=sigma_x, sigma_u=sigma_u, t_scale=t_scale,
            )

    ps.init()
    cloud = ps.register_point_cloud("pan_pipes", points, radius=0.001)
    cloud.set_color((0.7, 0.7, 0.7))
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
