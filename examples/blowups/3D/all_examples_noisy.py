import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import (
    Sample,
    UniformCurve,
    UniformSurface,
    add_point_noise,
    cone,
    cross_curve,
    cube_surface,
    figure8_space,
    helix,
    klein_bottle,
    jitter_normals,
    jitter_tangents,
    monkey_saddle,
    pentagram,
    plane_cross,
    sample,
    square_loop,
    space_lissajous,
    tetrahedron_surface,
    trefoil_knot,
    triangle_loop,
    whitney_umbrella,
)
from tangent_blowups.viz.nash_coordinates import visualize_components, visualize_quiver


def _flatten_sample(s: Sample) -> Sample:
    points = np.asarray(s.points, dtype=float)
    if points.ndim > 2:
        points = points.reshape(-1, points.shape[-1])

    tangents = s.tangents
    if tangents is not None:
        tangents = np.asarray(tangents, dtype=float)
        if tangents.ndim > 2:
            tangents = tangents.reshape(-1, tangents.shape[-1])

    normals = s.normals
    if normals is not None:
        normals = np.asarray(normals, dtype=float)
        if normals.ndim > 2:
            normals = normals.reshape(-1, normals.shape[-1])

    mask = s.singular_mask
    if mask is not None and mask.ndim > 1:
        mask = mask.reshape(-1)

    params = s.params
    if isinstance(params, tuple):
        params = tuple(np.asarray(p).reshape(-1) for p in params)
    else:
        params = np.asarray(params).reshape(-1) if np.asarray(params).ndim > 1 else params

    return Sample(
        points=points,
        params=params,
        tangents=tangents,
        normals=normals,
        singular_mask=mask,
        singular_indices=s.singular_indices,
    )


def build_curve_sample(
    curve,
    *,
    n: int = 1500,
    t_max: float = 2.0 * np.pi,
) -> Sample:
    strategy = UniformCurve(n=n, t_min=0.0, t_max=t_max, endpoint=False)
    s = sample(curve, strategy, with_tangents=True, with_normals=False)
    s.tangents = normalize_vectors(np.asarray(s.tangents, dtype=float))
    return s


def build_surface_sample(
    surface,
    *,
    nu: int = 70,
    nv: int = 70,
    u_bounds: tuple[float, float] = (-2.0, 2.0),
    v_bounds: tuple[float, float] = (-2.0, 2.0),
) -> Sample:
    strategy = UniformSurface(nu=nu, nv=nv, u_bounds=u_bounds, v_bounds=v_bounds)
    s = sample(surface, strategy, with_tangents=False, with_normals=True)
    return _flatten_sample(s)


def _set_axes_equal_3d(ax, points: np.ndarray):
    pts = np.asarray(points).reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def visualize_embedding_components(
    sample: BlownUpSample,
    *,
    alpha: float = 1.0,
    downsample: int = 1,
    cols: int = 4,
):
    """
    Visualize the Nash embedding by coloring the 3D cloud with each component.

    Uses BlownUpSample.embedding_vector(alpha=...), which returns:
        [x, y, z, sqrt(alpha/2) * P00, ..., sqrt(alpha/2) * P22]
    """
    pts = sample.spatial[::downsample]
    embed = sample.embedding_vector(alpha=alpha)[::downsample]
    n = sample.n

    if n == 3:
        base_labels = ["x", "y", "z"]
    elif n == 2:
        base_labels = ["x", "y"]
    else:
        base_labels = [f"x{i}" for i in range(n)]

    labels = base_labels + [f"P{i}{j}" for i in range(n) for j in range(n)]

    total = len(labels)
    rows = int(np.ceil(total / cols))
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))

    for idx, label in enumerate(labels):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        sc = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=embed[:, idx],
            s=6,
            cmap="viridis",
            alpha=0.9,
        )
        ax.set_title(label)
        _set_axes_equal_3d(ax, pts)
        if (idx % cols == cols - 1) or (idx == total - 1):
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Nash Embedding Components (R^3 x G(1,3) -> R^12)")
    plt.tight_layout()
    plt.show()


def _lift_and_plot_curve(curve_label: str, scenario: str, sample_in: Sample):
    print(f"\n=== Curve: {curve_label} | {scenario} ===")
    points = np.asarray(sample_in.points, dtype=float)
    tangents = normalize_vectors(np.asarray(sample_in.tangents, dtype=float))

    lifted = BlownUpSample.from_tangents(points, tangents)
    visualize_quiver(lifted, scale=0.2, downsample=12)
    visualize_components(lifted, components="all", downsample=4)
    visualize_embedding_components(lifted, alpha=1.0, downsample=4, cols=4)


def _lift_and_plot_surface(surface_label: str, scenario: str, sample_in: Sample):
    print(f"\n=== Surface: {surface_label} | {scenario} ===")
    points = np.asarray(sample_in.points, dtype=float)
    normals = normalize_vectors(np.asarray(sample_in.normals, dtype=float))

    lifted = BlownUpSample.from_normals(points, normals)
    visualize_quiver(lifted, scale=0.2, downsample=12)
    visualize_components(lifted, components="all", downsample=4)
    visualize_embedding_components(lifted, alpha=1.0, downsample=4, cols=4)


def main():
    curves = [
        ("Helix", helix(radius=2.0, pitch=0.5), 2.0 * np.pi),
        ("Figure8 Space", figure8_space(scale_xy=2.0, scale_z=1.2), 2.0 * np.pi),
        ("Space Lissajous", space_lissajous(a=3, b=4, c=5, delta_x=np.pi / 2, delta_y=np.pi / 4, scale=2.5), 2.0 * np.pi),
        ("Trefoil Knot", trefoil_knot(scale=1.0, z_scale=1.0), 2.0 * np.pi),
        ("Square Loop", square_loop(side=3.0), 4.0),
        ("Triangle Loop", triangle_loop(side=3.0), 3.0),
        ("Cross Curve", cross_curve(scale=1.5), 6.0),
        ("Pentagram", pentagram(scale=2.0), 5.0),
    ]
    surfaces = [
        ("Whitney Umbrella", whitney_umbrella(scale=1.0), (-2.0, 2.0), (-2.0, 2.0)),
        ("Monkey Saddle", monkey_saddle(scale_xy=1.0, scale_z=0.25), (-2.0, 2.0), (-2.0, 2.0)),
        ("Klein Bottle", klein_bottle(radius=2.0, scale=1.0), (0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)),
        ("Cone", cone(scale_r=1.0, scale_z=1.0), (0.0, 2.0), (0.0, 2.0 * np.pi)),
        ("Plane Cross", plane_cross(scale=1.2), (0.0, 2.0), (0.0, 1.0)),
        ("Cube Surface", cube_surface(scale=1.0), (0.0, 6.0), (0.0, 1.0)),
        ("Tetrahedron Surface", tetrahedron_surface(scale=1.2), (0.0, 4.0), (0.0, 1.0)),
    ]

    for idx, (name, curve, t_max) in enumerate(curves):
        base = build_curve_sample(curve, n=1500, t_max=t_max)
        rng_points = np.random.default_rng(100 + idx)
        rng_tangents = np.random.default_rng(200 + idx)

        noisy_points = add_point_noise(base, sigma=0.04, rng=rng_points)
        noisy_tangents = jitter_tangents(base, sigma=0.35, rng=rng_tangents, renormalize=True)
        noisy_both = jitter_tangents(noisy_points, sigma=0.35, rng=rng_tangents, renormalize=True)

        _lift_and_plot_curve(name, "Baseline (clean tangents)", base)
        _lift_and_plot_curve(name, "Noisy points (clean tangents)", noisy_points)
        _lift_and_plot_curve(name, "Jittered tangents (clean points)", noisy_tangents)
        _lift_and_plot_curve(name, "Noisy points + jittered tangents", noisy_both)

    for idx, (name, surface, u_bounds, v_bounds) in enumerate(surfaces):
        base = build_surface_sample(surface, nu=70, nv=70, u_bounds=u_bounds, v_bounds=v_bounds)
        rng_points = np.random.default_rng(300 + idx)
        rng_normals = np.random.default_rng(400 + idx)

        noisy_points = add_point_noise(base, sigma=0.04, rng=rng_points)
        noisy_normals = jitter_normals(base, sigma=0.3, rng=rng_normals, renormalize=True)
        noisy_both = jitter_normals(noisy_points, sigma=0.3, rng=rng_normals, renormalize=True)

        _lift_and_plot_surface(name, "Baseline (clean normals)", base)
        _lift_and_plot_surface(name, "Noisy points (clean normals)", noisy_points)
        _lift_and_plot_surface(name, "Jittered normals (clean points)", noisy_normals)
        _lift_and_plot_surface(name, "Noisy points + jittered normals", noisy_both)


if __name__ == "__main__":
    main()
