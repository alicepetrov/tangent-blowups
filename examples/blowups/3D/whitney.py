import numpy as np
import matplotlib.pyplot as plt

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import Sample, UniformSurface, sample, whitney_umbrella
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


def build_whitney_sample(
    *,
    nu: int = 80,
    nv: int = 80,
    u_bounds: tuple[float, float] = (-2.0, 2.0),
    v_bounds: tuple[float, float] = (-2.0, 2.0),
    scale: float = 1.0,
) -> Sample:
    """
    Sample a Whitney umbrella surface and return a flattened Sample.
    """
    surface = whitney_umbrella(scale=scale)
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
            cmap="magma",
            alpha=0.9,
        )
        ax.set_title(label)
        _set_axes_equal_3d(ax, pts)
        if (idx % cols == cols - 1) or (idx == total - 1):
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Nash Embedding Components (R^3 x G(1,3) -> R^12)")
    plt.tight_layout()
    plt.show()


def main():
    base = build_whitney_sample(nu=80, nv=80, u_bounds=(-2.0, 2.0), v_bounds=(-2.0, 2.0), scale=1.0)

    # Lift using normals (Gauss map), which is isometric to tangent planes for G(2,3)
    points = np.asarray(base.points, dtype=float)
    normals = normalize_vectors(np.asarray(base.normals, dtype=float))
    lifted = BlownUpSample.from_normals(points, normals)

    # 1) Visualize normals over the surface
    visualize_quiver(lifted, scale=0.2, downsample=12)

    # 2) Visualize the Grassmannian fiber components (P_ij) over R^3
    visualize_components(lifted, components="all", downsample=4)

    # 3) Explicit embedding visualization
    visualize_embedding_components(lifted, alpha=1.0, downsample=4, cols=4)


if __name__ == "__main__":
    main()
