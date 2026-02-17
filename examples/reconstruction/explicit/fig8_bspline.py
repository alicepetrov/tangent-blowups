from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

from tangent_blowups.geometry.grassmann import BlownUpSample
from tangent_blowups.reconstruction.explicit.bspline import fit_bspline_curve
from tangent_blowups.solvers.linalg import normalize_vectors
from tangent_blowups.testsupport import UniformCurve, figure8, sample


def build_oriented_fig8(
    *,
    n: int = 1200,
    scale: float = 2.0,
    seed: int = 7,
    jitter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample an oriented figure-8 curve.

    Returns:
        points: (N, 2)
        tangents: (N, 2) unit vectors
        params: (N,) curve parameter values
    """
    curve = figure8(scale=scale)
    strategy = UniformCurve(n=n, t_min=0.0, t_max=2.0 * np.pi, endpoint=False)
    s = sample(curve, strategy, with_tangents=True, with_normals=False)

    points = np.asarray(s.points, dtype=float)
    tangents = np.asarray(s.tangents, dtype=float)
    params = np.asarray(s.params, dtype=float)

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=jitter, size=points.shape)

    tangents = normalize_vectors(tangents)
    return points, tangents, params


def plot_fit(
    points: np.ndarray,
    blown_up_curve: np.ndarray,
    plain_curve: np.ndarray,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.scatter(points[:, 0], points[:, 1], s=10, color="slateblue", alpha=0.5, label="samples")
    ax.plot(
        blown_up_curve[:, 0],
        blown_up_curve[:, 1],
        color="black",
        linewidth=2.0,
        label="B-spline (blown-up)",
    )
    ax.plot(
        plain_curve[:, 0],
        plain_curve[:, 1],
        color="crimson",
        linewidth=1.6,
        linestyle="--",
        label="B-spline (spatial)",
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    points, tangents, params = build_oriented_fig8(
        n=1200,
        scale=2.0,
        seed=7,
        jitter=0.2,
    )

    lifted = BlownUpSample.from_tangents(points, tangents)
    result = fit_bspline_curve(
        lifted,
        alpha=1.0,
        degree=3,
        smoothing=100.0,
        periodic=True,
        parameter=params,
        n_eval=800,
    )

    tck_plain, u_plain = splprep(
        points.T,
        u=params,
        k=3,
        s=100.0,
        per=True,
    )
    u_min = float(np.min(u_plain))
    u_max = float(np.max(u_plain))
    plain_u = np.linspace(u_min, u_max, 800)
    plain_curve = np.stack(splev(plain_u, tck_plain), axis=-1)

    plot_fit(
        points,
        result.curve_spatial,
        plain_curve,
        title="Figure-8 B-spline Fit (Projected)",
    )


if __name__ == "__main__":
    main()
