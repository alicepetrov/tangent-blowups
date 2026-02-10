"""
3D geometry examples for tests and demos.

Includes:
- Space curves (regular and self-intersecting)
- Algebraic surfaces (including the Whitney umbrella)
"""

import numpy as np
from geom_types import ParametricCurve, ParametricSurface
from solvers.linalg import normalize_vectors

# -----------------------------------------------------------------------------
# Space Curves (3D)
# -----------------------------------------------------------------------------

def helix(radius: float = 1.0, pitch: float = 0.25) -> ParametricCurve:
    """
    Circular helix in 3D.
    Domain: t in [0, 2*pi] (or any interval for multiple turns).
    """

    def pos(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.stack(
            [
                radius * np.cos(t),
                radius * np.sin(t),
                pitch * t,
            ],
            axis=-1,
        )

    def tan(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        d = np.stack(
            [
                -radius * np.sin(t),
                radius * np.cos(t),
                np.full_like(t, pitch),
            ],
            axis=-1,
        )
        return normalize_vectors(d)

    def norm(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        n = np.stack(
            [
                -np.cos(t),
                -np.sin(t),
                np.zeros_like(t),
            ],
            axis=-1,
        )
        return normalize_vectors(n)

    return ParametricCurve(position=pos, tangent=tan, normal=norm)


def trefoil_knot(scale: float = 1.0, z_scale: float = 1.0) -> ParametricCurve:
    """
    Standard trefoil knot embedding (non-self-intersecting in 3D).
    Domain: t in [0, 2*pi]
    """

    def pos(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.stack(
            [
                scale * (np.sin(t) + 2.0 * np.sin(2.0 * t)),
                scale * (np.cos(t) - 2.0 * np.cos(2.0 * t)),
                z_scale * (-np.sin(3.0 * t)),
            ],
            axis=-1,
        )

    def tan(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        d = np.stack(
            [
                scale * (np.cos(t) + 4.0 * np.cos(2.0 * t)),
                scale * (-np.sin(t) + 4.0 * np.sin(2.0 * t)),
                z_scale * (-3.0 * np.cos(3.0 * t)),
            ],
            axis=-1,
        )
        return normalize_vectors(d)

    return ParametricCurve(position=pos, tangent=tan, normal=None)


def space_lissajous(
    a: float = 3.0,
    b: float = 2.0,
    c: float = 5.0,
    delta_x: float = np.pi / 2.0,
    delta_y: float = 0.0,
    scale: float = 1.0,
) -> ParametricCurve:
    """
    Generic 3D Lissajous space curve.
    Domain: t in [0, 2*pi]
    """

    def pos(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.stack(
            [
                scale * np.sin(a * t + delta_x),
                scale * np.sin(b * t + delta_y),
                scale * np.sin(c * t),
            ],
            axis=-1,
        )

    def tan(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        d = np.stack(
            [
                scale * a * np.cos(a * t + delta_x),
                scale * b * np.cos(b * t + delta_y),
                scale * c * np.cos(c * t),
            ],
            axis=-1,
        )
        return normalize_vectors(d)

    return ParametricCurve(position=pos, tangent=tan, normal=None)


# -----------------------------------------------------------------------------
# Self-Intersecting Space Curves (3D)
# -----------------------------------------------------------------------------

def figure8_space(scale_xy: float = 1.0, scale_z: float = 0.5) -> ParametricCurve:
    """
    Self-intersecting 3D figure-8 style curve.

    x = scale_xy * sin(t)
    y = scale_xy * sin(t) * cos(t)
    z = scale_z  * sin(2t)

    The curve intersects at the origin for t = 0, pi, 2*pi.
    Domain: t in [0, 2*pi]
    """

    def pos(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        s = np.sin(t)
        return np.stack(
            [
                scale_xy * s,
                scale_xy * s * np.cos(t),
                scale_z * np.sin(2.0 * t),
            ],
            axis=-1,
        )

    def tan(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        d = np.stack(
            [
                scale_xy * np.cos(t),
                scale_xy * np.cos(2.0 * t),
                2.0 * scale_z * np.cos(2.0 * t),
            ],
            axis=-1,
        )
        return normalize_vectors(d)

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=None,
    )


# -----------------------------------------------------------------------------
# Algebraic Surfaces in 3D
# -----------------------------------------------------------------------------

def whitney_umbrella(scale: float = 1.0) -> ParametricSurface:
    """
    Whitney umbrella (pinch point).

    Parametrization:
        x = scale * u * v
        y = scale * u
        z = scale * v^2

    Implicit form (up to scaling): x^2 = y^2 z
    """

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.stack(
            [
                scale * u * v,
                scale * u,
                scale * (v ** 2),
            ],
            axis=-1,
        )

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.stack(
            [
                scale * v,
                np.full_like(u, scale),
                np.zeros_like(u),
            ],
            axis=-1,
        )
        return normalize_vectors(du)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.stack(
            [
                scale * v,
                np.full_like(u, scale),
                np.zeros_like(u),
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                scale * u,
                np.zeros_like(u),
                2.0 * scale * v,
            ],
            axis=-1,
        )
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(
        position=pos,
        tangent=tan,
        normal=norm
    )


def monkey_saddle(scale_xy: float = 1.0, scale_z: float = 0.25) -> ParametricSurface:
    """
    Monkey saddle polynomial surface.

    Parametrization:
        x = scale_xy * u
        y = scale_xy * v
        z = scale_z * (u^3 - 3uv^2)
    """

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.stack(
            [
                scale_xy * u,
                scale_xy * v,
                scale_z * (u ** 3 - 3.0 * u * (v ** 2)),
            ],
            axis=-1,
        )

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.stack(
            [
                np.full_like(u, scale_xy),
                np.zeros_like(u),
                scale_z * (3.0 * (u ** 2) - 3.0 * (v ** 2)),
            ],
            axis=-1,
        )
        return normalize_vectors(du)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.stack(
            [
                np.full_like(u, scale_xy),
                np.zeros_like(u),
                scale_z * (3.0 * (u ** 2) - 3.0 * (v ** 2)),
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                np.zeros_like(u),
                np.full_like(u, scale_xy),
                scale_z * (-6.0 * u * v),
            ],
            axis=-1,
        )
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def cone(scale_r: float = 1.0, scale_z: float = 1.0) -> ParametricSurface:
    """
    Double cone parameterization.

    Parametrization:
        x = scale_r * u * cos(v)
        y = scale_r * u * sin(v)
        z = scale_z * u
    """

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        return np.stack(
            [
                scale_r * u * np.cos(v),
                scale_r * u * np.sin(v),
                scale_z * u,
            ],
            axis=-1,
        )

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.stack(
            [
                scale_r * np.cos(v),
                scale_r * np.sin(v),
                np.full_like(u, scale_z),
            ],
            axis=-1,
        )
        return normalize_vectors(du)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.stack(
            [
                scale_r * np.cos(v),
                scale_r * np.sin(v),
                np.full_like(u, scale_z),
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                -scale_r * u * np.sin(v),
                scale_r * u * np.cos(v),
                np.zeros_like(u),
            ],
            axis=-1,
        )
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


__all__ = [
    "helix",
    "trefoil_knot",
    "space_lissajous",
    "figure8_space",
    "whitney_umbrella",
    "monkey_saddle",
    "cone",
]
