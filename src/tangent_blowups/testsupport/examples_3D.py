"""
3D geometry examples for tests and demos.

Includes:
- Space curves (regular and self-intersecting)
- Algebraic surfaces (including the Whitney umbrella)
"""

import numpy as np
from .geom_types import ParametricCurve, ParametricSurface
from ..solvers.linalg import normalize_vectors


def _surface_frame(du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Gram-Schmidt an orthonormal tangent frame (..., 3, 2) from raw du, dv."""
    e1 = normalize_vectors(du)
    dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
    e2 = normalize_vectors(dv_perp)
    return np.stack([e1, e2], axis=-1)


def _surface_normal(du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Unit normal (..., 3) from raw du, dv via cross product."""
    return normalize_vectors(np.cross(du, dv))


# -----------------------------------------------------------------------------
# Space Curves (3D)
# -----------------------------------------------------------------------------

def _polyline_curve(vertices: np.ndarray) -> ParametricCurve:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must be shape (M, 3)")
    if verts.shape[0] < 2:
        raise ValueError("vertices must contain at least 2 points")

    # Ensure closed by repeating the first vertex if needed.
    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])

    edges = verts[1:] - verts[:-1]
    edge_dirs = normalize_vectors(edges)
    n_edges = edges.shape[0]

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, float(n_edges))
        idx = np.floor(s).astype(int)
        local = s - idx
        return verts[idx] + edges[idx] * local[..., None]

    def tan(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, float(n_edges))
        idx = np.floor(s).astype(int)
        return edge_dirs[idx]

    return ParametricCurve(position=pos, tangent=tan, normal=None)


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


def mobius_border_trefoil(R: float = 3.0) -> ParametricCurve:
    """
    Boundary curve of the 3-half-turn Mobius band — a trefoil knot in R^3.

    Parametrization (t in [0, 4*pi]):
        x = (R + cos(3t/2)) * cos(t)
        y = (R + cos(3t/2)) * sin(t)
        z = sin(3t/2)
    """

    def pos(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        a = R + np.cos(1.5 * t)
        return np.stack(
            [
                a * np.cos(t),
                a * np.sin(t),
                np.sin(1.5 * t),
            ],
            axis=-1,
        )

    def tan(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        a = R + np.cos(1.5 * t)
        a_prime = -1.5 * np.sin(1.5 * t)
        d = np.stack(
            [
                a_prime * np.cos(t) - a * np.sin(t),
                a_prime * np.sin(t) + a * np.cos(t),
                1.5 * np.cos(1.5 * t),
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

    The curve intersects at the origin for t = 0, pi.
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


def square_loop(side: float = 2.0, z: float = 0.0) -> ParametricCurve:
    """
    Square loop in the XY-plane with sharp corners.
    """
    half = 0.5 * side
    verts = np.array(
        [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ]
    )
    return _polyline_curve(verts)


def triangle_loop(side: float = 2.0, z: float = 0.0) -> ParametricCurve:
    """
    Equilateral triangle in the XY-plane with sharp corners.
    """
    h = np.sqrt(3.0) * 0.5 * side
    verts = np.array(
        [
            [0.0, 2.0 * h / 3.0, z],
            [-0.5 * side, -h / 3.0, z],
            [0.5 * side, -h / 3.0, z],
        ]
    )
    return _polyline_curve(verts)


def cross_curve(scale: float = 1.0) -> ParametricCurve:
    """
    Self-intersecting cross curve with sharp corners.
    """
    verts = np.array(
        [
            [-scale, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, -scale, 0.0],
            [0.0, 0.0, 0.0],
            [-scale, 0.0, 0.0],
        ]
    )
    return _polyline_curve(verts)


def pentagram(scale: float = 1.0, z: float = 0.0) -> ParametricCurve:
    """
    Self-intersecting 5-point star (pentagram) in the XY-plane.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.full_like(angles, z)], axis=-1)
    order = [0, 2, 4, 1, 3]
    verts = ring[order]
    return _polyline_curve(verts)


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
        du = np.stack([scale * v, np.full_like(u, scale), np.zeros_like(u)], axis=-1)
        dv = np.stack([scale * u, np.zeros_like(u), 2.0 * scale * v], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

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
        du = np.stack([np.full_like(u, scale_xy), np.zeros_like(u),
                       scale_z * (3.0 * (u ** 2) - 3.0 * (v ** 2))], axis=-1)
        dv = np.stack([np.zeros_like(u), np.full_like(u, scale_xy),
                       scale_z * (-6.0 * u * v)], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

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


def klein_bottle(radius: float = 2.0, scale: float = 1.0) -> ParametricSurface:
    """
    Klein bottle immersion in R^3 (self-intersecting).

    Parametrization (u, v in [0, 2*pi]):
        A = radius + cos(u/2) * sin(v) - sin(u/2) * sin(2v)
        x = scale * A * cos(u)
        y = scale * A * sin(u)
        z = scale * (sin(u/2) * sin(v) + cos(u/2) * sin(2v))
    """

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu = np.cos(u)
        su = np.sin(u)
        cuh = np.cos(0.5 * u)
        suh = np.sin(0.5 * u)
        sv = np.sin(v)
        s2v = np.sin(2.0 * v)

        a = radius + cuh * sv - suh * s2v
        x = scale * a * cu
        y = scale * a * su
        z = scale * (suh * sv + cuh * s2v)
        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu = np.cos(u)
        su = np.sin(u)
        cuh = np.cos(0.5 * u)
        suh = np.sin(0.5 * u)
        sv = np.sin(v)
        cv = np.cos(v)
        s2v = np.sin(2.0 * v)
        c2v = np.cos(2.0 * v)

        a = radius + cuh * sv - suh * s2v
        dadu = -0.5 * (suh * sv + cuh * s2v)
        dadv = cuh * cv - 2.0 * suh * c2v

        du = np.stack([
            scale * (dadu * cu - a * su),
            scale * (dadu * su + a * cu),
            scale * (0.5 * (cuh * sv - suh * s2v)),
        ], axis=-1)
        dv = np.stack([
            scale * dadv * cu,
            scale * dadv * su,
            scale * (suh * cv + 2.0 * cuh * c2v),
        ], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu = np.cos(u)
        su = np.sin(u)
        cuh = np.cos(0.5 * u)
        suh = np.sin(0.5 * u)
        sv = np.sin(v)
        cv = np.cos(v)
        s2v = np.sin(2.0 * v)
        c2v = np.cos(2.0 * v)

        a = radius + cuh * sv - suh * s2v
        dadu = -0.5 * (suh * sv + cuh * s2v)
        dadv = cuh * cv - 2.0 * suh * c2v

        du = np.stack(
            [
                scale * (dadu * cu - a * su),
                scale * (dadu * su + a * cu),
                scale * (0.5 * (cuh * sv - suh * s2v)),
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                scale * dadv * cu,
                scale * dadv * su,
                scale * (suh * cv + 2.0 * cuh * c2v),
            ],
            axis=-1,
        )
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def torus(R: float = 2.0, r: float = 1.0) -> ParametricSurface:
    """
    Torus of revolution.

    Parametrization (u, v in [0, 2*pi]):
        x = (R + r*cos(v)) * cos(u)
        y = (R + r*cos(v)) * sin(u)
        z = r * sin(v)

    R is the major radius (center of tube to center of torus),
    r is the minor radius (tube radius).
    """

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        a = R + r * cv
        return np.stack([a * cu, a * su, r * sv], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        a = R + r * cv
        du = np.stack([-a * su, a * cu, np.zeros_like(u)], axis=-1)
        dv = np.stack([-r * sv * cu, -r * sv * su, r * cv], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        # Outward normal: (cos(v)*cos(u), cos(v)*sin(u), sin(v))
        return np.stack([cv * cu, cv * su, sv], axis=-1)

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
        du = np.stack([scale_r * np.cos(v), scale_r * np.sin(v), np.full_like(u, scale_z)], axis=-1)
        dv = np.stack([-scale_r * u * np.sin(v), scale_r * u * np.cos(v), np.zeros_like(u)], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

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


def plane_cross(scale: float = 1.0) -> ParametricSurface:
    """
    Two perpendicular planes intersecting along the x-axis (self-intersecting).

    Domain:
      u in [0, 2), v in [0, 1]
    """
    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale
        b = (v - 0.5) * 2.0 * scale

        x = a
        y = np.zeros_like(a)
        z = np.zeros_like(a)

        m0 = face == 0
        m1 = face != 0

        y[m0] = b[m0]
        z[m0] = 0.0

        y[m1] = 0.0
        z[m1] = b[m1]

        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        e1 = np.stack([np.ones_like(v), np.zeros_like(v), np.zeros_like(v)], axis=-1)
        vy = np.where(face == 0, 1.0, 0.0)
        vz = np.where(face != 0, 1.0, 0.0)
        e2 = np.stack([np.zeros_like(v), vy, vz], axis=-1)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        nx = np.zeros_like(v)
        ny = np.zeros_like(v)
        nz = np.zeros_like(v)

        m0 = face == 0
        m1 = face != 0

        nz[m0] = 1.0
        ny[m1] = 1.0

        return normalize_vectors(np.stack([nx, ny, nz], axis=-1))

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def cylinders_tangent(
    radius: float = 1.0,
    half_axis: float = 1.5,
    half_angle: float = 0.5,
) -> ParametricSurface:
    """
    Two cylinders of the same radius with perpendicular axes, tangentially
    touching at the origin.

    Cylinder 1: axis along the x-axis, lying below z = 0.
        Parametrization: (a, r sin b, r(cos b - 1))
        Tangent frame: e1 = (1,0,0), e2 = (0, cos b, -sin b)
        Principal curvatures: kappa_1 = 0 (along e1), kappa_2 = 1/r (along e2)

    Cylinder 2: axis along the y-axis, lying above z = 0.
        Parametrization: (r sin b, a, r(1 - cos b))
        Tangent frame: e1 = (cos b, 0, sin b), e2 = (0, 1, 0)
        Principal curvatures: kappa_1 = 1/r (along e1), kappa_2 = 0 (along e2)

    At the origin both surfaces share the same position, the same tangent
    plane (z = 0), and the same scalar curvature invariants
    H = 1/(2r),  K = 0 everywhere -- but their principal *directions* are
    rotated 90 degrees:

        B_1 = [[0,    0  ],    B_2 = [[1/r,  0],
               [0,  1/r  ]]           [0,    0]]

    Same eigenvalues, different eigenvectors.  No scalar function of
    (kappa_1, kappa_2, H, K) can distinguish them.  The full shape-operator
    matrix -- encoded in the level-1 Grassmannian tangent Q_i -- is required.

    Domain: u in [0, 2), v in [0, 1].
      face 0 (u in [0,1)): Cylinder 1.  local s -> axis a, v -> azimuthal b.
      face 1 (u in [1,2)): Cylinder 2.  local s -> azimuthal b, v -> axis a.

    Args:
        radius:     cylinder radius r.
        half_axis:  half-extent along each cylinder's own axis.
        half_angle: half-extent of the azimuthal sampling window (radians).
    """
    r = float(radius)
    L = float(half_axis)
    A = float(half_angle)

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s    = u - face

        # axis parameter a ∈ (-L, L), azimuthal angle b ∈ (-A, A)
        a = (s - 0.5) * 2.0 * L
        b = (v - 0.5) * 2.0 * A

        x = np.empty_like(a)
        y = np.empty_like(a)
        z = np.empty_like(a)

        m0 = face == 0   # Cylinder 1: axis along x
        x[m0] = a[m0]
        y[m0] = r * np.sin(b[m0])
        z[m0] = r * (np.cos(b[m0]) - 1.0)

        m1 = face == 1   # Cylinder 2: axis along y
        x[m1] = r * np.sin(b[m1])
        y[m1] = a[m1]
        z[m1] = r * (1.0 - np.cos(b[m1]))

        return np.stack([x, y, z], axis=-1)

    def tang(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        b    = (v - 0.5) * 2.0 * A

        m0 = face == 0   # C1: e1 = (1,0,0),          e2 = (0, cos b, -sin b)
        m1 = face == 1   # C2: e1 = (cos b, 0, sin b), e2 = (0, 1, 0)

        e1 = np.stack([
            np.where(m0, 1.0, np.where(m1, np.cos(b), 0.0)),
            np.zeros_like(b),
            np.where(m1, np.sin(b), 0.0),
        ], axis=-1)  # (*shape, 3)

        e2 = np.stack([
            np.zeros_like(b),
            np.where(m0, np.cos(b), np.where(m1, 1.0, 0.0)),
            np.where(m0, -np.sin(b), 0.0),
        ], axis=-1)  # (*shape, 3)

        return np.stack([e1, e2], axis=-1)   # (*shape, 3, 2)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        b    = (v - 0.5) * 2.0 * A

        nx = np.zeros_like(b)
        ny = np.zeros_like(b)
        nz = np.zeros_like(b)

        m0 = face == 0   # C1: N = (0, sin b, cos b)
        ny[m0] = np.sin(b[m0])
        nz[m0] = np.cos(b[m0])

        m1 = face == 1   # C2: N = (-sin b, 0, cos b)
        nx[m1] = -np.sin(b[m1])
        nz[m1] =  np.cos(b[m1])

        return normalize_vectors(np.stack([nx, ny, nz], axis=-1))

    return ParametricSurface(position=pos, tangent=tang, normal=norm)


def plane_paraboloid_tangent(scale_xy: float = 1.0, scale_z: float = 0.25) -> ParametricSurface:
    """
    Plane and paraboloid tangent at the origin.

    Plane: z = 0
    Paraboloid: z = scale_z * (x^2 + y^2)

    Domain:
      u in [0, 2), v in [0, 1]
    """
    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        b = (v - 0.5) * 2.0 * scale_xy

        x = a
        y = b
        z = np.zeros_like(a)

        m1 = face != 0
        z[m1] = scale_z * (a[m1] ** 2 + b[m1] ** 2)

        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        b = (v - 0.5) * 2.0 * scale_xy

        du_z = np.where(face == 0, 0.0, 4.0 * scale_z * scale_xy * a)
        dv_z = np.where(face == 0, 0.0, 4.0 * scale_z * scale_xy * b)

        du = np.stack(
            [
                np.full_like(a, 2.0 * scale_xy),
                np.zeros_like(a),
                du_z,
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                np.zeros_like(a),
                np.full_like(a, 2.0 * scale_xy),
                dv_z,
            ],
            axis=-1,
        )
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        b = (v - 0.5) * 2.0 * scale_xy

        du_z = np.where(face == 0, 0.0, 4.0 * scale_z * scale_xy * a)
        dv_z = np.where(face == 0, 0.0, 4.0 * scale_z * scale_xy * b)

        du = np.stack(
            [
                np.full_like(a, 2.0 * scale_xy),
                np.zeros_like(a),
                du_z,
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                np.zeros_like(a),
                np.full_like(a, 2.0 * scale_xy),
                dv_z,
            ],
            axis=-1,
        )
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def paraboloids_tangent(
    k1: float = 0.25,
    k2: float = 1.0,
    scale_xy: float = 1.0,
) -> ParametricSurface:
    """
    Two cylindrical paraboloids tangent along the y-axis.

    Sheet 0:  z = k1 * x^2
    Sheet 1:  z = k2 * x^2

    Both sheets share position *and* tangent plane along the entire y-axis
    (x = 0), where z = 0 and the tangent plane is the xy-plane.  They differ
    only in curvature (k1 vs k2), so only the level-2 blow-up can metrically
    distinguish them near the tangency line.

    Domain:
      u in [0, 2), v in [0, 1]
      floor(u) selects the sheet (0 or 1).
    """
    curvatures = [float(k1), float(k2)]

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy   # x
        b = (v - 0.5) * 2.0 * scale_xy   # y
        k = np.where(face == 0, curvatures[0], curvatures[1])
        z = k * a ** 2
        return np.stack([a, b, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        k = np.where(face == 0, curvatures[0], curvatures[1])

        du = np.stack([
            np.full_like(a, 2.0 * scale_xy),
            np.zeros_like(a),
            2.0 * k * a * 2.0 * scale_xy,
        ], axis=-1)
        dv = np.stack([
            np.zeros_like(a),
            np.full_like(a, 2.0 * scale_xy),
            np.zeros_like(a),
        ], axis=-1)
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        k = np.where(face == 0, curvatures[0], curvatures[1])

        du = np.stack([
            np.full_like(a, 2.0 * scale_xy),
            np.zeros_like(a),
            2.0 * k * a * 2.0 * scale_xy,
        ], axis=-1)
        dv = np.stack([
            np.zeros_like(a),
            np.full_like(a, 2.0 * scale_xy),
            np.zeros_like(a),
        ], axis=-1)
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def hemisphere_paraboloid_tangent(
    radius: float = 1.0,
    k: float = 1.0,
    scale_xy: float = 0.8,
) -> ParametricSurface:
    """
    Hemisphere and paraboloid tangent at the origin.

    Sheet 0 (hemisphere):  z = R - sqrt(R^2 - x^2 - y^2),  centered at (0,0,R)
    Sheet 1 (paraboloid):  z = k * (x^2 + y^2)

    Both surfaces touch at the origin with tangent plane z = 0.  The
    hemisphere has Gaussian curvature 1/R^2 everywhere; the paraboloid
    has curvature 2k at the origin.  When k != 1/(2R) the curvatures
    differ, so only the level-2 blow-up can separate them near the
    tangency point.

    If k > 1/(2R) the paraboloid is steeper and they intersect along a
    circle, giving a ring of near-tangency.

    Points with x^2 + y^2 >= R^2 (outside the hemisphere domain) are
    returned as NaN and should be filtered.

    Domain:
      u in [0, 2), v in [0, 1]
      floor(u) selects the sheet (0 = hemisphere, 1 = paraboloid).
    """
    R = float(radius)

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        b = (v - 0.5) * 2.0 * scale_xy

        r2 = a ** 2 + b ** 2
        z_hemi = R - np.sqrt(np.maximum(R ** 2 - r2, 0.0))
        z_para = k * r2
        # Outside hemisphere domain -> NaN
        outside = r2 >= R ** 2
        z_hemi = np.where(outside, np.nan, z_hemi)

        z = np.where(face == 0, z_hemi, z_para)
        return np.stack([a, b, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        b = (v - 0.5) * 2.0 * scale_xy

        r2 = a ** 2 + b ** 2
        denom = np.sqrt(np.maximum(R ** 2 - r2, 1e-30))
        dz_da_hemi = a / denom
        dz_db_hemi = b / denom
        outside = r2 >= R ** 2
        dz_da_hemi = np.where(outside, np.nan, dz_da_hemi)
        dz_db_hemi = np.where(outside, np.nan, dz_db_hemi)

        dz_da = np.where(face == 0, dz_da_hemi, 2.0 * k * a)
        dz_db = np.where(face == 0, dz_db_hemi, 2.0 * k * b)

        c = 2.0 * scale_xy
        du = np.stack([np.full_like(a, c), np.zeros_like(a), dz_da * c], axis=-1)
        dv = np.stack([np.zeros_like(a), np.full_like(a, c), dz_db * c], axis=-1)

        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        s = u - face
        a = (s - 0.5) * 2.0 * scale_xy
        b = (v - 0.5) * 2.0 * scale_xy

        r2 = a ** 2 + b ** 2
        denom = np.sqrt(np.maximum(R ** 2 - r2, 1e-30))
        dz_da_hemi = a / denom
        dz_db_hemi = b / denom
        outside = r2 >= R ** 2
        dz_da_hemi = np.where(outside, np.nan, dz_da_hemi)
        dz_db_hemi = np.where(outside, np.nan, dz_db_hemi)

        dz_da = np.where(face == 0, dz_da_hemi, 2.0 * k * a)
        dz_db = np.where(face == 0, dz_db_hemi, 2.0 * k * b)

        c = 2.0 * scale_xy
        du = np.stack([np.full_like(a, c), np.zeros_like(a), dz_da * c], axis=-1)
        dv = np.stack([np.zeros_like(a), np.full_like(a, c), dz_db * c], axis=-1)

        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def nested_bowls(
    k1: float = 0.3,
    k2: float = 0.6,
    r_max: float = 0.8,
) -> ParametricSurface:
    """
    Two rotationally-symmetric paraboloid bowls on a shared disk.

    Sheet 0:  z = k1 * (x^2 + y^2)
    Sheet 1:  z = k2 * (x^2 + y^2)

    Both are sampled on the disk of radius ``r_max``.  Because the
    curvatures are close and the domain is compact, the two surfaces
    remain close and nearly parallel throughout.  Position and normal
    are similar everywhere, so bilateral and level-1 kernels struggle
    to separate them.  Only the level-2 blow-up (which encodes
    curvature) cleanly distinguishes the two sheets.

    Uses polar parametrisation (area-uniform in ``v``) to avoid
    wasting samples outside a circle:

        theta = 2 * pi * (u - floor(u)),   r = r_max * sqrt(v)

    Domain:
      u in [0, 2), v in [0, 1]
      floor(u) selects the sheet (0 or 1).
    """
    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        theta = 2.0 * np.pi * (u - face)
        r = r_max * np.sqrt(np.clip(v, 0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        k = np.where(face == 0, k1, k2)
        z = k * (x ** 2 + y ** 2)
        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        theta = 2.0 * np.pi * (u - face)
        r = r_max * np.sqrt(np.clip(v, 0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        k = np.where(face == 0, k1, k2)

        ct, st = np.cos(theta), np.sin(theta)
        denom_v = np.maximum(np.sqrt(np.clip(v, 1e-12, 1)), 1e-6)
        dr_dv = r_max / (2.0 * denom_v)

        # d/d(theta) direction
        dx_dt = -r * st * 2.0 * np.pi
        dy_dt = r * ct * 2.0 * np.pi
        dz_dt = k * 2.0 * (x * dx_dt + y * dy_dt)
        du_vec = np.stack([dx_dt, dy_dt, dz_dt], axis=-1)

        # d/dv direction
        dx_dv = ct * dr_dv
        dy_dv = st * dr_dv
        dz_dv = k * 2.0 * (x * dx_dv + y * dy_dv)
        dv_vec = np.stack([dx_dv, dy_dv, dz_dv], axis=-1)

        e1 = normalize_vectors(du_vec)
        dv_perp = dv_vec - np.sum(dv_vec * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.clip(np.floor(u).astype(int), 0, 1)
        theta = 2.0 * np.pi * (u - face)
        r = r_max * np.sqrt(np.clip(v, 0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        k = np.where(face == 0, k1, k2)

        ct, st = np.cos(theta), np.sin(theta)
        denom_v = np.maximum(np.sqrt(np.clip(v, 1e-12, 1)), 1e-6)
        dr_dv = r_max / (2.0 * denom_v)

        dx_dt = -r * st * 2.0 * np.pi
        dy_dt = r * ct * 2.0 * np.pi
        dz_dt = k * 2.0 * (x * dx_dt + y * dy_dt)
        du_vec = np.stack([dx_dt, dy_dt, dz_dt], axis=-1)

        dx_dv = ct * dr_dv
        dy_dv = st * dr_dv
        dz_dv = k * 2.0 * (x * dx_dv + y * dy_dv)
        dv_vec = np.stack([dx_dv, dy_dv, dz_dv], axis=-1)

        n = np.cross(du_vec, dv_vec)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def tangent_spheres(radius: float = 1.0) -> ParametricSurface:
    """
    Two spheres tangent at the origin.

    Centers: (-radius, 0, 0) and (+radius, 0, 0)

    Domain:
      u in [0, 2), v in [0, 1]
    """
    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        theta = 2.0 * np.pi * s
        phi = np.pi * v

        cx = np.where(face == 0, -radius, radius)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x = cx + radius * cos_theta * sin_phi
        y = radius * sin_theta * sin_phi
        z = radius * cos_phi

        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        theta = 2.0 * np.pi * s
        phi = np.pi * v

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        du = np.stack(
            [
                -2.0 * np.pi * radius * sin_theta * sin_phi,
                2.0 * np.pi * radius * cos_theta * sin_phi,
                np.zeros_like(sin_phi),
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                np.pi * radius * cos_theta * cos_phi,
                np.pi * radius * sin_theta * cos_phi,
                -np.pi * radius * sin_phi,
            ],
            axis=-1,
        )
        e1 = normalize_vectors(du)
        dv_perp = dv - np.sum(dv * e1, axis=-1, keepdims=True) * e1
        e2 = normalize_vectors(dv_perp)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 1)
        s = u - face
        theta = 2.0 * np.pi * s
        phi = np.pi * v

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        du = np.stack(
            [
                -2.0 * np.pi * radius * sin_theta * sin_phi,
                2.0 * np.pi * radius * cos_theta * sin_phi,
                np.zeros_like(sin_phi),
            ],
            axis=-1,
        )
        dv = np.stack(
            [
                np.pi * radius * cos_theta * cos_phi,
                np.pi * radius * sin_theta * cos_phi,
                -np.pi * radius * sin_phi,
            ],
            axis=-1,
        )
        n = np.cross(du, dv)
        return normalize_vectors(n)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def cube_surface(scale: float = 1.0) -> ParametricSurface:
    """
    Axis-aligned cube surface with sharp edges and corners.

    Domain:
      u in [0, 6), v in [0, 1]
    """
    half = float(scale)

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 5)
        s = u - face

        x = np.zeros_like(s)
        y = np.zeros_like(s)
        z = np.zeros_like(s)

        y0 = -half + 2.0 * half * s
        z0 = -half + 2.0 * half * v

        # +X face
        m0 = face == 0
        x[m0] = half
        y[m0] = y0[m0]
        z[m0] = z0[m0]

        # -X face
        m1 = face == 1
        x[m1] = -half
        y[m1] = -y0[m1]
        z[m1] = z0[m1]

        # +Y face
        m2 = face == 2
        y[m2] = half
        x[m2] = -y0[m2]
        z[m2] = z0[m2]

        # -Y face
        m3 = face == 3
        y[m3] = -half
        x[m3] = y0[m3]
        z[m3] = z0[m3]

        # +Z face
        m4 = face == 4
        z[m4] = half
        x[m4] = y0[m4]
        y[m4] = z0[m4]

        # -Z face
        m5 = face == 5
        z[m5] = -half
        x[m5] = y0[m5]
        y[m5] = -z0[m5]

        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 5)

        tx = np.zeros_like(v)
        ty = np.zeros_like(v)
        tz = np.zeros_like(v)

        m0 = face == 0  # +X, u-dir along +Y
        tx[m0], ty[m0], tz[m0] = 0.0, 1.0, 0.0

        m1 = face == 1  # -X, u-dir along -Y
        tx[m1], ty[m1], tz[m1] = 0.0, -1.0, 0.0

        m2 = face == 2  # +Y, u-dir along -X
        tx[m2], ty[m2], tz[m2] = -1.0, 0.0, 0.0

        m3 = face == 3  # -Y, u-dir along +X
        tx[m3], ty[m3], tz[m3] = 1.0, 0.0, 0.0

        m4 = face == 4  # +Z, u-dir along +X
        tx[m4], ty[m4], tz[m4] = 1.0, 0.0, 0.0

        m5 = face == 5  # -Z, u-dir along +X
        tx[m5], ty[m5], tz[m5] = 1.0, 0.0, 0.0

        e1 = np.stack([tx, ty, tz], axis=-1)

        vx = np.zeros_like(v)
        vy = np.zeros_like(v)
        vz = np.zeros_like(v)

        # v-direction: +Z for faces 0-3, +Y for face 4, -Y for face 5
        vz[face <= 3] = 1.0
        vy[face == 4] = 1.0
        vy[face == 5] = -1.0

        e2 = np.stack([vx, vy, vz], axis=-1)
        return np.stack([e1, e2], axis=-1)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 5)

        nx = np.zeros_like(v)
        ny = np.zeros_like(v)
        nz = np.zeros_like(v)

        nx[face == 0] = 1.0
        nx[face == 1] = -1.0
        ny[face == 2] = 1.0
        ny[face == 3] = -1.0
        nz[face == 4] = 1.0
        nz[face == 5] = -1.0

        return normalize_vectors(np.stack([nx, ny, nz], axis=-1))

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def tetrahedron_surface(scale: float = 1.0) -> ParametricSurface:
    """
    Regular tetrahedron surface with sharp edges and corners.

    Domain:
      u in [0, 4), v in [0, 1]
    """
    verts = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=float,
    )
    verts *= scale / np.linalg.norm(verts[0])

    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=int,
    )

    face_normals = []
    face_tangent_frames = []
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        e0 = v1 - v0
        e1_vec = v2 - v0
        n = normalize_vectors(np.cross(e0, e1_vec))
        face_normals.append(n)
        t0 = normalize_vectors(e0)
        e1_perp = e1_vec - np.dot(e1_vec, t0) * t0
        t1 = normalize_vectors(e1_perp)
        face_tangent_frames.append(np.stack([t0, t1], axis=-1))  # (3, 2)
    face_normals = np.asarray(face_normals).reshape(-1, 3)
    face_tangent_frames = np.array(face_tangent_frames)  # (4, 3, 2)

    def _barycentric(u_local: np.ndarray, v_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = u_local
        t = v_local
        over = (s + t) > 1.0
        s = np.where(over, 1.0 - s, s)
        t = np.where(over, 1.0 - t, t)
        return s, t

    def pos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 3)
        s = u - face
        t = v
        s, t = _barycentric(s, t)

        x = np.zeros_like(s)
        y = np.zeros_like(s)
        z = np.zeros_like(s)

        for fi in range(4):
            m = face == fi
            if not np.any(m):
                continue
            v0, v1, v2 = verts[faces[fi][0]], verts[faces[fi][1]], verts[faces[fi][2]]
            s_m = np.atleast_1d(s[m])
            t_m = np.atleast_1d(t[m])
            p = v0 + s_m[:, None] * (v1 - v0) + t_m[:, None] * (v2 - v0)
            x[m], y[m], z[m] = p[:, 0], p[:, 1], p[:, 2]

        return np.stack([x, y, z], axis=-1)

    def tan(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 3)
        return face_tangent_frames[face]  # (..., 3, 2)

    def norm(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        face = np.floor(u).astype(int)
        face = np.clip(face, 0, 3)
        return face_normals[face]

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


# -----------------------------------------------------------------------------
# Non-Orientable Surfaces
# -----------------------------------------------------------------------------
# Parametrizations adapted from the EuMaT "Non-Orientable Surfaces" reference
# (https://eumat.sourceforge.net/Programs/Examples/Non-Orientable%20Surfaces.html).
# Each factory returns a ParametricSurface with analytic first derivatives.
# Natural (u, v) domains are documented per surface; the caller chooses the
# sampling domain via the sampling strategy (consistent with klein_bottle/torus).


def _mobius_band(R: float, k: int) -> ParametricSurface:
    """Shared implementation for the k-half-turn Mobius band."""
    half_k = 0.5 * k

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        chv = np.cos(half_k * v)
        shv = np.sin(half_k * v)
        cv = np.cos(v)
        sv = np.sin(v)
        a = R + u * chv
        p = np.stack([a * cv, a * sv, u * shv], axis=-1)
        du = np.stack([chv * cv, chv * sv, shv], axis=-1)
        a_v = -half_k * u * shv
        dv_ = np.stack(
            [a_v * cv - a * sv, a_v * sv + a * cv, half_k * u * chv],
            axis=-1,
        )
        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def mobius_band_1(R: float = 3.0) -> ParametricSurface:
    """
    Mobius band with 1 half-turn (classical Mobius strip).

    Parametrization (u in [-1, 1], v in [0, 2*pi]):
        x = (R + u*cos(v/2)) * cos(v)
        y = (R + u*cos(v/2)) * sin(v)
        z = u * sin(v/2)
    """
    return _mobius_band(R, 1)


def mobius_band_3(R: float = 3.0) -> ParametricSurface:
    """
    Mobius band with 3 half-turns (trefoil-boundary Mobius band).

    Parametrization (u in [-1, 1], v in [0, 2*pi]):
        x = (R + u*cos(3v/2)) * cos(v)
        y = (R + u*cos(3v/2)) * sin(v)
        z = u * sin(3v/2)
    """
    return _mobius_band(R, 3)


def steiner_crosscap() -> ParametricSurface:
    """
    Steiner crosscap (real projective plane immersion, full domain).

    Parametrization (u in [0, 2*pi], v in [0, pi/2]):
        x = 0.5 * cos(u) * sin(2v)
        y = 0.5 * sin(u) * sin(2v)
        z = 0.5 * (cos(v)^2 - sin(v)^2 * cos(u)^2)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)
        x = 0.5 * cu * s2v
        y = 0.5 * su * s2v
        z = 0.5 * (cv * cv - sv * sv * cu * cu)
        p = np.stack([x, y, z], axis=-1)
        du = np.stack(
            [
                -0.5 * su * s2v,
                0.5 * cu * s2v,
                sv * sv * cu * su,
            ],
            axis=-1,
        )
        dv_ = np.stack(
            [
                cu * c2v,
                su * c2v,
                -0.5 * s2v * (1.0 + cu * cu),
            ],
            axis=-1,
        )
        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def steiner_crosscap_cut() -> ParametricSurface:
    """
    Steiner crosscap with half-u domain that exposes the self-intersection curve.

    Same parametrization as `steiner_crosscap`; natural domain
    (u in [0, pi], v in [0, pi/2]).
    """
    return steiner_crosscap()


def steiner_roman() -> ParametricSurface:
    """
    Steiner Roman surface.

    Parametrization (u in [0, pi], v in [0, pi]):
        x = 0.5 * sin(2u) * sin(v)^2
        y = 0.5 * sin(u) * sin(2v)
        z = 0.5 * cos(u) * sin(2v)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        c2u, s2u = np.cos(2.0 * u), np.sin(2.0 * u)
        cv, sv = np.cos(v), np.sin(v)
        c2v, s2v = np.cos(2.0 * v), np.sin(2.0 * v)
        p = np.stack(
            [
                0.5 * s2u * sv * sv,
                0.5 * su * s2v,
                0.5 * cu * s2v,
            ],
            axis=-1,
        )
        du = np.stack(
            [
                c2u * sv * sv,
                0.5 * cu * s2v,
                -0.5 * su * s2v,
            ],
            axis=-1,
        )
        dv_ = np.stack(
            [
                0.5 * s2u * s2v,
                su * c2v,
                cu * c2v,
            ],
            axis=-1,
        )
        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def busser_decic() -> ParametricSurface:
    """
    Busser decic surface (projective-plane variant).

    Parametrization (u in [0, 2*pi], v in [-pi/2, pi/2]):
        x = cos(u) * cos(v)
        y = sin(u) * cos(v)
        z = sin(2v) * cos(3u/2)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)
        c3u2 = np.cos(1.5 * u)
        s3u2 = np.sin(1.5 * u)
        p = np.stack([cu * cv, su * cv, s2v * c3u2], axis=-1)
        du = np.stack(
            [
                -su * cv,
                cu * cv,
                -1.5 * s2v * s3u2,
            ],
            axis=-1,
        )
        dv_ = np.stack(
            [
                -cu * sv,
                -su * sv,
                2.0 * c2v * c3u2,
            ],
            axis=-1,
        )
        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def boy_surface(
    m: int = 3,
    c0: float = 2.0,
    c1: float = 0.5,
    c2: float = 0.5,
    r: float = 4.0,
) -> ParametricSurface:
    """
    Boy surface — immersion of RP^2 in R^3.

    Parametrization (u in [0, 2*pi], v in [0, pi]):
        a     = c0 + c1*sin(2m*v - pi/3) + c2*sin(m*v - pi/6)
        b     = c0 + c1*sin(2m*v - pi/3) - c2*sin(m*v - pi/6)
        alpha = (pi/8) * sin(m*v)
        x1    = (a^2 - b^2)/sqrt(a^2+b^2) + a*cos(u) - b*sin(u)
        z1    = sqrt(a^2+b^2) + a*cos(u) + b*sin(u)
        X = r*(x1*cos(v) - z1*sin(alpha)*sin(v))
        Y = r*(x1*sin(v) + z1*sin(alpha)*cos(v))
        Z = r* z1*cos(alpha)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)

        phi1 = 2.0 * m * v - np.pi / 3.0
        phi2 = m * v - np.pi / 6.0
        s_phi1, c_phi1 = np.sin(phi1), np.cos(phi1)
        s_phi2, c_phi2 = np.sin(phi2), np.cos(phi2)

        common = c0 + c1 * s_phi1
        a = common + c2 * s_phi2
        b = common - c2 * s_phi2

        a_v = 2.0 * m * c1 * c_phi1 + m * c2 * c_phi2
        b_v = 2.0 * m * c1 * c_phi1 - m * c2 * c_phi2

        s2 = a * a + b * b
        s = np.sqrt(s2)
        diff = a * a - b * b

        x1 = diff / s + a * cu - b * su
        z1 = s + a * cu + b * su

        aa_v = a * a_v
        bb_v = b * b_v
        s_v = (aa_v + bb_v) / s
        diff_v = 2.0 * (aa_v - bb_v)
        quot_v = diff_v / s - diff * s_v / s2

        x1_v = quot_v + a_v * cu - b_v * su
        z1_v = s_v + a_v * cu + b_v * su

        x1_u = -a * su - b * cu
        z1_u = -a * su + b * cu

        alpha = (np.pi / 8.0) * np.sin(m * v)
        alpha_v = (m * np.pi / 8.0) * np.cos(m * v)
        sa = np.sin(alpha)
        ca = np.cos(alpha)

        X = r * (x1 * cv - z1 * sa * sv)
        Y = r * (x1 * sv + z1 * sa * cv)
        Z = r * z1 * ca
        p = np.stack([X, Y, Z], axis=-1)

        X_u = r * (x1_u * cv - z1_u * sa * sv)
        Y_u = r * (x1_u * sv + z1_u * sa * cv)
        Z_u = r * z1_u * ca
        du = np.stack([X_u, Y_u, Z_u], axis=-1)

        X_v = r * (
            x1_v * cv
            - x1 * sv
            - z1_v * sa * sv
            - z1 * ca * alpha_v * sv
            - z1 * sa * cv
        )
        Y_v = r * (
            x1_v * sv
            + x1 * cv
            + z1_v * sa * cv
            + z1 * ca * alpha_v * cv
            - z1 * sa * sv
        )
        Z_v = r * (z1_v * ca - z1 * sa * alpha_v)
        dv_ = np.stack([X_v, Y_v, Z_v], axis=-1)

        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def boy_surface_5(
    c0: float = 2.0, c1: float = 0.5, c2: float = 0.5, r: float = 4.0
) -> ParametricSurface:
    """Boy-like surface with m=5 (5 half-turns). See `boy_surface`."""
    return boy_surface(m=5, c0=c0, c1=c1, c2=c2, r=r)


def etruscan_venus(
    a: float = 4.0,
    b: float = 2.0,
    c: float = 2.0,
    e: float = 3.0,
    g: float = 12.0,
) -> ParametricSurface:
    """
    Etruscan Venus — a Klein-bottle-like immersion.

    Parametrization (u, v in (~0, 2*pi)):
        w   = b*sin(u) + e
        dx  = a*(cos(u) - cos(c*u))
        dy  = -g*sin(u)
        rxy = sqrt(dx^2 + dy^2)
        X = a*sin(u) - b*sin(c*u) - dy*w*cos(v)/rxy
        Y = g*cos(u) + dx*w*cos(v)/rxy
        Z = w*sin(v)

    Note: rxy -> 0 at u = 0 — sample strictly away from the seam.
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        ccu = np.cos(c * u)
        scu = np.sin(c * u)
        cv, sv = np.cos(v), np.sin(v)

        w = b * su + e
        dx = a * (cu - ccu)
        dy = -g * su
        rxy2 = dx * dx + dy * dy
        rxy = np.sqrt(rxy2)

        F = w * cv / rxy

        X = a * su - b * scu - dy * F
        Y = g * cu + dx * F
        Z = w * sv
        p = np.stack([X, Y, Z], axis=-1)

        w_u = b * cu
        dx_u = a * (-su + c * scu)
        dy_u = -g * cu
        rxy_u = (dx * dx_u + dy * dy_u) / rxy
        F_u = cv * (w_u / rxy - w * rxy_u / rxy2)

        X_u = a * cu - b * c * ccu - dy_u * F - dy * F_u
        Y_u = -g * su + dx_u * F + dx * F_u
        Z_u = w_u * sv
        du = np.stack([X_u, Y_u, Z_u], axis=-1)

        X_v = dy * w * sv / rxy
        Y_v = -dx * w * sv / rxy
        Z_v = w * cv
        dv_ = np.stack([X_v, Y_v, Z_v], axis=-1)

        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def jeener_bonan(s: float = 2.0, t: float = 4.0) -> ParametricSurface:
    """
    Jeener-Bonan 3-neck Klein bottle.

    Parametrization (u, v in [0, 2*pi]):
        w = ((s+1)/4) * cos((s+1)u + pi/t) + sqrt(2)
        X = s*cos(u) + cos(su) - w*sin(((s-1)/2)u)*cos(v)
        Y = s*sin(u) - sin(su) - w*cos(((s-1)/2)u)*cos(v)
        Z = w*sin(v)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        k1 = s + 1.0
        k2 = 0.5 * (s - 1.0)
        phase = k1 * u + np.pi / t
        cphase = np.cos(phase)
        sphase = np.sin(phase)

        w = (k1 / 4.0) * cphase + np.sqrt(2.0)
        w_u = -(k1 * k1 / 4.0) * sphase

        cu, su = np.cos(u), np.sin(u)
        csu, ssu = np.cos(s * u), np.sin(s * u)
        c_k2u, s_k2u = np.cos(k2 * u), np.sin(k2 * u)
        cv, sv = np.cos(v), np.sin(v)

        X = s * cu + csu - w * s_k2u * cv
        Y = s * su - ssu - w * c_k2u * cv
        Z = w * sv
        p = np.stack([X, Y, Z], axis=-1)

        X_u = (
            -s * su
            - s * ssu
            - (w_u * s_k2u + w * k2 * c_k2u) * cv
        )
        Y_u = (
            s * cu
            - s * csu
            - (w_u * c_k2u - w * k2 * s_k2u) * cv
        )
        Z_u = w_u * sv
        du = np.stack([X_u, Y_u, Z_u], axis=-1)

        X_v = w * s_k2u * sv
        Y_v = w * c_k2u * sv
        Z_v = w * cv
        dv_ = np.stack([X_v, Y_v, Z_v], axis=-1)

        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def jeener_bonan_2cc() -> ParametricSurface:
    """Jeener-Bonan with s=0, t=2 — surface with 2 crosscaps."""
    return jeener_bonan(s=0.0, t=2.0)


def busser_nonorientable() -> ParametricSurface:
    """
    Busser non-orientable surface.

    Parametrization (u in [0, 2*pi], v in [-pi/2, pi/2]):
        x = 0.5 * cos(u) * cos(v)
        y = 0.5 * sin(u) * cos(v)
        z = 0.5 * sin(v) - cos(u)^2 * sin(v/2)^3
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        s2u = np.sin(2.0 * u)
        svh = np.sin(0.5 * v)
        cvh = np.cos(0.5 * v)
        svh2 = svh * svh
        svh3 = svh2 * svh

        p = np.stack(
            [
                0.5 * cu * cv,
                0.5 * su * cv,
                0.5 * sv - cu * cu * svh3,
            ],
            axis=-1,
        )
        du = np.stack(
            [
                -0.5 * su * cv,
                0.5 * cu * cv,
                s2u * svh3,
            ],
            axis=-1,
        )
        dv_ = np.stack(
            [
                -0.5 * cu * sv,
                -0.5 * su * sv,
                0.5 * cv - 1.5 * cu * cu * svh2 * cvh,
            ],
            axis=-1,
        )
        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def petit_russian_hat() -> ParametricSurface:
    """
    Petit Russian Hat.

    Parametrization (u in [0, 2*pi], v in [0, pi]):
        x = (1 + sin(v)) * cos(u)
        y = 2 * sin(u) * sin(v)
        z = sin(2v)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        s2v, c2v = np.sin(2.0 * v), np.cos(2.0 * v)
        zero = np.zeros_like(cu * cv)
        p = np.stack(
            [
                (1.0 + sv) * cu,
                2.0 * su * sv,
                s2v + zero,
            ],
            axis=-1,
        )
        du = np.stack(
            [
                -(1.0 + sv) * su,
                2.0 * cu * sv,
                zero,
            ],
            axis=-1,
        )
        dv_ = np.stack(
            [
                cv * cu,
                2.0 * su * cv,
                2.0 * c2v + zero,
            ],
            axis=-1,
        )
        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def petit_russian_hat_cut() -> ParametricSurface:
    """Petit Russian Hat restricted to u in [pi, 2*pi]. See `petit_russian_hat`."""
    return petit_russian_hat()


def banchoff_klein(m: int = 1, a: float = 2.0, b: float = 1.0) -> ParametricSurface:
    """
    Banchoff Klein bottle immersion.

    Parametrization (u, v in [0, 2*pi]):
        T = a + b*cos(u)*cos(m*v/2) - (b/2)*sin(2u)*sin(m*v/2)
        X = T * cos(v)
        Y = T * sin(v)
        Z = b*cos(u)*sin(m*v/2) + (b/2)*sin(2u)*cos(m*v/2)
    """

    def raw(u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        cu, su = np.cos(u), np.sin(u)
        s2u, c2u = np.sin(2.0 * u), np.cos(2.0 * u)
        mhv = 0.5 * m * v
        cmh, smh = np.cos(mhv), np.sin(mhv)
        cv, sv = np.cos(v), np.sin(v)

        T = a + b * cu * cmh - 0.5 * b * s2u * smh
        Z = b * cu * smh + 0.5 * b * s2u * cmh

        p = np.stack([T * cv, T * sv, Z], axis=-1)

        T_u = -b * su * cmh - b * c2u * smh
        Z_u = -b * su * smh + b * c2u * cmh

        X_u = T_u * cv
        Y_u = T_u * sv
        du = np.stack([X_u, Y_u, Z_u], axis=-1)

        T_v = -0.5 * b * m * cu * smh - 0.25 * b * m * s2u * cmh
        Z_v = 0.5 * b * m * cu * cmh - 0.25 * b * m * s2u * smh

        X_v = T_v * cv - T * sv
        Y_v = T_v * sv + T * cv
        dv_ = np.stack([X_v, Y_v, Z_v], axis=-1)

        return p, du, dv_

    def pos(u, v):
        return raw(u, v)[0]

    def tan(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_frame(du, dv_)

    def norm(u, v):
        _, du, dv_ = raw(u, v)
        return _surface_normal(du, dv_)

    return ParametricSurface(position=pos, tangent=tan, normal=norm)


def banchoff_klein_3(a: float = 2.0, b: float = 1.0) -> ParametricSurface:
    """Banchoff Klein bottle with m=3 (3 half-turns). See `banchoff_klein`."""
    return banchoff_klein(m=3, a=a, b=b)


__all__ = [
    "helix",
    "trefoil_knot",
    "mobius_border_trefoil",
    "space_lissajous",
    "figure8_space",
    "square_loop",
    "triangle_loop",
    "cross_curve",
    "pentagram",
    "whitney_umbrella",
    "monkey_saddle",
    "klein_bottle",
    "cone",
    "plane_cross",
    "cylinders_tangent",
    "plane_paraboloid_tangent",
    "tangent_spheres",
    "cube_surface",
    "tetrahedron_surface",
    "mobius_band_1",
    "mobius_band_3",
    "steiner_crosscap",
    "steiner_crosscap_cut",
    "steiner_roman",
    "busser_decic",
    "boy_surface",
    "boy_surface_5",
    "etruscan_venus",
    "jeener_bonan",
    "jeener_bonan_2cc",
    "busser_nonorientable",
    "petit_russian_hat",
    "petit_russian_hat_cut",
    "banchoff_klein",
    "banchoff_klein_3",
]
