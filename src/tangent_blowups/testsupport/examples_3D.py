"""
3D geometry examples for tests and demos.

Includes:
- Space curves (regular and self-intersecting)
- Algebraic surfaces (including the Whitney umbrella)
"""

import numpy as np
from .geom_types import ParametricCurve, ParametricSurface
from ..solvers.linalg import normalize_vectors

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


__all__ = [
    "helix",
    "trefoil_knot",
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
]
