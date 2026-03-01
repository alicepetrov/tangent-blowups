"""
This module provides explicit 2D curve examples for testing tangent blowup handling.
"""

import numpy as np
from .geom_types import ParametricCurve

def circle(radius: float = 1.0) -> ParametricCurve:
    """
    Returns the ParametricCurve for a circle centered at (0,0).
    Domain: t in [0, 2*pi]
    """
    
    def pos(t: np.ndarray) -> np.ndarray:
        return np.stack([
            radius * np.cos(t), 
            radius * np.sin(t)
        ], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        # Normalized tangent: (-sin(t), cos(t))
        return np.stack([
            -np.sin(t), 
            np.cos(t)
        ], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        # Outward normal: (cos(t), sin(t))
        return np.stack([
            np.cos(t), 
            np.sin(t)
        ], axis=-1)

    return ParametricCurve(
        position=pos, 
        tangent=tan, 
        normal=norm,
    )

def tangent_circles(radius: float = 1.0) -> ParametricCurve:
    """
    Two tangent circles ("kissing circles") sharing a common tangent point.

    Centers: (0, +radius) and (0, -radius)
    Tangent point: (0, 0) with horizontal tangent direction.

    Domain: t in [0, 2), each unit interval traces one circle.
    """

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 2.0)
        m0 = s < 1.0
        theta = np.where(
            m0,
            2.0 * np.pi * s - 0.5 * np.pi,
            0.5 * np.pi - 2.0 * np.pi * (s - 1.0),
        )
        cy = np.where(m0, radius, -radius)
        x = radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        return np.stack([x, y], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 2.0)
        m0 = s < 1.0
        theta = np.where(
            m0,
            2.0 * np.pi * s - 0.5 * np.pi,
            0.5 * np.pi - 2.0 * np.pi * (s - 1.0),
        )
        sign = np.where(m0, 1.0, -1.0)
        tx = sign * (-np.sin(theta))
        ty = sign * (np.cos(theta))
        return np.stack([tx, ty], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        tvec = tan(t)
        return np.stack(
            [
                tvec[..., 1],
                -tvec[..., 0],
            ],
            axis=-1,
        )

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=norm,
    )

def tangent_circle_line(radius: float = 1.0, line_length: float = 4.0) -> ParametricCurve:
    """
    Circle tangent to a straight line.

    Circle center: (0, +radius) with radius = radius
    Line: y = 0 (x-axis)
    Tangency point: (0, 0) with horizontal tangent direction.

    Domain: t in [0, 2), each unit interval traces one component.
    """

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 2.0)
        m0 = s < 1.0
        theta = 2.0 * np.pi * s - 0.5 * np.pi

        x = np.zeros_like(s, dtype=float)
        y = np.zeros_like(s, dtype=float)

        x[m0] = radius * np.cos(theta[m0])
        y[m0] = radius + radius * np.sin(theta[m0])

        s1 = s[~m0] - 1.0
        x[~m0] = -0.5 * line_length + line_length * s1
        y[~m0] = 0.0

        return np.stack([x, y], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 2.0)
        m0 = s < 1.0
        theta = 2.0 * np.pi * s - 0.5 * np.pi

        tx = np.zeros_like(s, dtype=float)
        ty = np.zeros_like(s, dtype=float)

        tx[m0] = -np.sin(theta[m0])
        ty[m0] = np.cos(theta[m0])

        tx[~m0] = 1.0
        ty[~m0] = 0.0

        return np.stack([tx, ty], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        tvec = tan(t)
        return np.stack(
            [
                tvec[..., 1],
                -tvec[..., 0],
            ],
            axis=-1,
        )

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=norm,
    )


def tangent_sine_line(
    amplitude: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 2.0 * np.pi,
    phase: float = 0.0,
) -> ParametricCurve:
    """
    Sine wave tangent to a horizontal line.

    Sine: y = amplitude * sin(x + phase)
    Line: y = amplitude
    Tangency point: x = pi/2 - phase (assuming it lies in [x_min, x_max])

    Domain: t in [0, 2), each unit interval traces one component.
    """

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 2.0)
        m0 = s < 1.0

        x = np.zeros_like(s, dtype=float)
        y = np.zeros_like(s, dtype=float)

        x[m0] = x_min + (x_max - x_min) * s[m0]
        y[m0] = amplitude * np.sin(x[m0] + phase)

        s1 = s[~m0] - 1.0
        x[~m0] = x_min + (x_max - x_min) * s1
        y[~m0] = amplitude

        return np.stack([x, y], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 2.0)
        m0 = s < 1.0

        tx = np.zeros_like(s, dtype=float)
        ty = np.zeros_like(s, dtype=float)

        x = x_min + (x_max - x_min) * s
        dy_dx = amplitude * np.cos(x + phase)

        tx[m0] = 1.0
        ty[m0] = dy_dx[m0]

        tx[~m0] = 1.0
        ty[~m0] = 0.0

        denom = np.sqrt(tx * tx + ty * ty)
        denom[denom == 0] = 1.0

        return np.stack([tx / denom, ty / denom], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        tvec = tan(t)
        return np.stack(
            [
                tvec[..., 1],
                -tvec[..., 0],
            ],
            axis=-1,
        )

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=norm,
    )


def figure8(scale: float = 1.0) -> ParametricCurve:
    """
    Returns the ParametricCurve for a self-intersecting figure-8 curve.
    
    Parametrization (Lemniscate of Gerono):
      x = scale * sin(t)
      y = scale * sin(t) * cos(t)
    
    Domain: t in [0, 2*pi]
    Singularity: The curve self-intersects at the origin at t=pi (and endpoints).
    """

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.sin(t)
        c = np.cos(t)
        return np.stack([
            scale * s,
            scale * s * c
        ], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        dx = scale * np.cos(t)
        dy = scale * np.cos(2.0 * t)
        
        # Normalize
        denom = np.sqrt(dx*dx + dy*dy)
        denom[denom == 0] = 1.0 
        
        return np.stack([
            dx / denom,
            dy / denom
        ], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        tvec = tan(t)
        # CW rotation (t_y, -t_x): outward-pointing convention
        return np.stack([
            tvec[..., 1],
            -tvec[..., 0]
        ], axis=-1)

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=norm,
    )

def lissajous(a: float = 3.0, b: float = 2.0, delta: float = np.pi / 2, scale: float = 1.0) -> ParametricCurve:
    """
    Returns the ParametricCurve for a Lissajous curve.
    
    Parametrization:
      x = scale * sin(a*t + delta)
      y = scale * sin(b*t)
    
    Domain: t in [0, 2*pi] (Assuming a, b are integers, otherwise domain varies)
    
    Common examples:
      - a=1, b=2, delta=pi/2 : Similar to the Figure-8 (Lemniscate of Gerono)
      - a=3, b=2, delta=pi/2 : A more complex knot
    """
    
    def pos(t: np.ndarray) -> np.ndarray:
        return np.stack([
            scale * np.sin(a * t + delta),
            scale * np.sin(b * t)
        ], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        # Derivative of sin(kt) is k*cos(kt)
        dx = scale * a * np.cos(a * t + delta)
        dy = scale * b * np.cos(b * t)
        
        # Normalize
        denom = np.sqrt(dx*dx + dy*dy)
        
        # Avoid division by zero if velocity vanishes (rare for standard params)
        denom[denom == 0] = 1.0 
        
        return np.stack([
            dx / denom,
            dy / denom
        ], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        tvec = tan(t)
        # CW rotation (t_y, -t_x): outward-pointing convention
        return np.stack([
            tvec[..., 1],
            -tvec[..., 0]
        ], axis=-1)

    return ParametricCurve(
        position=pos, 
        tangent=tan, 
        normal=norm,
    )


def square(side: float = 2.0) -> ParametricCurve:
    """
    Axis-aligned square centered at the origin with sharp corners.

    Parametrization:
      t in [0, 4), each unit interval traces one edge.
    """
    half = 0.5 * side

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 4.0)
        x = np.zeros_like(s, dtype=float)
        y = np.zeros_like(s, dtype=float)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = (s >= 2.0) & (s < 3.0)
        m3 = s >= 3.0

        x[m0] = -half + side * s[m0]
        y[m0] = -half

        x[m1] = half
        y[m1] = -half + side * (s[m1] - 1.0)

        x[m2] = half - side * (s[m2] - 2.0)
        y[m2] = half

        x[m3] = -half
        y[m3] = half - side * (s[m3] - 3.0)

        return np.stack([x, y], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 4.0)
        tx = np.zeros_like(s, dtype=float)
        ty = np.zeros_like(s, dtype=float)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = (s >= 2.0) & (s < 3.0)
        m3 = s >= 3.0

        tx[m0], ty[m0] = 1.0, 0.0
        tx[m1], ty[m1] = 0.0, 1.0
        tx[m2], ty[m2] = -1.0, 0.0
        tx[m3], ty[m3] = 0.0, -1.0

        return np.stack([tx, ty], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 4.0)
        nx = np.zeros_like(s, dtype=float)
        ny = np.zeros_like(s, dtype=float)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = (s >= 2.0) & (s < 3.0)
        m3 = s >= 3.0

        nx[m0], ny[m0] = 0.0, -1.0
        nx[m1], ny[m1] = 1.0, 0.0
        nx[m2], ny[m2] = 0.0, 1.0
        nx[m3], ny[m3] = -1.0, 0.0

        return np.stack([nx, ny], axis=-1)

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=norm,
    )


def triangle(side: float = 2.0) -> ParametricCurve:
    """
    Equilateral triangle centered at the origin with sharp corners.

    Parametrization:
      t in [0, 3), each unit interval traces one edge.
    """
    h = np.sqrt(3.0) * 0.5 * side
    v0 = np.array([0.0, 2.0 * h / 3.0])
    v1 = np.array([-0.5 * side, -h / 3.0])
    v2 = np.array([0.5 * side, -h / 3.0])

    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    t0 = e0 / np.linalg.norm(e0)
    t1 = e1 / np.linalg.norm(e1)
    t2 = e2 / np.linalg.norm(e2)

    def pos(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 3.0)
        x = np.zeros_like(s, dtype=float)
        y = np.zeros_like(s, dtype=float)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = s >= 2.0

        s0 = s[m0]
        s1 = s[m1]
        s2 = s[m2]

        p0 = v1 + e0 * s0[:, None]
        p1 = v2 + e1 * (s1 - 1.0)[:, None]
        p2 = v0 + e2 * (s2 - 2.0)[:, None]

        if p0.size:
            x[m0], y[m0] = p0[:, 0], p0[:, 1]
        if p1.size:
            x[m1], y[m1] = p1[:, 0], p1[:, 1]
        if p2.size:
            x[m2], y[m2] = p2[:, 0], p2[:, 1]

        return np.stack([x, y], axis=-1)

    def tan(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 3.0)
        tx = np.zeros_like(s, dtype=float)
        ty = np.zeros_like(s, dtype=float)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = s >= 2.0

        tx[m0], ty[m0] = t0[0], t0[1]
        tx[m1], ty[m1] = t1[0], t1[1]
        tx[m2], ty[m2] = t2[0], t2[1]

        return np.stack([tx, ty], axis=-1)

    def norm(t: np.ndarray) -> np.ndarray:
        s = np.mod(t, 3.0)
        nx = np.zeros_like(s, dtype=float)
        ny = np.zeros_like(s, dtype=float)

        m0 = s < 1.0
        m1 = (s >= 1.0) & (s < 2.0)
        m2 = s >= 2.0

        n0 = np.array([t0[1], -t0[0]])
        n1 = np.array([t1[1], -t1[0]])
        n2 = np.array([t2[1], -t2[0]])

        nx[m0], ny[m0] = n0[0], n0[1]
        nx[m1], ny[m1] = n1[0], n1[1]
        nx[m2], ny[m2] = n2[0], n2[1]

        return np.stack([nx, ny], axis=-1)

    return ParametricCurve(
        position=pos,
        tangent=tan,
        normal=norm,
    )
