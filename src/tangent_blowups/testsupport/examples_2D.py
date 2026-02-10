"""
This module provides explicit 2D curve examples for testing tangent blowup handling.
"""

import numpy as np
from geom_types import ParametricCurve


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
        return np.stack([
            -tvec[..., 1],
            tvec[..., 0]
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
        # Rotate 90 degrees counter-clockwise: (x, y) -> (-y, x)
        return np.stack([
            -tvec[..., 1],
            tvec[..., 0]
        ], axis=-1)

    return ParametricCurve(
        position=pos, 
        tangent=tan, 
        normal=norm,
    )