"""
Multidimensional Visualization
------------------------------
Visualizes the fiber bundle R^n x G(k, n) by projecting points to their 
canonical spatial coordinates (R^n) and using color to display the 
components of the Grassmannian fiber (elements of the Projector P).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Registers 3D projection
from typing import Optional, Union, Tuple

from ..geometry.grassmann import BlownUpSample

def _setup_figure(spatial_dim: int, n_subplots: int = 1, cols: int = 3, figsize_base: int = 4):
    """
    Creates a figure and a list of axes (2D or 3D) automatically.
    """
    rows = int(np.ceil(n_subplots / cols))
    fig = plt.figure(figsize=(figsize_base * cols, figsize_base * rows))
    
    axes = []
    for i in range(n_subplots):
        if spatial_dim == 3:
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_aspect('equal')
        axes.append(ax)
        
    return fig, axes

def _plot_cloud(ax, points: np.ndarray, color_vals: np.ndarray, title: str, s: float = 2.0, alpha: float = 0.8):
    """
    Backend-agnostic scatter plot (handles 2D vs 3D logic).
    """
    spatial_dim = points.shape[1]
    
    # Create scatter
    if spatial_dim == 3:
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=color_vals, s=s, cmap='viridis', alpha=alpha)
        
        # 3D Equal Aspect Ratio Hack
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])
    else:
        sc = ax.scatter(points[:, 0], points[:, 1], 
                       c=color_vals, s=s, cmap='viridis', alpha=alpha)
    
    ax.set_title(title)
    return sc

def visualize_components(
    sample: BlownUpSample,
    components: Optional[Union[str, Tuple[int, int]]] = 'all',
    downsample: int = 1,
    shared_color_scale: bool = False,
):
    """
    Visualizes the lifted cloud.
    
    - 'all': Plots an (n x n) grid showing every component of the Projector P.
    - (i, j): Plots a single view colored by P[i, j].
    - 'trace': Plots the trace (should be constant k, useful sanity check).
    """
    points = sample.spatial[::downsample]
    # Get (N, n, n) projectors
    P = sample.projectors[::downsample]
    
    n = sample.n
    
    # Mode 1: Plot Specific Component
    if isinstance(components, tuple):
        i, j = components
        fig, axes = _setup_figure(n, n_subplots=1)
        ax = axes[0]
        
        vals = P[:, i, j]
        sc = _plot_cloud(ax, points, vals, title=f"Component $P_{{{i}{j}}}$")
        plt.colorbar(sc, ax=ax)
        plt.show()
        return

    # Mode 2: Plot All Components (Grid)
    if components == 'all':
        # We only strictly need the upper triangle for symmetric P, 
        # but the full grid is often easier to read mentally.
        n_plots = n * n
        fig, axes = _setup_figure(n, n_subplots=n_plots, cols=n)

        vmin = vmax = None
        if shared_color_scale:
            # Shared color scale across all components
            vmin = np.nanmin(P)
            vmax = np.nanmax(P)
        
        for idx, ax in enumerate(axes):
            row = idx // n
            col = idx % n
            
            vals = P[:, row, col]
            
            # Formatting title
            title = f"$P_{{{row},{col}}}$"
            if row == col:
                title += " (Diag)"
            
            sc = _plot_cloud(ax, points, vals, title=title, s=2.0, alpha=0.8)
            if shared_color_scale:
                # Force consistent color limits
                sc.set_clim(vmin, vmax)
            
            # Only add colorbars to the last column to save space
            if col == n - 1:
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        return

def visualize_quiver(sample: BlownUpSample, 
                     scale: float = 0.1, 
                     downsample: int = 10):
    """
    Visualizes the lift as vectors (arrows) attached to the points.
    Only works if the lift represents 1D lines (tangents or normals, k=1).
    """
    if sample.k != 1:
        print(f"Quiver plot only supports k=1 (vectors). Current k={sample.k}.")
        return

    points = sample.spatial[::downsample]
    # tangents: (N, n, 1) -> squeeze to (N, n)
    vectors = sample.basis[::downsample].squeeze(-1)
    
    spatial_dim = points.shape[1]
    
    fig = plt.figure(figsize=(10, 8))
    if spatial_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(points[:,0], points[:,1], points[:,2],
                  vectors[:,0], vectors[:,1], vectors[:,2],
                  length=scale, normalize=True)
    else:
        ax = fig.add_subplot(111)
        ax.quiver(points[:,0], points[:,1], 
                  vectors[:,0], vectors[:,1],
                  scale=1.0/scale, scale_units='xy')
        ax.set_aspect('equal')
        
    ax.set_title(f"Fiber Bundle Visualization (Quiver k={sample.k})")
    plt.show()
