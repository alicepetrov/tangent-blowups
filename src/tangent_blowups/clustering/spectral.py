"""
Spectral Clustering
-------------------
Spectral clustering utilities built on point cloud Laplacians.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from ..geometry.grassmann import BlownUpSample
from ..pointcloud.laplacian import (
    pointcloud_laplacian,
    lifted_pointcloud_laplacian,
    laplacian_spectrum,
)
from ..solvers.linalg import normalize_vectors


def spectral_embedding_from_laplacian(
    L: sparse.spmatrix | np.ndarray,
    *,
    n_components: int,
    drop_first: bool = False,  # FIX 1: Must be False for K-Means to work correctly
    which: Literal["SM", "LM"] = "SM",
    normalize_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a spectral embedding from a Laplacian.

    Returns:
        evals, evecs, embedding
    """
    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    # FIX 2: laplacian_spectrum handles the drop offset internally.
    evals, evecs = laplacian_spectrum(
        L,
        k=n_components,
        which=which,
        drop_first=drop_first,
        return_eigenvalues=True,
    )

    embedding = evecs
    if normalize_rows:
        embedding = normalize_vectors(embedding)

    return evals, evecs, embedding


def spectral_clustering_from_laplacian(
    L: sparse.spmatrix | np.ndarray,
    n_clusters: int,
    *,
    drop_first: bool = False,
    which: Literal["SM", "LM"] = "SM",
    normalize_rows: bool = True,
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run k-means on the spectral embedding of a Laplacian.

    Returns:
        labels, evals, evecs, embedding
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    evals, evecs, embedding = spectral_embedding_from_laplacian(
        L,
        n_components=n_clusters,
        drop_first=drop_first,
        which=which,
        normalize_rows=normalize_rows,
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = kmeans.fit_predict(embedding)
    return labels, evals, evecs, embedding


def spectral_clustering_pointcloud(
    points: np.ndarray,
    n_clusters: int,
    *,
    k: int | None = 16,
    radius: float | None = None,
    h: float | Literal["local"] | None = "local",  # FIX 3: Expose local bandwidth
    laplacian_normalized: bool = True,
    symmetrize: bool = True,
    include_self: bool = False,
    drop_first: bool = False,
    which: Literal["SM", "LM"] = "SM",
    normalize_rows: bool = True,
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral clustering on a Euclidean point cloud.
    """
    L = pointcloud_laplacian(
        points,
        k=k,
        radius=radius,
        h=h,
        normalized=laplacian_normalized,
        symmetrize=symmetrize,
        include_self=include_self,
    )

    return spectral_clustering_from_laplacian(
        L,
        n_clusters,
        drop_first=drop_first,
        which=which,
        normalize_rows=normalize_rows,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )


def _coerce_blown_up(
    points_or_sample: np.ndarray | BlownUpSample,
    subspace_basis: np.ndarray | None,
) -> BlownUpSample:
    if isinstance(points_or_sample, BlownUpSample):
        if subspace_basis is not None:
            raise ValueError("subspace_basis must be None when passing BlownUpSample.")
        return points_or_sample
    if subspace_basis is None:
        raise ValueError("subspace_basis is required when passing raw points.")
    return BlownUpSample(points_or_sample, subspace_basis)


def spectral_clustering_lifted(
    points_or_sample: np.ndarray | BlownUpSample,
    subspace_basis: np.ndarray | None,
    n_clusters: int,
    *,
    k: int | None = 16,
    radius: float | None = None,
    h: float | Literal["local"] | None = "local",  # FIX 3: Expose local bandwidth
    alpha: float = 1.0,
    subspace_metric: Literal["chordal", "geodesic"] = "chordal",
    laplacian_normalized: bool = True,
    symmetrize: bool = True,
    include_self: bool = False,
    drop_first: bool = False,
    which: Literal["SM", "LM"] = "SM",
    normalize_rows: bool = True,
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral clustering on lifted points in R^n x G(k, n).
    """
    sample = _coerce_blown_up(points_or_sample, subspace_basis)
    L = lifted_pointcloud_laplacian(
        sample,
        k=k,
        radius=radius,
        h=h,
        alpha=alpha,
        subspace_metric=subspace_metric,
        normalized=laplacian_normalized,
        symmetrize=symmetrize,
        include_self=include_self,
    )

    return spectral_clustering_from_laplacian(
        L,
        n_clusters,
        drop_first=drop_first,
        which=which,
        normalize_rows=normalize_rows,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )


__all__ = [
    "spectral_embedding_from_laplacian",
    "spectral_clustering_from_laplacian",
    "spectral_clustering_pointcloud",
    "spectral_clustering_lifted",
]