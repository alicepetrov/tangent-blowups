"""
Spectral Clustering
-------------------
Spectral clustering utilities built on point cloud Laplacians.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN, KMeans

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
    drop_first: bool = False,
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
    cluster_method: Literal["kmeans", "dbscan"] = "kmeans",
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run k-means or DBSCAN on the spectral embedding of a Laplacian.

    n_clusters controls the embedding dimension for all clustering methods.

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

    if cluster_method == "kmeans":
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = kmeans.fit_predict(embedding)
    elif cluster_method == "dbscan":
        if dbscan_eps <= 0.0:
            raise ValueError("dbscan_eps must be positive.")
        if dbscan_min_samples <= 0:
            raise ValueError("dbscan_min_samples must be positive.")
        dbscan = DBSCAN(
            eps=float(dbscan_eps),
            min_samples=int(dbscan_min_samples),
        )
        labels = dbscan.fit_predict(embedding)
    else:
        raise ValueError(
            f"Unknown cluster_method '{cluster_method}'. Expected 'kmeans' or 'dbscan'."
        )
    return labels, evals, evecs, embedding


def spectral_clustering_pointcloud(
    points: np.ndarray,
    n_clusters: int,
    *,
    k: int | None = 16,
    radius: float | None = None,
    h: float | Literal["local"] | None = "local",
    laplacian_normalized: bool = True,
    symmetrize: bool = True,
    include_self: bool = False,
    drop_first: bool = False,
    which: Literal["SM", "LM"] = "SM",
    normalize_rows: bool = True,
    cluster_method: Literal["kmeans", "dbscan"] = "kmeans",
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral clustering on a Euclidean point cloud.

    cluster_method selects k-means or DBSCAN for clustering the embedding.
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
        cluster_method=cluster_method,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
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
    h: float | Literal["local"] | None = "local",
    alpha: float = 1.0,
    subspace_metric: Literal["chordal", "geodesic"] = "chordal",
    laplacian_normalized: bool = True,
    symmetrize: bool = True,
    include_self: bool = False,
    drop_first: bool = False,
    which: Literal["SM", "LM"] = "SM",
    normalize_rows: bool = True,
    cluster_method: Literal["kmeans", "dbscan"] = "kmeans",
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spectral clustering on lifted points in R^n x G(k, n).

    cluster_method selects k-means or DBSCAN for clustering the embedding.
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
        cluster_method=cluster_method,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )


__all__ = [
    "spectral_embedding_from_laplacian",
    "spectral_clustering_from_laplacian",
    "spectral_clustering_pointcloud",
    "spectral_clustering_lifted",
]
