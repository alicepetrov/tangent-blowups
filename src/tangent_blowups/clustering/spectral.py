"""
Spectral clustering via Laplacian eigenmaps.

Two public functions:

- `spectral_embedding(L, n_components)` — the smallest-eigenvalue eigenmap of a
  graph Laplacian.
- `spectral_clustering(L, n_clusters)` — spectral embedding followed by
  k-means.

Both operate on any pre-built Laplacian. Build yours via
`tangent_blowups.geometry.kernels.lifted_laplacian` (self-tuning product kernel
by default).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans

from ..solvers.eigen import laplacian_spectrum
from ..solvers.linalg import normalize_vectors


def spectral_embedding(
    L: sparse.spmatrix | np.ndarray,
    n_components: int,
    *,
    drop_first: bool = True,
    normalize_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Laplacian eigenmap: the `n_components` smallest non-trivial eigenvectors of L.

    Args:
        L:              Graph Laplacian (sparse or dense, square).
        n_components:   Number of eigenvector coordinates to return.
        drop_first:     Drop the trivial constant eigenvector (eigenvalue ~ 0),
                        standard for connected graphs. Default True.
        normalize_rows: Unit-normalize each row of the embedding (Ng-Jordan-Weiss
                        preprocessing for k-means). Default True.

    Returns:
        (evals, embedding) -- eigenvalues (n_components,) and embedding
        (N, n_components). Eigenvectors are ordered by ascending eigenvalue.
    """
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    # laplacian_spectrum handles the drop_first offset internally (it fetches
    # k+1 eigenpairs and drops the trivial one), so we pass n_components directly.
    evals, evecs = laplacian_spectrum(
        L, k=n_components, which="SM", drop_first=drop_first, return_eigenvalues=True,
    )

    embedding = normalize_vectors(evecs) if normalize_rows else evecs
    return evals, embedding


def spectral_clustering(
    L: sparse.spmatrix | np.ndarray,
    n_clusters: int,
    *,
    n_components: int | None = None,
    random_state: int | None = 0,
    n_init: int = 10,
) -> np.ndarray:
    """
    Spectral clustering: eigenmap embedding + k-means.

    Args:
        L:            Graph Laplacian.
        n_clusters:   Number of clusters.
        n_components: Embedding dimension. Defaults to `n_clusters`.
        random_state: k-means seed.
        n_init:       k-means restarts.

    Returns:
        Integer labels of shape (N,).
    """
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}.")

    dim = n_components if n_components is not None else n_clusters
    _, embedding = spectral_embedding(L, n_components=dim, drop_first=True)
    return KMeans(
        n_clusters=n_clusters, n_init=n_init, random_state=random_state,
    ).fit_predict(embedding)


__all__ = ["spectral_embedding", "spectral_clustering"]
