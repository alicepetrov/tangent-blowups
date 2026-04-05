"""
Jet Fitting Curvature Estimation (Cazals & Pouget, SGP 2003)
============================================================
Estimates principal, mean, and Gaussian curvature on a point cloud by
fitting an osculating *n*-jet (truncated Taylor expansion of the height
function) at each sample point.

Algorithm
---------
For each point *p* with unit normal *n*:

1. Project k-NN into the local tangent frame (t1, t2, n) centred at *p*.
2. Build the Vandermonde matrix for the monomial basis
   {x^i y^j : 1 <= i+j <= degree} (the constant term is zero because
   neighbours are centred at the origin).
3. Solve the (weighted) least-squares system  M @ coeffs ≈ z.
4. Extract curvature from the degree-2 coefficients via the first and
   second fundamental forms (Table 1 of Cazals & Pouget).

The pre-conditioning strategy from Section 2.3 of the paper is used:
each monomial column x^i y^j is divided by h^{i+j} where h is the
average distance to neighbours, making the condition number invariant
under homothetic scaling.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def jet_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    k: int = 30,
    degree: int = 2,
) -> dict[str, np.ndarray]:
    """
    Estimate curvature via polynomial jet fitting.

    Args:
        points:  (N, 3) point positions.
        normals: (N, 3) point normals (need not be unit; will be normalised).
        k:       number of nearest neighbours used for fitting.
        degree:  polynomial degree of the jet (>= 2).

    Returns:
        dict with keys:
            ``k1``              – (N,) minimum principal curvature
            ``k2``              – (N,) maximum principal curvature
            ``mean_curvature``  – (N,) mean curvature H = (k1+k2)/2
            ``gaussian_curvature`` – (N,) Gaussian curvature K = k1*k2
            ``principal_dir1``  – (N, 3) direction of k1
            ``principal_dir2``  – (N, 3) direction of k2
            ``condition``       – (N,) condition number of the (pre-conditioned)
                                  Vandermonde system (for diagnostics)
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    N = len(points)
    if degree < 2:
        raise ValueError("degree must be >= 2 for curvature estimation")

    # Normalise normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-15, norms, 1.0)
    normals = normals / norms

    # k-NN
    tree = cKDTree(points)
    k_eff = min(k, N - 1)
    _, nn_idx = tree.query(points, k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]  # exclude self

    # Monomial exponents for total degree 1..degree (skip constant)
    exponents = _monomial_exponents(degree)  # list of (i, j)
    n_coeffs = len(exponents)

    # Output arrays
    k1 = np.full(N, np.nan)
    k2 = np.full(N, np.nan)
    d1 = np.zeros((N, 3))
    d2 = np.zeros((N, 3))
    cond = np.full(N, np.nan)

    for idx in range(N):
        ni = normals[idx]
        if not np.isfinite(ni).all():
            continue
        t1, t2 = _tangent_basis(ni)

        # Local coordinates
        nbs = points[nn_idx[idx]] - points[idx]  # (k_eff, 3)
        x = nbs @ t1
        y = nbs @ t2
        z = nbs @ ni

        # Pre-conditioning scale h (average ||projected neighbour||)
        r = np.sqrt(x**2 + y**2)
        h = np.mean(r)
        if h < 1e-30:
            continue

        # Vandermonde matrix with pre-conditioning
        M = np.empty((k_eff, n_coeffs))
        for col, (ei, ej) in enumerate(exponents):
            M[:, col] = (x ** ei) * (y ** ej) / (h ** (ei + ej))

        # Solve (weighted) least-squares
        try:
            coeffs_scaled, residuals, rank, sv = np.linalg.lstsq(M, z, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Condition number
        if sv is not None and len(sv) > 0 and sv[-1] > 0:
            cond[idx] = sv[0] / sv[-1]
        elif sv is not None and len(sv) > 0:
            cond[idx] = np.inf

        # Undo pre-conditioning: true coefficient of x^i y^j = scaled / h^{i+j}
        coeffs = np.empty(n_coeffs)
        for col, (ei, ej) in enumerate(exponents):
            coeffs[col] = coeffs_scaled[col] / (h ** (ei + ej))

        # Extract degree-1 and degree-2 coefficients
        # Monomial ordering: (1,0), (0,1), (2,0), (1,1), (0,2), ...
        a1 = coeffs[0]   # x   -> df/dx
        a2 = coeffs[1]   # y   -> df/dy
        a3 = coeffs[2]   # x^2 -> (1/2) d^2f/dx^2  ... but we fit f = sum a_{ij} x^i y^j
        a4 = coeffs[3]   # xy  -> d^2f/dxdy
        a5 = coeffs[4]   # y^2 -> (1/2) d^2f/dy^2

        # Curvature from the Weingarten map (Table 1, Cazals & Pouget)
        # f(u,v) = a1*u + a2*v + a3*u^2 + a4*u*v + a5*v^2 + ...
        # First fundamental form coefficients
        E = 1.0 + a1**2
        F = a1 * a2
        G = 1.0 + a2**2
        denom = np.sqrt(a1**2 + 1.0 + a2**2)

        # Second fundamental form coefficients
        e = 2.0 * a3 / denom
        f = a4 / denom
        g = 2.0 * a5 / denom

        # Principal curvatures via the generalized eigenvalue problem
        #   II @ v = k * I @ v
        # where I = first fundamental form, II = second fundamental form.
        # This is correct even when I^{-1} II is non-symmetric (a1, a2 != 0).
        I_mat = np.array([[E, F], [F, G]])
        II_mat = np.array([[e, f], [f, g]])

        try:
            evals, evecs = np.linalg.eig(np.linalg.solve(I_mat, II_mat))
        except np.linalg.LinAlgError:
            continue

        # Eigenvalues are real (principal curvatures); sort ascending
        evals = evals.real
        evecs = evecs.real
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        k1[idx], k2[idx] = evals[0], evals[1]

        # Map eigenvectors back to R^3
        # X_u = (1, 0, a1), X_v = (0, 1, a2)  in local (t1, t2, n) frame
        Xu = np.array([1.0, 0.0, a1])
        Xv = np.array([0.0, 1.0, a2])
        R = np.column_stack([t1, t2, ni])  # local->world rotation
        for j, (arr, ev) in enumerate([(d1, evecs[:, 0]), (d2, evecs[:, 1])]):
            v_local = ev[0] * Xu + ev[1] * Xv
            v_world = R @ v_local
            v_norm = np.linalg.norm(v_world)
            if v_norm > 1e-15:
                arr[idx] = v_world / v_norm

    H = 0.5 * (k1 + k2)
    K = k1 * k2

    return {
        "k1": k1,
        "k2": k2,
        "mean_curvature": H,
        "gaussian_curvature": K,
        "principal_dir1": d1,
        "principal_dir2": d2,
        "condition": cond,
    }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _monomial_exponents(degree: int) -> list[tuple[int, int]]:
    """
    Monomial exponents (i, j) for total degree 1..degree, ordered by
    increasing total degree then increasing j within each degree.

    Example for degree=3:
        (1,0), (0,1),  (2,0), (1,1), (0,2),  (3,0), (2,1), (1,2), (0,3)
    """
    exps = []
    for d in range(1, degree + 1):
        for j in range(d + 1):
            exps.append((d - j, j))
    return exps


def _tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal tangent vectors perpendicular to unit normal *n*."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2
