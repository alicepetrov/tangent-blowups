"""
Frenet-Serret invariants for 3D space curves via iterated Nash blow-up.

For a space curve  (d = 1, n = 3, codimension 2), two iterations of the
blow-up construction give the full Frenet-Serret data.

Level 1 — curvature and Frenet frame
--------------------------------------
The shape operator at each point is a 2-vector in the local normal-plane basis::

    b_i = B^(1)_i[:, 0, 0]  in  R^2

where N_i in R^{3x2} is the complement of t_i.  The **curvature vector**

    kn_i = N_i @ b_i  in  R^3

is frame-independent (N_i @ b_i = kappa_i * n_i regardless of the
orientation of N_i), so

    kappa_i  = ||kn_i||
    n_i      = kn_i / kappa_i       (Frenet principal normal)
    binorm_i = t_i x n_i            (binormal)

Level 2 — torsion via d(kn)/ds regression
------------------------------------------
Torsion is extracted by regressing the R^3 differences

    kn_j - kn_i  ~  [d(kn)/ds]_i * s_ij,    s_ij = t_i . (x_j - x_i)

over level-1 k-NN neighbours.  Because both kn_i and kn_j live in the
same global R^3 (not in local per-point frames), the difference is a
genuine first-order approximation to d(kn)/ds — no parallel transport
or frame alignment is needed.

By Frenet-Serret  d(kn)/ds = -kappa^2 t + (dkappa/ds) n + kappa*tau b,
so projecting the regression result onto the Frenet frame gives::

    dkappa/ds  =  [d(kn)/ds] . n
    kappa*tau  =  [d(kn)/ds] . binorm
    tau        =  kappa*tau / kappa

The signed torsion is returned; its sign convention matches the
orientation of the complement frame N_i from numpy's Householder QR,
which is consistent (and constant for a helix) but may differ from the
classical right-hand Frenet convention by a global sign.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .iterated_grassmann import BlowUpLevel


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class FrenetSerretInvariants:
    """
    Frenet-Serret frame and differential invariants for a point-cloud space
    curve, estimated via the Nash blow-up construction (d=1, n=3).

    All arrays have leading axis N (number of sample points).

    Attributes:
        tangent   : (N, 3)  unit tangent t_i.
        normal    : (N, 3)  Frenet principal normal n_i (zero where kappa ~ 0).
        binormal  : (N, 3)  binormal = t_i x n_i.
        curvature : (N,)    kappa_i = ||kn_i|| = ||N_i b_i||.
        torsion   : (N,)    signed torsion tau_i (see module docstring).
        dkappa_ds : (N,)    d kappa / ds along the curve.
    """
    tangent:   np.ndarray   # (N, 3)
    normal:    np.ndarray   # (N, 3)
    binormal:  np.ndarray   # (N, 3)
    curvature: np.ndarray   # (N,)
    torsion:   np.ndarray   # (N,)
    dkappa_ds: np.ndarray   # (N,)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_frenet_serret(
    level0: BlowUpLevel,
    level1: BlowUpLevel,
    *,
    k: int = 16,
    lam: float = 1e-3,
) -> FrenetSerretInvariants:
    """
    Extract Frenet-Serret invariants from a one-level Nash blow-up.

    ``level1`` must have been produced by ``level0.lift()`` so that
    ``level1.curvature_ops`` is populated.

    The torsion is estimated by a weighted ridge regression of the R^3
    curvature-vector differences  kn_j - kn_i  against intrinsic
    displacements  s_ij = t_i · (x_j - x_i),  with k-NN taken in the
    level-1 Chordal-Sasaki embedding (sheet-aware for self-intersecting
    curves).

    Args:
        level0: Level-0 BlowUpLevel (d=1, n_orig=3).
        level1: Level-1 BlowUpLevel produced by ``level0.lift()``.
        k:      k-NN count for the torsion regression.
        lam:    Ridge regularisation parameter (>= 0).

    Returns:
        :class:`FrenetSerretInvariants`.

    Raises:
        ValueError: if ``level0.d != 1``, ``level0.n_orig != 3``, or
                    ``level1.curvature_ops`` is None.
    """
    if level0.d != 1:
        raise ValueError(f"Frenet-Serret requires d=1; got d={level0.d}")
    if level0.n_orig != 3:
        raise ValueError(
            f"Frenet-Serret requires n_orig=3; got n_orig={level0.n_orig}"
        )
    if level1.curvature_ops is None:
        raise ValueError(
            "level1.curvature_ops is None. "
            "Pass a BlowUpLevel produced by level0.lift()."
        )

    N = level0.N

    # ------------------------------------------------------------------
    # Level-1: curvature vector, Frenet normal, binormal
    # ------------------------------------------------------------------

    # Unit tangent (N, 3)
    tangent = level0.frame[:, :, 0].copy()

    # Complement frame N_i: (N, 3, 2) — orthonormal basis for t_i^perp
    N_frames = level0._complement_frames()   # (N, 3, 2)

    # Shape operator in local normal-plane coords: (N, 2)
    # b_i = B^(1)[i, :, 0, 0]
    b = level1.curvature_ops[:, :, 0, 0].copy()   # (N, 2)

    # Curvature vector in ambient R^3 — frame-independent:
    #   kn_i = N_i @ b_i = kappa_i * n_i
    # Because n_i is in span(N_i), N_i @ (N_i^T n_i) = n_i exactly.
    kn = np.einsum("nij,nj->ni", N_frames, b)    # (N, 3)

    # Curvature magnitude
    kappa = np.linalg.norm(kn, axis=1)           # (N,)
    kappa_safe = np.where(kappa > 1e-10, kappa, 1.0)

    # Frenet normal (zero at near-inflection points)
    normal = kn / kappa_safe[:, np.newaxis]       # (N, 3)
    normal[kappa < 1e-10] = 0.0

    # Binormal
    binormal = np.cross(tangent, normal)           # (N, 3)

    # ------------------------------------------------------------------
    # Torsion: regress d(kn)/ds in R^3 over level-1 k-NN
    # ------------------------------------------------------------------
    # Using kn_j - kn_i (global R^3 differences) avoids the frame-
    # alignment problem that arises when differencing b in local N_i coords.

    k_eff = min(k, N - 1)
    tree = cKDTree(level1.embedded)
    _, nn_idx = tree.query(level1.embedded, k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]   # (N, k_eff) — exclude self

    dkn_ds = np.zeros((N, 3), dtype=float)

    for i in range(N):
        ti        = tangent[i]              # (3,) unit tangent
        neighbors = nn_idx[i]              # (k_eff,)

        # Intrinsic displacement (scalar for d=1): s_j = t_i . (x_j - x_i)
        dx = level0.embedded[neighbors] - level0.embedded[i]   # (k_eff, 3)
        s  = dx @ ti                                             # (k_eff,)

        # Curvature-vector differences in R^3 (no frame issue)
        dkn = kn[neighbors] - kn[i]                             # (k_eff, 3)

        # Gaussian weights from level-1 distances (self-tuning bandwidth)
        d_l1  = level1.embedded[neighbors] - level1.embedded[i]  # (k_eff, D1)
        dist2 = np.einsum("ki,ki->k", d_l1, d_l1)               # (k_eff,)
        bw    = dist2.max() if dist2.max() > 0.0 else 1.0
        w     = np.exp(-dist2 / bw)                              # (k_eff,)

        # Scalar ridge regression for d(kn)/ds (d=1 case):
        #   minimise  sum_j w_j (s_j a - dkn_j)^2 + lam ||a||^2
        #   -> a = (sum_j w_j s_j dkn_j) / (sum_j w_j s_j^2 + lam)
        ws    = w * s                    # (k_eff,)
        denom = float(ws @ s) + lam     # scalar
        if abs(denom) < 1e-15:
            continue
        dkn_ds[i] = ws @ dkn / denom    # (3,)

    # ------------------------------------------------------------------
    # Decompose d(kn)/ds via Frenet-Serret:
    #   d(kn)/ds = -kappa^2 t + (dkappa/ds) n + kappa*tau b
    # ------------------------------------------------------------------
    dkappa_ds = np.einsum("ni,ni->n", dkn_ds, normal)    # (N,) = · n
    kappa_tau = np.einsum("ni,ni->n", dkn_ds, binormal)  # (N,) = · b

    # tau = kappa_tau / kappa  (signed; near-zero kappa -> tau = 0)
    torsion = np.where(kappa > 1e-10, kappa_tau / kappa_safe, 0.0)  # (N,)

    return FrenetSerretInvariants(
        tangent=tangent,
        normal=normal,
        binormal=binormal,
        curvature=kappa,
        torsion=torsion,
        dkappa_ds=dkappa_ds,
    )


__all__ = [
    "FrenetSerretInvariants",
    "extract_frenet_serret",
]
