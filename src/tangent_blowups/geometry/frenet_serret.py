"""
Frenet-Serret invariants for 3D space curves via iterated Nash blow-up.

For a space curve  (d = 1, n = 3, codimension 2), two regressions in
global R^3 give the full Frenet-Serret data without local-frame issues.

Step 1 — curvature vector via projector regression
----------------------------------------------------
Rather than going through the shape-operator in local normal-plane coords,
we regress the tangent projectors directly:

    P_i = t_i t_i^T   in  R^{3x3}

Since  dP/ds = kappa*n t^T + t (kappa*n)^T  (Frenet-Serret), and
kappa*n is perpendicular to t, we recover

    kappa*n_i  =  [dP/ds]_i  t_i

by a weighted ridge regression of  P_j - P_i  against  s_ij = t_i.(x_j-x_i).

This is equivalent to regressing dt/ds = kappa*n but is orientation-
independent: P_i = t_i t_i^T is the same whether t_i points "forward" or
"backward", avoiding sign-flip artefacts for closed/self-intersecting curves.

Step 2 — torsion via d(kappa*n)/ds regression
-----------------------------------------------
Regress  (kappa*n)_j - (kappa*n)_i  against  s_ij  over the same
level-1 k-NN neighbours (sheet-aware).  By Frenet-Serret

    d(kn)/ds = -kappa^2 t + (dkappa/ds) n + kappa*tau b,

so projecting onto the Frenet frame gives::

    dkappa/ds  =  [d(kn)/ds] . n
    kappa*tau  =  [d(kn)/ds] . binorm
    tau        =  kappa*tau / kappa

No local frame alignment is needed in either step because both
P_i and kn_i live in a fixed global ambient space.
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
        curvature : (N,)    kappa_i = ||(dP/ds)_i t_i||.
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

    Uses two global-R^3 regressions that avoid local-frame alignment issues:

    1. Regress tangent projectors  P_j - P_i  against arc-length  s_ij  to
       obtain  dP/ds = kappa*n t^T + t (kappa*n)^T, then extract
       kappa*n = (dP/ds) t.

    2. Regress  (kappa*n)_j - (kappa*n)_i  against  s_ij  and decompose via
       Frenet-Serret to get torsion and d(kappa)/ds.

    k-NN is built in the level-1 Chordal-Sasaki embedding, which is
    sheet-aware for self-intersecting / closely wound curves.

    Args:
        level0: Level-0 BlowUpLevel (d=1, n_orig=3).
        level1: Level-1 BlowUpLevel (used only for sheet-aware k-NN).
        k:      Neighbourhood size for both regressions.
        lam:    Ridge regularisation parameter (>= 0).

    Returns:
        :class:`FrenetSerretInvariants`.

    Raises:
        ValueError: if ``level0.d != 1`` or ``level0.n_orig != 3``.
    """
    if level0.d != 1:
        raise ValueError(f"Frenet-Serret requires d=1; got d={level0.d}")
    if level0.n_orig != 3:
        raise ValueError(
            f"Frenet-Serret requires n_orig=3; got n_orig={level0.n_orig}"
        )

    N = level0.N
    x       = level0.embedded              # (N, 3) positions
    tangent = level0.frame[:, :, 0].copy() # (N, 3) unit tangents

    # ------------------------------------------------------------------
    # Build k-NN in level-1 embedding (sheet-aware)
    # ------------------------------------------------------------------
    k_eff = min(k, N - 1)
    tree   = cKDTree(level1.embedded)
    _, nn_idx = tree.query(level1.embedded, k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]   # (N, k_eff) — exclude self

    # Pre-compute tangent projectors P_i = t_i t_i^T  (N, 3, 3)
    P = np.einsum("ni,nj->nij", tangent, tangent)

    # ------------------------------------------------------------------
    # Step 1: regress P_j - P_i against s_ij to get dP/ds, then kappa*n
    # ------------------------------------------------------------------
    # dP/ds = (kappa*n) t^T + t (kappa*n)^T
    # => (dP/ds) t = (kappa*n)(t.t) + t((kappa*n).t) = kappa*n  [since kn perp t]
    kn = np.zeros((N, 3), dtype=float)

    for i in range(N):
        ti        = tangent[i]        # (3,)
        neighbors = nn_idx[i]         # (k_eff,)

        dx   = x[neighbors] - x[i]   # (k_eff, 3)
        s    = dx @ ti                # (k_eff,) arc-length proxy

        dP   = P[neighbors] - P[i]   # (k_eff, 3, 3)

        # Gaussian weights from level-1 distances
        d_l1  = level1.embedded[neighbors] - level1.embedded[i]
        dist2 = np.einsum("ki,ki->k", d_l1, d_l1)
        bw    = dist2.max() if dist2.max() > 0.0 else 1.0
        w     = np.exp(-dist2 / bw)  # (k_eff,)

        ws    = w * s                 # (k_eff,)
        denom = float(ws @ s) + lam
        if abs(denom) < 1e-15:
            continue

        dP_ds    = np.einsum("k,kij->ij", ws, dP) / denom  # (3, 3)
        kn[i]    = dP_ds @ ti                               # (3,)  = kappa*n

    # Curvature magnitude and Frenet frame
    kappa      = np.linalg.norm(kn, axis=1)              # (N,)
    kappa_safe = np.where(kappa > 1e-10, kappa, 1.0)
    normal     = kn / kappa_safe[:, np.newaxis]           # (N, 3)
    normal[kappa < 1e-10] = 0.0
    binormal   = np.cross(tangent, normal)                # (N, 3)

    # ------------------------------------------------------------------
    # Step 2: regress d(kn)/ds for torsion (same neighbours / weights)
    # ------------------------------------------------------------------
    dkn_ds = np.zeros((N, 3), dtype=float)

    for i in range(N):
        ti        = tangent[i]
        neighbors = nn_idx[i]

        dx   = x[neighbors] - x[i]
        s    = dx @ ti
        dkn  = kn[neighbors] - kn[i]           # (k_eff, 3)

        d_l1  = level1.embedded[neighbors] - level1.embedded[i]
        dist2 = np.einsum("ki,ki->k", d_l1, d_l1)
        bw    = dist2.max() if dist2.max() > 0.0 else 1.0
        w     = np.exp(-dist2 / bw)

        ws    = w * s
        denom = float(ws @ s) + lam
        if abs(denom) < 1e-15:
            continue
        dkn_ds[i] = ws @ dkn / denom

    # ------------------------------------------------------------------
    # Decompose d(kn)/ds via Frenet-Serret:
    #   d(kn)/ds = -kappa^2 t + (dkappa/ds) n + kappa*tau b
    # ------------------------------------------------------------------
    dkappa_ds = np.einsum("ni,ni->n", dkn_ds, normal)    # (N,)
    kappa_tau = np.einsum("ni,ni->n", dkn_ds, binormal)  # (N,)
    torsion   = np.where(kappa > 1e-10, kappa_tau / kappa_safe, 0.0)

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
