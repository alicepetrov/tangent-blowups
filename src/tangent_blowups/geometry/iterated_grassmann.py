"""
Iterated Tangent Blow-Up
------------------------
Implements the iterated Nash blow-up construction from point–tangent data.

At each level ell, every sample point i carries:
  - Phi_i^(ell) in R^{D_ell}        -- the embedded position
  - U_i^(ell)   in R^{D_ell x d}    -- orthonormal tangent frame

The tangent projector P_i^(ell) = U_i^(ell) @ U_i^(ell).T encodes the
tangent plane of the ell-th lifted manifold at that point.

The ambient dimension grows as:
  D_0 = n,  D_ell = D_{ell-1} + D_{ell-1}^2  (= D_{ell-1} * (1 + D_{ell-1}))

Each lift step:
  1. Embeds via the Chordal-Sasaki isometry:
       Phi^(ell) = (Phi^(ell-1),  sqrt(alpha/2) * vec(P^(ell-1)))
  2. Estimates the curvature operator B_i by weighted ridge regression of
     Grassmann tangent coordinates against intrinsic displacements.
  3. Constructs the tangent projector P^(ell) from product tangent vectors,
     each assembled from a horizontal (position) and vertical (curvature)
     component.

The ell-th blow-up metric telescopes as:
  d_ell^2 = ||x_i - x_j||^2 + sum_{m=0}^{ell} (alpha/2) ||P_i^(m) - P_j^(m)||_F^2

Reference: theoretical notes on iterated tangent blow-ups / Nash blow-ups.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ambient_dim_sequence(n: int, num_levels: int) -> list[int]:
    """
    Returns the ambient dimension at each blow-up level.

    D_0 = n,  D_ell = D_{ell-1} + D_{ell-1}^2  for ell >= 1.

    Args:
        n:          original ambient dimension (D_0).
        num_levels: number of lift steps.

    Returns:
        List of length num_levels + 1: [D_0, D_1, ..., D_{num_levels}].
    """
    dims = [n]
    for _ in range(num_levels):
        d = dims[-1]
        dims.append(d + d * d)
    return dims


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BlowUpLevel:
    """
    Data for one level of the iterated tangent blow-up.

    At level ell, stores for each of N sample points:
      - embedded[i]:  Phi_i^(ell) in R^{D_ell},    shape (N, D)
      - frame[i]:     U_i^(ell)   in R^{D_ell x d}, shape (N, D, d)

    The tangent projector P_i^(ell) = U_i^(ell) @ U_i^(ell).T is available
    via the .projectors property.
    """

    def __init__(
        self,
        embedded: np.ndarray,  # (N, D)
        frame: np.ndarray,     # (N, D, d)
        level: int = 0,
    ) -> None:
        self.embedded = np.asarray(embedded, dtype=float)
        self.frame = np.asarray(frame, dtype=float)
        self.level = int(level)

        self.N, self.D = self.embedded.shape
        if self.frame.ndim != 3 or self.frame.shape[:2] != (self.N, self.D):
            raise ValueError(
                f"frame must have shape (N={self.N}, D={self.D}, d), "
                f"got {self.frame.shape}"
            )
        self.d = self.frame.shape[2]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_point_tangents(
        cls,
        points: np.ndarray,  # (N, n)
        frames: np.ndarray,  # (N, n, d)  or  (N, n) for d=1
    ) -> "BlowUpLevel":
        """
        Construct the level-0 blow-up from point-tangent samples.

        Args:
            points: (N, n) spatial coordinates.
            frames: (N, n, d) orthonormal tangent frames. For curves (d=1)
                    a (N, n) array of unit tangent vectors is also accepted.
        """
        points = np.asarray(points, dtype=float)
        frames = np.asarray(frames, dtype=float)
        if frames.ndim == 2:
            frames = frames[:, :, np.newaxis]  # (N, n) -> (N, n, 1)

        N, n, d = frames.shape

        # Orthonormalize each frame via reduced QR
        orth = np.empty_like(frames)
        for i in range(N):
            Q, _ = np.linalg.qr(frames[i])  # Q: (n, d) reduced
            orth[i] = Q[:, :d]

        return cls(embedded=points, frame=orth, level=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def projectors(self) -> np.ndarray:
        """(N, D, D) array of projectors P_i = U_i U_i^T."""
        return np.einsum("nik,njk->nij", self.frame, self.frame)

    def embedding_vector(self, alpha: float) -> np.ndarray:
        """
        Chordal-Sasaki isometric embedding into R^{D + D^2}:
          Phi_next_i = (Phi_i,  sqrt(alpha/2) * vec(P_i))

        Returns shape (N, D + D^2).
        """
        P_flat = self.projectors.reshape(self.N, -1)   # (N, D^2)
        scale = np.sqrt(alpha / 2.0)
        return np.hstack([self.embedded, scale * P_flat])  # (N, D + D^2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complement_frames(self) -> np.ndarray:
        """
        Returns (N, D, D-d) orthonormal bases N_i spanning (U_i)^perp,
        computed as the last D-d columns of the full QR factorisation of U_i.
        """
        N, D, d = self.N, self.D, self.d
        n_comp = D - d
        N_out = np.empty((N, D, n_comp), dtype=float)
        for i in range(N):
            Q, _ = np.linalg.qr(self.frame[i], mode="complete")  # Q: (D, D)
            N_out[i] = Q[:, d:]   # last D-d columns
        return N_out

    @staticmethod
    def _fit_curvature(
        t: np.ndarray,      # (k, d)     intrinsic displacements
        C: np.ndarray,      # (k, m, d)  Grassmann tangent coords  (m = n_comp)
        w: np.ndarray,      # (k,)       non-negative weights
        d: int,
        n_comp: int,
        lam: float,
    ) -> np.ndarray:
        """
        Weighted ridge regression for the curvature operator B_i.

        Finds B_flat (n_comp*d, d) such that  B_flat @ t_j ≈ vec(C_j),
        by solving the normal equations:
          (T^T W T + lam I) B_flat^T = T^T W C_flat

        Returns B_flat of shape (n_comp*d, d), or zeros on failure.
        """
        if t.shape[0] == 0 or n_comp == 0:
            return np.zeros((n_comp * d, d), dtype=float)

        c_flat = C.reshape(len(t), n_comp * d)     # (k, n_comp*d)
        wt = w[:, np.newaxis] * t                  # (k, d)
        TWT = wt.T @ t                             # (d, d)
        TWc = wt.T @ c_flat                        # (d, n_comp*d)
        reg = TWT + lam * np.eye(d)

        try:
            B_flat_T = np.linalg.solve(reg, TWc)  # (d, n_comp*d)
        except np.linalg.LinAlgError:
            return np.zeros((n_comp * d, d), dtype=float)

        return B_flat_T.T                          # (n_comp*d, d)

    # ------------------------------------------------------------------
    # Core: lift to the next level
    # ------------------------------------------------------------------

    def lift(
        self,
        *,
        k: int = 16,
        alpha: float = 1.0,
        lam: float = 1e-3,
    ) -> "BlowUpLevel":
        """
        Compute one step of the iterated blow-up.

        Given level-(ell-1) data (Phi^{ell-1}, U^{ell-1}), produces
        level-ell data (Phi^{ell}, U^{ell}):

          1. Embed:
               Phi^{ell} = (Phi^{ell-1},  sqrt(alpha/2) * vec(P^{ell-1}))

          2. Estimate curvature operator B_i by weighted ridge regression:
               C_ij = N_i^T (P_j - P_i) U_i  ~  B_i t_ij
             where t_ij = U_i^T (Phi_j^{ell-1} - Phi_i^{ell-1}) and
             k-NN is taken in Phi^{ell}.

          3. Construct tangent vectors for each basis direction a = 1..d:
               g_a = (U_i e_a,  sqrt(alpha/2) * vec(N_i A_a U_i^T + U_i A_a^T N_i^T))
             where A_a = B_i(e_a). Stack into G_i, QR-orthonormalize, form P^{ell}.

        Args:
            k:     number of nearest neighbours for curvature estimation.
            alpha: Chordal-Sasaki weight (alpha_ell in the theory).
            lam:   ridge regularisation parameter (>= 0).

        Returns:
            BlowUpLevel at level ell = self.level + 1.
        """
        N, D, d = self.N, self.D, self.d
        n_comp = D - d       # complement dimension
        D_next = D + D * D   # = D * (1 + D)

        # ----------------------------------------------------------------
        # Step 1: Embed into R^{D_next}
        # ----------------------------------------------------------------
        Phi_next = self.embedding_vector(alpha)   # (N, D_next)
        scale = np.sqrt(alpha / 2.0)

        # ----------------------------------------------------------------
        # Step 2: Estimate curvature operators B_i
        # ----------------------------------------------------------------
        k_eff = min(k, N - 1)

        # k-NN in the level-ell embedded space
        tree = cKDTree(Phi_next)
        _, nn_idx = tree.query(Phi_next, k=k_eff + 1)  # (N, k_eff+1) incl. self
        nn_idx = nn_idx[:, 1:]                          # (N, k_eff) excl. self

        P_all = self.projectors          # (N, D, D)
        N_frames = self._complement_frames()  # (N, D, n_comp)

        # B_all[i] shape (n_comp, d, d):  B_all[i][:, :, a] = B_i(e_a) in R^{n_comp x d}
        B_all = np.zeros((N, n_comp, d, d), dtype=float)

        for i in range(N):
            Ui = self.frame[i]          # (D, d)
            Ni = N_frames[i]            # (D, n_comp)
            Pi = P_all[i]               # (D, D)
            neighbors = nn_idx[i]       # (k_eff,)

            # Gaussian weights with self-tuning bandwidth (= farthest neighbour dist)
            d_phi = Phi_next[neighbors] - Phi_next[i]     # (k_eff, D_next)
            dist2 = np.einsum("ki,ki->k", d_phi, d_phi)   # (k_eff,)
            bw = dist2.max() if dist2.max() > 0.0 else 1.0
            w = np.exp(-dist2 / bw)                        # (k_eff,)

            # Intrinsic coords: t_ij = U_i^T (Phi_j^{ell-1} - Phi_i^{ell-1})
            d_emb = self.embedded[neighbors] - self.embedded[i]  # (k_eff, D)
            t = d_emb @ Ui                                        # (k_eff, d)

            # Grassmann tangent coords: C_ij = N_i^T (P_j - P_i) U_i
            delta_P = P_all[neighbors] - Pi               # (k_eff, D, D)
            dP_Ui = delta_P @ Ui                          # (k_eff, D, d)
            # einsum: C[k,p,d] = sum_m N_i.T[p,m] * dP_Ui[k,m,d]
            C = np.einsum("pm,kmd->kpd", Ni.T, dP_Ui)   # (k_eff, n_comp, d)

            B_flat = self._fit_curvature(t, C, w, d, n_comp, lam)  # (n_comp*d, d)
            B_all[i] = B_flat.reshape(n_comp, d, d)
            # B_all[i][:, :, a] = B_flat[:, a].reshape(n_comp, d) = B_i(e_a)

        # ----------------------------------------------------------------
        # Step 3: Construct tangent projectors P_i^{ell}
        # ----------------------------------------------------------------
        U_next = np.empty((N, D_next, d), dtype=float)

        for i in range(N):
            Ui = self.frame[i]   # (D, d)
            Ni = N_frames[i]     # (D, n_comp)
            Bi = B_all[i]        # (n_comp, d, d)

            # Build G_i = [g_1, ..., g_d] in R^{D_next x d}
            G = np.empty((D_next, d), dtype=float)
            for a in range(d):
                # Horizontal component: u_a = U_i e_a
                u_a = Ui[:, a]                            # (D,)

                # Vertical component from the curvature operator
                A_a = Bi[:, :, a]                         # (n_comp, d) = B_i(e_a)
                # Delta_P_a = N_i A_a U_i^T + U_i A_a^T N_i^T  in R^{D x D}
                Delta_P_a = Ni @ A_a @ Ui.T + Ui @ A_a.T @ Ni.T  # (D, D)

                G[:D, a] = u_a
                G[D:, a] = scale * Delta_P_a.ravel()

            # Reduced QR: Q has shape (D_next, d) with orthonormal columns
            Q, _ = np.linalg.qr(G)
            U_next[i] = Q[:, :d]

        return BlowUpLevel(embedded=Phi_next, frame=U_next, level=self.level + 1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BlowUpLevel(level={self.level}, N={self.N}, D={self.D}, d={self.d})"
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def iterated_blowup(
    points: np.ndarray,
    frames: np.ndarray,
    num_levels: int,
    *,
    k: int = 16,
    alpha: float = 1.0,
    lam: float = 1e-3,
) -> list[BlowUpLevel]:
    """
    Compute the iterated tangent blow-up up to the requested number of levels.

    Args:
        points:     (N, n) spatial coordinates.
        frames:     (N, n, d) orthonormal tangent frames. A (N, n) array is
                    accepted for curves (d=1).
        num_levels: number of lift steps. 0 returns only level-0 data.
        k:          nearest neighbours for curvature estimation at each level.
        alpha:      Chordal-Sasaki weight (applied uniformly at all levels).
        lam:        ridge regularisation for curvature estimation.

    Returns:
        List [level_0, level_1, ..., level_{num_levels}].
    """
    levels: list[BlowUpLevel] = [BlowUpLevel.from_point_tangents(points, frames)]
    for _ in range(num_levels):
        levels.append(levels[-1].lift(k=k, alpha=alpha, lam=lam))
    return levels


__all__ = [
    "BlowUpLevel",
    "iterated_blowup",
    "ambient_dim_sequence",
]
