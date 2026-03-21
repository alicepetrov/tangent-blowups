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
  d_ell^2 = ||x_i - x_j||^2 + sum_{m=0}^{ell-1} (alpha_m/2) ||P_i^(m) - P_j^(m)||_F^2

(The level-ell embedding includes projectors from levels 0 through ell-1;
the level-ell projectors are appended only by the next lift.)

Reference: theoretical notes on iterated tangent blow-ups / Nash blow-ups.
"""
from __future__ import annotations

from dataclasses import dataclass

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
        n_orig: int | None = None,
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
        # n_orig: dimension of the original ambient space (first n_orig coords of
        # embedded are always the original spatial positions, at every level).
        # Defaults to D at level 0 (where embedded == points, so D == n).
        self.n_orig: int = int(n_orig) if n_orig is not None else self.D
        self.curvature_ops: np.ndarray | None = None  # (N, n_comp, d, d); set by lift()
        # Layout of vectorised projector blocks within self.embedded.
        # Each entry is (start_col, n_cols, scale) where
        #   embedded[:, start:start+n_cols] = scale * vec(P^(m))
        # At level 0 this is empty; lift() appends one entry per lift step.
        self._proj_blocks: list[tuple[int, int, float]] = []

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

        # Orthonormalize each frame via reduced QR.
        # Householder QR has a sign ambiguity: (Q, R) and (-Q, -R) are
        # both valid.  We enforce det(R) > 0 so that Q preserves the
        # orientation of the input frame.  Without this, random sign
        # flips in the tangent frames create conflicting targets for any
        # downstream loss that depends on frame orientation.
        orth = np.empty_like(frames)
        for i in range(N):
            Q, R = np.linalg.qr(frames[i])  # Q: (n, d) reduced
            if np.linalg.det(R[:d, :d]) < 0:
                Q[:, 0] *= -1
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
        # Scale ridge relative to data so shrinkage is independent of
        # bandwidth / effective sample size.
        tr_TWT = np.trace(TWT)
        ridge = lam * (tr_TWT / d) if tr_TWT > 0 else lam
        reg = TWT + ridge * np.eye(d)

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

            # Gaussian weights with self-tuning bandwidth (q-th neighbor dist)
            d_phi = Phi_next[neighbors] - Phi_next[i]     # (k_eff, D_next)
            dist2 = np.einsum("ki,ki->k", d_phi, d_phi)   # (k_eff,)
            sorted_d2 = np.sort(dist2)
            q = min(7, k_eff - 1)
            bw = sorted_d2[q] if sorted_d2[q] > 0 else 1.0
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

            # Reduced QR: Q has shape (D_next, d) with orthonormal columns.
            # Enforce det(R) > 0 to avoid sign ambiguity (see from_point_tangents).
            Q, R = np.linalg.qr(G)
            if np.linalg.det(R[:d, :d]) < 0:
                Q[:, 0] *= -1
            U_next[i] = Q[:, :d]

        result = BlowUpLevel(
            embedded=Phi_next, frame=U_next, level=self.level + 1, n_orig=self.n_orig,
        )
        result.curvature_ops = B_all
        # Track where each level's vectorised projectors sit in the embedding.
        result._proj_blocks = list(self._proj_blocks) + [
            (D, D * D, scale)  # block for P^(self.level): starts at col D, has D^2 cols
        ]
        return result

    # ------------------------------------------------------------------
    # Automatic parameter selection
    # ------------------------------------------------------------------

    def auto_alpha(self, *, k: int = 30) -> float:
        """
        Compute a data-driven weight parameter alpha for the lift.

        Sets alpha = median_spatial^2 / median_projector^2, balancing the
        spatial and angular scales so that neither dominates the lifted
        metric.  This is the default alpha recommended in the paper
        (Section 7, Implementation).

        Args:
            k:  k-NN neighbourhood size used to sample pairwise distances.

        Returns:
            alpha > 0.
        """
        k_eff = min(k, self.N - 1)
        tree = cKDTree(self.embedded)
        dist, idx = tree.query(self.embedded, k=k_eff + 1)
        idx = idx[:, 1:]   # exclude self
        dist = dist[:, 1:]

        # Spatial distances: use original positions
        positions = self.embedded[:, :self.n_orig]
        rows = np.repeat(np.arange(self.N), k_eff)
        cols = idx.ravel()
        dx = positions[rows] - positions[cols]
        dist2_spatial = np.einsum("ij,ij->i", dx, dx)

        # Projector distances: ||P_i - P_j||_F^2 = 2d - 2||U_i^T U_j||_F^2
        U = self.frame                                        # (N, D, d)
        M = np.einsum("eka,ekb->eab", U[rows], U[cols])      # (E, d, d)
        inner_sq = np.einsum("eab,eab->e", M, M)
        dist2_proj = 2.0 * self.d - 2.0 * inner_sq
        dist2_proj = np.maximum(dist2_proj, 0.0)              # numerical safety

        med_spatial = float(np.median(np.sqrt(dist2_spatial)))
        med_proj = float(np.median(np.sqrt(dist2_proj)))

        if med_proj <= 0.0:
            return 1.0
        return (med_spatial / med_proj) ** 2

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BlowUpLevel(level={self.level}, N={self.N}, D={self.D}, d={self.d})"
        )


# ---------------------------------------------------------------------------
# Differential invariant containers
# ---------------------------------------------------------------------------

@dataclass
class Level0Invariants:
    """
    Differential invariants at blow-up level 0.

    Basic geometric quantities of the original submanifold: position,
    orthonormal tangent frame, and orthonormal normal frame.
    """
    points: np.ndarray        # (N, n)      position in original ambient space
    tangent_frame: np.ndarray # (N, n, d)   orthonormal tangent basis
    normal_frame: np.ndarray  # (N, n, n-d) orthonormal normal basis


@dataclass
class Level1Invariants:
    """
    Differential invariants at blow-up level 1 (second fundamental form).

    shape_operator[i, α, a, b] = h_{α,a,b} is the symmetrised second
    fundamental form in normal direction α, symmetric in tangent indices (a, b).

    For hypersurfaces (n_comp == 1), principal curvatures/directions and
    Gaussian curvature are also provided; otherwise they are None.
    """
    shape_operator: np.ndarray               # (N, n_comp, d, d)
    mean_curvature: np.ndarray               # (N, n_comp)  H_α = tr(h_α) / d
    total_curvature: np.ndarray              # (N, n_comp)  ||h_α||_F
    principal_curvatures: np.ndarray | None  # (N, d)       hypersurface only
    principal_directions: np.ndarray | None  # (N, d, d)    hypersurface only
    gaussian_curvature: np.ndarray | None    # (N,)         hypersurface only


@dataclass
class Level2Invariants:
    """
    Differential invariants at blow-up level 2 (Codazzi tensor).

    curvature_gradient[i, α, a, b, c] = ∂_c h_{α,a,b} approximates the
    covariant derivative of the shape operator (Codazzi tensor), estimated
    by regressing shape-operator differences against intrinsic displacements
    using the level-0 geometry.
    """
    curvature_gradient: np.ndarray   # (N, n_comp, d, d, d)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_level0(level: BlowUpLevel) -> Level0Invariants:
    """
    Extract level-0 differential invariants: position, tangent, and normal frame.

    Args:
        level: A level-0 BlowUpLevel.

    Returns:
        Level0Invariants with fields ``points``, ``tangent_frame``, ``normal_frame``.

    Raises:
        ValueError: if ``level.level != 0``.
    """
    if level.level != 0:
        raise ValueError(f"Expected a level-0 BlowUpLevel, got level={level.level}")
    return Level0Invariants(
        points=level.embedded,
        tangent_frame=level.frame,
        normal_frame=level._complement_frames(),
    )


def extract_level1(level: BlowUpLevel) -> Level1Invariants:
    """
    Extract level-1 differential invariants from a lifted BlowUpLevel.

    The input must have been produced by ``BlowUpLevel.lift()`` so that
    ``curvature_ops`` is populated (typically a level-1 object).

    The curvature operator B_all[i] of shape (n_comp, d, d) encodes the
    second fundamental form via::

        h[i, α, a, b] = B_all[i, α, b, a]

    which is then symmetrised over (a, b).

    Args:
        level: A BlowUpLevel with ``curvature_ops`` set (output of ``.lift()``).

    Returns:
        Level1Invariants.

    Raises:
        ValueError: if ``curvature_ops`` is None.
    """
    if level.curvature_ops is None:
        raise ValueError(
            "curvature_ops is not set. Pass a BlowUpLevel produced by .lift()."
        )
    B_all = level.curvature_ops   # (N, n_comp, d, d); B_all[i, α, b, a] = B_i(e_a)[α, b]
    N, n_comp, d, _ = B_all.shape

    # h[i, α, a, b] = B_all[i, α, b, a]  →  transpose last two axes then symmetrize
    raw_h = B_all.transpose(0, 1, 3, 2)               # (N, n_comp, d, d) with axes (i,α,a,b)
    shape_op = (raw_h + raw_h.transpose(0, 1, 3, 2)) / 2.0  # symmetrize over (a, b)

    # Mean curvature: H_α = tr(h_α) / d
    mean_curv = np.trace(shape_op, axis1=2, axis2=3) / d    # (N, n_comp)

    # Total curvature: ||h_α||_F
    total_curv = np.sqrt(np.einsum("nabc,nabc->na", shape_op, shape_op))  # (N, n_comp)

    if n_comp == 1:
        S = shape_op[:, 0, :, :]                       # (N, d, d)
        evals, evecs = np.linalg.eigh(S)               # evals (N,d) ascending, evecs (N,d,d)
        principal_curvatures: np.ndarray | None = evals
        principal_directions: np.ndarray | None = evecs
        gaussian_curvature: np.ndarray | None = np.prod(evals, axis=1)  # (N,)
    else:
        principal_curvatures = None
        principal_directions = None
        gaussian_curvature = None

    return Level1Invariants(
        shape_operator=shape_op,
        mean_curvature=mean_curv,
        total_curvature=total_curv,
        principal_curvatures=principal_curvatures,
        principal_directions=principal_directions,
        gaussian_curvature=gaussian_curvature,
    )


def extract_level2(
    level0: BlowUpLevel,
    level1: BlowUpLevel,
    level1_inv: Level1Invariants,
    *,
    k: int = 16,
    lam: float = 1e-3,
) -> Level2Invariants:
    """
    Estimate level-2 differential invariants: the gradient of the shape operator.

    Approximates the covariant derivative ∂_c h_{α,a,b} of the shape operator
    by regressing shape-operator differences against level-0 intrinsic
    displacements (exact in geodesic normal coordinates at each point).

    k-NN is performed in the level-1 Chordal-Sasaki metric (``level1.embedded``)
    rather than in position space, consistent with how :meth:`BlowUpLevel.lift`
    selects neighbours for curvature estimation.  This prevents cross-component
    contamination near tangential intersection points, where both components are
    close in position space but separated in the level-1 metric (because the
    flat component has zero projector change between its own points, while a
    cross-component step has a large projector change).

    The result::

        curvature_gradient[i, α, a, b, c] ≈ ∂_c h_{α,a,b}

    encodes the Codazzi tensor.

    Args:
        level0:     The level-0 BlowUpLevel (original positions and tangent frames).
        level1:     The level-1 BlowUpLevel (Chordal-Sasaki embedding for k-NN).
        level1_inv: Level1Invariants returned by :func:`extract_level1`.
        k:          Number of nearest neighbours for regression.
        lam:        Ridge regularisation parameter.

    Returns:
        Level2Invariants with ``curvature_gradient`` of shape (N, n_comp, d, d, d).
    """
    N = level0.N
    d = level0.d
    h_all = level1_inv.shape_operator   # (N, n_comp, d, d)
    n_comp = h_all.shape[1]

    k_eff = min(k, N - 1)
    # k-NN in the level-1 Chordal-Sasaki space, consistent with lift()
    tree = cKDTree(level1.embedded)
    _, nn_idx = tree.query(level1.embedded, k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]   # (N, k_eff) excluding self

    grad_h = np.zeros((N, n_comp, d, d, d), dtype=float)

    for i in range(N):
        Ui = level0.frame[i]          # (n, d)
        neighbors = nn_idx[i]         # (k_eff,)

        # Intrinsic displacements in level-0 position space
        d_emb = level0.embedded[neighbors] - level0.embedded[i]   # (k_eff, n)
        t = d_emb @ Ui                                              # (k_eff, d)

        # Gaussian weights from level-1 distances (q-th neighbor bandwidth)
        d_l1 = level1.embedded[neighbors] - level1.embedded[i]    # (k_eff, D1)
        dist2 = np.einsum("ki,ki->k", d_l1, d_l1)
        sorted_d2 = np.sort(dist2)
        q_bw = min(7, k_eff - 1)
        bw = sorted_d2[q_bw] if sorted_d2[q_bw] > 0 else 1.0
        w = np.exp(-dist2 / bw)                     # (k_eff,)

        dh = h_all[neighbors] - h_all[i]            # (k_eff, n_comp, d, d)
        dh_flat = dh.reshape(k_eff, n_comp * d * d)

        wt = w[:, np.newaxis] * t                   # (k_eff, d)
        TWT_raw = wt.T @ t                           # (d, d)
        tr_TWT = np.trace(TWT_raw)
        ridge = lam * (tr_TWT / d) if tr_TWT > 0 else lam
        TWT = TWT_raw + ridge * np.eye(d)            # (d, d)
        TWdh = wt.T @ dh_flat                        # (d, n_comp*d^2)

        try:
            G_flat = np.linalg.solve(TWT, TWdh)     # (d, n_comp*d^2)
        except np.linalg.LinAlgError:
            continue

        # G_flat[c, α*d²+a*d+b] = ∂_c h_{α,a,b}
        # grad_h[i, α, a, b, c] = G_flat.T[α*d²+a*d+b, c]
        grad_h[i] = G_flat.T.reshape(n_comp, d, d, d)

    return Level2Invariants(curvature_gradient=grad_h)


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
    # Invariant containers
    "Level0Invariants",
    "Level1Invariants",
    "Level2Invariants",
    # Extraction functions
    "extract_level0",
    "extract_level1",
    "extract_level2",
]
