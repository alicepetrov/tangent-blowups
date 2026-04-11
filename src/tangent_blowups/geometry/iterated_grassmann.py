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

def _build_vech_w_duplication(d: int) -> np.ndarray:
    """
    Weighted duplication matrix D_w of shape (d², d*(d+1)//2).

    Maps vech_w(B) → vec(B) for a symmetric d×d matrix B, using the
    norm-preserving convention: diagonal entries stored as-is,
    off-diagonal entries scaled by √2.  Satisfies D_w^T D_w = I_m.
    """
    m = d * (d + 1) // 2
    D_w = np.zeros((d * d, m))
    col = 0
    for a in range(d):
        for b in range(a, d):
            if a == b:
                D_w[a * d + a, col] = 1.0
            else:
                D_w[a * d + b, col] = 1.0 / np.sqrt(2.0)
                D_w[b * d + a, col] = 1.0 / np.sqrt(2.0)
            col += 1
    return D_w


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
    def from_normals(
        cls,
        points: np.ndarray,  # (N, n)
        normals: np.ndarray,  # (N, n)
    ) -> "BlowUpLevel":
        """Construct a level-0 blow-up from point positions and normals.

        Builds (n-1)-dimensional tangent frames perpendicular to each normal.

        Args:
            points:  (N, n) spatial coordinates.
            normals: (N, n) unit normal vectors (will be re-normalised).
        """
        from ..solvers.linalg import normalize_vectors

        points = np.asarray(points, dtype=float)
        normals = np.asarray(normalize_vectors(normals), dtype=float)
        N, n = points.shape
        d = n - 1  # tangent dimension

        frames = np.empty((N, n, d), dtype=float)
        for i in range(N):
            # Householder-style: QR of normal gives normal as first col,
            # remaining cols span the tangent plane.
            Q, _ = np.linalg.qr(
                np.column_stack([normals[i, :, np.newaxis],
                                 np.eye(n)[:, :d]]),
                mode="reduced",
            )
            # Q[:,0] ~ normal, Q[:,1:] spans tangent plane
            frames[i] = Q[:, 1:]

        return cls.from_point_tangents(points, frames)

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

    def distance_matrix(
        self,
        other: "BlowUpLevel | None" = None,
        *,
        alpha: float = 1.0,
        subspace_metric: str = "chordal",
    ) -> np.ndarray:
        """Pairwise product-metric distances.

        d(i,j) = sqrt( ||x_i - x_j||^2 + alpha * d_sub(U_i, U_j)^2 )

        Args:
            other: second sample (default: self).
            alpha: weight on the subspace component.
            subspace_metric: ``'chordal'`` or ``'geodesic'``.

        Returns:
            (N, M) distance matrix.
        """
        if other is None:
            other = self

        A = self.embedded
        B = other.embedded
        a2 = np.sum(A * A, axis=1)[:, np.newaxis]
        b2 = np.sum(B * B, axis=1)[np.newaxis, :]
        dist_x_sq = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)

        if subspace_metric == "chordal":
            P = self.projectors
            Q = other.projectors
            inner = np.einsum("nij,mij->nm", P, Q)
            dist_u_sq = np.maximum(
                (self.d + other.d - 2.0 * inner) / 2.0, 0.0)
        elif subspace_metric == "geodesic":
            dist_u_sq = np.empty((self.N, other.N), dtype=float)
            for i in range(self.N):
                Ui = self.frame[i]
                overlaps = np.einsum("dk,mde->mke", Ui, other.frame)
                s = np.clip(
                    np.linalg.svd(overlaps, compute_uv=False), 0.0, 1.0)
                thetas = np.arccos(s)
                dist_u_sq[i] = np.sum(thetas * thetas, axis=1)
        else:
            raise ValueError(
                f"Unknown subspace_metric '{subspace_metric}'. "
                "Expected 'chordal' or 'geodesic'.")

        return np.sqrt(dist_x_sq + alpha * dist_u_sq)

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
        # bandwidth / effective sample size.  Absolute floor ensures
        # invertibility even when lam=0.
        tr_TWT = np.trace(TWT)
        ridge = lam * (tr_TWT / d) if tr_TWT > 0 else lam
        ridge = max(ridge, 1e-12 * (tr_TWT / d if tr_TWT > 0 else 1.0))
        reg = TWT + ridge * np.eye(d)

        try:
            B_flat_T = np.linalg.solve(reg, TWc)  # (d, n_comp*d)
        except np.linalg.LinAlgError:
            return np.zeros((n_comp * d, d), dtype=float)

        return B_flat_T.T                          # (n_comp*d, d)

    @staticmethod
    def _fit_curvature_symmetric(
        t: np.ndarray,      # (k, d)     intrinsic displacements
        C: np.ndarray,      # (k, m, d)  Grassmann tangent coords  (m = n_comp)
        w: np.ndarray,      # (k,)       non-negative weights
        d: int,
        n_comp: int,
        lam: float,
        D_w: np.ndarray,    # (d², d*(d+1)//2)  weighted duplication matrix
    ) -> np.ndarray:
        """
        Symmetric-constrained weighted ridge regression for B_i.

        Reparametrises each d×d block B_α via vech_w (its d(d+1)/2 unique
        entries with √2 scaling on off-diagonals) so the solution is
        symmetric by construction.  Solves:

          (D_w^T (T^T W T ⊗ I_d) D_w + ridge I_m) s_α = D_w^T vec(R_α)

        Returns B_flat of shape (n_comp*d, d), or zeros on failure.
        """
        if t.shape[0] == 0 or n_comp == 0:
            return np.zeros((n_comp * d, d), dtype=float)

        m = d * (d + 1) // 2

        c_flat = C.reshape(len(t), n_comp * d)     # (k, n_comp*d)
        wt = w[:, np.newaxis] * t                   # (k, d)
        TWT = wt.T @ t                              # (d, d)
        TWc = wt.T @ c_flat                         # (d, n_comp*d)

        tr_TWT = np.trace(TWT)
        ridge = lam * (tr_TWT / d) if tr_TWT > 0 else lam
        ridge = max(ridge, 1e-12 * (tr_TWT / d if tr_TWT > 0 else 1.0))

        # Gram matrix in vech_w space: G = D_w^T (TWT ⊗ I_d) D_w
        G = D_w.T @ np.kron(TWT, np.eye(d)) @ D_w  # (m, m)
        reg = G + ridge * np.eye(m)

        # RHS for each α:  D_w^T vec(R_α)
        # TWc reshaped gives R_α blocks of shape (d, d)
        R_all = TWc.reshape(d, n_comp, d).transpose(1, 0, 2)  # (n_comp, d, d)
        all_vecs = R_all.reshape(n_comp, d * d).T              # (d², n_comp)
        rhs = D_w.T @ all_vecs                                 # (m, n_comp)

        try:
            s_all = np.linalg.solve(reg, rhs)                  # (m, n_comp)
        except np.linalg.LinAlgError:
            return np.zeros((n_comp * d, d), dtype=float)

        # Reconstruct symmetric B_α from D_w @ s_α
        B_vecs = D_w @ s_all                                   # (d², n_comp)
        B_flat = B_vecs.T.reshape(n_comp, d, d).reshape(n_comp * d, d)
        return B_flat

    # ------------------------------------------------------------------
    # Core: lift to the next level
    # ------------------------------------------------------------------

    def lift(
        self,
        *,
        k: int = 20,
        alpha: float = 1.0,
        lam: float = 0.0,
        weight_config: "WeightConfig | None" = None,
        symmetric: bool = False,
    ) -> "BlowUpLevel":
        """
        Compute one step of the iterated blow-up.

        Given level-(ell-1) data (Phi^{ell-1}, U^{ell-1}), produces
        level-ell data (Phi^{ell}, U^{ell}):

          1. Embed:
               Phi^{ell} = (Phi^{ell-1},  sqrt(alpha/2) * vec(P^{ell-1}))

          2. Estimate curvature operator B_i by weighted ridge regression:
               C_ij = N_i^T (P_j - P_i) U_i  ~  B_i t_ij
             where t_ij = U_i^T (Phi_j^{ell-1} - Phi_i^{ell-1}).

          3. Construct tangent vectors for each basis direction a = 1..d:
               g_a = (U_i e_a,  sqrt(alpha/2) * vec(N_i A_a U_i^T + U_i A_a^T N_i^T))
             where A_a = B_i(e_a). Stack into G_i, QR-orthonormalize, form P^{ell}.

        Args:
            k:     number of nearest neighbours for curvature estimation.
            alpha: Chordal-Sasaki weight (alpha_ell in the theory).
            lam:   ridge regularisation parameter (>= 0).
            weight_config: Regression weight configuration.  If None,
                defaults to product kernel with sigma_x=0.5, sigma_u=0.5.
            symmetric: If True, constrain each d×d curvature block to be
                symmetric via vech_w reparametrisation (solves the true
                constrained optimum).  If False (default), solve
                unconstrained and symmetrise in ``extract_level1``.

        Returns:
            BlowUpLevel at level ell = self.level + 1.
        """
        from .weight_config import WeightConfig, product_weights
        if weight_config is None:
            weight_config = WeightConfig()
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

        # k-NN in the lifted Chordal-Sasaki space (separates sheets)
        tree = cKDTree(Phi_next)
        _, nn_idx = tree.query(Phi_next, k=k_eff + 1)  # (N, k_eff+1) incl. self
        nn_idx = nn_idx[:, 1:]                          # (N, k_eff) excl. self

        P_all = self.projectors          # (N, D, D)
        N_frames = self._complement_frames()  # (N, D, n_comp)

        # B_all[i] shape (n_comp, d, d):  B_all[i][:, :, a] = B_i(e_a) in R^{n_comp x d}
        B_all = np.zeros((N, n_comp, d, d), dtype=float)

        # Precompute duplication matrix for symmetric solver
        D_w = _build_vech_w_duplication(d) if symmetric else None

        for i in range(N):
            Ui = self.frame[i]          # (D, d)
            Ni = N_frames[i]            # (D, n_comp)
            Pi = P_all[i]               # (D, D)
            neighbors = nn_idx[i]       # (k_eff,)

            # Intrinsic coords: t_ij = U_i^T (Phi_j^{ell-1} - Phi_i^{ell-1})
            d_emb = self.embedded[neighbors] - self.embedded[i]  # (k_eff, D)
            t = d_emb @ Ui                                        # (k_eff, d)

            # Grassmann tangent coords: C_ij = N_i^T (P_j - P_i) U_i
            delta_P = P_all[neighbors] - Pi               # (k_eff, D, D)

            # Regression weights (product kernel or uniform)
            dx = self.embedded[neighbors, :self.n_orig] - self.embedded[i, :self.n_orig]
            spatial_d2 = np.einsum("ki,ki->k", dx, dx)
            angular_d2 = np.einsum("kab,kab->k", delta_P, delta_P)
            w = product_weights(spatial_d2, angular_d2, weight_config)
            dP_Ui = delta_P @ Ui                          # (k_eff, D, d)
            # einsum: C[k,p,d] = sum_m N_i.T[p,m] * dP_Ui[k,m,d]
            C = np.einsum("pm,kmd->kpd", Ni.T, dP_Ui)   # (k_eff, n_comp, d)

            if symmetric:
                B_flat = self._fit_curvature_symmetric(
                    t, C, w, d, n_comp, lam, D_w,
                )
            else:
                B_flat = self._fit_curvature(t, C, w, d, n_comp, lam)
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
    k: int = 20,
    lam: float = 0.0,
    weight_config: "WeightConfig | None" = None,
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
    from .weight_config import WeightConfig, product_weights
    if weight_config is None:
        weight_config = WeightConfig()

    N = level0.N
    d = level0.d
    h_all = level1_inv.shape_operator   # (N, n_comp, d, d)
    n_comp = h_all.shape[1]

    k_eff = min(k, N - 1)
    # k-NN in the level-1 Chordal-Sasaki space, consistent with lift()
    tree = cKDTree(level1.embedded)
    _, nn_idx = tree.query(level1.embedded, k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]   # (N, k_eff) excluding self

    # Level-1 projectors for angular distance
    P1_all = level1.projectors  # (N, D1, D1)

    grad_h = np.zeros((N, n_comp, d, d, d), dtype=float)

    for i in range(N):
        Ui = level0.frame[i]          # (n, d)
        neighbors = nn_idx[i]         # (k_eff,)

        # Intrinsic displacements in level-0 position space
        d_emb = level0.embedded[neighbors] - level0.embedded[i]   # (k_eff, n)
        t = d_emb @ Ui                                              # (k_eff, d)

        # Regression weights (product kernel or uniform)
        dx = level0.embedded[neighbors] - level0.embedded[i]       # spatial
        spatial_d2 = np.einsum("ki,ki->k", dx, dx)
        dP = P1_all[neighbors] - P1_all[i]                         # angular
        angular_d2 = np.einsum("kab,kab->k", dP, dP)
        w = product_weights(spatial_d2, angular_d2, weight_config)

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
    k: int = 20,
    alpha: float = 1.0,
    lam: float = 0.0,
    weight_config: "WeightConfig | None" = None,
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
        weight_config: Regression weight configuration for curvature estimation.

    Returns:
        List [level_0, level_1, ..., level_{num_levels}].
    """
    levels: list[BlowUpLevel] = [BlowUpLevel.from_point_tangents(points, frames)]
    for _ in range(num_levels):
        levels.append(levels[-1].lift(k=k, alpha=alpha, lam=lam,
                                      weight_config=weight_config))
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
