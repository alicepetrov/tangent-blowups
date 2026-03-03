"""
Neural Poisson Manifold Reconstruction in Blow-Up Space.

Implements Poisson manifold reconstruction (Kohlbrenner et al., "Poisson
Manifold Reconstruction -- Beyond Co-dimension One") using neural implicit
functions in the iterated tangent blow-up space.

For a d-dimensional manifold lifted to D-dimensional blow-up space,
learns F: R^D -> R^{D-d} whose zero-set approximates the lifted manifold.

The energy (paper Eq. 10, general co-dimension, Sec 5.5) is:

    E(F) = || *(grad f_1 ^ ... ^ grad f_{codim})  -  tau_S ||^2
         + alpha * sum_s ||F(s)||^2
         + beta  * sum_k ||grad f_k||^2

where * denotes the Hodge star and tau_S is the tangent d-blade of the
manifold at the sample points.

Loss components:

1. **Exterior-product fit** (Eq. 10): the Hodge dual of the gradient
   exterior product must match the tangent d-blade.  For d=1 this reduces
   to the generalized cross product matching the unit tangent.  This is
   the primary constraint and is NOT satisfied by the trivial solution
   F = 0, because the target tangent blade is nonzero.

2. **Screening**: F(Phi_i) = 0 on the sample points.

3. **Orthonormality** (Sec 5.3, Dirichlet regularization): G G^T ~ I
   for the gradient matrix G, breaking the SL(codim) ambiguity and
   preferring orthonormal gradient frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ...geometry.iterated_grassmann import BlowUpLevel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PoissonConfig:
    """Configuration for neural Poisson reconstruction."""
    hidden_dim: int = 256
    n_layers: int = 4
    lr: float = 1e-3
    n_epochs: int = 2000
    lambda_fit: float = 10.0
    lambda_align: float = 10.0
    lambda_screen: float = 10.0
    lambda_ortho: float = 5.0
    off_manifold_std: float = 0.2
    off_manifold_refresh: int = 200
    batch_size: Optional[int] = None
    device: str = "cuda"
    print_every: int = 200


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ImplicitNet(nn.Module):
    """MLP  F : R^D -> R^{codim}  with smooth (Softplus) activations."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.Softplus(beta=10))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def _compute_jacobian(
    model: nn.Module,
    x: torch.Tensor,
    *,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute F(x) and the full Jacobian dF/dx.

    Args:
        model:  F : R^D -> R^codim
        x:      (N, D) input points.
        create_graph: whether to build the computational graph through the
                      Jacobian (needed for training, not for projection).

    Returns:
        F_val:  (N, codim)
        J:      (N, codim, D)   Jacobian matrix
    """
    x = x.detach().requires_grad_(True)
    F_val = model(x)
    codim = F_val.shape[1]

    grads = []
    for k in range(codim):
        g = torch.autograd.grad(
            F_val[:, k].sum(), x,
            create_graph=create_graph,
            retain_graph=True,
        )[0]                              # (N, D)
        grads.append(g)

    J = torch.stack(grads, dim=1)         # (N, codim, D)
    return F_val, J


# ---------------------------------------------------------------------------
# Exterior algebra helpers
# ---------------------------------------------------------------------------

def _perm_sign(perm: list[int]) -> int:
    """Sign of a permutation (+1 even, -1 odd)."""
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if visited[i]:
            continue
        j = i
        cycle_len = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len > 1:
            sign *= (-1) ** (cycle_len - 1)
    return sign


def _hodge_star_of_gradients(
    J: torch.Tensor,
    hodge_data: list[tuple[int, list[int]]],
) -> torch.Tensor:
    """
    Hodge star of the exterior product of codim gradient vectors.

    Given J = (N, codim, D) where each row is a gradient vector,
    computes the Hodge dual of g_1 ^ ... ^ g_{codim}, yielding a
    d-form with C(D, d) components.

    For d = 1 this is the generalized cross product.

    Args:
        J:          (N, codim, D) gradient matrix.
        hodge_data: list of (sign, complement_indices) for each d-tuple,
                    as returned by _precompute_hodge_data.

    Returns:
        (N, C(D, d))  Hodge-dual blade coordinates.
    """
    components = []
    for sign, I_c in hodge_data:
        minor = J[:, :, I_c]                    # (N, codim, codim)
        det_val = torch.det(minor)              # (N,)
        components.append(sign * det_val)
    return torch.stack(components, dim=1)


def _tangent_blade(
    tangent_frames: torch.Tensor,
    d_tuples: list[tuple[int, ...]],
) -> torch.Tensor:
    """
    Plücker coordinates of the tangent d-blade  t_1 ^ ... ^ t_d.

    For d = 1 this is simply the tangent vector.
    For d = 2 this gives the C(D, 2) pairwise minors.

    Args:
        tangent_frames: (N, D, d) orthonormal tangent frame.
        d_tuples:       sorted list of d-element index tuples.

    Returns:
        (N, C(D, d))  blade coordinates.
    """
    components = []
    for I in d_tuples:
        sub = tangent_frames[:, I, :]            # (N, d, d)
        det_val = torch.det(sub)                 # (N,)
        components.append(det_val)
    return torch.stack(components, dim=1)


def _precompute_hodge_data(
    D: int, d: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, list[int]]]]:
    """
    Precompute index tuples and signs for the Hodge-star computation.

    Returns:
        d_tuples:   list of d-element tuples  (i_1, ..., i_d)  with i_1 < ... < i_d
        hodge_data: list of (sign, complement_indices) for each d-tuple
    """
    all_idx = list(range(D))
    d_tuples = list(combinations(all_idx, d))
    hodge_data: list[tuple[int, list[int]]] = []
    for I in d_tuples:
        I_c = [j for j in all_idx if j not in I]
        perm = list(I) + I_c
        sign = _perm_sign(perm)
        hodge_data.append((sign, I_c))
    return d_tuples, hodge_data


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def poisson_loss(
    F_on: torch.Tensor,             # (N, codim)
    J_on: torch.Tensor,             # (N, codim, D)
    tangent_frames: torch.Tensor,   # (N, D, d)
    J_off: torch.Tensor,            # (M, codim, D)
    config: PoissonConfig,
    d_tuples: list[tuple[int, ...]],
    hodge_data: list[tuple[int, list[int]]],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Neural Poisson manifold-reconstruction loss.

    Four complementary terms:

    1. **Exterior-product fit** (Eq. 10):  the Hodge dual of the gradient
       exterior product must match the tangent d-blade.  Provides correct
       orientation and is not satisfied by the trivial solution F = 0.

    2. **Tangent-orthogonality** (alignment):  each gradient must be
       orthogonal to the tangent space.  Provides a strong determinant-free
       gradient signal that works even for large codimension where the fit
       loss suffers from vanishing gradients through the determinant.

    3. **Screening**:  F vanishes on the manifold samples.

    4. **Orthonormality** (Sec 5.3):  G G^T ~ I on both on-manifold and
       off-manifold points.  The on-manifold term forces unit-norm
       gradients, preventing the trivial solution.
    """
    codim = J_on.shape[1]
    eye = torch.eye(codim, device=J_on.device).unsqueeze(0)

    # 1. Exterior-product fit (on manifold)
    predicted_blade = _hodge_star_of_gradients(J_on, hodge_data)
    target_blade = _tangent_blade(tangent_frames, d_tuples)
    loss_fit = torch.mean((predicted_blade - target_blade) ** 2)

    # 2. Tangent-orthogonality (on manifold)
    tangent_proj = torch.bmm(J_on, tangent_frames)        # (N, codim, d)
    loss_align = torch.mean(tangent_proj ** 2)

    # 3. Screening
    loss_screen = torch.mean(F_on ** 2)

    # 4. Gradient orthonormality (on + off manifold)
    gram_on = torch.bmm(J_on, J_on.transpose(1, 2))
    gram_off = torch.bmm(J_off, J_off.transpose(1, 2))
    loss_ortho = 0.5 * (
        torch.mean((gram_on - eye) ** 2)
        + torch.mean((gram_off - eye) ** 2)
    )

    total = (
        config.lambda_fit * loss_fit
        + config.lambda_align * loss_align
        + config.lambda_screen * loss_screen
        + config.lambda_ortho * loss_ortho
    )

    diagnostics = {
        "fit": loss_fit.item(),
        "align": loss_align.item(),
        "screen": loss_screen.item(),
        "ortho": loss_ortho.item(),
        "total": total.item(),
    }
    return total, diagnostics


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Output of a neural Poisson reconstruction."""
    reconstructed: np.ndarray       # (M, D) points in lifted space
    spatial: np.ndarray             # (M, n_orig) original-space coordinates
    residuals: np.ndarray           # (M,)  ||F(x)||
    training_losses: list[float]


# ---------------------------------------------------------------------------
# Main reconstructor
# ---------------------------------------------------------------------------

class NeuralPoissonReconstructor:
    """
    Neural Poisson manifold reconstruction in blow-up space.

    Given a ``BlowUpLevel`` (lifted point cloud with tangent frames), learns
    F : R^D -> R^{D-d} whose zero-set approximates the lifted manifold.

    Typical usage::

        level0 = BlowUpLevel.from_point_tangents(points, tangents)
        level1 = level0.lift(k=16, alpha=1.0)

        recon = NeuralPoissonReconstructor(level1)
        recon.fit()
        result = recon.reconstruct()
    """

    def __init__(
        self,
        level: BlowUpLevel,
        config: PoissonConfig | None = None,
    ) -> None:
        self.level = level
        self.config = config or PoissonConfig()

        self.D = level.D
        self.d = level.d
        self.codim = self.D - self.d
        self.n_orig = level.n_orig

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = ImplicitNet(
            self.D, self.codim,
            self.config.hidden_dim, self.config.n_layers,
        ).to(self.device)

        # ---------- precompute Hodge-star index data ----------
        self._d_tuples, self._hodge_data = _precompute_hodge_data(
            self.D, self.d,
        )

        # ---------- prepare tensors ----------
        self._pts_on = torch.tensor(
            level.embedded, dtype=torch.float32, device=self.device,
        )
        self._tangent_frames = torch.tensor(
            level.frame, dtype=torch.float32, device=self.device,
        )
        self._training_losses: list[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, n_epochs: int | None = None) -> list[float]:
        """Train the implicit network.  Returns per-epoch total losses."""
        n_epochs = n_epochs or self.config.n_epochs
        cfg = self.config

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=cfg.lr * 0.01,
        )

        N = self._pts_on.shape[0]
        batch_size = cfg.batch_size or N

        losses: list[float] = []
        pts_off_full = self._pts_on.clone()          # placeholder

        for epoch in range(n_epochs):
            # Refresh off-manifold samples periodically
            if epoch % cfg.off_manifold_refresh == 0:
                pts_off_full = (
                    self._pts_on
                    + torch.randn_like(self._pts_on) * cfg.off_manifold_std
                )

            # Mini-batch (optional)
            if batch_size < N:
                idx = torch.randperm(N, device=self.device)[:batch_size]
                pts_on = self._pts_on[idx]
                frames = self._tangent_frames[idx]
                pts_off = pts_off_full[idx]
            else:
                pts_on = self._pts_on
                frames = self._tangent_frames
                pts_off = pts_off_full

            optimizer.zero_grad()

            F_on, J_on = _compute_jacobian(self.model, pts_on)
            _, J_off = _compute_jacobian(self.model, pts_off)

            loss, diag = poisson_loss(
                F_on, J_on, frames, J_off, cfg,
                self._d_tuples, self._hodge_data,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(diag["total"])
            if cfg.print_every and epoch % cfg.print_every == 0:
                print(
                    f"Epoch {epoch:5d}: "
                    f"total={diag['total']:.4f}  "
                    f"fit={diag['fit']:.4f}  "
                    f"align={diag['align']:.4f}  "
                    f"screen={diag['screen']:.5f}  "
                    f"ortho={diag['ortho']:.4f}"
                )

        self._training_losses = losses
        return losses

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate F at arbitrary points.  Returns (N, codim)."""
        x = torch.tensor(points, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()

    # ------------------------------------------------------------------
    # Projection to zero-set
    # ------------------------------------------------------------------

    def project_to_zero_set(
        self,
        points: np.ndarray,
        *,
        n_iters: int = 30,
        lr: float = 0.5,
        batch_size: int = 4096,
    ) -> ReconstructionResult:
        """
        Project points onto F^{-1}(0) via damped Newton iteration:

            x <- x  -  lr * J^T (J J^T)^{-1} F(x)

        where J = dF/dx is the Jacobian and J^T(JJ^T)^{-1} is the right
        pseudo-inverse.  This is the standard normal-space Newton step for
        under-determined systems (D > codim).
        """
        self.model.eval()
        N_total = len(points)
        all_recon: list[np.ndarray] = []
        all_resid: list[np.ndarray] = []

        for start in range(0, N_total, batch_size):
            end = min(start + batch_size, N_total)
            x = torch.tensor(
                points[start:end], dtype=torch.float32, device=self.device,
            )
            for _ in range(n_iters):
                F_val, J = _compute_jacobian(
                    self.model, x, create_graph=False,
                )
                # J^+ F  =  J^T (J J^T + eps I)^{-1} F
                JJT = torch.bmm(J, J.transpose(1, 2))
                reg = 1e-6 * torch.eye(
                    self.codim, device=self.device,
                ).unsqueeze(0)
                JJT = JJT + reg
                alpha = torch.linalg.solve(
                    JJT, F_val.unsqueeze(-1),
                )                                          # (B, codim, 1)
                dx = torch.bmm(
                    J.transpose(1, 2), alpha,
                ).squeeze(-1)                              # (B, D)
                x = (x - lr * dx).detach()

            with torch.no_grad():
                F_final = self.model(x)
                resid = torch.norm(F_final, dim=1).cpu().numpy()

            all_recon.append(x.cpu().numpy())
            all_resid.append(resid)

        self.model.train()
        recon = np.concatenate(all_recon, axis=0)
        residuals = np.concatenate(all_resid, axis=0)

        return ReconstructionResult(
            reconstructed=recon,
            spatial=recon[:, :self.n_orig],
            residuals=residuals,
            training_losses=self._training_losses,
        )

    # ------------------------------------------------------------------
    # Full reconstruction pipeline
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        *,
        n_search: int = 50,
        search_std: float = 0.2,
        n_iters: int = 30,
        lr: float = 0.5,
        residual_threshold: float = 0.01,
        batch_size: int = 4096,
    ) -> ReconstructionResult:
        """
        Full reconstruction pipeline:

        1. Generate a dense search cloud around on-manifold points.
        2. Project to F^{-1}(0) via Newton iteration.
        3. Filter by ``residual_threshold``.

        Returns a :class:`ReconstructionResult` with the filtered points.
        """
        pts_base = self.level.embedded
        pts_dense = np.repeat(pts_base, n_search, axis=0)
        pts_dense += np.random.normal(0, search_std, pts_dense.shape)

        result = self.project_to_zero_set(
            pts_dense, n_iters=n_iters, lr=lr, batch_size=batch_size,
        )

        mask = result.residuals < residual_threshold
        return ReconstructionResult(
            reconstructed=result.reconstructed[mask],
            spatial=result.spatial[mask],
            residuals=result.residuals[mask],
            training_losses=self._training_losses,
        )
