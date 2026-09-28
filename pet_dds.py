"""
pet_dds.py
==========
The two core objects of the reproduction:

  1. ScoreModelTrainer      -- trains the score network s_theta(x, t)
                               via denoising score matching
                               (Song & Ermon 2019, paper's ref [5]).

  2. PETDDSDeltaReconstructor
                            -- performs the reverse diffusion while
                               enforcing data consistency, i.e. the
                               PET-DDS-delta sampler of Webber et al.
                               (2024), Eq. (10).

Eq. (10) of the paper, one iteration k of N = 100, given iterate
x_{t_{k+1}}:

   (a) x_{t_k}^{(s)}   <- s_theta(x_{t_{k+1}}, t_{k+1})                [diffusion]
   (b) x_{t_{k+1}}^0   <- gamma_{t_{k+1}}^{-1} (x_{t_{k+1}}
                            + nu_{k+1}^2 x_{t_k}^{(s)})                [Tweedie]
   (c) Phi_j(x)        <- L_j(|x|) + (1/n_sub) [ lambda_RDP RDP_z(x)
                            - lambda_DDS ||x - x_{t_{k+1}}^0||^2 ]    [objective]
   (d) for i=1..p:  x^i <- GD( delta * Phi_{j+i}(x^{i-1}) )           [data consistency]
   (e) j <- j + i;  z ~ N(0, I)                                       [resample]
   (f) x_{t_k} <- gamma_{t_k} x^p
                    - nu_{t_{k+1}} sqrt(nu_{t_k}^2 - eta^2) x^{(s)}
                    + eta_{t_{k+1}} z                                    [reverse update]

Since our network predicts eps, not the score, we use s_theta = -eps/nu.

TWO HONEST CHANGES vs THE PAPER
-------------------------------
(1) RDP_z -> rdp_2d.  The paper uses an *axial* relative difference prior
    to couple neighbouring z slices for 3D reconstruction.  In 2D there
    is no z axis, so we substitute an in-plane RDP of the same form.

(2) The paper prints -lambda_DDS * ||x - x_0||^2 inside Phi.  If taken
    literally it would *push away* from the Tweedie estimate, which
    cannot be intended.  We minimise +lambda_DDS * ||x - x_0||^2, i.e.
    we pull TOWARD the Tweedie estimate, which is the correct reading.

Finally, the paper selects delta = 0.2 for the model-free NLL scale of
its data.  In our implementation the Poisson NLL is normalised by the
total measured counts (to make it dimensionless), so gradients are much
smaller and a larger delta (about 20) is needed to give a comparable step.
This is discussed explicitly in the reconstruct script.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

from score_model import ScoreUNet


# ============================================================================
# Cosine variance-preserving diffusion schedule (Nichol & Dhariwal 2021)
# ---------------------------------------------------------------------------
# The forward process is  x_t = gamma_t x_0 + nu_t eps  with
#         gamma_t^2 + nu_t^2 = 1  (variance-preserving),
# and  eta_t = 0.1 * nu_t  is the stochasticity level used in the
# re-noising step (f) of Eq. (10).
# ============================================================================
def make_diffusion_schedule(N=100, device="cpu"):
    s = 0.008
    t = torch.linspace(0, 1, N + 1, device=device)
    f = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = torch.clamp(f / f[0], min=1e-5, max=0.9999)
    gamma = torch.sqrt(alpha_bar)          # signal scale
    nu    = torch.sqrt(1.0 - alpha_bar)    # noise scale
    eta   = 0.1 * nu                       # reverse-process stochasticity
    return gamma, nu, eta


# ============================================================================
# Poisson negative log-likelihood  --  the L_j(|x|) term in Eq. (10c)
# ---------------------------------------------------------------------------
# The forward model is  q = A x + b  (paper Eq. 1).  The standard Poisson
# negative log-likelihood is
#
#        L_j(x) = sum_over_bins( q - m * log(q) )
#
# We normalise by sum(m) so the objective is dimensionless; this makes the
# value independent of the absolute count level, which is what lets the
# same delta be used across count fractions.  The `dose_fraction` argument
# scales the forward projection back to the low-count scale so the NLL is
# compared to the correct measurement magnitude.
# ============================================================================
def poisson_nll(x, measured, proj, scatter=None, dose_fraction=1.0):
    q = dose_fraction * proj(x)
    if scatter is not None:
        q = q + scatter
    q = torch.clamp(q, min=1e-3, max=1e4)
    nll = torch.sum(q - measured * torch.log(q))
    return nll / (measured.sum() + 1e-8)


# ============================================================================
# In-plane Relative Difference Prior
# ---------------------------------------------------------------------------
# The paper's 3D RDP_z couples neighbouring axial slices; here we use the
# same functional form but only on the 2D in-plane neighbours.  For a pair
# of neighbouring intensities a, b the RDP contribution is
#
#        (a - b)^2 / (a + b + beta * |a - b| + eps)
#
# which is a smooth edge-preserving prior: it penalises differences but
# flattens out at strong edges.
# ============================================================================
def rdp_2d(x, beta=1.0, delta=1e-9):
    dx      = x[..., :-1] - x[..., 1:]
    denom_x = torch.clamp(x[..., :-1] + x[..., 1:]
                          + beta * torch.abs(dx) + delta, min=1e-6)
    rdp_x   = torch.sum(dx ** 2 / denom_x)

    dy      = x[..., :-1, :] - x[..., 1:, :]
    denom_y = torch.clamp(x[..., :-1, :] + x[..., 1:, :]
                          + beta * torch.abs(dy) + delta, min=1e-6)
    rdp_y   = torch.sum(dy ** 2 / denom_y)

    return rdp_x + rdp_y


# ============================================================================
# Trainer for the score model
# ---------------------------------------------------------------------------
# This is denoising score matching with the variance-preserving schedule.
# For each batch we sample t uniformly in {1, ..., N}, form the noisy image
# x_t = gamma_t x_0 + nu_t eps, and regress the network onto eps:
#
#        L_dsm(theta) = E[ || eps_theta(x_t, t) - eps ||^2 ]
#
# This is the VP analogue of Song & Ermon's NCSN objective and reduces to
# the standard DDPM 'simple' loss.  It is the training procedure the paper
# refers to as "denoising score matching [5]" in Section II.
# ============================================================================
class ScoreModelTrainer(pl.LightningModule):
    def __init__(self, lr=1e-4, N=100, base_ch=32, time_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.N = N
        self.model = ScoreUNet(base_ch=base_ch, time_dim=time_dim)
        gamma, nu, eta = make_diffusion_schedule(N)
        self.register_buffer("gamma", gamma)
        self.register_buffer("nu",    nu)
        self.register_buffer("eta",   eta)

    def training_step(self, batch, batch_idx):
        x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        B  = x0.shape[0]

        # Sample diffusion times uniformly in {1, ..., N}
        t   = torch.randint(1, self.N + 1, (B,), device=x0.device)
        eps = torch.randn_like(x0)

        # Forward process:  x_t = gamma_t x_0 + nu_t eps
        g_t = self.gamma[t].view(B, 1, 1, 1)
        n_t = self.nu[t].view(B, 1, 1, 1)
        x_t = g_t * x0 + n_t * eps

        # Network prediction and DSM loss
        eps_pred = self.model(x_t, t)
        loss     = F.mse_loss(eps_pred, eps)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


# ============================================================================
# PET-DDS-delta sampler
# ---------------------------------------------------------------------------
# See the module docstring above for the full mapping to Eq. (10).
# ============================================================================
class PETDDSDeltaReconstructor(pl.LightningModule):
    def __init__(self, score_model, proj, measured_sinogram,
                 scatter_sinogram=None,
                 dose_fraction=1.0,
                 N=100, p=5, delta=20.0,
                 lambda_DDS=1.0, lambda_RDP=1e-4, n_sub=21,
                 img_shape=(128, 128), verbose=True,
                 nll_cutoff_k=None, skip_first=0,
                 use_ancestral=True, grad_probe=False):
        super().__init__()
        self.score_model       = score_model
        self.proj              = proj
        self.measured_sinogram = measured_sinogram
        self.scatter_sinogram  = scatter_sinogram
        self.dose_fraction     = dose_fraction

        # Hyperparameters from Section III of the paper
        self.N          = N          # 100 diffusion steps
        self.p          = p          # 5 reconstruction steps per diffusion step
        self.delta      = delta      # GD step size (see docstring on why 20, not 0.2)
        self.lambda_DDS = lambda_DDS # 1.0
        self.lambda_RDP = lambda_RDP # 1e-4
        self.n_sub      = n_sub      # 21 (subset count, here a scalar divisor)

        self.img_shape    = img_shape
        self.verbose      = verbose
        self.nll_cutoff_k = N if nll_cutoff_k is None else nll_cutoff_k
        self.skip_first   = skip_first
        self.use_ancestral = use_ancestral
        self.grad_probe    = grad_probe

        gamma, nu, eta = make_diffusion_schedule(N)
        self.register_buffer("gamma", gamma)
        self.register_buffer("nu",    nu)
        self.register_buffer("eta",   eta)

    # ------------------------------------------------------------------
    def forward(self):
        """
        One full reverse trajectory from pure noise to a reconstructed
        image, with a data-consistency block inserted at every step.
        """
        device  = self.device
        N       = self.N
        p       = self.p
        start_k = N - self.skip_first

        # Initialise at the highest noise level of the trajectory
        n_start = self.nu[start_k]
        x_next  = n_start * torch.randn(1, 1, *self.img_shape, device=device)

        for k in reversed(range(start_k)):
            t_next, t_curr = k + 1, k
            g_next, n_next = self.gamma[t_next], self.nu[t_next]
            g_curr, n_curr = self.gamma[t_curr], self.nu[t_curr]

            # --- (a) Diffusion step: score at x_{t_{k+1}} ---
            t_batch  = torch.tensor([t_next], device=device)
            eps_pred = self.score_model(x_next, t_batch)

            # --- (b) Tweedie estimate:  x0 = (x - nu eps) / gamma ---
            x0_tweedie = (x_next - n_next * eps_pred) / g_next
            x0_hat     = x0_tweedie.detach().clone()

            # --- (c)-(d) Data-consistency gradient descent ---
            grad_norm = 0.0
            for _ in range(p):
                x0_hat = x0_hat.detach().requires_grad_(True)
                x_pos  = torch.clamp(x0_hat, min=0.0)   # non-negativity

                # DDS anchor to the Tweedie estimate (fixed across p iterations)
                dds = torch.sum((x0_hat - x0_tweedie.detach()) ** 2)

                # Poisson data fidelity
                if k < self.nll_cutoff_k:
                    nll = poisson_nll(x_pos,
                                      self.measured_sinogram,
                                      self.proj,
                                      self.scatter_sinogram,
                                      dose_fraction=self.dose_fraction)
                else:
                    nll = torch.tensor(0.0, device=x_pos.device)

                # In-plane RDP (replaces the paper's axial RDP_z)
                if self.lambda_RDP > 0.0:
                    rdp = rdp_2d(x_pos)
                else:
                    rdp = torch.tensor(0.0, device=x_pos.device)

                # Objective Phi_j  --  note: n_sub divides ONLY the RDP+DDS
                # terms, exactly as printed in Eq. (10c).
                E = nll + (self.lambda_RDP * rdp
                           + self.lambda_DDS * dds) / self.n_sub

                grad = torch.autograd.grad(E, x0_hat)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                if self.grad_probe and k == start_k - 1:
                    grad_norm = float(grad.abs().max())

                with torch.no_grad():
                    x0_hat = x0_hat - self.delta * grad   # GD step with delta

            if self.grad_probe and k == start_k - 1:
                print(f"  [PROBE] k={k}  max|grad| = {grad_norm:.3e}")

            # --- (f) Reverse update / re-noising ---
            with torch.no_grad():
                x0_hat = torch.clamp(x0_hat, min=0.0, max=1.0)
                if self.use_ancestral:
                    # Eq. (10f) with eta_t = 0.1 * nu_t
                    eta_t = self.eta[t_next]
                    coef  = torch.sqrt(
                        torch.clamp(n_curr ** 2 - eta_t ** 2, min=0.0))
                    x_next = (g_curr * x0_hat
                              + coef * eps_pred
                              + eta_t * torch.randn_like(x0_hat))
                else:
                    # Deterministic DDIM limit (eta = 0)
                    x_next = g_curr * x0_hat + n_curr * eps_pred

            if self.verbose and (k % 20 == 0 or k == start_k - 1):
                print(f"  k={k:3d}  x0 range=[{x0_hat.min():.3f}, "
                      f"{x0_hat.max():.3f}]  x0 std={x0_hat.std():.3f}")

        return x_next.detach()

    # ------------------------------------------------------------------
    def reconstruct(self, num_samples=1, seed=None):
        """
        Draw several independent reverse trajectories from the posterior
        by re-running forward() with different random seeds.  Returns
        the mean, standard deviation, and coefficient of variation,
        which is exactly the uncertainty map described in Figure 1 of
        the paper.
        """
        recons = []
        for s in range(num_samples):
            if seed is not None:
                torch.manual_seed(seed + s)
            recons.append(self.forward().squeeze().cpu().numpy())

        recons = np.stack(recons, axis=0)
        mean_r = np.mean(recons, axis=0)
        std_r  = np.std(recons,  axis=0)
        return mean_r, std_r, std_r / (np.abs(mean_r) + 1e-8)

    def configure_optimizers(self):
        return None
