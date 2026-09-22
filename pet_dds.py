"""
pet_dds.py — 2D PET-DDS-delta.

CONVENTIONS:
  * Model predicts NOISE eps.
  * Score = -eps / nu_t.
  * Objective to MINIMISE:
        E = (1/n_sub) * (NLL + lambda_RDP * RDP + lambda_DDS * DDS)
    where NLL is the negative Poisson log-likelihood.
    NLL is only active for k > 20 (see notes in forward()).
  * Update: x <- x - delta * grad(E)   (gradient DESCENT)
  * DDPM ancestral update: x_t = g_curr * x0_hat + n_curr * z
  * Training t in [0, N] to match sampler.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Tuple
from score_model import ScoreUNet


def make_diffusion_schedule(N=100, device="cpu"):
    """Cosine VP schedule. Index 0 = clean, index N = pure noise."""
    s = 0.008
    t = torch.linspace(0, 1, N + 1, device=device)
    f = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = torch.clamp(f / f[0], min=1e-5, max=0.9999)
    gamma = torch.sqrt(alpha_bar)
    nu = torch.sqrt(1.0 - alpha_bar)
    eta = 0.1 * nu
    return gamma, nu, eta


def poisson_nll(x, measured, proj, scatter=None):
    """Negative Poisson log-likelihood (to be MINIMISED)."""
    predicted = proj(x)
    if scatter is not None:
        predicted = predicted + scatter
    predicted = torch.clamp(predicted, min=1e-3, max=1e4)
    return -torch.sum(measured * torch.log(predicted) - predicted)


def rdp_2d(x, beta=1.0, delta=1e-9):
    """
    In-plane RDP using SLICING (no torch.roll, no wrap-around).
    x has shape (..., H, W).
    """
    # Horizontal neighbours
    dx = x[..., :-1] - x[..., 1:]
    denom_x = torch.clamp(
        x[..., :-1] + x[..., 1:] + beta * torch.abs(dx) + delta, min=1e-6)
    rdp_x = torch.sum(dx ** 2 / denom_x)
    # Vertical neighbours
    dy = x[..., :-1, :] - x[..., 1:, :]
    denom_y = torch.clamp(
        x[..., :-1, :] + x[..., 1:, :] + beta * torch.abs(dy) + delta, min=1e-6)
    rdp_y = torch.sum(dy ** 2 / denom_y)
    return rdp_x + rdp_y


class ScoreModelTrainer(pl.LightningModule):
    """
    Noise-prediction denoising score matching:
        x_t = gamma_t * x0 + nu_t * eps
        loss = MSE(model(x_t, t), eps)
    """
    def __init__(self, lr=1e-4, N=100, base_ch=32, time_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.N = N
        self.model = ScoreUNet(base_ch=base_ch, time_dim=time_dim)
        gamma, nu, eta = make_diffusion_schedule(N)
        self.register_buffer("gamma", gamma)
        self.register_buffer("nu", nu)
        self.register_buffer("eta", eta)

    def training_step(self, batch, batch_idx):
        x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        B = x0.shape[0]
        # t in [0, N] inclusive
        t = torch.randint(0, self.N + 1, (B,), device=x0.device)
        eps = torch.randn_like(x0)
        g_t = self.gamma[t].view(B, 1, 1, 1)
        n_t = self.nu[t].view(B, 1, 1, 1)
        x_t = g_t * x0 + n_t * eps
        eps_pred = self.model(x_t, t)
        loss = F.mse_loss(eps_pred, eps)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


class PETDDSDeltaReconstructor(pl.LightningModule):
    """
    2D PET-DDS-delta sampler.
    NLL is only active for k > 20, so the diffusion prior can complete
    the image at small t without the Poisson likelihood corrupting it.
    """
    def __init__(
        self,
        score_model: nn.Module,
        proj,
        measured_sinogram: torch.Tensor,
        scatter_sinogram: Optional[torch.Tensor] = None,
        N: int = 100,
        p: int = 5,
        delta: float = 0.005,
        lambda_RDP: float = 1e-4,
        lambda_DDS: float = 1000.0,
        n_sub: int = 21,
        img_shape: Tuple[int, int] = (128, 128),
        verbose: bool = True,
        nll_cutoff_k: int = 10,
    ):
        super().__init__()
        self.score_model = score_model
        self.proj = proj
        self.measured_sinogram = measured_sinogram
        self.scatter_sinogram = scatter_sinogram
        self.N = N
        self.p = p
        self.delta = delta
        self.lambda_RDP = lambda_RDP
        self.lambda_DDS = lambda_DDS
        self.n_sub = n_sub
        self.img_shape = img_shape
        self.verbose = verbose
        self.nll_cutoff_k = nll_cutoff_k
        gamma, nu, eta = make_diffusion_schedule(N)
        self.register_buffer("gamma", gamma)
        self.register_buffer("nu", nu)
        self.register_buffer("eta", eta)

    def forward(self):
        device = self.device
        N = self.N
        p = self.p

        x_next = torch.randn(1, 1, *self.img_shape, device=device)

        for k in reversed(range(N)):
            t_next = k + 1
            t_curr = k
            g_next = self.gamma[t_next]
            n_next = self.nu[t_next]
            g_curr = self.gamma[t_curr]
            n_curr = self.nu[t_curr]

            # ---- Step 1: model predicts noise ----
            t_batch = torch.tensor([t_next], device=device)
            eps_pred = self.score_model(x_next, t_batch)

            # ---- Step 2: Tweedie estimate x0_hat (clean image estimate) ----
            x0_hat = ((x_next - n_next * eps_pred) / g_next).detach()

            # ---- Step 3-4: p GD steps on x0_hat directly ----
            for _ in range(p):
                x0_hat = x0_hat.detach().requires_grad_(True)
                x_pos = torch.clamp(x0_hat, min=0.0)
                rdp = rdp_2d(x_pos)
                # DDS: keep x0 close to the diffusion estimate
                dds = torch.sum((x0_hat - (x_next - n_next * eps_pred)
                                 / g_next) ** 2)

                # ---- NLL only active when the image is still noisy ----
                if k < self.nll_cutoff_k:
                    nll = poisson_nll(x_pos, self.measured_sinogram,
                                      self.proj, self.scatter_sinogram)
                else:
                    nll = torch.tensor(0.0, device=x_pos.device)

                E = (1.0 / self.n_sub) * (
                    nll
                    + self.lambda_RDP * rdp
                    + self.lambda_DDS * dds
                )
                grad = torch.autograd.grad(E, x0_hat)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad = torch.clamp(grad, min=-1.0, max=1.0)
                with torch.no_grad():
                    x0_hat = x0_hat - self.delta * grad

            # ---- Step 5: DDPM ancestral update ----
            with torch.no_grad():
                x0_hat = torch.clamp(x0_hat, min=0.0, max=1.0)
                z = torch.randn_like(x_next)
                if k > 0:
                    x_next = g_curr * x0_hat + n_curr * z
                else:
                    x_next = g_curr * x0_hat

            if self.verbose and (k % 20 == 0 or k == N - 1):
                print(f"  k={k:3d}  x0_hat range=[{x0_hat.min():.3f}, "
                      f"{x0_hat.max():.3f}]  x0_hat std={x0_hat.std():.3f}  "
                      f"x_next std={x_next.std():.3f}")

        return x_next.detach()

    def reconstruct(self, num_samples=1, seed=None):
        recons = []
        for s in range(num_samples):
            if seed is not None:
                torch.manual_seed(seed + s)
            recons.append(self.forward().squeeze().cpu().numpy())
        recons = np.stack(recons, axis=0)
        mean_r = np.mean(recons, axis=0)
        std_r = np.std(recons, axis=0)
        return (mean_r, std_r,
                std_r / (np.abs(mean_r) + 1e-8))

    def configure_optimizers(self):
        """Inference only — no optimizer."""
        return None
