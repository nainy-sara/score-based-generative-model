"""
score_model.py
==============
2D time-conditioned U-Net for the score model.

PAPER CONTEXT
-------------
Webber et al. (2024), Section II, says:

  "We train an unconditional time-dependent SGM s_theta(x, t) to model
   the score function from a training set of high-quality 2D transverse
   brain slices extracted from 3D full-count images, via denoising score
   matching [5]."

Reference [5] is Song & Ermon (2019), 'Generative Modeling by Estimating
Gradients of the Data Distribution', NeurIPS 32.

We use the variance-preserving (VP) formulation, so the forward process is

        x_t = gamma_t * x_0 + nu_t * eps,     eps ~ N(0, I)

with the cosine schedule (Nichol & Dhariwal).  Song & Ermon's original
NCSN used variance-exploding noise; the two are mathematically equivalent
after the substitution sigma <-> nu_t.  We parameterise the network to
predict the NOISE eps, so the score is recovered as

        s_theta(x_t, t) = - eps_theta(x_t, t) / nu_t

which is exactly the mapping used by Eq. (10) of the paper.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding of the diffusion time t.
    Identical in spirit to the Transformer positional encoding: after
    this layer the network can tell *which* noise level it is denoising.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with time conditioning.
    Structure:  SiLU -> GroupNorm -> Conv -> + time MLP -> SiLU -> GroupNorm
                -> Conv -> + skip.
    """
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1    = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2    = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        # Inject the time information as a per-channel additive bias.
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ScoreUNet(nn.Module):
    """
    Small U-Net that predicts the noise eps from a noisy image x_t.

    Input  : x of shape (B, 1, H, W)   and   t of shape (B,)   [integer]
    Output : eps_theta(x, t) of shape  (B, 1, H, W)
    """
    def __init__(self, base_ch=32, time_dim=128):
        super().__init__()
        # --- Time embedding MLP (same shape convention as DDPM) ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # --- Encoder (3 down-sampling levels) ---
        self.enc1       = ResidualBlock(1,                base_ch,     time_dim)
        self.enc2       = ResidualBlock(base_ch,          base_ch * 2, time_dim)
        self.enc3       = ResidualBlock(base_ch * 2,      base_ch * 4, time_dim)
        self.pool       = nn.MaxPool2d(2)
        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(base_ch * 4,      base_ch * 4, time_dim)
        # --- Decoder (U-Net skip connections) ---
        self.up3        = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3       = ResidualBlock(base_ch * 8,      base_ch * 4, time_dim)
        self.up2        = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2       = ResidualBlock(base_ch * 4,      base_ch * 2, time_dim)
        self.up1        = nn.ConvTranspose2d(base_ch * 2, base_ch,     2, stride=2)
        self.dec1       = ResidualBlock(base_ch * 2,      base_ch,     time_dim)
        # --- Final 1x1 conv to a single noise channel ---
        self.out_conv   = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x,                  t_emb)
        e2 = self.enc2(self.pool(e1),      t_emb)
        e3 = self.enc3(self.pool(e2),      t_emb)
        b  = self.bottleneck(self.pool(e3), t_emb)
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)
