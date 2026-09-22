"""
score_model.py — 2D time-conditioned U-Net.
The model predicts the NOISE eps (not the score).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class ScoreUNet(nn.Module):
    """Input: (B, 1, H, W), t: (B,). Output: (B, 1, H, W) predicting eps."""
    def __init__(self, base_ch=32, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = ResidualBlock(1, base_ch, time_dim)
        self.enc2 = ResidualBlock(base_ch, base_ch * 2, time_dim)
        self.enc3 = ResidualBlock(base_ch * 2, base_ch * 4, time_dim)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_ch * 4, base_ch * 4, time_dim)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_ch * 8, base_ch * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ResidualBlock(base_ch * 2, base_ch, time_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        b = self.bottleneck(self.pool(e3), t_emb)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)
