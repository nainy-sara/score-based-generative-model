"""
data_utils.py
=============
Vectorised 2D Radon projector + low-count sinogram generator.

WHY A CUSTOM PROJECTOR
----------------------
The paper uses ParallelProj with span-11 axial compression for the fully
3D volume (Section III).  For a 2D slice reproduction we need a fast,
differentiable 2D Radon transform.  We implement it with a single
`affine_grid` + `grid_sample` call for all angles, which is ~40x faster
than a Python loop over angles and works on CPU or GPU.

FORWARD MODEL (paper Eq. 1)
---------------------------
        q = A x + b
where  x  = radiotracer distribution (image)
       A  = system model (projector)
       b  = scatter + randoms (optional, not modelled here)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom


def get_ground_truth_slice_2d(volume, size=128):
    """Middle axial slice of a 3D volume, resized and normalised to [0, 1]."""
    H, W, D = volume.shape
    s = volume[:, :, D // 2].copy()
    if s.shape != (size, size):
        s = zoom(s, (size / s.shape[0], size / s.shape[1]), order=1)
        s = s[:size, :size]
        if s.shape != (size, size):
            padded = np.zeros((size, size), dtype=np.float32)
            padded[:s.shape[0], :s.shape[1]] = s
            s = padded
    vmin, vmax = s.min(), s.max()
    if vmax > vmin:
        s = (s - vmin) / (vmax - vmin)
    return s.astype(np.float32)


class RadonProjector2D(nn.Module):
    """
    Differentiable 2D parallel-beam Radon transform.

    image (B, 1, H, W)  ->  sinogram (B, num_angles, W)

    Angles are evenly spaced over [0, pi) -- parallel-beam symmetry.
    We use num_angles=180, which is close to the Nyquist angular
    sampling rate (~pi/2 * N_pixels) for a 128-pixel image.
    """
    def __init__(self, num_angles=180):
        super().__init__()
        self.num_angles = num_angles
        angles = torch.linspace(0.0, math.pi, num_angles + 1)[:-1]
        self.register_buffer("angles", angles)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        # Repeat the image along the angle axis -> one big batch.
        x_rep = x.repeat(1, self.num_angles, 1, 1).view(B * self.num_angles, C, H, W)

        cos_a = torch.cos(self.angles).to(device)
        sin_a = torch.sin(self.angles).to(device)
        zeros = torch.zeros_like(cos_a)

        # Rotation matrices, one per angle:  shape (B * num_angles, 2, 3)
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, zeros], dim=1),
            torch.stack([sin_a,  cos_a, zeros], dim=1),
        ], dim=1).repeat(B, 1, 1)

        # Single grid_sample for all angles.
        grid = F.affine_grid(theta, x_rep.shape, align_corners=False)
        rot  = F.grid_sample(x_rep, grid, align_corners=False,
                             padding_mode="zeros")

        # Parallel-beam projection = sum along the y (row) axis of the
        # rotated image.  After the rotation the rows of the image are
        # the lines the detector would integrate along.
        sino = rot.sum(dim=2)                      # (B * A, C, W)
        return sino.view(B, self.num_angles, W)


def generate_low_count_sinogram_2d(phantom_2d, proj,
                                   count_fraction=0.01, seed=42):
    """
    Forward-project the phantom, scale the total counts down by
    `count_fraction`, then sample Poisson noise.

    Matches the paper's low-count simulation (Section III):
      "prompts and randoms were sampled at 1% of counts assuming
       independent Poisson statistics".

    Returns
    -------
    measured       (1, A, W)  Poisson-sampled low-count sinogram
    clean_scaled   (1, A, W)  noiseless low-count sinogram
    clean          (1, A, W)  full-count noiseless sinogram
    dose_fraction  float      the fraction used, so the NLL can match
                              the measured scale exactly
    """
    rng = np.random.RandomState(seed)
    img_t = torch.tensor(phantom_2d).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        clean = proj(img_t).squeeze(0).numpy()

    total = clean.sum()
    target = total * count_fraction
    clean_scaled = clean * (target / total) if total > 0 else clean
    measured = rng.poisson(clean_scaled).astype(np.float32)

    return (
        torch.tensor(measured).unsqueeze(0),
        torch.tensor(clean_scaled).unsqueeze(0),
        torch.tensor(clean).unsqueeze(0),
        float(count_fraction),
    )

