"""
data_utils.py — DICOM loading + VECTORIZED 2D Radon projector.
"""
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
from scipy.ndimage import zoom


def load_brain_pet_volume(pet_dir, target_size=128, verbose=True):
    """Load PET DICOM series -> (H, W, D) numpy in [0, 1]."""
    if not os.path.isdir(pet_dir):
        raise FileNotFoundError(f"PET directory not found: {pet_dir}")

    dcm_files = sorted(glob.glob(os.path.join(pet_dir, "**", "*.dcm"),
                                 recursive=True))
    if not dcm_files:
        dcm_files = sorted([
            f for f in glob.glob(os.path.join(pet_dir, "**", "*"), recursive=True)
            if os.path.isfile(f)
            and not f.endswith((".txt", ".csv", ".json", ".md"))
        ])
    if verbose:
        print(f"Found {len(dcm_files)} DICOM files")

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            slices.append(ds.pixel_array.astype(np.float32))
        except Exception as e:
            if verbose:
                print(f"  skip {os.path.basename(f)}: {e}")

    if not slices:
        raise RuntimeError("No readable DICOM slices.")

    volume = np.stack(slices, axis=0)  # (D, H, W)
    if verbose:
        print(f"Raw volume: {volume.shape}")

    D_target = max(20, min(120, volume.shape[0]))
    volume = zoom(volume, (
        D_target / volume.shape[0],
        target_size / volume.shape[1],
        target_size / volume.shape[2],
    ), order=1)
    volume = np.transpose(volume, (1, 2, 0))  # (H, W, D)

    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)

    if verbose:
        print(f"Final volume: {volume.shape}, range=[{volume.min():.3f}, {volume.max():.3f}]")
    return volume.astype(np.float32)


def make_real_brain_slices_2d(volume, num_slices=3000, size=128, seed=0):
    """Extract 2D slices with random slice index + flip + intensity jitter."""
    rng = np.random.RandomState(seed)
    H, W, D = volume.shape
    all_slices = []
    for _ in range(num_slices):
        z = rng.randint(0, D)
        s = volume[:, :, z].copy()
        if s.shape != (size, size):
            s = zoom(s, (size / s.shape[0], size / s.shape[1]), order=1)
            s = s[:size, :size]
            if s.shape != (size, size):
                padded = np.zeros((size, size), dtype=np.float32)
                padded[:s.shape[0], :s.shape[1]] = s
                s = padded
        if rng.rand() > 0.5:
            s = s[:, ::-1].copy()
        s = s * rng.uniform(0.9, 1.1)
        s = np.clip(s, 0.0, 1.0)
        all_slices.append(s.astype(np.float32))
    arr = np.stack(all_slices, axis=0)
    return torch.tensor(arr).unsqueeze(1)


def get_ground_truth_slice_2d(volume, size=128):
    """Middle axial slice as reconstruction target."""
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
    VECTORIZED 2D Radon projector.
    ONE grid_sample call for all angles -> ~40x faster than the loop version.
    image (B, 1, H, W) -> sinogram (B, num_angles, W).
    """
    def __init__(self, num_angles=60):
        super().__init__()
        self.num_angles = num_angles
        angles = torch.linspace(0.0, math.pi, num_angles + 1)[:-1]
        self.register_buffer("angles", angles)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Repeat image along angle axis: (B*A, C, H, W)
        x_rep = x.repeat(1, self.num_angles, 1, 1).view(
            B * self.num_angles, C, H, W
        )

        # Build rotation matrices: (A, 2, 3) -> (B*A, 2, 3)
        cos_a = torch.cos(self.angles).to(device)
        sin_a = torch.sin(self.angles).to(device)
        zeros = torch.zeros_like(cos_a)
        theta = torch.stack([
            torch.stack([cos_a, -sin_a, zeros], dim=1),
            torch.stack([sin_a,  cos_a, zeros], dim=1),
        ], dim=1).repeat(B, 1, 1)

        # One affine_grid + one grid_sample
        grid = F.affine_grid(theta, x_rep.shape, align_corners=False)
        rot = F.grid_sample(x_rep, grid, align_corners=False,
                            padding_mode="zeros")
        sino = rot.sum(dim=2)  # (B*A, C, W)
        return sino.view(B, self.num_angles, W)


def generate_low_count_sinogram_2d(phantom_2d, proj, count_fraction=0.01, seed=42):
    """Forward project, scale to count_fraction, Poisson sample."""
    rng = np.random.RandomState(seed)
    img_t = torch.tensor(phantom_2d).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        clean = proj(img_t).squeeze(0).numpy()
    total = clean.sum()
    target = total * count_fraction
    clean_scaled = clean * (target / total) if total > 0 else clean
    measured = rng.poisson(clean_scaled).astype(np.float32)
    return (torch.tensor(measured).unsqueeze(0),
            torch.tensor(clean_scaled).unsqueeze(0),
            torch.tensor(clean).unsqueeze(0))
