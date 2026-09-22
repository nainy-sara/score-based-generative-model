"""reconstruct_pet_dds.py — 2D PET-DDS-delta on real brain slice."""

import os, glob
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func

from pet_dds import PETDDSDeltaReconstructor, ScoreModelTrainer
from data_utils import (
    load_brain_pet_volume,
    get_ground_truth_slice_2d,
    RadonProjector2D,
    generate_low_count_sinogram_2d,
)


PET_DIR = "/kaggle/input/datasets/grantmcnatt/mri-and-pet-dice-similarity-dataset/data/BrainTumorPET"


def compute_metrics(recon, gt):
    recon = np.nan_to_num(recon, nan=0.0)
    gt = np.nan_to_num(gt, nan=0.0)
    recon = recon / (recon.max() + 1e-8)
    gt = gt / (gt.max() + 1e-8)
    mse = float(np.mean((recon - gt) ** 2))
    nrmse = float(np.sqrt(mse) / (gt.max() - gt.min() + 1e-8) * 100.0)
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-8)))
    ssim_val = float(ssim_func(gt, recon, data_range=1.0))
    return nrmse, psnr, ssim_val * 100.0


def load_best_ckpt():
    best = sorted(glob.glob("checkpoints/best*.ckpt"), key=os.path.getmtime)
    if best:
        return best[-1]
    all_c = sorted(glob.glob("checkpoints/*.ckpt"), key=os.path.getmtime)
    return all_c[-1] if all_c else None


def main():
    pl.seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading DICOM from {PET_DIR}...")
    volume = load_brain_pet_volume(PET_DIR, target_size=128)
    phantom = get_ground_truth_slice_2d(volume, size=128)
    print(f"GT slice: {phantom.shape}, range=[{phantom.min():.3f}, {phantom.max():.3f}]")

    proj = RadonProjector2D(num_angles=60).to(device)

    measured, clean_scaled, clean = generate_low_count_sinogram_2d(
        phantom, proj, count_fraction=0.01, seed=42)
    measured = measured.to(device)
    print(f"Sinogram: sum={measured.sum():.0f}, max={measured.max()}")

    ckpt = load_best_ckpt()
    if ckpt is None:
        print("No checkpoint. Run train_score.py first.")
        return
    print(f"Loading: {ckpt}")
    score_model = ScoreModelTrainer.load_from_checkpoint(
        ckpt, map_location=device)
    score_model.eval()
    score_model.to(device)

    # ---- Sanity check at multiple t values ----
    print("\n--- Score model sanity check ---")
    with torch.no_grad():
        gt_t = torch.tensor(phantom, device=device).unsqueeze(0).unsqueeze(0)
        for t_check in [90, 75, 50, 25]:
            g_tc = score_model.gamma[t_check]
            n_tc = score_model.nu[t_check]
            eps_c = torch.randn_like(gt_t)
            x_noisy_c = g_tc * gt_t + n_tc * eps_c
            eps_pred_c = score_model.model(x_noisy_c,
                                           torch.tensor([t_check], device=device))
            x0_c = (x_noisy_c - n_tc * eps_pred_c) / g_tc
            mse_c = float(((x0_c - gt_t) ** 2).mean())
            print(f"  t={t_check:3d}: Tweedie MSE = {mse_c:.4f}")

    print("\nRunning PET-DDS-delta 2D on REAL brain slice...")
    rec = PETDDSDeltaReconstructor(
        score_model=score_model.model,
        proj=proj,
        measured_sinogram=measured,
        N=100,
        p=5,
        delta=0.005,           # small GD step
        lambda_RDP=1e-4,
        lambda_DDS=100.0,      # strong leash
        n_sub=21,              # match paper
        img_shape=(128, 128),
        verbose=True,
    ).to(device)

    mean_r, std_r, cv_r = rec.reconstruct(num_samples=3, seed=0)

    nrmse, psnr, ssim_val = compute_metrics(mean_r, phantom)
    print(f"\n=== 2D PET-DDS-delta on REAL brain ===")
    print(f"NRMSE: {nrmse:.2f}%  PSNR: {psnr:.2f} dB  SSIM: {ssim_val:.2f}%")

    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(phantom, cmap="gray"); ax[0].set_title("GT brain"); ax[0].axis("off")
    ax[1].imshow(np.log1p(clean.squeeze().numpy()), cmap="gray", aspect="auto"); ax[1].set_title("Clean sino"); ax[1].axis("off")
    ax[2].imshow(np.log1p(measured.cpu().squeeze().numpy()), cmap="gray", aspect="auto"); ax[2].set_title("1% sino"); ax[2].axis("off")
    ax[3].imshow(mean_r, cmap="gray"); ax[3].set_title(f"Recon\nNRMSE={nrmse:.1f}%"); ax[3].axis("off")
    ax[4].imshow(std_r, cmap="hot"); ax[4].set_title("Uncertainty"); ax[4].axis("off")
    plt.tight_layout()
    plt.savefig("pet_dds_result.png", dpi=120)
    print("Saved: pet_dds_result.png")


if __name__ == "__main__":
    main()
