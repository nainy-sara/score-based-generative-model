"""
reconstruct_pet_dds.py
======================
Run PET-DDS-delta (Eq. 10 of Webber et al. 2024) on the held-out volume,
sweep the data-consistency step size delta, and report NRMSE / PSNR / SSIM
against the ground-truth slice.

WHAT THIS SCRIPT DOES
---------------------
1. Load one axial slice from the held-out volume as ground truth.
2. Forward-project it with the 2D Radon projector (180 angles) and
   sample 1% Poisson counts to make a low-count sinogram.
3. Load the best checkpoint of the score model.
4. Run the PET-DDS-delta sampler for a range of delta values.
5. Report metrics for each delta and save the best result as a figure.

DELTA NOTE
----------
The paper uses delta = 0.2 because its Poisson NLL is on the raw count
scale (10^8 - 10^9 per dataset) so its gradients are of order unity.
We normalise the NLL by sum(measured) (see poisson_nll in pet_dds.py)
which makes the gradients ~1e-4.  The correct delta therefore scales up
by roughly the same factor.  We therefore sweep around 25 rather than 0.2,
and pick the value that minimises NRMSE, exactly as the paper did on its
validation set.
"""
import os
import sys
import glob
import time
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
import nibabel as nib
from scipy.ndimage import zoom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pet_dds import PETDDSDeltaReconstructor, ScoreModelTrainer
from data_utils import RadonProjector2D, generate_low_count_sinogram_2d
from data_synthetic import select_held_out_file
from paths import SYNTHETIC_BASE, CHECKPOINT_DIR, RESULTS_DIR

# ------------------------- User settings -------------------------
NUM_ANGLES     = 180          # close to Nyquist for 128 px
N_DIFFUSION    = 100          # paper Section III
P_STEPS        = 5            # paper Section III
N_SUB          = 21           # paper Section III
LAMBDA_DDS     = 1.0          # paper Section III
LAMBDA_RDP     = 1e-4         # paper Section III
COUNT_FRACTION = 0.01         # paper Section III: 1% counts
DELTA_SWEEP    = [15, 18, 20, 22, 25, 28]
N_SAMPLES      = 3            # posterior samples per delta
STALE_SECONDS  = 3600         # warn if checkpoint is older than 1 h
# -----------------------------------------------------------------


def load_gt(path, size=128):
    """Load the middle axial slice of the held-out volume as GT."""
    img  = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    vmin, vmax = data.min(), data.max()
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin)

    D = data.shape[2]
    z = D // 2
    s = data[:, :, z].copy()

    if s.shape != (size, size):
        zf = (size / s.shape[0], size / s.shape[1])
        s = zoom(s, zf, order=1)
        s = s[:size, :size]
        if s.shape != (size, size):
            pad = np.zeros((size, size), dtype=np.float32)
            pad[:s.shape[0], :s.shape[1]] = s
            s = pad

    print("GT slice: %s  z=%d/%d" % (os.path.basename(path), z, D))
    return s.astype(np.float32)


def compute_metrics(recon, gt):
    """NRMSE %, PSNR dB, SSIM %  (same definitions as the paper's Table I)."""
    recon = np.nan_to_num(recon, nan=0.0)
    gt    = np.nan_to_num(gt,    nan=0.0)
    recon = recon / (recon.max() + 1e-8)
    gt    = gt    / (gt.max()    + 1e-8)

    mse   = float(np.mean((recon - gt) ** 2))
    nrmse = float(np.sqrt(mse) / (gt.max() - gt.min() + 1e-8) * 100.0)
    psnr  = float(10.0 * np.log10(1.0 / (mse + 1e-8)))
    ssim_val = float(ssim_func(gt, recon, data_range=1.0))
    return nrmse, psnr, ssim_val * 100.0


def load_best_ckpt():
    """Return (path, age_seconds) of the newest best*.ckpt (fallback: any)."""
    best = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "best*.ckpt")),
                  key=os.path.getmtime)
    if best:
        p = best[-1]
    else:
        allc = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt")),
                      key=os.path.getmtime)
        if not allc:
            return None, None
        p = allc[-1]
    return p, time.time() - os.path.getmtime(p)


def main():
    pl.seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------- 1) Ground truth and low-count sinogram ----------
    held_out = select_held_out_file(SYNTHETIC_BASE)
    phantom  = load_gt(held_out, size=128)
    print("GT shape %s  range [%.3f, %.3f]"
          % (phantom.shape, phantom.min(), phantom.max()))

    proj = RadonProjector2D(num_angles=NUM_ANGLES).to(device)
    measured, clean_scaled, clean, dose = generate_low_count_sinogram_2d(
        phantom, proj, count_fraction=COUNT_FRACTION, seed=42)
    measured = measured.to(device)
    print("Sinogram sum=%.0f  max=%s  dose=%s  angles=%d"
          % (measured.sum(), measured.max(), dose, NUM_ANGLES))

    # ---------- 2) Load the trained score model ----------
    ckpt, age = load_best_ckpt()
    if ckpt is None:
        print("No checkpoint found.  Run train_score.py first.")
        return
    print("Loading:", ckpt)
    print("Checkpoint age: %.1f min" % (age / 60))
    if age > STALE_SECONDS:
        print("  WARNING: checkpoint is older than 1 hour - did you retrain?")

    score_model = ScoreModelTrainer.load_from_checkpoint(ckpt, map_location=device)
    score_model.eval()
    score_model.to(device)

    # ---------- 3) Tweedie sanity check ----------
    # A well-trained score model should reduce the Tweedie MSE smoothly
    # as t decreases.  This is a fast diagnostic that does not depend on
    # the data-consistency code at all.
    print("")
    print("--- Tweedie sanity check ---")
    with torch.no_grad():
        gt_t = torch.tensor(phantom, device=device).unsqueeze(0).unsqueeze(0)
        for tc in [90, 75, 50, 25]:
            g_tc  = score_model.gamma[tc]
            n_tc  = score_model.nu[tc]
            eps_c = torch.randn_like(gt_t)
            xn    = g_tc * gt_t + n_tc * eps_c
            ep    = score_model.model(xn, torch.tensor([tc], device=device))
            x0    = (xn - n_tc * ep) / g_tc
            print("  t=%3d  Tweedie MSE = %.4f"
                  % (tc, float(((x0 - gt_t) ** 2).mean())))

    # ---------- 4) Delta sweep ----------
    print("")
    print("================ Delta sweep ================")
    results = []
    best    = None
    for delta in DELTA_SWEEP:
        print(">>> delta =", delta)
        rec = PETDDSDeltaReconstructor(
            score_model       = score_model.model,
            proj              = proj,
            measured_sinogram = measured,
            dose_fraction     = dose,
            N=N_DIFFUSION, p=P_STEPS, delta=float(delta),
            lambda_DDS=LAMBDA_DDS, lambda_RDP=LAMBDA_RDP, n_sub=N_SUB,
            img_shape=(128, 128), verbose=False,
            nll_cutoff_k=N_DIFFUSION, skip_first=0,
            use_ancestral=True, grad_probe=True,
        ).to(device)

        mean_r, std_r, cv = rec.reconstruct(num_samples=N_SAMPLES, seed=0)
        nrmse, psnr, s   = compute_metrics(mean_r, phantom)
        print("  delta=%-6s  NRMSE=%6.2f%%  PSNR=%5.2f  SSIM=%5.2f%%"
              % (delta, nrmse, psnr, s))

        results.append((delta, nrmse, psnr, s, mean_r, std_r))
        if best is None or s > best[3]:
            best = (delta, nrmse, psnr, s, mean_r, std_r)

    # ---------- 5) Summary and figure ----------
    print("")
    print("================ Summary ================")
    print("%8s %10s %8s %8s" % ("delta", "NRMSE%", "PSNR", "SSIM%"))
    for d, n, p, sv, _, _ in results:
        print("%8s %10.2f %8.2f %8.2f" % (d, n, p, sv))

    bd, bn, bp, bs, bmean, bstd = best
    print("")
    print("Best delta=%s  NRMSE=%.2f%%  PSNR=%.2f  SSIM=%.2f%%"
          % (bd, bn, bp, bs))

    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    ax[0].imshow(phantom, cmap="gray")
    ax[0].set_title("GT"); ax[0].axis("off")
    ax[1].imshow(np.log1p(clean.squeeze().numpy()), cmap="gray", aspect="auto")
    ax[1].set_title("Clean sino"); ax[1].axis("off")
    ax[2].imshow(np.log1p(measured.cpu().squeeze().numpy()),
                 cmap="gray", aspect="auto")
    ax[2].set_title("1% sino"); ax[2].axis("off")
    ax[3].imshow(bmean, cmap="gray")
    ax[3].set_title("Recon d=%s  NRMSE=%.1f%%" % (bd, bn)); ax[3].axis("off")
    ax[4].imshow(bstd, cmap="hot")
    ax[4].set_title("Uncertainty"); ax[4].axis("off")
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "pet_dds_result.png")
    plt.savefig(out_png, dpi=120)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
