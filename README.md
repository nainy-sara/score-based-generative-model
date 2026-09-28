# 2D PET-DDS-δ on Kaggle — reproduction of Webber et al. (2024)

This folder contains a cleaned, paper-annotated 2D reproduction of the
PET-DDS-δ algorithm from:

> Webber, Mizuno, Howes, Hammers, King & Reader (2024).
> *Generative-Model-Based Fully 3D PET Image Reconstruction by
> Conditional Diffusion Sampling.* arXiv:2412.04319v1.

## Files

| File | Role |
|---|---|
| `paths.py` | Kaggle I/O paths (data + output) in one place |
| `score_model.py` | 2D time-conditioned U-Net predicting the noise eps |
| `data_utils.py` | Vectorised 2D Radon projector + low-count sinogram |
| `data_synthetic.py` | NIfTI slice loader with held-out exclusion |
| `pet_dds.py` | `ScoreModelTrainer` + `PETDDSDeltaReconstructor` (Eq. 10) |
| `train_score.py` | Train the score model |
| `reconstruct_pet_dds.py` | Reconstruction + delta sweep + metrics + figure |
| `run_all.py` | Train then reconstruct in one command |
| `requirements.txt` | Python packages |

## Kaggle setup

1. Create a Kaggle notebook and **add the dataset**
   `lbvigilantdata/synthetic-neurodegenerative-brain-image-data-set`.
2. Upload the `.py` files to `/kaggle/working/Andrew Reader/test2/`
   (same directory as in `paths.py`).
3. In a code cell, run:

   ```bash
   %cd "/kaggle/working/Andrew Reader/test2"
   !python run_all.py
   ```

   Or, one script at a time:

   ```bash
   %cd "/kaggle/working/Andrew Reader/test2"
   !python train_score.py
   !python reconstruct_pet_dds.py
   ```

## Outputs

* Best model checkpoint:  `checkpoints/best*.ckpt`
* Result figure:          `results/pet_dds_result.png`

## Paper fidelity — what is and is not reproduced

Faithful to the paper:

* Eq. (10) is implemented line-by-line: diffusion step, Tweedie estimate,
  data-consistency objective, `p` gradient-descent steps with step size δ,
  and the ancestral re-noising step with η_t = 0.1 ν_t.
* Hyperparameters from Section III: N = 100, p = 5, λ_RDP = 1e-4,
  λ_DDS = 1.0, n_sub = 21, 1 % counts.
* Denoising score matching training (Song & Ermon 2019 — the paper's ref [5]).

Deliberate deviations:

* **2D, not 3D.** The paper reconstructs 128×128×120 volumes; here we work
  on a single axial slice.
* **RDP_z replaced by rdp_2d.** The paper's axial relative difference prior
  is undefined in 2D; we use an in-plane RDP of the same functional form.
* **δ ≈ 20, not 0.2.** The Poisson NLL here is normalised by Σm, so its
  gradients are ~10⁴ × smaller than the paper's raw-count scale.
* **Custom 2D Radon projector** instead of ParallelProj with span-11
  axial compression.
* **No OSEM / MAP-EM baselines.** Only the SGM method is reproduced.
* **Synthetic Kaggle data**, not the paper's 55 real [¹⁸F]DPA-714 subjects.

## Notes on the DDS anchor sign

The paper writes `-λ_DDS · ||x − x₀||²` inside Φ.  Taken literally this
would push the iterate away from the Tweedie estimate, which cannot be
intended.  We minimise `+λ_DDS · ||x − x₀||²`, i.e. pull toward the
Tweedie estimate — the physically correct reading.
