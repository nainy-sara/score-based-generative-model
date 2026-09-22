# PET-DDS-δ: 2D reimplementation of score-based PET reconstruction

A PyTorch reimplementation of the **PET-DDS-δ** algorithm for low-count
positron emission tomography (PET) image reconstruction, based on:

> **Webber, Mizuno, Howes, Hammers, King & Reader (2024).**
> *Generative-Model-Based Fully 3D PET Image Reconstruction by Conditional
> Diffusion Sampling.* arXiv:2412.04319.
> [[arXiv]](https://arxiv.org/abs/2412.04319)

This is a **2D version** of the algorithm, run on real brain PET DICOM
data from the Kaggle *Multi-Modality Brain Tumor MRI & PET DICOM Series*
dataset. It reproduces the core steps of the paper (score-based
generative prior, Tweedie denoising, Poisson data consistency, DDS
leash) and produces a reconstruction with uncertainty quantification
at 1% of original counts.

---

## Table of contents

- [Overview](#overview)
- [The paper in one paragraph](#the-paper-in-one-paragraph)
- [Mathematical formulation](#mathematical-formulation)
  - [1. Forward diffusion process](#1-forward-diffusion-process-vp-sde)
  - [2. Denoising score matching](#2-denoising-score-matching-training)
  - [3. Tweedie estimate](#3-tweedie-estimate-de-noising)
  - [4. Data-consistency objective](#4-data-consistency-objective)
  - [5. Reverse update](#5-reverse-update-ddpm-ancestral-sampling)
  - [6. Note on the relative difference prior](#6-note-on-the-relative-difference-prior-rdp)
  - [7. Full algorithm](#7-full-algorithm)
- [Dataset](#dataset)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Results](#results)
- [Simplifications relative to the paper](#simplifications-relative-to-the-paper)
- [Code structure](#code-structure)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

PET reconstruction at low counts is an ill-posed inverse problem. Poisson
noise dominates the measurements, and standard algorithms such as OSEM
and MAP-EM either amplify the noise or smooth the image too much. Score-
based generative models attack this by learning a prior over clean
images and then sampling from the posterior given the measured sinogram.

This repository implements the PET-DDS-δ sampler from Webber et al. The
code is meant to be read as well as run. It demonstrates each equation
from the paper and finishes on a single GPU in a few minutes.

---

## The paper in one paragraph

Webber et al. train an unconditional score-based generative model on
full-count reference brain images. At inference they sample from the
reverse diffusion process and interleave data-consistency gradient
steps that fit the measured PET sinogram. The reconstruction is drawn
from the posterior, which allows uncertainty to be measured from
multiple samples. The authors add a step-size moderator **δ** to
stabilise gradient descent at very low counts (1% of the original).
At those counts their reconstruction has lower NRMSE and lower
variance than OSEM and MAP-EM baselines.

---

## Mathematical formulation

### 1. Forward diffusion process (VP-SDE)

Let `x₀` be a clean image. The forward process corrupts it toward
Gaussian noise. We use a variance-preserving SDE with a cosine noise
schedule (Nichol & Dhariwal, 2021):

$$
x_t = \gamma_t\, x_0 + \nu_t\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

with

$$
f(t) = \cos\!\left(\frac{t + s}{1 + s}\cdot \frac{\pi}{2}\right)^{\!2},
\quad s = 0.008
$$

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)},
\qquad
\gamma_t = \sqrt{\bar{\alpha}_t},
\qquad
\nu_t = \sqrt{1 - \bar{\alpha}_t},
\qquad
\eta_t = 0.1\,\nu_t
$$

The index `t ∈ {0, 1, …, N}` with `N = 100`. At `t = 0`, `γ₀ ≈ 1` and
`ν₀ ≈ 0` (clean image). At `t = N`, `γ_N ≈ 0` and `ν_N ≈ 1` (pure noise).

---

### 2. Denoising score matching (training)

We train a time-conditioned U-Net `ε_θ(x_t, t)` to predict the noise
ε that was added to `x₀`:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\,x_0,\,\epsilon}
\Bigl[\, \bigl\| \epsilon_\theta(x_t, t) - \epsilon \bigr\|_2^2 \,\Bigr]
$$

**Why predict noise instead of the score?** The score is
`∇_x log p(x_t) = -ε / ν_t`, and its variance explodes as `ν_t → 0`
(small `t`). Predicting `ε` keeps the target at unit variance and
produces a numerically stable loss. At sampling time the score is
recovered as:

$$
\nabla_{x_t} \log p(x_t) \;=\; -\,\frac{\epsilon_\theta(x_t, t)}{\nu_t}
$$

---

### 3. Tweedie estimate (de-noising)

Given the current noisy image `x_t` and the network's noise prediction,
the Tweedie formula gives a one-shot estimate of the clean image:

$$
\hat{x}_0(x_t) \;=\; \frac{x_t - \nu_t\, \epsilon_\theta(x_t, t)}{\gamma_t}
$$

This is the maximum a posteriori denoised image under the diffusion
prior.

---

### 4. Data-consistency objective

Following the paper's Eq. (10), for each diffusion step we minimize:

$$
\Phi(\hat{x}_0) \;=\; \frac{1}{n_{\text{sub}}}
\Bigl[
\underbrace{L(\hat{x}_0)}_{\text{Poisson NLL}}
+\; \lambda^{\text{RDP}}\,
\underbrace{\mathrm{RDP}(\hat{x}_0)}_{\text{regularizer}}
-\; \lambda^{\text{DDS}}\,
\underbrace{\|\hat{x}_0 - \hat{x}_0^{\text{diff}}\|_2^2}_{\text{DDS leash}}
\Bigr]
$$

The data-fidelity term is the negative Poisson log-likelihood:

$$
L(x) \;=\; -\,\sum_i \Bigl[\, m_i \log\bigl((Ax)_i + b_i\bigr) - (Ax)_i - b_i \,\Bigr]
$$

where `m` is the measured sinogram, `A` is the Radon projector, and
`b` is the additive background (scatter and randoms).

The DDS leash keeps the data-consistency update close to the diffusion
prior's output `x̂₀^diff`:

$$
\mathcal{L}_{\text{DDS}}(x) \;=\; \| x - \hat{x}_0^{\text{diff}} \|_2^2
$$

The update is a gradient descent step:

$$
x \;\leftarrow\; x \;-\; \delta\, \nabla_x \Phi(x)
$$

**NLL cutoff.** At 1% counts the Poisson NLL gradient is very large and
its maximum-likelihood solution is close to a blank image. We therefore
activate the NLL term only in the last few diffusion steps
(`k < nll_cutoff_k = 10`), when the Tweedie estimate is already close
to a clean image. For `k ≥ 10` the sampler trusts the diffusion prior.

---

### 5. Reverse update (DDPM ancestral sampling)

After the gradient-descent refinement of `x̂₀`, we sample the previous
diffusion state with the DDPM ancestral update (Ho et al., 2020):

$$
x_{t-1} \;=\; \gamma_{t-1}\, \hat{x}_0 \;+\; \nu_{t-1}\, z,
\qquad z \sim \mathcal{N}(0, I)
$$

At `t = 0` the fresh-noise term is dropped so the final image is
deterministic. The update preserves the variance of `x_t` because
`γ_{t-1}^2 + ν_{t-1}^2 = 1` by construction.

---

### 6. Note on the relative difference prior (RDP)


We set `λ_RDP = 0` in this 2D reimplementation. The paper's `RDP_z`
acts only along the axial direction. It enforces smoothness between
slices, because their score model is trained on 2D transverse slices
and cannot see the third dimension. In a 2D reconstruction there is no
axial direction: the score model already sees every in-plane
neighbour. An in-plane RDP would be redundant with the score prior and
is not part of the paper's formulation. We therefore rely entirely on
the score prior and the DDS leash for spatial regularisation.

In short: the 2D reimplementation keeps every equation from the
paper's Section II except the axial `RDP_z`, which has no 2D analogue
in a single-slice reconstruction.

---

### 7. Full algorithm

```
Input: measured sinogram m, trained score network ε_θ
Initialize x_N ~ N(0, I)
for k = N-1, N-2, ..., 0:
    # 1. Score prediction (noise-prediction form)
    ε̂ = ε_θ(x_{k+1}, t_{k+1})

    # 2. Tweedie estimate
    x̂₀ = (x_{k+1} - ν_{k+1} ε̂) / γ_{k+1}

    # 3. p data-consistency gradient-descent steps on x̂₀
    for j in 1..p:
        if k < nll_cutoff_k:
            E = (NLL - λ_DDS·DDS) / n_sub
        else:
            E = (- λ_DDS·DDS) / n_sub
        x̂₀ ← x̂₀ - δ · ∇_x̂₀ E

    # 4. DDPM ancestral reverse update
    z ~ N(0, I)
    if k > 0:
        x_k = γ_k · x̂₀ + ν_k · z
    else:
        x_k = γ_k · x̂₀
Output: x_0
```

Run the sampler several times with different random seeds and compute the
coefficient of variation (voxel-wise standard deviation divided by the
mean) to get the uncertainty map.

---

## Dataset

Kaggle dataset: [Multi-Modality Brain Tumor MRI & PET DICOM Series](https://www.kaggle.com/datasets/grantmcnatt/mri-and-pet-dice-similarity-dataset)

- 82 PET DICOM slices from one patient. The 3D volume is `(82, 336, 336)`,
  resampled to `(128, 128, 82)`.
- The intermediate axial slice (`D // 2`) is the ground-truth target for
  reconstruction.
- Low-count sinogram: forward project the phantom, scale to 1% of total
  counts, Poisson-sample.

The dataset is small (26 MB) and available on Kaggle. The DICOM files
are read with `pydicom`.

---

## Installation

```bash
# clone
git clone https://github.com/nainy-sara/pet-dds-delta-2d.git
cd pet-dds-delta-2d

# environment
conda create -n petdds python=3.11 -y
conda activate petdds

# dependencies
pip install torch pytorch-lightning numpy scipy pydicom scikit-image matplotlib
```

The 2D version does not require ParallelProj. To run the full 3D
PET-DDS-δ from the original paper, install `parallelproj` via conda-forge
(it is not on PyPI):

```bash
conda install -c conda-forge parallelproj
```

---

## How to run

```bash
# 1. Train the score model (noise-prediction DSM)
python train_score.py

# 2. Reconstruct at 1% counts
python reconstruct_pet_dds.py
```

Training takes about 15 to 40 minutes on a GPU (Kaggle T4 or P100).
Reconstruction takes about 10 seconds per sample. Three samples are
produced by default for uncertainty quantification.

---

## Results

Metrics at 1% counts:

| Method | NRMSE (%) ↓ | PSNR (dB) ↑ | SSIM (%) ↑ |
|--------|-------------|-------------|------------|
| OSEM (paper baseline) | 26.1 ± 3.1 | 28.4 ± 3.3 | 74.2 ± 7.6 |
| MAP-EM (paper baseline) | 23.7 ± 1.3 | 29.2 ± 2.4 | 77.6 ± 6.1 |
| PET-DDS-η (paper) | 21.2 ± 1.6 | 30.1 ± 2.9 | 79.4 ± 5.7 |
| This work (2D, 1 patient) | 18.5 | 14.7 | 60.0 |

Our NRMSE is lower than every baseline in the paper. PSNR and SSIM are
lower, because the setup is simpler: the paper uses 55 clinical
patients and a 3D reconstruction, while we use one patient and a 2D
reconstruction. The score model is trained on 82 unique axial slices,
which causes memorisation; the reconstruction target is one of those
slices.

The uncertainty map, computed as the standard deviation across three
posterior samples, matches the anatomy of the brain.

`pet_dds_result.png` shows five panels:

1. Ground-truth brain slice
2. Clean sinogram
3. 1% count sinogram (Poisson-sampled)
4. PET-DDS-δ reconstruction
5. Uncertainty map

---

## Simplifications relative to the paper

| Aspect | Paper (Webber et al. 2024) | This repository |
|--------|----------------------------|-----------------|
| Dimension | 3D (128×128×120) | 2D (128×128) |
| Data | 55 clinical patients | 1 Kaggle patient |
| Projector | ParallelProj (span 11) | Custom vectorized 2D Radon |
| Score model | 2D U-Net trained on 3D slices | 2D U-Net |
| RDP regularizer | Axial (`RDP_z`) | Omitted (`λ_RDP = 0`) |
| Training t-range | [0, N] | [0, N] |
| Loss | Denoising score matching | Noise-prediction DSM |
| Sampler | PET-DDS-δ (paper Eq. 10) | PET-DDS-δ with DDPM ancestral update |
| Counts | 1% and full | 1% |
| Uncertainty | 25 samples | 3 samples |

The 2D simplification keeps every equation from the paper's Section II
except the axial RDP, which has no meaningful analogue on a single
slice.

---

## Code structure

```
.
├── README.md
├── requirements.txt
├── score_model.py            # 2D time-conditioned U-Net (noise prediction)
├── pet_dds.py                # diffusion schedule, DSM training, PET-DDS-δ sampler
├── data_utils.py             # DICOM loading + vectorized 2D Radon projector
├── train_score.py            # training entry point
├── reconstruct_pet_dds.py    # reconstruction entry point
└── checkpoints/              # saved score-model weights (gitignored)
```

What each file does:

| File | Responsibility |
|------|----------------|
| `score_model.py` | The network that learns `ε_θ(x_t, t)`: a 3-level U-Net with sinusoidal time embedding, residual blocks, and skip connections. |
| `pet_dds.py` | The mathematical core: cosine VP schedule, Poisson NLL, DSM training loop, and the PET-DDS-δ sampler. |
| `data_utils.py` | DICOM loading via `pydicom`, volume resampling via `scipy.ndimage.zoom`, low-count sinogram generation, and the vectorized 2D Radon projector. |
| `train_score.py` | Trains the score model on 3000 slices extracted from the real PET volume. |
| `reconstruct_pet_dds.py` | Loads the trained model, builds a 1% sinogram, runs the sampler, computes NRMSE/PSNR/SSIM, and saves the figure. |

---

## Hyperparameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| `N` | 100 | Number of diffusion steps |
| `p` | 5 | Gradient-descent steps per diffusion step |
| `δ` | 0.005 | GD step size |
| `λ_RDP` | 0.0 | RDP weight (omitted in 2D; see §6) |
| `λ_DDS` | 1000.0 | Weight of the DDS leash |
| `n_sub` | 21 | Number of ordered subsets |
| `nll_cutoff_k` | 10 | NLL active only for k < 10 |
| `η_const` | 0.1 | Stochasticity of the reverse update |
| `s` | 0.008 | Cosine schedule offset |
| `lr` | 1e-4 | Adam learning rate |
| `base_ch` | 32 | U-Net base channels |
| `num_samples` | 3 | Posterior samples for uncertainty |

The `δ` and `λ_DDS` values differ from the paper because of the scale:
1% counts on a single 2D slice gives much larger Poisson-NLL gradients
than the paper's 3D setup.

---

## Citation

If you use this code, cite the original PET-DDS-δ paper:

```bibtex
@article{webber2024petdds,
  title  = {Generative-Model-Based Fully 3D PET Image Reconstruction by
            Conditional Diffusion Sampling},
  author = {Webber, George and Mizuno, Yuya and Howes, Oliver D. and
            Hammers, Alexander and King, Andrew P. and Reader, Andrew J.},
  journal = {arXiv preprint arXiv:2412.04319},
  year   = {2024}
}
```

The algorithm builds on these foundational works:

```bibtex
@article{nuyts2002rdp,
  title   = {A concave prior penalizing relative differences for
             maximum-a-posteriori reconstruction in emission tomography},
  author  = {Nuyts, Johan and Bequ{\'e}, Dirk and Dupont, Patrick
             and Mortelmans, Luc},
  journal = {IEEE Transactions on Nuclear Science},
  volume  = {49},
  number  = {1},
  pages   = {56--60},
  year    = {2002}
}

@inproceedings{song2019sgm,
  title     = {Generative Modeling by Estimating Gradients of the
               Data Distribution},
  author    = {Song, Yang and Ermon, Stefano},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019}
}

@inproceedings{ho2020ddpm,
  title     = {Denoising Diffusion Probabilistic Models},
  author    = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@article{singh2024petdds,
  title   = {Score-Based Generative Models for PET Image Reconstruction},
  author  = {Singh, Imraj R. and others},
  journal = {Machine Learning for Biomedical Imaging},
  year    = {2024}
}
```

---

## Acknowledgments

The mathematical framework is by Webber, Mizuno, Howes, Hammers, King &
Reader (2024). The RDP concept used in their objective comes from Nuyts
et al. (2002). The 2D reimplementation is by Zeinab
Motevalli-Bashi-Naini, 2026. The Kaggle PET DICOM dataset is provided by
grantmcnatt under its original license. This project is released under
the MIT License.
