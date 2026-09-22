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

Webber et al. (2024) do not write the explicit formula for the Relative
Difference Prior. They mention "an axial relative difference prior
(RDP)" in the Theory section and use the symbol `RDP_z` in their
Eq. (10), but the formula itself is not given. The `z` subscript means
axial (between adjacent 3D slices).

The explicit RDP formula is older and comes from:

> **Nuyts, J., Bequ, D., Dupont, P., & Mortelmans, L. (2002).**
> *A concave prior penalizing relative differences for
> maximum-a-posteriori reconstruction in emission tomography.*
> IEEE Transactions on Nuclear Science, 49(1), 56–60.

Its standard form is:

$$
\mathrm{RDP}(x) \;=\; \sum_{\langle i,j \rangle}
\frac{(x_i - x_j)^2}{\,x_i + x_j + \beta\,|x_i - x_j| + \delta\,}
$$

with `β = 1.0` and `δ = 1e-9`.

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
