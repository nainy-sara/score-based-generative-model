# score-based-generative-model
# PET-DDS-δ: 2D Reimplementation of Score-Based PET Reconstruction

A clean, from-scratch **PyTorch** reimplementation of the **PET-DDS-δ**
algorithm for low-count positron emission tomography (PET) image
reconstruction, based on:

> **Webber, Mizuno, Howes, Hammers, King & Reader (2024).**
> *Generative-Model-Based Fully 3D PET Image Reconstruction by Conditional
> Diffusion Sampling.* arXiv:2412.04319.
> [[arXiv]](https://arxiv.org/abs/2412.04319)

This repository implements a **2D version** of the algorithm on **real brain
PET DICOM data** from the Kaggle *Multi-Modality Brain Tumor MRI & PET
DICOM Series* dataset. It reproduces the core algorithmic steps of the
paper — score-based generative modeling, Tweedie denoising, Poisson data
consistency, RDP regularization, and DDS leash — and produces a
reconstruction with uncertainty quantification at **1% of original counts**.

---

## Table of contents

- [Overview](#overview)
- [The paper in one paragraph](#the-paper-in-one-paragraph)
- [Mathematical formulation](#mathematical-formulation)
  - [Forward diffusion process](#1-forward-diffusion-process-vp-sde)
  - [Denoising score matching](#2-denoising-score-matching-training)
  - [Tweedie estimate](#3-tweedie-estimate-de-noising)
  - [Data-consistency objective](#4-data-consistency-objective)
  - [Reverse update](#5-reverse-update-ddpm-ancestral-sampling)
  - [Algorithm summary](#6-full-algorithm)
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

PET reconstruction from low-count data is an ill-posed inverse problem
dominated by Poisson noise. Conventional algorithms such as OSEM and
MAP-EM either amplify noise or over-smooth the image. Score-based
generative models (SGMs) address this by learning a prior over
high-quality images and sampling from the posterior conditioned on the
measured sinogram.

This repository provides a compact, readable, and fully reproducible
implementation of the **PET-DDS-δ** sampler introduced by Webber et al.
The code is written for teaching purposes and for demonstration of the
underlying mathematics. It runs end-to-end on a single GPU in minutes.

---

## The paper in one paragraph

Webber et al. train an unconditional score-based generative model on
full-count reference brain images, then at inference time sample from
the reverse diffusion process while interleaving data-consistency
gradient steps that fit the measured PET sinogram. The resulting
reconstruction is drawn from the posterior distribution, allowing
uncertainty quantification via multiple samples. They introduce a
step-size moderator **δ** to stabilize gradient descent at very low
counts (1% of original). At 1% counts their reconstructions achieve
lower NRMSE and lower variance than OSEM and MAP-EM baselines.

---

## Mathematical formulation

### 1. Forward diffusion process (VP-SDE)

Let `x₀` be a clean image. The forward process gradually corrupts it
toward Gaussian noise. We use a **variance-preserving SDE** with a
cosine noise schedule (Nichol & Dhariwal, 2021):

$$
x_t = \gamma_t\, x_0 + \nu_t\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

where the coefficients are defined by:

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

We train a time-conditioned U-Net `ε_θ(x_t, t)` to **predict the noise
ε** that was added to `x₀`. The training loss is the standard
**noise-prediction denoising score matching** objective:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\,x_0,\,\epsilon}
\Bigl[\, \bigl\| \epsilon_\theta(x_t, t) - \epsilon \bigr\|_2^2 \,\Bigr]
$$

**Why predict noise instead of the score?** The score is
`∇_x log p(x_t) = -ε / ν_t`, whose variance explodes as `ν_t → 0`
(small `t`). Predicting `ε` keeps the target unit-variance and
produces a numerically stable loss. At sampling time the score is
recovered via:

$$
\nabla_{x_t} \log p(x_t) \;=\; -\,\frac{\epsilon_\theta(x_t, t)}{\nu_t}
$$

---

### 3. Tweedie estimate (de-noising)

Given the current noisy image `x_t` and the network's noise prediction,
the **Tweedie formula** gives a one-shot estimate of the clean image:

$$
\hat{x}_0(x_t) \;=\; \frac{x_t - \nu_t\, \epsilon_\theta(x_t, t)}{\gamma_t}
$$

This estimate is the *maximum a posteriori* denoised image under
the diffusion prior.

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

We use the **negative Poisson log-likelihood** as the data-fidelity term:

$$
L(x) \;=\; -\,\sum_i \Bigl[\, m_i \log\bigl((Ax)_i + b_i\bigr) - (Ax)_i - b_i \,\Bigr]
$$

where `m` is the measured sinogram, `A` is the Radon projector, and
`b` is the additive background (scatter + randoms). The **Relative
Difference Prior (RDP)** regularizer is:

$$
\mathrm{RDP}(x) \;=\; \sum_{\langle i,j \rangle}
\frac{(x_i - x_j)^2}{x_i + x_j + \beta\,|x_i - x_j| + \delta}
$$

with `β = 1.0` and `δ = 1e-9`. Finally, the **DDS leash** ensures the
data-consistency update does not stray too far from the diffusion
prior's output `x̂₀^diff`:

$$
\mathcal{L}_{\text{DDS}}(x) \;=\; \| x - \hat{x}_0^{\text{diff}} \|_2^2
$$

The update is a **gradient descent** step:

$$
x \;\leftarrow\; x \;-\; \delta\, \nabla_x \Phi(x)
$$

**NLL cutoff.** At 1% counts the Poisson NLL gradient is enormous and
its maximum-likelihood solution is close to a blank image. We therefore
**only activate the NLL term in the last few diffusion steps**
(`k < nll_cutoff_k = 10`), when the Tweedie estimate is already
close to a clean image. For `k ≥ 10` the sampler trusts the
diffusion prior entirely.

---

### 5. Reverse update (DDPM ancestral sampling)

After the gradient-descent refinement of `x̂₀`, we sample the
previous diffusion state via the **DDPM ancestral update**:

$$
x_{t-1} \;=\; \gamma_{t-1}\, \hat{x}_0 \;+\; \nu_{t-1}\, z,
\qquad z \sim \mathcal{N}(0, I)
$$

At `t = 0` the fresh-noise term is omitted so that the final image
is deterministic. This update preserves the variance of `x_t`
because `γ_{t-1}^2 + ν_{t-1}^2 = 1` by construction.

---

### 6. Full algorithm

Putting everything together, the **PET-DDS-δ** sampler is:
