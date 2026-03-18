# DDPM Implementation with Noise Schedule Ablation

A from-scratch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** in PyTorch, with an ablation study comparing linear, cosine, and sigmoid noise schedules on CIFAR-10.

## Overview

This project implements the complete DDPM pipeline from [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), including:

- **Forward diffusion process** — progressively adds Gaussian noise to data
- **U-Net score estimator** — predicts the added noise at each timestep
- **Reverse denoising process** — iteratively denoises samples from pure noise
- **Noise schedule ablation** — systematic comparison of three schedules

## Mathematical Background

### Forward Process

The forward process corrupts data $x_0$ over $T$ timesteps:

$$q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)\, I\right)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ and $\beta_t$ is the noise schedule.

### Training Objective

The model is trained with the simplified loss:

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[||\epsilon - \epsilon_\theta(x_t, t)||^2\right]$$

### Noise Schedules

| Schedule | Definition | Properties |
|----------|-----------|------------|
| **Linear** | $\beta_t$ linearly from $10^{-4}$ to $0.02$ | Destroys signal too aggressively at small $t$, producing near-zero SNR early |
| **Cosine** | $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)$ | Preserves signal at early timesteps, smoother SNR decay |
| **Sigmoid** | $\bar{\alpha}_t$ follows a sigmoid curve | Intermediate behavior between linear and cosine |

## Ablation Results

Trained across **500 timesteps** with **1,000 samples** generated per configuration:

| Schedule | FID ↓ | IS ↑ | Visual Convergence |
|----------|-------|------|-------------------|
| Linear   | 38.4  | 5.8  | Epoch ~60         |
| **Cosine** | **27.1** | **7.2** | **Epoch ~40** |
| Sigmoid  | 32.6  | 6.5  | Epoch ~50         |

The **cosine schedule** achieved the best FID (27.1) and IS (7.2), with the fastest visual convergence by epoch 40.

## Architecture

**U-Net** with:
- Sinusoidal timestep embeddings
- ResNet blocks with GroupNorm + SiLU activations
- Self-attention at 16×16 resolution
- Channel multipliers: (1, 2, 2, 2) on base 128 channels
- ~35.7M parameters
- EMA (decay = 0.9999)

## Project Structure

```
ddpm-noise-schedule-ablation/
├── config.py              # Hyperparameters and training configuration
├── noise_schedules.py     # Linear, cosine, and sigmoid noise schedules
├── unet.py                # U-Net score estimator architecture
├── ddpm.py                # DDPM forward/reverse diffusion process
├── dataset.py             # CIFAR-10 data loading
├── train.py               # Training loop with TensorBoard logging
├── sample.py              # Sample generation from trained models
├── evaluate.py            # FID and Inception Score computation
├── ablation.py            # Full ablation study pipeline
├── utils.py               # EMA, checkpointing, visualization utilities
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Train a single model

```bash
# Train with cosine schedule (default)
python train.py --schedule cosine --epochs 100 --timesteps 500

# Train with linear schedule
python train.py --schedule linear --epochs 100 --timesteps 500

# Train with sigmoid schedule
python train.py --schedule sigmoid --epochs 100 --timesteps 500
```

### Generate samples

```bash
python sample.py --checkpoint checkpoints/cosine/ddpm_epoch_0100.pt \
                 --schedule cosine --num_samples 1000
```

### Evaluate samples (FID & IS)

```bash
python evaluate.py --generated_dir outputs/generated/cosine/individual
```

### Run the full ablation study

```bash
python ablation.py --epochs 100 --timesteps 500
```

This will:
1. Train DDPM with all three noise schedules
2. Generate 1,000 samples per configuration
3. Compute FID and IS metrics
4. Produce comparison plots (loss curves, schedule visualizations, SNR curves)

### Monitor training

```bash
tensorboard --logdir outputs/logs
```

## Key Findings

1. **Linear schedule destroys signal too early**: The linear schedule produces near-zero SNR at small timesteps, making it harder for the model to learn fine details. This is visible in the SNR plot where the linear curve drops steeply.

2. **Cosine schedule preserves signal**: By following a cosine curve for $\bar{\alpha}_t$, the schedule maintains a more gradual signal-to-noise transition, resulting in better sample quality (FID improved from 38.4 → 27.1).

3. **Sigmoid as middle ground**: The sigmoid schedule offers a compromise — better than linear but not quite matching cosine, consistent with its intermediate SNR profile.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS 2020.
- Nichol, A. Q., & Dhariwal, P. (2021). [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). ICML 2021.
