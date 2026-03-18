"""
Noise schedule implementations for DDPM.

Three schedules are provided:
  - Linear:  betas increase linearly from beta_start to beta_end.
  - Cosine:  alpha_bar_t = cos^2((t/T + s) / (1+s) * pi/2), which preserves
             signal at small t and avoids the near-zero SNR problem of linear.
  - Sigmoid: betas follow a sigmoid curve, offering a middle ground between
             linear and cosine.
"""

import torch
import numpy as np


def linear_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_bar = torch.cos(((t + s) / (1 + s)) * (np.pi / 2)) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def sigmoid_schedule(timesteps: int, start: float = -3.0, end: float = 3.0) -> torch.Tensor:
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    v_start = torch.sigmoid(torch.tensor(start))
    v_end = torch.sigmoid(torch.tensor(end))
    alphas_bar = (-((t * (end - start) + start).sigmoid() - v_start) / (v_end - v_start) + 1)
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def get_schedule(name: str, timesteps: int, **kwargs) -> torch.Tensor:
    if name == "linear":
        return linear_schedule(timesteps, kwargs.get("beta_start", 1e-4), kwargs.get("beta_end", 0.02))
    elif name == "cosine":
        return cosine_schedule(timesteps, kwargs.get("cosine_s", 0.008))
    elif name == "sigmoid":
        return sigmoid_schedule(timesteps, kwargs.get("sigmoid_start", -3.0), kwargs.get("sigmoid_end", 3.0))
    else:
        raise ValueError(f"Unknown schedule: {name}. Choose from 'linear', 'cosine', 'sigmoid'.")


def compute_schedule_quantities(betas: torch.Tensor):
    """Pre-compute all quantities needed for the diffusion forward/reverse process."""
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
    log_one_minus_alphas_bar = torch.log(1.0 - alphas_bar)
    sqrt_recip_alphas_bar = torch.sqrt(1.0 / alphas_bar)
    sqrt_recip_alphas_bar_minus_one = torch.sqrt(1.0 / alphas_bar - 1.0)

    # Posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
    posterior_mean_coef1 = betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar)
    posterior_mean_coef2 = (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "alphas_bar_prev": alphas_bar_prev,
        "sqrt_alphas_bar": sqrt_alphas_bar,
        "sqrt_one_minus_alphas_bar": sqrt_one_minus_alphas_bar,
        "log_one_minus_alphas_bar": log_one_minus_alphas_bar,
        "sqrt_recip_alphas_bar": sqrt_recip_alphas_bar,
        "sqrt_recip_alphas_bar_minus_one": sqrt_recip_alphas_bar_minus_one,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
