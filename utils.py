"""Utility functions for visualization, EMA, and checkpointing."""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from dataset import unnormalize


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Copy shadow params into model."""
        for m_param, s_param in zip(model.parameters(), self.shadow.parameters()):
            m_param.data.copy_(s_param.data)


def save_checkpoint(model, optimizer, epoch, loss, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config.__dict__,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def save_samples_grid(samples: torch.Tensor, path: str, nrow: int = 10):
    """Save a grid of generated samples."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = unnormalize(samples).clamp(0, 1)
    grid = make_grid(samples, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(loss_dict: dict, save_path: str):
    """Plot training loss curves for multiple schedules."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name, alpha=0.8)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss by Noise Schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_schedule_comparison(num_timesteps: int, save_path: str, cosine_s: float = 0.008):
    """Plot alpha_bar curves for all three noise schedules."""
    from noise_schedules import get_schedule, compute_schedule_quantities

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    schedules = {
        "Linear": get_schedule("linear", num_timesteps),
        "Cosine": get_schedule("cosine", num_timesteps, cosine_s=cosine_s),
        "Sigmoid": get_schedule("sigmoid", num_timesteps),
    }

    plt.figure(figsize=(10, 6))
    t = np.arange(num_timesteps)
    for name, betas in schedules.items():
        q = compute_schedule_quantities(betas)
        plt.plot(t, q["alphas_bar"].numpy(), label=name, linewidth=2)

    plt.xlabel("Timestep t")
    plt.ylabel(r"$\bar{\alpha}_t$")
    plt.title("Signal Retention by Noise Schedule")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_snr_comparison(num_timesteps: int, save_path: str, cosine_s: float = 0.008):
    """Plot log-SNR curves for all three schedules."""
    from noise_schedules import get_schedule, compute_schedule_quantities

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    schedules = {
        "Linear": get_schedule("linear", num_timesteps),
        "Cosine": get_schedule("cosine", num_timesteps, cosine_s=cosine_s),
        "Sigmoid": get_schedule("sigmoid", num_timesteps),
    }

    plt.figure(figsize=(10, 6))
    t = np.arange(num_timesteps)
    for name, betas in schedules.items():
        q = compute_schedule_quantities(betas)
        snr = q["alphas_bar"] / (1 - q["alphas_bar"])
        log_snr = torch.log10(snr + 1e-10).numpy()
        plt.plot(t, log_snr, label=name, linewidth=2)

    plt.xlabel("Timestep t")
    plt.ylabel(r"$\log_{10}$ SNR")
    plt.title("Log Signal-to-Noise Ratio by Schedule")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
