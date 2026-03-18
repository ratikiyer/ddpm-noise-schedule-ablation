"""
Denoising Diffusion Probabilistic Model (DDPM).

Implements the forward noising process and reverse denoising process
as described in Ho et al. (2020). The forward process is defined as:

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

The model is trained via the simplified objective:

    L = E_{t, x_0, eps} [|| eps - eps_theta(x_t, t) ||^2]
"""

import torch
import torch.nn as nn

from noise_schedules import get_schedule, compute_schedule_quantities


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.num_timesteps = config.num_timesteps
        self.device = config.device

        betas = get_schedule(
            config.noise_schedule,
            config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            cosine_s=config.cosine_s,
            sigmoid_start=config.sigmoid_start,
            sigmoid_end=config.sigmoid_end,
        )
        quantities = compute_schedule_quantities(betas)
        for k, v in quantities.items():
            self.register_buffer(k, v)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward process: sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute the simplified DDPM training objective."""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recip_alphas_bar_minus_one, t, x_t.shape) * noise
        )

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Compute the posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor):
        """Compute the model's predicted mean and variance for p(x_{t-1} | x_t)."""
        predicted_noise = self.model(x_t, t)
        x_0_pred = self.predict_x0_from_noise(x_t, t, predicted_noise)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        mean, variance, log_variance = self.q_posterior_mean_variance(x_0_pred, x_t, t)
        return mean, variance, log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        mean, _, log_variance = self.p_mean_variance(x_t, t_batch)
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        return mean + torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(self, shape: tuple, return_intermediates: bool = False) -> torch.Tensor:
        """Generate samples via the full reverse process."""
        device = self.betas.device
        x = torch.randn(shape, device=device)
        intermediates = []

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)
            if return_intermediates and t % (self.num_timesteps // 10) == 0:
                intermediates.append(x.cpu())

        if return_intermediates:
            return x, intermediates
        return x

    @staticmethod
    def _extract(tensor: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract values from tensor at index t, reshaped for broadcasting."""
        out = tensor.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))
