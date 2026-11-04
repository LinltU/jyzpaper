"""Diffusion model for a scalar voltage innovation conditioned on real history."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .humidity_conditional_ddpm import (
    ConditionalResidualBlock,
    SinusoidalTimeEmbedding,
    _extract,
)


class HistoryConditionedResidualDenoiser(nn.Module):
    """Predict scalar diffusion noise from observed history and known context."""

    def __init__(
        self,
        history_dim,
        target_context_dim,
        num_stages,
        num_devices,
        hidden_dim=128,
        depth=3,
        dropout=0.05,
    ):
        super().__init__()
        if min(history_dim, target_context_dim, num_stages, num_devices) <= 0:
            raise ValueError("All context dimensions must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.history_dim = int(history_dim)
        self.target_context_dim = int(target_context_dim)
        self.num_stages = int(num_stages)
        self.num_devices = int(num_devices)
        self.noisy_projection = nn.Linear(1, hidden_dim)
        self.history_projection = nn.Sequential(
            nn.LayerNorm(history_dim),
            nn.Linear(history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_projection = nn.Sequential(
            nn.Linear(target_context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        stage_dim = min(24, max(8, hidden_dim // 8))
        device_dim = min(32, max(8, hidden_dim // 8))
        self.stage_embedding = nn.Embedding(num_stages, stage_dim)
        self.device_embedding = nn.Embedding(num_devices, device_dim)
        self.static_projection = nn.Sequential(
            nn.Linear(stage_dim + device_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock(hidden_dim, dropout) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        noisy_residual,
        timesteps,
        history_context,
        target_context,
        stage_indices,
        device_indices,
    ):
        if noisy_residual.ndim != 2 or noisy_residual.shape[1] != 1:
            raise ValueError("noisy_residual must have shape [batch, 1]")
        batch_size = noisy_residual.shape[0]
        if tuple(history_context.shape) != (batch_size, self.history_dim):
            raise ValueError("history_context has an unexpected shape")
        if tuple(target_context.shape) != (
            batch_size,
            self.target_context_dim,
        ):
            raise ValueError("target_context has an unexpected shape")
        for name, values in (
            ("timesteps", timesteps),
            ("stage_indices", stage_indices),
            ("device_indices", device_indices),
        ):
            if tuple(values.shape) != (batch_size,):
                raise ValueError(name + " must have shape [batch]")

        static = self.static_projection(torch.cat([
            self.stage_embedding(stage_indices),
            self.device_embedding(device_indices),
        ], dim=-1))
        context = (
            self.time_embedding(timesteps)
            + self.history_projection(history_context)
            + self.target_projection(target_context)
            + static
        )
        hidden = self.noisy_projection(noisy_residual)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output_projection(F.silu(self.output_norm(hidden)))


class HistoryConditionedResidualDDPM(nn.Module):
    """DDPM over one-step voltage innovation with immutable observed history."""

    def __init__(
        self,
        history_dim,
        target_context_dim,
        num_stages,
        num_devices,
        hidden_dim=128,
        depth=3,
        dropout=0.05,
        diffusion_steps=30,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta schedule must satisfy 0 < start < end < 1")
        self.history_dim = int(history_dim)
        self.target_context_dim = int(target_context_dim)
        self.num_stages = int(num_stages)
        self.num_devices = int(num_devices)
        self.diffusion_steps = int(diffusion_steps)
        self.denoiser = HistoryConditionedResidualDenoiser(
            history_dim=history_dim,
            target_context_dim=target_context_dim,
            num_stages=num_stages,
            num_devices=num_devices,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat([
            torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]
        ])
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        ).clamp_min(1e-20)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer(
            "posterior_variance", posterior_variance
        )
        self.register_buffer(
            "posterior_mean_coefficient_1",
            betas * torch.sqrt(alpha_bars_previous) / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coefficient_2",
            (1.0 - alpha_bars_previous)
            * torch.sqrt(alphas)
            / (1.0 - alpha_bars),
        )

    def q_sample(self, clean_residual, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_residual)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean_residual.shape)
            * clean_residual
            + _extract(
                self.sqrt_one_minus_alpha_bars,
                timesteps,
                clean_residual.shape,
            )
            * noise
        )

    def loss(
        self,
        clean_residual,
        history_context,
        target_context,
        stage_indices,
        device_indices,
        timesteps=None,
        noise=None,
    ):
        batch_size = clean_residual.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion_steps,
                (batch_size,),
                device=clean_residual.device,
            )
        if noise is None:
            noise = torch.randn_like(clean_residual)
        noisy = self.q_sample(clean_residual, timesteps, noise)
        predicted = self.denoiser(
            noisy,
            timesteps,
            history_context,
            target_context,
            stage_indices,
            device_indices,
        )
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(
        self,
        history_context,
        target_context,
        stage_indices,
        device_indices,
        clip_denoised=5.0,
    ):
        batch_size = history_context.shape[0]
        if target_context.shape[0] != batch_size:
            raise ValueError("condition tensors must have equal batch size")
        model_device = self.betas.device
        history_context = history_context.to(model_device)
        target_context = target_context.to(model_device)
        stage_indices = stage_indices.to(model_device)
        device_indices = device_indices.to(model_device)
        sample = torch.randn(batch_size, 1, device=model_device)

        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full(
                (batch_size,), step, device=model_device, dtype=torch.long
            )
            predicted_noise = self.denoiser(
                sample,
                timesteps,
                history_context,
                target_context,
                stage_indices,
                device_indices,
            )
            predicted_clean = (
                sample
                - _extract(
                    self.sqrt_one_minus_alpha_bars,
                    timesteps,
                    sample.shape,
                )
                * predicted_noise
            ) / torch.sqrt(_extract(self.alpha_bars, timesteps, sample.shape))
            if clip_denoised is not None:
                predicted_clean = predicted_clean.clamp(
                    -clip_denoised, clip_denoised
                )
            posterior_mean = (
                _extract(
                    self.posterior_mean_coefficient_1,
                    timesteps,
                    sample.shape,
                )
                * predicted_clean
                + _extract(
                    self.posterior_mean_coefficient_2,
                    timesteps,
                    sample.shape,
                )
                * sample
            )
            if step > 0:
                variance = _extract(
                    self.posterior_variance, timesteps, sample.shape
                )
                sample = posterior_mean + torch.sqrt(variance) * torch.randn_like(sample)
            else:
                sample = posterior_mean
        return sample
