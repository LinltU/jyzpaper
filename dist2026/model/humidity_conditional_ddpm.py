"""Compact conditional DDPM for humidity-aware sensor window generation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(values, timesteps, target_shape):
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *([1] * (len(target_shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be at least 4")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[1]))
        return embedding


class ConditionalResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.input_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, context):
        residual = hidden
        mixed = self.norm(hidden) + self.context_projection(context)
        value, gate = self.input_projection(mixed).chunk(2, dim=-1)
        mixed = value * torch.sigmoid(gate)
        mixed = self.output_projection(self.dropout(F.silu(mixed)))
        return residual + mixed


class ConditionalDenoiser(nn.Module):
    """Predict DDPM noise from a flattened sensor window and conditions."""

    def __init__(
        self,
        input_dim,
        num_stages,
        num_devices,
        hidden_dim=192,
        depth=4,
        dropout=0.05,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_stages <= 0 or num_devices <= 0:
            raise ValueError("num_stages and num_devices must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.input_dim = input_dim
        self.num_stages = num_stages
        self.num_devices = num_devices
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        stage_dim = min(32, max(8, hidden_dim // 8))
        device_dim = min(48, max(8, hidden_dim // 8))
        self.stage_embedding = nn.Embedding(num_stages, stage_dim)
        self.device_embedding = nn.Embedding(num_devices, device_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(stage_dim + device_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            ConditionalResidualBlock(hidden_dim, dropout) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, noisy_sample, timesteps, stage_indices, device_indices):
        if noisy_sample.ndim != 2 or noisy_sample.shape[1] != self.input_dim:
            raise ValueError(
                "noisy_sample must have shape [batch, {}]".format(self.input_dim)
            )
        batch_size = noisy_sample.shape[0]
        expected = (batch_size,)
        if tuple(timesteps.shape) != expected:
            raise ValueError("timesteps must have shape [batch]")
        if tuple(stage_indices.shape) != expected:
            raise ValueError("stage_indices must have shape [batch]")
        if tuple(device_indices.shape) != expected:
            raise ValueError("device_indices must have shape [batch]")

        stage_condition = self.stage_embedding(stage_indices)
        device_condition = self.device_embedding(device_indices)
        condition = self.condition_projection(
            torch.cat([stage_condition, device_condition], dim=-1)
        )
        context = self.time_embedding(timesteps) + condition
        hidden = self.input_projection(noisy_sample)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output_projection(F.silu(self.output_norm(hidden)))


class ConditionalDDPM(nn.Module):
    """DDPM wrapper with a linear beta schedule and ancestral sampling."""

    def __init__(
        self,
        input_dim,
        num_stages,
        num_devices,
        hidden_dim=192,
        depth=4,
        dropout=0.05,
        diffusion_steps=50,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta schedule must satisfy 0 < start < end < 1")

        self.input_dim = input_dim
        self.num_stages = num_stages
        self.num_devices = num_devices
        self.diffusion_steps = diffusion_steps
        self.denoiser = ConditionalDenoiser(
            input_dim=input_dim,
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
            torch.ones(1, dtype=alpha_bars.dtype),
            alpha_bars[:-1],
        ])
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        ).clamp_min(1e-20)
        posterior_mean_coefficient_1 = (
            betas * torch.sqrt(alpha_bars_previous) / (1.0 - alpha_bars)
        )
        posterior_mean_coefficient_2 = (
            (1.0 - alpha_bars_previous)
            * torch.sqrt(alphas)
            / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coefficient_1", posterior_mean_coefficient_1
        )
        self.register_buffer(
            "posterior_mean_coefficient_2", posterior_mean_coefficient_2
        )

    def q_sample(self, clean_sample, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_sample)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean_sample.shape)
            * clean_sample
            + _extract(
                self.sqrt_one_minus_alpha_bars, timesteps, clean_sample.shape
            )
            * noise
        )

    def loss(
        self,
        clean_sample,
        stage_indices,
        device_indices,
        timesteps=None,
        noise=None,
    ):
        batch_size = clean_sample.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.diffusion_steps, (batch_size,), device=clean_sample.device
            )
        if noise is None:
            noise = torch.randn_like(clean_sample)
        noisy_sample = self.q_sample(clean_sample, timesteps, noise)
        predicted_noise = self.denoiser(
            noisy_sample, timesteps, stage_indices, device_indices
        )
        return F.mse_loss(predicted_noise, noise)

    def forward(self, noisy_sample, timesteps, stage_indices, device_indices):
        return self.denoiser(
            noisy_sample, timesteps, stage_indices, device_indices
        )

    @torch.no_grad()
    def sample(self, stage_indices, device_indices, clip_denoised=5.0):
        if stage_indices.ndim != 1 or device_indices.ndim != 1:
            raise ValueError("condition tensors must have shape [batch]")
        if stage_indices.shape != device_indices.shape:
            raise ValueError("stage and device conditions must have equal length")
        model_device = self.betas.device
        stage_indices = stage_indices.to(model_device)
        device_indices = device_indices.to(model_device)
        sample = torch.randn(
            stage_indices.shape[0], self.input_dim, device=model_device
        )

        for step in reversed(range(self.diffusion_steps)):
            timesteps = torch.full(
                (sample.shape[0],), step, device=model_device, dtype=torch.long
            )
            predicted_noise = self.denoiser(
                sample, timesteps, stage_indices, device_indices
            )
            alpha_bar = _extract(self.alpha_bars, timesteps, sample.shape)
            predicted_clean = (
                sample
                - _extract(
                    self.sqrt_one_minus_alpha_bars, timesteps, sample.shape
                )
                * predicted_noise
            ) / torch.sqrt(alpha_bar)
            if clip_denoised is not None:
                predicted_clean = predicted_clean.clamp(
                    min=-clip_denoised, max=clip_denoised
                )
            posterior_mean = (
                _extract(
                    self.posterior_mean_coefficient_1, timesteps, sample.shape
                )
                * predicted_clean
                + _extract(
                    self.posterior_mean_coefficient_2, timesteps, sample.shape
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
