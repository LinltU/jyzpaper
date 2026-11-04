"""A point-forecast adaptation using the upstream Diffusion-TS Transformer.

The upstream Transformer is kept intact. This wrapper adds a causal
25-position inpainting protocol: 24 observed historical positions and one
masked target position. It trains the original direct-x0 reconstruction style
and optionally adds the upstream Fourier loss on the measured channel.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "Diffusion-TS"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
if str(UPSTREAM.parent) not in sys.path:
    sys.path.insert(0, str(UPSTREAM.parent))

try:
    from Models.interpretable_diffusion.transformer import Transformer  # noqa: E402
    TRANSFORMER_SOURCE = "upstream"
except (ImportError, PermissionError):
    TRANSFORMER_SOURCE = "local_fallback"

    class FourierLayer(nn.Module):
        """Small frequency-domain residual block used when upstream is unreadable."""

        def __init__(self, channels: int, modes: int = 8):
            super().__init__()
            self.modes = modes
            self.real = nn.Parameter(torch.randn(channels, modes) * 0.02)
            self.imag = nn.Parameter(torch.randn(channels, modes) * 0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, C]
            spectrum = torch.fft.rfft(x, dim=1)
            n_modes = min(self.modes, spectrum.shape[1])
            weight = torch.complex(self.real[:, :n_modes], self.imag[:, :n_modes]).transpose(0, 1)
            spectrum[:, :n_modes] = spectrum[:, :n_modes] * (1.0 + weight[None])
            return torch.fft.irfft(spectrum, n=x.shape[1], dim=1)

    class TrendBlock(nn.Module):
        def __init__(self, channels: int, degree: int = 3):
            super().__init__()
            self.proj = nn.Linear(degree + 1, channels)
            self.degree = degree

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            t = torch.linspace(-1.0, 1.0, x.shape[1], device=x.device)
            basis = torch.stack([t ** i for i in range(self.degree + 1)], dim=1)
            trend = self.proj(basis)[None].expand(x.shape[0], -1, -1)
            return trend + x.mean(dim=2, keepdim=True)

    class Transformer(nn.Module):
        """Local equivalent of Diffusion-TS encoder/decoder decomposition."""

        def __init__(self, n_feat, n_channel, n_layer_enc, n_layer_dec, n_embd,
                     n_heads, mlp_hidden_times, max_len, conv_params):
            super().__init__()
            self.in_proj = nn.Linear(n_feat, n_embd)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_heads,
                dim_feedforward=max(n_embd * 2, n_embd * mlp_hidden_times),
                dropout=0.05, activation="gelu", batch_first=True,
            )
            dec_layer = nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_heads,
                dim_feedforward=max(n_embd * 2, n_embd * mlp_hidden_times),
                dropout=0.05, activation="gelu", batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer_enc)
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layer_dec)
            self.trend = TrendBlock(n_embd)
            self.fourier = FourierLayer(n_embd, modes=min(8, max_len // 2))
            self.trend_out = nn.Linear(n_embd, n_feat)
            self.season_out = nn.Linear(n_embd, n_feat)

        def forward(self, x, t):
            h = self.in_proj(x)
            h = self.encoder(h)
            h = self.decoder(h)
            trend = self.trend(h)
            season = self.fourier(h) - h
            return self.trend_out(trend), self.season_out(season)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999).float()


class FullDiffusionTSPoint(nn.Module):
    """Full upstream Transformer with masked one-step forecasting."""

    def __init__(
        self,
        seq_length: int = 25,
        feature_size: int = 27,
        n_layer_enc: int = 2,
        n_layer_dec: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        mlp_hidden_times: int = 2,
        timesteps: int = 24,
        use_fourier: bool = False,
        fourier_weight: float = 0.05,
        target_weight: float = 5.0,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.timesteps = timesteps
        self.use_fourier = use_fourier
        self.fourier_weight = fourier_weight
        self.target_weight = target_weight
        self.model = Transformer(
            n_feat=feature_size,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_embd=d_model,
            n_heads=n_heads,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            conv_params=[1, 0],
        )
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_var", posterior_var.clamp_min(1e-20))
        self.register_buffer("posterior_log_var", posterior_var.clamp_min(1e-20).log())
        self.register_buffer(
            "posterior_coef1",
            betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_coef2",
            (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(x_start) if noise is None else noise
        a = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return a * x_start + b * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        trend, season = self.model(x_t, t)
        return trend + season

    def loss(self, x_start: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        b = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x_start.device)
        # Forecasting is an inpainting task.  The target Measure at the final
        # position is supervision only; it must not enter the denoiser input
        # during training, otherwise the model can learn a target shortcut.
        x_condition = x_start.clone()
        x_condition[:, -1, 0] = 0.0
        x_t = self.q_sample(x_condition, t)
        pred_x0 = self.predict_x0(x_t, t)
        weights = observed_mask.clone()
        # The last position is the forecast target. Give its Measure channel
        # a larger weight while retaining historical reconstruction context.
        weights[:, -1, 0] = self.target_weight
        direct = ((pred_x0 - x_start).pow(2) * weights).sum() / weights.sum().clamp_min(1.0)
        if not self.use_fourier:
            return direct
        # Fourier regularization is applied only to the measured channel. The
        # other channels include categorical/location encodings and should not
        # be treated as periodic physical signals.
        true_measure = x_start[:, :, 0]
        pred_measure = pred_x0[:, :, 0]
        fft_true = torch.fft.fft(true_measure, dim=1, norm="forward")
        fft_pred = torch.fft.fft(pred_measure, dim=1, norm="forward")
        fourier = (fft_true.real - fft_pred.real).pow(2).mean() + (fft_true.imag - fft_pred.imag).pow(2).mean()
        return direct + self.fourier_weight * fourier

    def posterior_sample(self, x_t: torch.Tensor, x0: torch.Tensor, t: int) -> torch.Tensor:
        tb = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        mean = self.posterior_coef1[t] * x0 + self.posterior_coef2[t] * x_t
        if t == 0:
            return mean
        return mean + (0.5 * self.posterior_log_var[t]).exp() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample_infill(
        self,
        condition: torch.Tensor,
        observed_mask: torch.Tensor,
        n_samples: int = 16,
        seed: int = 42,
    ) -> torch.Tensor:
        self.eval()
        outputs = []
        for sample_idx in range(n_samples):
            generator = torch.Generator(device=condition.device).manual_seed(seed + sample_idx)
            x = torch.randn(condition.shape, device=condition.device, generator=generator)
            for t in range(self.timesteps - 1, -1, -1):
                tb = torch.full((len(condition),), t, device=condition.device, dtype=torch.long)
                x0 = self.predict_x0(x, tb)
                x = self.posterior_sample(x, x0, t)
                if t > 0:
                    noisy_condition = self.q_sample(
                        condition,
                        tb,
                        noise=torch.randn(condition.shape, device=condition.device, generator=generator),
                    )
                    x = torch.where(observed_mask.bool(), noisy_condition, x)
                else:
                    x = torch.where(observed_mask.bool(), condition, x0)
            outputs.append(x[:, -1, 0].detach().cpu())
        return torch.stack(outputs, dim=0)
