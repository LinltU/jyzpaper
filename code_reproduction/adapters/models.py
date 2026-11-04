"""Small CPU-friendly reproductions of the transferable parts of the papers.

The upstream projects remain under ``upstream/``.  This module is intentionally
independent from their dataset loaders because those loaders assume ETT,
Weather, NASA, SWaT, or other paper-specific formats.  The classes below keep
the algorithmic ideas that are meaningful for this project: conditional
diffusion, trend/state conditioning, masked reconstruction, consistency
regularization, and conditional tabular diffusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -np.log(10000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding.shape[1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding


class ConditionalDenoiser(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int = 96, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim + time_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, noisy_residual: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time = sinusoidal_embedding(t, self.time_dim)
        return self.net(torch.cat([noisy_residual, time, condition], dim=1)).squeeze(1)


class ConditionalResidualDiffusion:
    """A compact conditional DDPM for one-step innovation residuals.

    It follows the transferable core of TimeGrad/TimeDiff/DiffPLF/CCDM and
    Diffusion-TS, but predicts the next Measure innovation rather than a whole
    public benchmark trajectory.  This is deliberate: the local task is
    one-step online prediction with known target environmental covariates.
    """

    def __init__(
        self,
        history_dim: int,
        context_dim: int,
        n_devices: int,
        seq_len: int = 24,
        mode: str = "persistence",
        steps: int = 16,
        hidden_dim: int = 96,
        seed: int = 42,
    ):
        self.mode = mode
        self.steps = steps
        self.seed = seed
        self.device = torch.device("cpu")
        self.seq_len = int(seq_len)
        self.history_dim = int(history_dim)
        self.context_dim = int(context_dim)
        self.n_devices = max(1, int(n_devices))
        # four phase indicators and device identity are condition variables,
        # not independent measured channels.
        # The history tensor is flattened before conditioning, so its
        # contribution is sequence length x channel count (not just the
        # channel count).  Keeping this explicit prevents silent truncation of
        # the device/phase condition when seq_len changes.
        self.condition_dim = self.seq_len * self.history_dim + self.context_dim + 4 + self.n_devices
        self.model = ConditionalDenoiser(self.condition_dim, hidden_dim=hidden_dim).to(self.device)
        beta = torch.linspace(1e-4, 0.02, steps, dtype=torch.float32)
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.base_scale = 1.0

    def _base(self, x: torch.Tensor) -> torch.Tensor:
        # x is standardized and the first channel is Measure.
        measure = x[:, :, 0]
        if self.mode == "trend":
            tail = measure[:, -min(6, measure.shape[1]) :]
            slope = (tail[:, -1] - tail[:, 0]) / max(tail.shape[1] - 1, 1)
            return tail[:, -1] + torch.clamp(slope, -0.35, 0.35)
        if self.mode == "state_space":
            weights = torch.arange(1, min(8, measure.shape[1]) + 1, dtype=measure.dtype, device=measure.device)
            tail = measure[:, -len(weights) :]
            return (tail * weights).sum(dim=1) / weights.sum()
        if self.mode == "nonstationary":
            # Use the latest value as the center; residual normalization is
            # handled during fit/sample using a local scale.
            return measure[:, -1]
        return measure[:, -1]

    def _condition(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        stage: torch.Tensor,
        device_index: torch.Tensor,
    ) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        if self.mode == "channel":
            # Channel-aware contrastive weighting: emphasize the three direct
            # measured channels while retaining all derived context channels.
            weights = torch.ones(x.shape[2], device=x.device)
            weights[:3] = torch.tensor([1.25, 1.0, 1.15], device=x.device)
            flat = (x * weights.view(1, 1, -1)).reshape(x.shape[0], -1)
        phase = torch.nn.functional.one_hot(stage.clamp(0, 3), num_classes=4).float()
        device = torch.nn.functional.one_hot(
            device_index.clamp(0, self.n_devices - 1), num_classes=self.n_devices
        ).float()
        return torch.cat([flat, context, phase, device], dim=1)

    def fit(
        self,
        x: np.ndarray,
        context: np.ndarray,
        target: np.ndarray,
        stage: np.ndarray,
        device_index: np.ndarray,
        epochs: int = 14,
        batch_size: int = 256,
        lr: float = 8e-4,
    ) -> list[dict]:
        seed_everything(self.seed)
        x_t = torch.from_numpy(x).float()
        context_t = torch.from_numpy(context).float()
        stage_t = torch.from_numpy(stage).long()
        device_t = torch.from_numpy(device_index).long()
        with torch.no_grad():
            base = self._base(x_t)
        residual = torch.from_numpy(target).float() - base
        scale = float(residual.std().clamp_min(0.05))
        self.base_scale = scale
        dataset = TensorDataset(x_t, context_t, residual, stage_t, device_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=2e-4)
        history = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            losses = []
            for batch_x, batch_context, batch_residual, batch_stage, batch_device in loader:
                batch_x = batch_x.to(self.device)
                batch_context = batch_context.to(self.device)
                batch_residual = (batch_residual / scale).to(self.device)
                batch_stage = batch_stage.to(self.device)
                batch_device = batch_device.to(self.device)
                t = torch.randint(0, self.steps, (len(batch_x),), device=self.device)
                alpha_bar = self.alpha_bar[t]
                noise = torch.randn_like(batch_residual)
                noisy = alpha_bar.sqrt() * batch_residual + (1.0 - alpha_bar).sqrt() * noise
                condition = self._condition(batch_x, batch_context, batch_stage, batch_device)
                prediction = self.model(noisy[:, None], t.float() / max(self.steps - 1, 1), condition)
                loss = (prediction - noise).pow(2).mean()
                if self.mode == "nonstationary":
                    loss = loss + 0.02 * batch_residual.abs().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        return history

    @torch.no_grad()
    def sample(
        self,
        x: np.ndarray,
        context: np.ndarray,
        stage: np.ndarray,
        device_index: np.ndarray,
        n_samples: int = 48,
    ) -> np.ndarray:
        self.model.eval()
        x_t = torch.from_numpy(x).float().to(self.device)
        context_t = torch.from_numpy(context).float().to(self.device)
        stage_t = torch.from_numpy(stage).long().to(self.device)
        device_t = torch.from_numpy(device_index).long().to(self.device)
        condition = self._condition(x_t, context_t, stage_t, device_t)
        base = self._base(x_t)
        n = len(x_t)
        generated = []
        for sample_index in range(n_samples):
            generator = torch.Generator(device=self.device).manual_seed(self.seed + sample_index)
            current = torch.randn(n, device=self.device, generator=generator)
            for step in range(self.steps - 1, -1, -1):
                t = torch.full((n,), step / max(self.steps - 1, 1), device=self.device)
                pred_noise = self.model(current[:, None], t, condition)
                alpha = self.alpha[step]
                alpha_bar = self.alpha_bar[step]
                mean = (current - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
                if step > 0:
                    noise = torch.randn(n, device=self.device, generator=generator)
                    mean = mean + self.beta[step].sqrt() * noise
                current = mean
            generated.append((base + current * self.base_scale).cpu().numpy())
        return np.stack(generated, axis=0)


class MaskedConsistencyImputer(nn.Module):
    """CPU-friendly masked reconstruction model for CSDI/MTSCI/TSDE/CoSTI."""

    def __init__(
        self,
        time_len: int,
        channels: int,
        hidden_dim: int = 128,
        architecture: str = "mlp",
    ):
        super().__init__()
        self.time_len = time_len
        self.channels = channels
        self.architecture = architecture
        if architecture == "gru":
            self.gru = nn.GRU(
                channels * 2,
                max(16, hidden_dim // 2),
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.output = nn.Linear(max(16, hidden_dim // 2) * 2, channels)
        elif architecture == "conv":
            self.net = nn.Sequential(
                nn.Conv1d(channels * 2, hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, channels, kernel_size=3, padding=1),
            )
        else:
            input_dim = time_len * channels * 2
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, time_len * channels),
            )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([values, mask], dim=2)
        if self.architecture == "gru":
            encoded, _ = self.gru(joined)
            return self.output(encoded)
        if self.architecture == "conv":
            return self.net(joined.transpose(1, 2)).transpose(1, 2)
        return self.net(joined.reshape(len(values), -1)).reshape(
            len(values), self.time_len, self.channels
        )


class ImputationAdapter:
    def __init__(
        self,
        time_len: int,
        channels: int,
        seed: int = 42,
        architecture: str = "mlp",
        hidden_dim: int = 128,
    ):
        self.model = MaskedConsistencyImputer(
            time_len,
            channels,
            hidden_dim=hidden_dim,
            architecture=architecture,
        )
        self.seed = seed
        self.time_len = time_len
        self.channels = channels

    def _mask(self, batch: torch.Tensor, strategy: str, missing_ratio: float) -> torch.Tensor:
        """Build paper-style training masks (1 observed, 0 hidden)."""
        if strategy == "block":
            mask = torch.ones_like(batch)
            block = max(2, int(round(batch.shape[1] * missing_ratio)))
            for row in range(len(batch)):
                start = int(torch.randint(0, max(1, batch.shape[1] - block + 1), (1,)))
                mask[row, start : start + block, :] = 0.0
            return mask
        if strategy == "channel":
            mask = torch.ones_like(batch)
            for row in range(len(batch)):
                channel = int(torch.randint(0, batch.shape[2], (1,)))
                mask[row, :, channel] = 0.0
            return mask
        if strategy == "iif":
            # TSDE's IIF idea mixes imputation, interpolation and forecasting
            # masks instead of learning from one artificial missing pattern.
            selector = int(torch.randint(0, 3, (1,)))
            if selector == 1:
                return self._mask(batch, "block", missing_ratio)
            if selector == 2:
                mask = torch.ones_like(batch)
                tail = max(2, int(round(batch.shape[1] * missing_ratio)))
                mask[:, -tail:, :] = 0.0
                return mask
        return (torch.rand_like(batch) > missing_ratio).float()

    def fit(
        self,
        values: np.ndarray,
        epochs: int = 12,
        batch_size: int = 256,
        mask_strategy: str = "random",
        missing_ratio: float = 0.2,
        consistency_weight: float = 0.05,
        smoothness_weight: float = 0.0,
        noise_scale: float = 0.02,
    ) -> list[dict]:
        seed_everything(self.seed)
        data = torch.from_numpy(values).float()
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        history = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            losses = []
            for (batch,) in loader:
                random_mask = self._mask(batch, mask_strategy, missing_ratio)
                # Complementary masks reproduce the intra-consistency idea from
                # MTSCI while retaining a single compact network.
                complement = 1.0 - random_mask
                noisy = batch * random_mask + torch.randn_like(batch) * noise_scale * (1.0 - random_mask)
                pred_a = self.model(noisy, random_mask)
                pred_b = self.model(batch * complement, complement)
                masked_loss_a = ((1.0 - random_mask) * (pred_a - batch).pow(2)).sum() / (1.0 - random_mask).sum().clamp_min(1.0)
                masked_loss_b = ((1.0 - complement) * (pred_b - batch).pow(2)).sum() / (1.0 - complement).sum().clamp_min(1.0)
                consistency = (pred_a - pred_b).pow(2).mean()
                smoothness = (pred_a[:, 1:] - pred_a[:, :-1]).pow(2).mean()
                loss = (
                    masked_loss_a
                    + masked_loss_b
                    + consistency_weight * consistency
                    + smoothness_weight * smoothness
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        return history

    @torch.no_grad()
    def impute(self, values: np.ndarray, mask: np.ndarray, stochastic: bool = False, n_samples: int = 8) -> np.ndarray:
        self.model.eval()
        values_t = torch.from_numpy(values).float()
        mask_t = torch.from_numpy(mask).float()
        outputs = []
        for index in range(max(1, n_samples if stochastic else 1)):
            masked = values_t * mask_t
            if stochastic:
                seed_everything(self.seed + index)
                masked = masked + (1.0 - mask_t) * torch.randn_like(masked) * 0.02
            prediction = self.model(masked, mask_t)
            outputs.append((mask_t * values_t + (1.0 - mask_t) * prediction).numpy())
        return np.stack(outputs, axis=0)

    @torch.no_grad()
    def reconstruct_by_partitions(
        self,
        values: np.ndarray,
        partitions: int = 4,
        stochastic: bool = False,
        n_samples: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct every cell once, using disjoint temporal masks.

        This is used by the anomaly-detection reproductions.  It avoids the
        identity shortcut that occurs when a masked imputer is evaluated with
        an all-observed mask.
        """
        values = np.asarray(values, dtype=np.float32)
        reconstruction = np.zeros_like(values)
        uncertainty = np.zeros_like(values)
        for partition in range(partitions):
            mask = np.ones_like(values, dtype=np.float32)
            mask[:, partition::partitions, :] = 0.0
            samples = self.impute(values, mask, stochastic=stochastic, n_samples=n_samples)
            hidden = 1.0 - mask
            reconstruction += hidden * samples.mean(axis=0)
            uncertainty += hidden * samples.std(axis=0)
        return reconstruction, uncertainty


def run_treeffuser(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_samples: int = 48,
    n_steps: int = 24,
    n_repeats: int = 3,
    n_estimators: int = 120,
    return_model: bool = False,
) -> np.ndarray | tuple[np.ndarray, object]:
    """Call the official Treeffuser implementation when its dependencies work."""
    src = UPSTREAM / "Treeffuser" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from treeffuser import Treeffuser

    model = Treeffuser(
        n_repeats=n_repeats,
        n_estimators=n_estimators,
        early_stopping_rounds=15,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=15,
        n_jobs=2,
        seed=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    samples = model.sample(
        X_test,
        n_samples=n_samples,
        n_parallel=min(8, n_samples),
        n_steps=n_steps,
        seed=42,
        verbose=False,
    )
    if return_model:
        return samples, model
    return samples
