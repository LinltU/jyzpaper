import torch
import torch.nn as nn
import torch.nn.functional as F

from .fmlp_ltsf import SeriesDecomposition


class HumidityStageFMLP(nn.Module):
    """Shared FMLP encoder with humidity-stage residual experts.

    The model predicts ordered P10/P50/P90 quantiles. A shared temporal encoder
    learns behavior common to all humidity regimes. Four compact expert heads
    then learn residual corrections for dry, transition, humid and very-humid
    stages. Routing is smooth around 50%, 65% and 80% humidity boundaries.
    """

    STAGE_NAMES = ("dry", "transition", "humid", "very_humid")
    QUANTILES = (0.1, 0.5, 0.9)

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        hidden_dim=64,
        expert_hidden_dim=32,
        dropout=0.15,
        kernel_size=7,
        target_index=0,
        humidity_index=2,
        routing_temperature=2.5,
        learned_gate_scale=0.25,
        num_devices=0,
        device_embedding_dim=0,
        known_covariate_dim=0,
    ):
        super().__init__()
        if not 0 <= target_index < enc_in:
            raise ValueError(f"target_index {target_index} is outside enc_in={enc_in}")
        if not 0 <= humidity_index < enc_in:
            raise ValueError(f"humidity_index {humidity_index} is outside enc_in={enc_in}")
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        if num_devices < 0 or device_embedding_dim < 0 or known_covariate_dim < 0:
            raise ValueError("static context dimensions must be non-negative")
        if (num_devices == 0) != (device_embedding_dim == 0):
            raise ValueError(
                "num_devices and device_embedding_dim must both be zero or both be positive"
            )

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.hidden_dim = hidden_dim
        self.target_index = target_index
        self.humidity_index = humidity_index
        self.routing_temperature = routing_temperature
        self.learned_gate_scale = learned_gate_scale
        self.num_devices = num_devices
        self.device_embedding_dim = device_embedding_dim
        self.known_covariate_dim = known_covariate_dim

        self.decomposition = SeriesDecomposition(kernel_size)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)
        self.trend_projection = nn.Linear(seq_len, pred_len)

        self.feature_norm = nn.LayerNorm(enc_in)
        self.feature_mixer = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_temporal_projection = nn.Linear(seq_len, pred_len)
        self.base_projection = nn.Linear(1, hidden_dim)
        self.shared_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.device_embedding = (
            nn.Embedding(num_devices + 1, device_embedding_dim)
            if device_embedding_dim > 0
            else None
        )
        static_context_dim = device_embedding_dim + known_covariate_dim
        self.static_projection = (
            nn.Sequential(
                nn.Linear(static_context_dim, hidden_dim),
                nn.GELU(),
            )
            if static_context_dim > 0
            else None
        )

        self.global_quantile_head = nn.Linear(hidden_dim, 3)
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, 3),
            )
            for _ in self.STAGE_NAMES
        ])

        self.learned_gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, len(self.STAGE_NAMES)),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Start from interpretable humidity priors before learning corrections.
        final_gate = self.learned_gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

        # Start with a moderate quantile interval in standardized residual units.
        with torch.no_grad():
            self.global_quantile_head.bias[1:].fill_(-1.5)

    def _stage_priors(self, humidity):
        temperature = self.routing_temperature
        above_50 = torch.sigmoid((humidity - 50.0) / temperature)
        above_65 = torch.sigmoid((humidity - 65.0) / temperature)
        above_80 = torch.sigmoid((humidity - 80.0) / temperature)

        priors = torch.stack([
            1.0 - above_50,
            above_50 * (1.0 - above_65),
            above_65 * (1.0 - above_80),
            above_80,
        ], dim=-1)
        return priors / priors.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _routing_weights(self, x, routing_features=None):
        if routing_features is None:
            humidity = x[:, -1, self.humidity_index]
            previous_humidity = x[:, -2, self.humidity_index]
            humidity_trend = humidity - previous_humidity
        else:
            if routing_features.dim() != 2 or routing_features.size(1) != 2:
                raise ValueError(
                    "routing_features must have shape [batch, 2] containing "
                    "current humidity and humidity trend"
                )
            humidity = routing_features[:, 0]
            humidity_trend = routing_features[:, 1]

        priors = self._stage_priors(humidity)
        normalized_routing = torch.stack([
            humidity / 100.0,
            humidity_trend.clamp(-20.0, 20.0) / 20.0,
        ], dim=-1)
        learned_logits = self.learned_gate(normalized_routing)
        logits = torch.log(priors.clamp_min(1e-8)) + self.learned_gate_scale * learned_logits
        return torch.softmax(logits, dim=-1), priors

    @staticmethod
    def _ordered_quantiles(raw_parameters):
        median = raw_parameters[..., 0:1]
        lower = median - F.softplus(raw_parameters[..., 1:2])
        upper = median + F.softplus(raw_parameters[..., 2:3])
        return torch.cat([lower, median, upper], dim=-1)

    def _static_context(self, x, device_indices, known_covariates):
        if self.static_projection is None:
            return None

        context = []
        if self.device_embedding is not None:
            if device_indices is None or device_indices.dim() != 1:
                raise ValueError("device_indices must have shape [batch]")
            if device_indices.size(0) != x.size(0):
                raise ValueError("device_indices batch size does not match x")
            if device_indices.min() < 0 or device_indices.max() > self.num_devices:
                raise ValueError("device_indices contains an out-of-range value")
            context.append(self.device_embedding(device_indices.long()))

        if self.known_covariate_dim > 0:
            if (
                known_covariates is None
                or known_covariates.dim() != 2
                or known_covariates.size(0) != x.size(0)
                or known_covariates.size(1) != self.known_covariate_dim
            ):
                raise ValueError(
                    "known_covariates must have shape [batch, {}]".format(
                        self.known_covariate_dim
                    )
                )
            context.append(known_covariates)

        return self.static_projection(torch.cat(context, dim=-1))

    def forward(
        self,
        x,
        routing_features=None,
        device_indices=None,
        known_covariates=None,
    ):
        if x.dim() != 3:
            raise ValueError(f"Expected [batch, time, channels], got {tuple(x.shape)}")
        if x.size(1) != self.seq_len or x.size(2) != self.enc_in:
            raise ValueError(
                f"Expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )

        target = x[:, :, self.target_index:self.target_index + 1]
        seasonal, trend = self.decomposition(target)
        base_forecast = (
            self.seasonal_projection(seasonal.squeeze(-1))
            + self.trend_projection(trend.squeeze(-1))
        ).unsqueeze(-1)

        context = self.feature_mixer(self.feature_norm(x))
        context = self.context_temporal_projection(context.permute(0, 2, 1))
        context = context.permute(0, 2, 1)
        shared = context + self.base_projection(base_forecast)
        static_context = self._static_context(x, device_indices, known_covariates)
        if static_context is not None:
            shared = shared + static_context[:, None, :]
        shared = shared + self.shared_block(shared)

        global_parameters = self.global_quantile_head(shared)
        expert_parameters = torch.stack(
            [expert(shared) for expert in self.expert_heads],
            dim=2,
        )
        gates, priors = self._routing_weights(x, routing_features)
        expert_residual = (
            expert_parameters * gates[:, None, :, None]
        ).sum(dim=2)
        quantiles = self._ordered_quantiles(global_parameters + expert_residual)

        return {
            "quantiles": quantiles,
            "gates": gates,
            "stage_priors": priors,
            "expert_residuals": expert_parameters,
        }
