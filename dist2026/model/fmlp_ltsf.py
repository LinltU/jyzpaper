import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Centered moving average over the temporal dimension."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding, 1)
        end = x[:, -1:, :].repeat(1, padding, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_average(x)
        return x - trend, trend


class FMLP_LTSF(nn.Module):
    """Multivariate FMLP forecaster with an explicit target channel.

    A linear decomposition branch forecasts the historical target series. A
    second branch mixes every input feature at each time step and projects that
    context across the forecast horizon. This makes temperature, humidity and
    device attributes affect the prediction instead of being discarded.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        hidden_dim=64,
        dropout=0.2,
        kernel_size=7,
        individual=False,
        use_mlp=True,
        target_index=0,
    ):
        super().__init__()
        if not 0 <= target_index < enc_in:
            raise ValueError(f"target_index {target_index} is outside enc_in={enc_in}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual  # Kept for configuration compatibility.
        self.use_mlp = use_mlp
        self.target_index = target_index

        self.decomposition = SeriesDecomposition(kernel_size)
        self.seasonal_projection = nn.Linear(seq_len, pred_len)
        self.trend_projection = nn.Linear(seq_len, pred_len)

        self.context_norm = nn.LayerNorm(enc_in)
        self.feature_mixer = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_temporal_projection = nn.Linear(seq_len, pred_len)
        self.context_output = nn.Linear(hidden_dim, 1)

        if use_mlp:
            self.fusion = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
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

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"FMLP_LTSF expects [batch, time, channels], got {tuple(x.shape)}"
            )
        if x.size(1) != self.seq_len or x.size(2) != self.channels:
            raise ValueError(
                "FMLP_LTSF input shape mismatch: "
                f"expected [batch, {self.seq_len}, {self.channels}], got {tuple(x.shape)}"
            )

        target = x[:, :, self.target_index:self.target_index + 1]
        seasonal, trend = self.decomposition(target)
        seasonal = self.seasonal_projection(seasonal.squeeze(-1))
        trend = self.trend_projection(trend.squeeze(-1))
        base_forecast = (seasonal + trend).unsqueeze(-1)

        context = self.feature_mixer(self.context_norm(x))
        context = self.context_temporal_projection(context.permute(0, 2, 1))
        context = context.permute(0, 2, 1)
        context_forecast = self.context_output(context)

        if self.use_mlp:
            return self.fusion(torch.cat([base_forecast, context_forecast], dim=-1))
        return base_forecast + context_forecast
