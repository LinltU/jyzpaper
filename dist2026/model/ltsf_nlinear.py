import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """Moving-average block used to expose the time-series trend."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Split a sequence into seasonal residual and moving trend."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class NLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prediction_layer = nn.Linear(hidden_dim, 1)
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x * self.interaction_layer(x)
        return F.softplus(self.prediction_layer(self.feature_layer(x))).squeeze(-1)


class DecompositionNLinear(nn.Module):
    """DLinear/NLinear-compatible model used by the legacy GUI checkpoints."""

    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, kernel_size=25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decomposition = series_decomp(kernel_size)
        self.seasonal_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.trend_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        seasonal, trend = self.decomposition(x.unsqueeze(1))
        seasonal = seasonal.squeeze(1)
        trend = trend.squeeze(1)
        seasonal = seasonal * self.interaction_layer(seasonal)
        trend = trend * self.interaction_layer(trend)
        combined = torch.cat(
            [self.seasonal_layer(seasonal), self.trend_layer(trend)],
            dim=1,
        )
        return self.fusion_layer(combined).squeeze(-1)
