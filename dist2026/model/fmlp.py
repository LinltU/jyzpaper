import torch
import torch.nn as nn


class FMLP(nn.Module):
    """Feature-gated MLP for tabular regression.

    The previous implementation reshaped each row to a sequence of length one
    before applying a moving-average decomposition. For a one-step sequence the
    seasonal component is always zero, so the decomposition added parameters
    without adding information. This model keeps the useful feature gating and
    uses a residual MLP directly on the tabular features.
    """

    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, kernel_size=7):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size  # Kept for configuration compatibility.

        self.input_norm = nn.LayerNorm(input_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.feature_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

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
        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"FMLP expects [batch, {self.input_dim}], got {tuple(x.shape)}"
            )

        normalized = self.input_norm(x)
        gated = normalized * self.feature_gate(normalized)
        residual = self.input_projection(gated)
        hidden = residual + self.feature_block(residual)
        return self.output_layer(hidden).squeeze(-1)
