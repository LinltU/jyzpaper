import math
import torch
import torch.nn as nn

# 定义相对湿度转换为绝对湿度的函数
# （如有需要可移除，主模型只依赖SimpleTransformer）
def absolute_humidity(T, RH):
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=4, dropout=0.2, 
                 positional_encoding=True, feed_forward_dim=256):
        super(SimpleTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_positional_encoding = positional_encoding
        
        # 更强大的特征嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        # 可选的位置编码，对时序数据有帮助
        if positional_encoding:
            self.register_buffer('pos_encoding', self._create_positional_encoding(100, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
    def _create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.embedding(x)
        if self.use_positional_encoding:
            if x.size(0) <= self.pos_encoding.size(0):
                x = x + self.pos_encoding[:x.size(0), :]
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.output_layer(x)
        return x.squeeze(-1) 