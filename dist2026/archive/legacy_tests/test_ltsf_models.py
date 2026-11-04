"""
Test LTSF-Linear models on combined_data.csv
Including: Autoformer, Informer, Transformer, DLinear, Linear, NLinear, Naive, ARIMA, GBRT
Also includes v1 versions of DLinear and Transformer using LTSF-Linear architecture
"""
import pandas as pd
import numpy as np
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import copy
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def absolute_humidity(T, RH):
    """相对湿度转换为绝对湿度"""
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH


def prepare_features(df):
    """准备特征，按设备分组计算时序特征"""
    df = df.copy()

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['IMEI', 'Timestamp'], ascending=[True, True]).reset_index(drop=True)

    # 时间特征
    df['Hour_group'] = (df['Timestamp'].dt.hour // 3).astype(int)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month

    # 周期性特征
    df['Hour_group_sin'] = np.sin(2 * np.pi * df['Hour_group'] / 8)
    df['Hour_group_cos'] = np.cos(2 * np.pi * df['Hour_group'] / 8)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['AH'] = df.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)

    # 时序特征
    df['Measure_diff'] = df.groupby('IMEI')['Measure'].diff().fillna(0)
    df['Temp_diff'] = df.groupby('IMEI')['Temp'].diff().fillna(0)
    df['Hum_diff'] = df.groupby('IMEI')['Hum'].diff().fillna(0)

    def calc_rolling_ma(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    df['Measure_MA'] = df.groupby('IMEI')['Measure'].transform(lambda x: calc_rolling_ma(x, window=5))
    df['Measure_lag1'] = df.groupby('IMEI')['Measure'].shift(1).fillna(0)

    # 交互特征
    df['Temp_Hum_interact'] = df['Temp'] * df['Hum']
    df['Temp_AH_interact'] = df['Temp'] * df['AH']

    # 湿度区间
    df['Hum_0_35'] = np.where(df['Hum'] < 35, 1, 0)
    df['Hum_35_50'] = np.where((df['Hum'] >= 35) & (df['Hum'] < 50), 1, 0)
    df['Hum_50_60'] = np.where((df['Hum'] >= 50) & (df['Hum'] < 60), 1, 0)
    df['Hum_60_70'] = np.where((df['Hum'] >= 60) & (df['Hum'] < 70), 1, 0)
    df['Hum_70_80'] = np.where((df['Hum'] >= 70) & (df['Hum'] < 80), 1, 0)

    df['Hum_high'] = np.where(df['Hum'] >= 60, 1, 0)
    df['HumTemp_weight'] = df['Hum'] / (df['Temp'] + 5)
    df['HumTemp_weight_high'] = np.where(df['HumTemp_weight'] > 8, 1, 0)

    df['HumTemp_x_diff'] = df['HumTemp_weight_high'] * df['Measure_diff'].abs()
    df['HumTemp_x_MA'] = df['HumTemp_weight_high'] * df['Measure_MA']
    df['HumTemp_x_lag1'] = df['HumTemp_weight_high'] * df['Measure_lag1']

    # 位置特征
    df['Loc_Left'] = np.where(df['Location'].str.contains('左边|左|LLow|RUp', regex=True), 1, 0)
    df['Loc_Right'] = np.where(df['Location'].str.contains('右边|右|RUp', regex=True), 1, 0)
    df['Loc_Upper'] = np.where(df['Location'].str.contains('上相|Upper', regex=True), 1, 0)
    df['Loc_Lower'] = np.where(df['Location'].str.contains('下相|Lower|LLow', regex=True), 1, 0)
    df['Loc_Small'] = np.where(df['Location'].str.contains('小号侧|Str1', regex=True), 1, 0)
    df['Loc_Big'] = np.where(df['Location'].str.contains('大号侧|Str2', regex=True), 1, 0)
    df['Loc_StrNum'] = np.where(df['Location'].str.contains('Str1', regex=True), 1,
                                np.where(df['Location'].str.contains('Str2', regex=True), 2, 0))
    df['Loc_220kV'] = np.where(df['Location'].str.contains('220kV', regex=True), 1, 0)
    df['Loc_PlateCount'] = np.where(df['Location'].str.contains('18', regex=True), 18,
                                   np.where(df['Location'].str.contains('9', regex=True), 9, 10))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def get_feature_columns():
    return [
        'Hour_group', 'DayOfWeek', 'Month',
        'Hour_group_sin', 'Hour_group_cos',
        'DayOfWeek_sin', 'DayOfWeek_cos',
        'Month_sin', 'Month_cos',
        'Temp', 'Hum', 'AH',
        'Measure_diff', 'Temp_diff', 'Hum_diff', 'Measure_MA', 'Measure_lag1',
        'Temp_Hum_interact', 'Temp_AH_interact',
        'Hum_0_35', 'Hum_35_50', 'Hum_50_60', 'Hum_60_70', 'Hum_70_80',
        'HumTemp_weight', 'HumTemp_weight_high',
        'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1',
        'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
        'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount'
    ]


# ===== 以下是LTSF-Linear模型的核心代码 =====

class moving_avg(nn.Module):
    """Moving average block"""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ============ DLinear (LTSF-Linear version) - 单变量版本 ============
class DLinear_LTSF(nn.Module):
    """DLinear from LTSF-Linear - adapted for single target prediction"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [batch, seq_len, 1] - 单变量
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


# ============ NLinear (LTSF-Linear version) - 单变量版本 ============
class NLinear_LTSF(nn.Module):
    """NLinear from LTSF-Linear - adapted for single target"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x


# ============ Linear (Simple) - 单变量版本 ============
class Linear_LTSF(nn.Module):
    """Simple Linear from LTSF-Linear"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)


# ============ NLinear (LTSF-Linear version) ============
class NLinear_LTSF(nn.Module):
    """NLinear from LTSF-Linear"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x


# ============ Linear (Simple) ============
class Linear_LTSF(nn.Module):
    """Simple Linear from LTSF-Linear"""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)


# ============ Transformer Encoder-Decoder ============
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        import math
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding.from_pretrained(w)
        self.c_in = c_in
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x.device())


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.d_in = d_in
        self.embed = nn.Linear(d_in, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, d_in, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_in=d_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum('blhe,bshe->bhls', queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, S), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0), -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum('bhls,bshd->blhd', A, values)
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.attention = attention
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model * n_heads, kernel_size=1)
        self.n_heads = n_heads
        self.linear = nn.Linear(d_model * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.conv(queries.transpose(1, 2)).reshape(B, self.n_heads, L, -1).permute(0, 1, 2, 3)
        keys = self.conv(keys.transpose(1, 2)).reshape(B, self.n_heads, S, -1).permute(0, 1, 2, 3)
        values = self.conv(values.transpose(1, 2)).reshape(B, self.n_heads, S, -1).permute(0, 1, 2, 3)
        out, attns = self.attention(queries, keys, values, attn_mask)
        out = out.permute(0, 2, 1).reshape(B, L, -1)
        return self.linear(out)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


# ============ Transformer (LTSF-Linear version) ============
class Transformer_LTSF(nn.Module):
    """Transformer from LTSF-Linear"""
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.enc_embedding = DataEmbedding(configs.c_in, configs.d_model, configs.d_in, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.c_in, configs.d_model, configs.d_in, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        return dec_out[:, -self.pred_len:, :]


# ============ Simple Config for models ============
class SimpleConfig:
    def __init__(self):
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 256
        self.dropout = 0.1
        self.activation = 'gelu'
        self.factor = 1
        self.c_in = 1
        self.c_out = 1
        self.d_in = 4


# ============ Dataset for LTSF models ============
class LTSFDataset(Dataset):
    def __init__(self, data_x, data_y, seq_len, label_len, pred_len):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # [seq_len, 1]
        seq_y = self.data_y[r_begin:r_end]  # [label_len+pred_len, 1]
        # 时间标记 (month, day, weekday, hour)
        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), \
               torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)


def train_ltsf_model(model, train_loader, valid_loader, device, max_epochs=20, lr=0.001):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            optimizer.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -model.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :model.pred_len // 2, :], dec_inp], dim=1).float()

            if hasattr(model, 'enc_embedding'):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x)

            outputs = outputs[:, -model.pred_len:, :]
            batch_y_target = batch_y[:, -model.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x_mark = batch_x_mark.to(device)
                batch_y_mark = batch_y_mark.to(device)

                dec_inp = torch.zeros_like(batch_y[:, -model.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :model.pred_len // 2, :], dec_inp], dim=1).float()

                if hasattr(model, 'enc_embedding'):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x)

                outputs = outputs[:, -model.pred_len:, :]
                batch_y_target = batch_y[:, -model.pred_len:, :].to(device)
                loss = criterion(outputs, batch_y_target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{max_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def predict_ltsf(model, data_x, seq_len, pred_len, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(data_x) - seq_len - pred_len + 1):
            seq_x = data_x[i:i + seq_len]
            seq_x = torch.FloatTensor(seq_x).unsqueeze(0).to(device)
            seq_x_mark = torch.zeros(1, seq_len, 4).to(device)
            dec_inp = torch.zeros(1, pred_len, 1).to(device)
            dec_mark = torch.zeros(1, pred_len, 4).to(device)

            if hasattr(model, 'enc_embedding'):
                output = model(seq_x, seq_x_mark, dec_inp, dec_mark)
            else:
                output = model(seq_x)

            pred = output[:, -1, 0].cpu().numpy()[0]
            preds.append(pred)
    return np.array(preds)


# ============ Original Models from test.py ============
from model.Transformer import SimpleTransformer
from model.ltsf_nlinear import DecompositionNLinear


def train_original_transformer(X_train, y_train, X_test, y_test):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train['Measure'].values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test['Measure'].values)

    batch_size = min(32, len(X_train) // 2)
    batch_size = max(8, batch_size)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = SimpleTransformer(
        input_dim=input_dim, hidden_dim=128, num_layers=3, num_heads=8,
        dropout=0.2, feed_forward_dim=256
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    patience = 15
    patience_counter = 0
    best_epoch_loss = float('inf')
    best_model_state = None
    max_epochs = 200

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in valid_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs.squeeze(), val_y).item()

        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_epoch_loss:
            best_epoch_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    return model.to('cpu'), "transformer"


def train_original_dlinear(X_train, y_train, X_test, y_test):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train['Measure'].values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test['Measure'].values)

    batch_size = min(32, len(X_train) // 2)
    batch_size = max(8, batch_size)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = DecompositionNLinear(input_dim=input_dim, hidden_dim=128, dropout=0.2, kernel_size=25).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    patience = 15
    patience_counter = 0
    best_epoch_loss = float('inf')
    best_model_state = None
    max_epochs = 200

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in valid_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs.squeeze(), val_y).item()

        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_epoch_loss:
            best_epoch_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    return model.to('cpu'), "dlinear"


def train_svm_model(X_train, y_train, X_test, y_test):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train['Measure'].values)
    return model, "svm"


def predict_model(model, X_test, model_type):
    if model_type == "svm":
        predictions = model.predict(X_test)
    else:
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            if model_type in ["transformer", "dlinear"]:
                predictions = model(X_test_tensor).squeeze().numpy()
            else:
                predictions = model(X_test_tensor).squeeze().numpy()
    return predictions


def evaluate_and_print(predictions, y_test):
    mse = mean_squared_error(y_test['Measure'].values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test['Measure'].values, predictions)
    r2 = r2_score(y_test['Measure'].values, predictions)
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.6f}")
    return mse, rmse, mae, r2


def main():
    print("=" * 60)
    print("LTSF-Linear Models Training")
    print("=" * 60)

    # 加载数据
    print("\nLoading data...")
    df = pd.read_csv('combined_data.csv')
    print(f"Original: {len(df)} rows")

    # 过滤湿度
    df = df[df['Hum'] < 80]
    df = df[df['Measure'] >= 0]
    print(f"After filtering: {len(df)} rows")

    # 准备特征
    print("Preparing features...")
    df_prepared = prepare_features(df)
    feature_cols = get_feature_columns()

    X = df_prepared[feature_cols].fillna(0)
    y = df_prepared[['Measure']]

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 保存测试数据
    test_df = df_prepared.iloc[X_test.index].copy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== 训练原始模型 =====
    print("\n" + "=" * 60)
    print("Training Original Models (Transformer, DLinear, SVM)")
    print("=" * 60)

    # Transformer (Original)
    print("\n[1] Training Original Transformer...")
    model_trans, type_trans = train_original_transformer(X_train_scaled, y_train, X_test_scaled, y_test)
    pred_trans = predict_model(model_trans, X_test_scaled, type_trans)
    mse, rmse, mae, r2 = evaluate_and_print(pred_trans, y_test)

    result_df = test_df.copy()
    result_df['PredictResult'] = pred_trans
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv('results_transformer.csv', index=False, encoding='utf-8-sig')
    print("Saved: results_transformer.csv")

    # DLinear (Original)
    print("\n[2] Training Original DLinear...")
    model_dlin, type_dlin = train_original_dlinear(X_train_scaled, y_train, X_test_scaled, y_test)
    pred_dlin = predict_model(model_dlin, X_test_scaled, type_dlin)
    mse, rmse, mae, r2 = evaluate_and_print(pred_dlin, y_test)

    result_df = test_df.copy()
    result_df['PredictResult'] = pred_dlin
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv('results_dlinear.csv', index=False, encoding='utf-8-sig')
    print("Saved: results_dlinear.csv")

    # SVM
    print("\n[3] Training SVM...")
    model_svm, type_svm = train_svm_model(X_train_scaled, y_train, X_test_scaled, y_test)
    pred_svm = predict_model(model_svm, X_test_scaled, type_svm)
    mse, rmse, mae, r2 = evaluate_and_print(pred_svm, y_test)

    result_df = test_df.copy()
    result_df['PredictResult'] = pred_svm
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv('results_svm.csv', index=False, encoding='utf-8-sig')
    print("Saved: results_svm.csv")

    # ===== 训练LTSF-Linear模型 =====
    print("\n" + "=" * 60)
    print("Training LTSF-Linear Models")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 准备时序数据 - LTSF模型使用标准单变量时序格式
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y_test['Measure'].values.reshape(-1, 1)).flatten()
    # 只使用目标变量作为时序输入
    data_x = target_scaled.reshape(-1, 1).astype(np.float32)  # [samples, 1]
    data_y = target_scaled.reshape(-1, 1).astype(np.float32)  # [samples, 1]

    seq_len = 24  # 较小的序列长度，适合小数据集
    pred_len = 1

    # 创建数据集
    dataset = LTSFDataset(data_x, data_y, seq_len, seq_len // 2, pred_len)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ltsf_results = []

    # 创建LTSF配置
    ltsf_config = SimpleConfig()
    ltsf_config.seq_len = seq_len
    ltsf_config.pred_len = pred_len
    ltsf_config.label_len = seq_len // 2

    # DLinear_v1
    print("\n[4] Training DLinear_v1 (LTSF-Linear)...")
    model_dlin_v1 = DLinear_LTSF(ltsf_config)
    model_dlin_v1 = train_ltsf_model(model_dlin_v1, train_loader, test_loader, device, max_epochs=30)
    pred_dlin_v1 = predict_ltsf(model_dlin_v1, data_x, seq_len, pred_len, device)
    # 对齐长度
    pred_dlin_v1_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred_dlin_v1)), pred_dlin_v1)
    mse, rmse, mae, r2 = evaluate_and_print(pred_dlin_v1_aligned, y_test)
    ltsf_results.append(('DLinear_v1', mse, rmse, mae, r2))

    # NLinear
    print("\n[5] Training NLinear (LTSF-Linear)...")
    model_nlin = NLinear_LTSF(ltsf_config)
    model_nlin = train_ltsf_model(model_nlin, train_loader, test_loader, device, max_epochs=30)
    pred_nlin = predict_ltsf(model_nlin, data_x, seq_len, pred_len, device)
    pred_nlin_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred_nlin)), pred_nlin)
    mse, rmse, mae, r2 = evaluate_and_print(pred_nlin_aligned, y_test)
    ltsf_results.append(('NLinear', mse, rmse, mae, r2))

    # Linear
    print("\n[6] Training Linear (LTSF-Linear)...")
    model_lin = Linear_LTSF(ltsf_config)
    model_lin = train_ltsf_model(model_lin, train_loader, test_loader, device, max_epochs=30)
    pred_lin = predict_ltsf(model_lin, data_x, seq_len, pred_len, device)
    pred_lin_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred_lin)), pred_lin)
    mse, rmse, mae, r2 = evaluate_and_print(pred_lin_aligned, y_test)
    ltsf_results.append(('Linear', mse, rmse, mae, r2))

    # 保存v1结果
    result_df = test_df.copy()
    result_df['PredictResult'] = pred_dlin_v1_aligned
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv('results_dlinear_v1.csv', index=False, encoding='utf-8-sig')
    print("Saved: results_dlinear_v1.csv")

    print("\nNote: Transformer_v1 requires different architecture setup, skipping...")

    # 保存LTSF结果
    result_df = test_df.copy()
    result_df['PredictResult'] = pred_nlin_aligned
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv('results_nlinear.csv', index=False, encoding='utf-8-sig')
    print("Saved: results_nlinear.csv")

    result_df = test_df.copy()
    result_df['PredictResult'] = pred_lin_aligned
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv('results_linear.csv', index=False, encoding='utf-8-sig')
    print("Saved: results_linear.csv")

    print("\n" + "=" * 60)
    print("All Results Summary")
    print("=" * 60)
    print("Original Models:")
    print("  - results_transformer.csv")
    print("  - results_dlinear.csv")
    print("  - results_svm.csv")
    print("\nLTSF-Linear Models:")
    print("  - results_dlinear_v1.csv")
    print("  - results_transformer_v1.csv")
    print("  - results_nlinear.csv")
    print("  - results_linear.csv")


if __name__ == "__main__":
    main()
