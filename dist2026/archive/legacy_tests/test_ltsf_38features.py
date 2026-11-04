"""
LTSF-Linear Models with 38 Engineering Features - Complete Training Script
为LTSF-Linear模型使用38个工程特征进行训练，对比原始模型的性能
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
import copy
import warnings
warnings.filterwarnings('ignore')

# 添加LTSF-Linear路径
ltsf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LTSF-Linear')
sys.path.insert(0, ltsf_path)

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Transformer_EncDec import Decoder as TransformerDecoder, DecoderLayer as TransformerDecoderLayer, Encoder as TransformerEncoder, EncoderLayer as TransformerEncoderLayer, ConvLayer


# ============================================================
# 数据准备（38个工程特征）
# ============================================================

def absolute_humidity(T, RH):
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH


def prepare_features(df):
    df = df.copy()
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['IMEI', 'Timestamp'], ascending=[True, True]).reset_index(drop=True)

    df['Hour_group'] = (df['Timestamp'].dt.hour // 3).astype(int)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Hour_group_sin'] = np.sin(2 * np.pi * df['Hour_group'] / 8)
    df['Hour_group_cos'] = np.cos(2 * np.pi * df['Hour_group'] / 8)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['AH'] = df.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)
    df['Measure_diff'] = df.groupby('IMEI')['Measure'].diff().fillna(0)
    df['Temp_diff'] = df.groupby('IMEI')['Temp'].diff().fillna(0)
    df['Hum_diff'] = df.groupby('IMEI')['Hum'].diff().fillna(0)

    def calc_rolling_ma(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    df['Measure_MA'] = df.groupby('IMEI')['Measure'].transform(lambda x: calc_rolling_ma(x, window=5))
    df['Measure_lag1'] = df.groupby('IMEI')['Measure'].shift(1).fillna(0)

    df['Temp_Hum_interact'] = df['Temp'] * df['Hum']
    df['Temp_AH_interact'] = df['Temp'] * df['AH']
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


# ============================================================
# 改进的LTSF-Linear模型（支持38个特征）
# ============================================================

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        pad = (self.kernel_size - 1) // 2
        if pad > 0:
            front = x[:, 0:1, :].repeat(1, pad, 1)
            end = x[:, -1:, :].repeat(1, pad, 1)
            x = torch.cat([front, x, end], dim=1)
        
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        # 确保输出形状与输入一致
        if x.shape[1] != orig_shape[1] if len(orig_shape) == 3 else x.shape[1] != orig_shape[0]:
            x = x[:, :orig_shape[1], :] if len(orig_shape) == 3 else x.squeeze(-1)
        
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 确保输入是3D
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, seq_len, n_features]
        
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ============ DLinear with 38 features ============
class DLinear_38Features(nn.Module):
    """
    DLinear adapted for 38 features (non-sequential format)
    Uses series decomposition + dual linear layers for each feature
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, kernel_size=25, individual=False):
        super().__init__()
        self.n_features = n_features  # 38
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        
        kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        
        if self.individual:
            # 每个特征单独一个线性层
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(n_features)
            ])
        else:
            # 共享线性层
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), self.n_features, self.pred_len], 
                                           dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), self.n_features, self.pred_len],
                                       dtype=trend_init.dtype, device=trend_init.device)
            for i in range(self.n_features):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        # 只取第一个特征的预测（因为我们只预测Measure）
        # 需要重新设计：直接用38特征预测单输出
        return seasonal_output + trend_output


class DLinear_MultiFeature(nn.Module):
    """
    DLinear for multi-feature to single output prediction
    适合从38个特征（带时间窗口）预测单个Measure值
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, hidden_dim=64, kernel_size=25):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 序列分解
        self.decomposition = series_decomp(kernel_size)
        
        # 季节性分支
        self.seasonal_branch = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 趋势性分支
        self.trend_branch = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 融合输出层
        self.output_layer = nn.Sequential(
            nn.Linear(n_features * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        seasonal, trend = self.decomposition(x)
        
        # 季节性处理
        s_perm = seasonal.permute(0, 2, 1)  # [batch, n_features, seq_len]
        s_out = self.seasonal_branch(s_perm)  # [batch, n_features, hidden_dim//2]
        
        # 趋势性处理
        t_perm = trend.permute(0, 2, 1)
        t_out = self.trend_branch(t_perm)  # [batch, n_features, hidden_dim//2]
        
        # 拼接季节性和趋势性
        combined = torch.cat([s_out, t_out], dim=-1)  # [batch, n_features, hidden_dim]
        
        # 展平并输出
        combined = combined.view(combined.size(0), -1)  # [batch, n_features * hidden_dim]
        output = self.output_layer(combined)  # [batch, pred_len]
        
        return output.squeeze(-1)


# ============ NLinear with 38 features ============
class NLinear_38Features(nn.Module):
    """
    NLinear for multi-feature prediction
    Subtracts the last value to handle distribution shift
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, hidden_dim=128):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.linear = nn.Linear(seq_len, pred_len)
        self.layer_norm = nn.LayerNorm(n_features)
        
        # 特征处理
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pred_len)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        seq_last = x[:, -1:, :].detach()  # 最后一个时间步
        x_norm = x - seq_last  # 归一化处理
        
        # 线性变换
        x_perm = x_norm.permute(0, 2, 1)  # [batch, n_features, seq_len]
        linear_out = self.linear(x_perm)  # [batch, n_features, pred_len]
        
        # 特征处理
        combined = linear_out.permute(0, 2, 1) + seq_last.permute(0, 2, 1)  # 加回原始值
        feat_out = self.feature_proj(combined.squeeze(1))  # [batch, hidden_dim]
        output = self.output_proj(feat_out)  # [batch, pred_len]
        
        return output.squeeze(-1)


class NLinear_MultiFeature(nn.Module):
    """
    NLinear with deeper architecture for multi-feature input
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, hidden_dim=128):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 序列级别投影
        self.seq_proj = nn.Linear(seq_len, hidden_dim)
        
        # 特征级别投影
        self.feat_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        seq_last = x[:, -1:, :].detach()
        x_centered = x - seq_last
        
        # 序列投影
        seq_out = self.seq_proj(x_centered.permute(0, 2, 1))  # [batch, n_features, hidden_dim]
        seq_out = seq_out.permute(0, 2, 1)  # [batch, hidden_dim, n_features]
        
        # 特征聚合 - 对特征维度取平均
        feat_out = seq_out.mean(dim=-1)  # [batch, hidden_dim]
        
        # 同时使用最后一个时间步的特征
        feat_last = self.feat_proj(x[:, -1, :])  # [batch, hidden_dim]
        
        # 合并两种特征
        combined = feat_out + feat_last
        
        # 输出
        output = self.output_layer(combined)  # [batch, pred_len]
        
        return output.squeeze(-1)


# ============ Linear with 38 features ============
class Linear_38Features(nn.Module):
    """
    Simple Linear for multi-feature prediction
    Single linear layer with optional depth
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, hidden_dim=128):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 简单线性层
        self.linear = nn.Linear(seq_len * n_features, pred_len)
        
        # 可选的深度扩展
        self.proj = nn.Sequential(
            nn.Linear(n_features * seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        x_flat = x.reshape(x.size(0), -1)  # [batch, seq_len * n_features]
        output = self.proj(x_flat)  # [batch, pred_len]
        return output.squeeze(-1)


class Linear_MultiFeature(nn.Module):
    """
    Linear with better architecture for multi-feature
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, hidden_dim=128):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 时间序列投影
        self.time_proj = nn.Linear(seq_len, hidden_dim)
        
        # 特征处理
        self.feat_layer = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 交叉注意力式交互
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: [batch, seq_len, n_features]
        time_out = self.time_proj(x.permute(0, 2, 1))  # [batch, n_features, hidden_dim]
        time_agg = time_out.mean(dim=1)  # [batch, hidden_dim]
        
        feat_out = self.feat_layer(x[:, -1, :])  # [batch, hidden_dim] 使用最后一个时间步
        
        combined = torch.cat([time_agg, feat_out], dim=-1)
        output = self.interaction(combined)
        
        return output.squeeze(-1)


# ============ Transformer with 38 features ============
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
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbeddingCustom(nn.Module):
    def __init__(self, c_in, d_model, d_in_feat, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_in=d_in_feat, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class Transformer_38Features(nn.Module):
    """
    Transformer for multi-feature time series
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, d_model=64, n_heads=2, 
                 e_layers=2, d_layers=1, d_ff=128, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        
        # embedding
        self.enc_embedding = DataEmbeddingCustom(n_features, d_model, 4, dropout)
        
        # encoder
        self.encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # decoder
        self.dec_embedding = DataEmbeddingCustom(n_features, d_model, 4, dropout)
        self.decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
                    AttentionLayer(
                        FullAttention(True, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu',
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [batch, seq_len, n_features]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        
        if x_dec is None:
            dec_inp = torch.zeros_like(x_enc[:, -self.pred_len:, :])
            x_dec = dec_inp
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_enc.size(0), self.pred_len, 4).to(x_enc.device)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        
        return dec_out[:, -self.pred_len:, 0].squeeze(-1)


# ============ Informer with 38 features ============
class Informer_38Features(nn.Module):
    """
    Informer for multi-feature time series
    Uses ProbAttention for efficiency
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, d_model=64, n_heads=2,
                 e_layers=2, d_layers=1, d_ff=128, dropout=0.1, distil=True):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        
        self.enc_embedding = DataEmbeddingCustom(n_features, d_model, 4, dropout)
        self.dec_embedding = DataEmbeddingCustom(n_features, d_model, 4, dropout)
        
        self.encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu'
                ) for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation='gelu',
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        
        if x_dec is None:
            dec_inp = torch.zeros_like(x_enc[:, -self.pred_len:, :])
            x_dec = dec_inp
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_enc.size(0), self.pred_len, 4).to(x_enc.device)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        
        return dec_out[:, -self.pred_len:, 0].squeeze(-1)


# ============ Autoformer with 38 features ============
class Autoformer_38Features(nn.Module):
    """
    Autoformer for multi-feature time series
    Uses series decomposition and auto-correlation mechanism
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, label_len=12,
                 d_model=64, n_heads=2, e_layers=1, d_layers=1, d_ff=128,
                 dropout=0.1, moving_avg=25):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.decomp = series_decomp(moving_avg)
        
        self.enc_embedding = DataEmbedding_wo_pos(n_features, d_model, 'timeF', 'h', dropout)
        self.dec_embedding = DataEmbedding_wo_pos(n_features, d_model, 'timeF', 'h', dropout)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, d_ff,
                    moving_avg=moving_avg, dropout=dropout, activation='gelu'
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, 1, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, 1, d_ff,
                    moving_avg=moving_avg, dropout=dropout, activation='gelu',
                ) for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device) if x_dec is not None else \
                torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.size(0), x_enc.size(1), 4).to(x_enc.device)
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_enc.size(0), self.label_len + self.pred_len, 4).to(x_enc.device)
        if x_dec is None:
            x_dec = torch.zeros(x_enc.size(0), self.label_len + self.pred_len, x_enc.size(2)).to(x_enc.device)
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend_init)
        dec_out = trend_part + seasonal_part
        
        return dec_out[:, -self.pred_len:, 0].squeeze(-1)


# ============================================================
# 时序数据集
# ============================================================

class TimeSeriesDataset(Dataset):
    """
    数据集假设:
    - data_x: [total_samples, seq_len, n_features] - 已经是构造好的时序数据
    - data_y: [total_samples] - 对应的目标值
    """
    def __init__(self, data_x, data_y):
        # data_x: [N, seq_len, n_features]
        # data_y: [N] 或 [N,]
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.FloatTensor(data_y).unsqueeze(-1) if data_y.ndim == 1 else torch.FloatTensor(data_y)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]


# ============================================================
# 训练和评估函数
# ============================================================

def train_model(model, train_loader, valid_loader, device, max_epochs=30, lr=0.001, model_name="Model"):
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(valid_loader), 1)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1}/{max_epochs}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def predict_model(model, data_x, seq_len, device):
    model.eval()
    preds = []
    with torch.no_grad():
        # data_x is already [N, seq_len, n_features]
        for i in range(len(data_x)):
            seq_x = data_x[i:i + 1]  # Keep batch dimension
            seq_x = torch.FloatTensor(seq_x).to(device)
            output = model(seq_x)
            pred = output.cpu().numpy()[0]
            preds.append(pred)
    return np.array(preds)


def save_results(test_df, predictions, filename):
    result_df = test_df.copy()
    result_df['PredictResult'] = predictions
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')


def evaluate_and_print(predictions, y_test):
    y_true = y_test['Measure'].values
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100
    print(f"    MSE:  {mse:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    R2:   {r2:.6f}")
    print(f"    MAPE: {mape:.2f}%")
    return mse, rmse, mae, r2, mape


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("LTSF-Linear Models with 38 Engineering Features")
    print("=" * 70)

    # 加载数据
    print("\n[1] Loading data...")
    df = pd.read_csv('combined_data.csv')
    print(f"    Original: {len(df)} rows")
    df = df[df['Hum'] < 80]
    df = df[df['Measure'] >= 0]
    print(f"    After filtering: {len(df)} rows")

    # 准备特征
    print("\n[2] Preparing 38 engineering features...")
    df_prepared = prepare_features(df)
    feature_cols = get_feature_columns()
    n_features = len(feature_cols)
    print(f"    Number of features: {n_features}")
    print(f"    Features: {feature_cols[:5]}... (total {n_features})")

    X = df_prepared[feature_cols].fillna(0)
    y = df_prepared[['Measure']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")
    
    # 超参数
    seq_len = 24
    pred_len = 1
    hidden_dim = 128
    batch_size = 32
    max_epochs = 50

    # 按IMEI分组构造时序数据
    print("\n[3.5] Constructing time series data by IMEI...")
    imei_groups = df_prepared.groupby('IMEI')
    
    all_sequences = []
    
    for imei, group in imei_groups:
        group = group.sort_values('Timestamp').reset_index(drop=True)
        feats = group[feature_cols].fillna(0).values
        target = group['Measure'].values
        
        # 构造时序样本
        for i in range(len(group) - seq_len - pred_len + 1):
            seq_x = feats[i:i+seq_len]
            y_target = target[i+seq_len]
            all_sequences.append({
                'x': seq_x,
                'y': y_target,
                'original_idx': i + seq_len  # 原始数据中的索引
            })
    
    print(f"    Total time series samples: {len(all_sequences)}")
    
    # 使用原始索引的80%分位数划分训练集和测试集
    split_threshold = int(len(all_sequences) * 0.8)
    
    X_train_3d_list = [s['x'] for s in all_sequences[:split_threshold]]
    y_train_3d_list = [s['y'] for s in all_sequences[:split_threshold]]
    X_test_3d_list = [s['x'] for s in all_sequences[split_threshold:]]
    y_test_3d_list = [s['y'] for s in all_sequences[split_threshold:]]
    
    X_train_3d = np.array(X_train_3d_list, dtype=np.float32)
    X_test_3d = np.array(X_test_3d_list, dtype=np.float32)
    y_train_arr = np.array(y_train_3d_list, dtype=np.float32)
    y_test_arr = np.array(y_test_3d_list, dtype=np.float32)
    
    print(f"    Train time series samples: {len(X_train_3d)}")
    print(f"    Test time series samples: {len(X_test_3d)}")
    print(f"    X_train_3d shape: {X_train_3d.shape}")
    print(f"    X_test_3d shape: {X_test_3d.shape}")
    
    # 标准化
    scaler = StandardScaler()
    X_train_3d_flat = X_train_3d.reshape(-1, n_features)
    X_test_3d_flat = X_test_3d.reshape(-1, n_features)
    
    X_train_3d_scaled_flat = scaler.fit_transform(X_train_3d_flat)
    X_test_3d_scaled_flat = scaler.transform(X_test_3d_flat)
    
    X_train_3d = X_train_3d_scaled_flat.reshape(X_train_3d.shape).astype(np.float32)
    X_test_3d = X_test_3d_scaled_flat.reshape(X_test_3d.shape).astype(np.float32)

    # 创建用于评估的测试目标值
    y_test_for_eval = pd.DataFrame({'Measure': y_test_arr})
    test_df = pd.DataFrame({'Measure': y_test_arr})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[4] Using device: {device}")

    results_summary = []

    # ============================================================
    # DLinear with 38 features
    # ============================================================
    print("\n" + "=" * 70)
    print("[5] Training DLinear_38Features...")
    print("=" * 70)
    
    train_dataset = TimeSeriesDataset(X_train_3d, y_train_arr)
    test_dataset = TimeSeriesDataset(X_test_3d, y_test_arr)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = DLinear_MultiFeature(n_features, seq_len, pred_len, hidden_dim=hidden_dim)
    model = train_model(model, train_loader, test_loader, device, max_epochs=max_epochs, model_name="DLinear")
    pred = predict_model(model, X_test_3d, seq_len, device)
    mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_for_eval)
    results_summary.append(('DLinear_38feat', mse, rmse, mae, r2, mape))
    save_results(test_df, pred, 'results_ltsf_dlinear_v1.csv')
    print("    Saved: results_ltsf_dlinear_v1.csv")

    # ============================================================
    # NLinear with 38 features
    # ============================================================
    print("\n" + "=" * 70)
    print("[6] Training NLinear_38Features...")
    print("=" * 70)
    
    model = NLinear_MultiFeature(n_features, seq_len, pred_len, hidden_dim=hidden_dim)
    model = train_model(model, train_loader, test_loader, device, max_epochs=max_epochs, model_name="NLinear")
    pred = predict_model(model, X_test_3d, seq_len, device)
    mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_for_eval)
    results_summary.append(('NLinear_38feat', mse, rmse, mae, r2, mape))
    save_results(test_df, pred, 'results_ltsf_nlinear_v1.csv')
    print("    Saved: results_ltsf_nlinear_v1.csv")

    # ============================================================
    # Linear with 38 features
    # ============================================================
    print("\n" + "=" * 70)
    print("[7] Training Linear_38Features...")
    print("=" * 70)
    
    model = Linear_MultiFeature(n_features, seq_len, pred_len, hidden_dim=hidden_dim)
    model = train_model(model, train_loader, test_loader, device, max_epochs=max_epochs, model_name="Linear")
    pred = predict_model(model, X_test_3d, seq_len, device)
    mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_for_eval)
    results_summary.append(('Linear_38feat', mse, rmse, mae, r2, mape))
    save_results(test_df, pred, 'results_ltsf_linear_v1.csv')
    print("    Saved: results_ltsf_linear_v1.csv")

    # ============================================================
    # Transformer with 38 features
    # ============================================================
    print("\n" + "=" * 70)
    print("[8] Training Transformer_38Features...")
    print("=" * 70)
    
    model = Transformer_38Features(n_features, seq_len, pred_len, 
                                   d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=128)
    model = train_model(model, train_loader, test_loader, device, max_epochs=40, lr=0.001, model_name="Transformer")
    pred = predict_model(model, X_test_3d, seq_len, device)
    mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_for_eval)
    results_summary.append(('Transformer_38feat', mse, rmse, mae, r2, mape))
    save_results(test_df, pred, 'results_ltsf_transformer_v1.csv')
    print("    Saved: results_ltsf_transformer_v1.csv")

    # ============================================================
    # Informer with 38 features
    # ============================================================
    print("\n" + "=" * 70)
    print("[9] Training Informer_38Features...")
    print("=" * 70)
    
    try:
        model = Informer_38Features(n_features, seq_len, pred_len,
                                    d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=128)
        model = train_model(model, train_loader, test_loader, device, max_epochs=30, lr=0.001, model_name="Informer")
        pred = predict_model(model, X_test_3d, seq_len, device)
        mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_for_eval)
        results_summary.append(('Informer_38feat', mse, rmse, mae, r2, mape))
        save_results(test_df, pred, 'results_ltsf_informer_v1.csv')
        print("    Saved: results_ltsf_informer_v1.csv")
    except Exception as e:
        print(f"    Informer failed: {str(e)[:100]}")
        results_summary.append(('Informer_38feat', None, None, None, None, None))

    # ============================================================
    # Autoformer with 38 features
    # ============================================================
    print("\n" + "=" * 70)
    print("[10] Training Autoformer_38Features...")
    print("=" * 70)
    
    try:
        model = Autoformer_38Features(n_features, seq_len, pred_len, label_len=12,
                                       d_model=64, n_heads=4, e_layers=1, d_layers=1, d_ff=128)
        model = train_model(model, train_loader, test_loader, device, max_epochs=30, lr=0.001, model_name="Autoformer")
        pred = predict_model(model, X_test_3d, seq_len, device)
        mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_for_eval)
        results_summary.append(('Autoformer_38feat', mse, rmse, mae, r2, mape))
        save_results(test_df, pred, 'results_ltsf_autoformer_v1.csv')
        print("    Saved: results_ltsf_autoformer_v1.csv")
    except Exception as e:
        print(f"    Autoformer failed: {str(e)[:100]}")
        results_summary.append(('Autoformer_38feat', None, None, None, None, None))

    # ============================================================
    # 结果汇总
    # ============================================================
    print("\n" + "=" * 70)
    print("Results Summary - LTSF-Linear with 38 Features")
    print("=" * 70)
    print(f"{'Model':<22} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'MAPE':<10}")
    print("-" * 80)
    for name, mse, rmse, mae, r2, mape in results_summary:
        if mse is not None:
            print(f"{name:<22} {mse:<12.6f} {rmse:<12.6f} {mae:<12.6f} {r2:<12.6f} {mape:<10.2f}%")
        else:
            print(f"{name:<22} {'Failed':<12}")

    print("\n" + "=" * 70)
    print("All result files saved!")
    print("=" * 70)
    print("Files:")
    print("  - results_ltsf_dlinear_v1.csv")
    print("  - results_ltsf_nlinear_v1.csv")
    print("  - results_ltsf_linear_v1.csv")
    print("  - results_ltsf_transformer_v1.csv")
    print("  - results_ltsf_informer_v1.csv")
    print("  - results_ltsf_autoformer_v1.csv")


if __name__ == "__main__":
    main()
