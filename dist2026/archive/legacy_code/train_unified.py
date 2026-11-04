"""
===========================================
统一模型训练脚本 - 绝缘子等值盐密预测
===========================================
整合所有模型算法，统一训练流程，输出对比结果

模型分类：
  1. 表格模型 (Tabular) - 单步预测
  2. LTSF-Linear模型 (时序) - 多步预测  
  3. FMLP-LTSF模型 - FMLP的时序版本
  4. 丰富特征模型 - 39/53维特征工程

统一配置：
  - 数据划分: 70%训练 / 15%验证 / 15%测试
  - 评估指标: MSE, RMSE, MAE, R², MAPE
  - 模型命名: 类别_子类型_特征数
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
LTSF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LTSF-Linear')
sys.path.insert(0, LTSF_PATH)

# 导入LTSF-Linear层
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Transformer_EncDec import Decoder as TransformerDecoder, DecoderLayer as TransformerDecoderLayer, Encoder as TransformerEncoder, EncoderLayer as TransformerEncoderLayer, ConvLayer

# 导入自定义模型
from model.fmlp import FMLP
from model.fmlp_ltsf import FMLP_LTSF

# ============================================================
# 统一配置
# ============================================================

class Config:
    """统一配置类"""
    # 数据配置
    DATA_PATH = 'combined_data.csv'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # 时序参数
    SEQ_LEN = 24
    PRED_LEN = 6
    LABEL_LEN = 12
    
    # 训练参数
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 15
    EARLY_STOP_PATIENCE = 15
    
    # 模型参数
    HIDDEN_DIM = 64
    D_MODEL = 64
    N_HEADS = 2
    E_LAYERS = 1
    D_LAYERS = 1
    D_FF = 128
    DROPOUT = 0.2
    KERNEL_SIZE = 7
    MOVING_AVG = 25
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 特征配置
    USE_RICH_FEATURES = True  # True: 53 features; False: 39 features
    USE_38_FEATURES = False   # Legacy compatibility flag
    
    # 数据过滤
    FILTER_HUM = False  # 是否过滤Hum<80
    FILTER_NEGATIVE = False  # 是否过滤负值


# ============================================================
# 特征工程 (统一版本)
# ============================================================

def absolute_humidity(T, RH):
    """相对湿度转换为绝对湿度"""
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH


def prepare_features(df, feature_type='rich'):
    """
    准备特征
    feature_type: 'basic' (39 features) / 'rich' (53 features)
    """
    df = df.copy()
    
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['IMEI', 'Timestamp'], ascending=[True, True]).reset_index(drop=True)
    
    # ===== 时间特征 =====
    df['Hour'] = df['Timestamp'].dt.hour
    df['Hour_group'] = (df['Timestamp'].dt.hour // 3).astype(int)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Hour_group_sin'] = np.sin(2 * np.pi * df['Hour_group'] / 8)
    df['Hour_group_cos'] = np.cos(2 * np.pi * df['Hour_group'] / 8)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # ===== 环境特征 =====
    df['AH'] = df.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)
    
    # ===== 时序特征 =====
    df['Measure_diff'] = df.groupby('IMEI')['Measure'].diff().fillna(0)
    df['Temp_diff'] = df.groupby('IMEI')['Temp'].diff().fillna(0)
    df['Hum_diff'] = df.groupby('IMEI')['Hum'].diff().fillna(0)
    
    def calc_rolling_ma(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    df['Measure_MA'] = df.groupby('IMEI')['Measure'].transform(lambda x: calc_rolling_ma(x, window=5))
    df['Measure_MA_10'] = df.groupby('IMEI')['Measure'].transform(lambda x: calc_rolling_ma(x, window=10))
    df['Measure_lag1'] = df.groupby('IMEI')['Measure'].shift(1).fillna(0)
    df['Measure_lag2'] = df.groupby('IMEI')['Measure'].shift(2).fillna(0)
    
    # ===== 湿度区域特征 =====
    df['Hum_0_25'] = np.where(df['Hum'] < 25, 1, 0)
    df['Hum_25_35'] = np.where((df['Hum'] >= 25) & (df['Hum'] < 35), 1, 0)
    df['Hum_35_45'] = np.where((df['Hum'] >= 35) & (df['Hum'] < 45), 1, 0)
    df['Hum_45_55'] = np.where((df['Hum'] >= 45) & (df['Hum'] < 55), 1, 0)
    df['Hum_55_65'] = np.where((df['Hum'] >= 55) & (df['Hum'] < 65), 1, 0)
    df['Hum_65_75'] = np.where((df['Hum'] >= 65) & (df['Hum'] < 75), 1, 0)
    df['Hum_75_85'] = np.where((df['Hum'] >= 75) & (df['Hum'] < 85), 1, 0)
    df['Hum_85_100'] = np.where(df['Hum'] >= 85, 1, 0)
    df['Hum_high'] = np.where(df['Hum'] >= 65, 1, 0)
    df['Hum_very_high'] = np.where(df['Hum'] >= 80, 1, 0)
    
    # ===== 位置特征 =====
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
    
    # ===== 交互特征 =====
    df['Temp_Hum'] = df['Temp'] * df['Hum']
    df['Temp_AH'] = df['Temp'] * df['AH']
    df['HumTemp_weight'] = df['Hum'] / (df['Temp'] + 5)
    df['HumTemp_weight_high'] = np.where(df['HumTemp_weight'] > 8, 1, 0)
    df['Hum_high_x_Temp'] = df['Hum_high'] * df['Temp']
    df['Hum_high_x_AH'] = df['Hum_high'] * df['AH']
    df['Hum_high_x_MA'] = df['Hum_high'] * df['Measure_MA']
    df['Hum_high_x_lag1'] = df['Hum_high'] * df['Measure_lag1']
    df['HumTemp_x_diff'] = df['HumTemp_weight_high'] * df['Measure_diff'].abs()
    df['HumTemp_x_MA'] = df['HumTemp_weight_high'] * df['Measure_MA']
    df['HumTemp_x_lag1'] = df['HumTemp_weight_high'] * df['Measure_lag1']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    return df


def get_feature_columns(feature_type='rich'):
    """获取特征列名"""
    if feature_type == 'rich':
        return [
            # Target history (1)
            'Measure',
            # 时间特征 (12个)
            'Hour', 'Hour_group', 'DayOfWeek', 'Month',
            'Hour_sin', 'Hour_cos', 'Hour_group_sin', 'Hour_group_cos',
            'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos',
            # 环境特征 (3个)
            'Temp', 'Hum', 'AH',
            # 时序特征 (7个)
            'Measure_diff', 'Temp_diff', 'Hum_diff',
            'Measure_MA', 'Measure_MA_10', 'Measure_lag1', 'Measure_lag2',
            # 湿度区域 (10个)
            'Hum_0_25', 'Hum_25_35', 'Hum_35_45', 'Hum_45_55', 'Hum_55_65',
            'Hum_65_75', 'Hum_75_85', 'Hum_85_100', 'Hum_high', 'Hum_very_high',
            # 位置特征 (9个)
            'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
            'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount',
            # 交互特征 (11个)
            'Temp_Hum', 'Temp_AH', 'HumTemp_weight', 'HumTemp_weight_high',
            'Hum_high_x_Temp', 'Hum_high_x_AH', 'Hum_high_x_MA', 'Hum_high_x_lag1',
            'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1',
        ]  # 53 features
    else:  # basic 39 features
        return [
            'Measure',
            'Hour_group', 'DayOfWeek', 'Month',
            'Hour_group_sin', 'Hour_group_cos',
            'DayOfWeek_sin', 'DayOfWeek_cos',
            'Month_sin', 'Month_cos',
            'Temp', 'Hum', 'AH',
            'Measure_diff', 'Temp_diff', 'Hum_diff', 'Measure_MA', 'Measure_lag1',
            'Temp_Hum', 'Temp_AH',
            'Hum_0_25', 'Hum_25_35', 'Hum_35_45', 'Hum_45_55', 'Hum_55_65',
            'Hum_65_75', 'Hum_75_85', 'Hum_85_100',
            'HumTemp_weight', 'HumTemp_weight_high',
            'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1',
            'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
            'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount'
        ]  # 39 features


# ============================================================
# 数据处理工具
# ============================================================

def create_sequences_by_device(df, feature_cols, seq_len, pred_len):
    """按设备分组创建滑动窗口序列"""
    hum_idx = feature_cols.index('Hum')
    X_list, y_list, w_list, idx_list = [], [], [], []
    
    for imei, group in df.groupby('IMEI', sort=False):
        group = group.sort_values('Timestamp')
        data = group[feature_cols + ['Measure']].values
        original_indices = group.index.to_numpy()
        n = len(data)
        
        for i in range(n - seq_len - pred_len + 1):
            X = data[i:i+seq_len, :-1]
            y = data[i+seq_len:i+seq_len+pred_len, -1]
            
            avg_hum = np.mean(data[i:i+seq_len, hum_idx])
            if avg_hum >= 85: w = 3.0
            elif avg_hum >= 75: w = 2.5
            elif avg_hum >= 65: w = 2.0
            elif avg_hum >= 55: w = 1.5
            else: w = 1.0
            
            X_list.append(X)
            y_list.append(y)
            w_list.append(w)
            idx_list.append(original_indices[i + seq_len + pred_len - 1])
    
    return np.array(X_list), np.array(y_list), np.array(w_list), np.array(idx_list)


def split_by_device_time(df, train_ratio, val_ratio, seq_len):
    """Split every device chronologically while retaining lookback context.

    Validation and test partitions include the preceding ``seq_len`` rows as
    input context. Their forecast targets still start strictly after the
    training and validation boundaries respectively.
    """
    train_parts, val_parts, test_parts = [], [], []

    for _, group in df.groupby('IMEI', sort=False):
        group = group.sort_values('Timestamp')
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[max(0, train_end - seq_len):val_end])
        test_parts.append(group.iloc[max(0, val_end - seq_len):])

    return (
        pd.concat(train_parts).copy(),
        pd.concat(val_parts).copy(),
        pd.concat(test_parts).copy(),
    )


def create_sequences_simple(df, feature_cols, seq_len, pred_len):
    """简单滑动窗口(不按设备分组)"""
    hum_idx = feature_cols.index('Hum')
    X_list, y_list, w_list, idx_list = [], [], [], []
    
    data = df[feature_cols + ['Measure']].values
    n = len(data)
    
    for i in range(n - seq_len - pred_len + 1):
        X = data[i:i+seq_len, :-1]
        y = data[i+seq_len:i+seq_len+pred_len, -1]
        
        avg_hum = np.mean(data[i:i+seq_len, hum_idx])
        if avg_hum >= 85: w = 3.0
        elif avg_hum >= 75: w = 2.5
        elif avg_hum >= 65: w = 2.0
        elif avg_hum >= 55: w = 1.5
        else: w = 1.0
        
        X_list.append(X)
        y_list.append(y)
        w_list.append(w)
        idx_list.append(i + seq_len + pred_len // 2)
    
    return np.array(X_list), np.array(y_list), np.array(w_list), np.array(idx_list)


class TimeSeriesDataset(Dataset):
    """时序数据集"""
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
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))
        
        return (torch.FloatTensor(seq_x), torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark))


# ============================================================
# 模型定义
# ============================================================

# ----- LTSF-Linear 基线模型 -----

class DLinear_LTSF(nn.Module):
    """DLinear - 序列分解+双线性层"""
    def __init__(self, seq_len, pred_len, enc_in=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class NLinear_LTSF(nn.Module):
    """NLinear - 归一化线性层"""
    def __init__(self, seq_len, pred_len, enc_in=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x


class Linear_LTSF(nn.Module):
    """Linear - 简单线性层"""
    def __init__(self, seq_len, pred_len, enc_in=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)


# ----- 位置编码 -----

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


# ----- Transformer系列 -----

class Transformer_LTSF(nn.Module):
    """Transformer - 编码器-解码器架构"""
    def __init__(self, seq_len, label_len, pred_len, enc_in=1, d_model=64, n_heads=2, 
                 e_layers=2, d_layers=1, d_ff=128, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.enc_embedding = DataEmbeddingCustom(enc_in, d_model, 4, dropout)
        self.dec_embedding = DataEmbeddingCustom(enc_in, d_model, 4, dropout)
        
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
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        
        if x_dec is None:
            x_dec = torch.zeros_like(x_enc[:, -self.pred_len:, :])
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_enc.size(0), self.pred_len, 4).to(x_enc.device)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        
        return dec_out[:, -self.pred_len:, 0].squeeze(-1)


class Informer_LTSF(nn.Module):
    """Informer - ProbAttention"""
    def __init__(self, seq_len, label_len, pred_len, enc_in=1, d_model=64, n_heads=2,
                 e_layers=2, d_layers=1, d_ff=128, dropout=0.1, distil=True):
        super().__init__()
        self.pred_len = pred_len
        self.enc_embedding = DataEmbeddingCustom(enc_in, d_model, 4, dropout)
        self.dec_embedding = DataEmbeddingCustom(enc_in, d_model, 4, dropout)
        
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
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)
        
        if x_dec is None:
            x_dec = torch.zeros_like(x_enc[:, -self.pred_len:, :])
        if x_mark_dec is None:
            x_mark_dec = torch.zeros(x_enc.size(0), self.pred_len, 4).to(x_enc.device)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out)
        
        return dec_out[:, -self.pred_len:, 0].squeeze(-1)


class Autoformer_LTSF(nn.Module):
    """Autoformer - 自相关机制"""
    def __init__(self, seq_len, label_len, pred_len, enc_in=1, d_model=64, n_heads=2,
                 e_layers=1, d_layers=1, d_ff=128, dropout=0.1, moving_avg=25):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.decomp = series_decomp(moving_avg)
        
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, 'timeF', 'h', dropout)
        self.dec_embedding = DataEmbedding_wo_pos(enc_in, d_model, 'timeF', 'h', dropout)
        
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
# 训练和评估函数
# ============================================================

def train_model(model, train_loader, valid_loader, device, config, model_type='default'):
    """
    统一训练函数
    model_type: 'tabular' - 表格模型 (2D输入)
                'ltsf' - LTSF模型 (3D输入，单变量)
                'fmlp_ltsf' - FMLP-LTSF (3D输入，多变量)
                'transformer' - Transformer系列 (3D输入)
    """
    model = model.to(device)
    
    if model_type == 'tabular':
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif model_type in ['fmlp_ltsf', 'ltsf']:
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    else:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_epochs = config.MAX_EPOCHS if model_type != 'tabular' else 200
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        
        if model_type == 'tabular':
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
        elif model_type in ['fmlp_ltsf', 'ltsf']:
            for batch_X, batch_y, batch_w in train_loader:
                batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze(-1)
                sample_loss = criterion(outputs, batch_y).mean(dim=1)
                loss = (sample_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
        else:  # transformer系列
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x_mark = batch_x_mark.to(device)
                batch_y_mark = batch_y_mark.to(device)
                
                optimizer.zero_grad()
                dec_inp = torch.zeros_like(batch_y[:, -config.PRED_LEN:, :]).float()
                dec_inp = torch.cat([batch_y[:, :config.LABEL_LEN, :], dec_inp], dim=1).float()
                
                if hasattr(model, 'enc_embedding'):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x)
                
                outputs = outputs[:, -config.PRED_LEN:, :]
                batch_y_target = batch_y[:, -config.PRED_LEN:, :].to(device)
                loss = criterion(outputs.squeeze(-1), batch_y_target.squeeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
        
        avg_train_loss = train_loss / max(len(train_loader), 1)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            if model_type == 'tabular':
                for batch_X, batch_y in valid_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            elif model_type in ['fmlp_ltsf', 'ltsf']:
                for batch_X, batch_y, batch_w in valid_loader:
                    batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
                    outputs = model(batch_X).squeeze(-1)
                    sample_loss = criterion(outputs, batch_y).mean(dim=1)
                    loss = (sample_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
                    val_loss += loss.item()
            else:
                for batch_x, batch_y, batch_x_mark, batch_y_mark in valid_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_x_mark = batch_x_mark.to(device)
                    batch_y_mark = batch_y_mark.to(device)
                    
                    dec_inp = torch.zeros_like(batch_y[:, -config.PRED_LEN:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :config.LABEL_LEN, :], dec_inp], dim=1).float()
                    
                    if hasattr(model, 'enc_embedding'):
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = model(batch_x)
                    
                    outputs = outputs[:, -config.PRED_LEN:, :]
                    batch_y_target = batch_y[:, -config.PRED_LEN:, :].to(device)
                    loss = criterion(outputs.squeeze(-1), batch_y_target.squeeze(-1))
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / max(len(valid_loader), 1)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch [{epoch+1:3d}/{max_epochs}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"      Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_loss


def evaluate_model(predictions, y_true, mode='multi_step'):
    """
    评估模型
    mode: 'single_step' - 单步预测
          'multi_step' - 多步预测(取均值)
    """
    if mode == 'multi_step':
        pred_mean = predictions.reshape(-1)
        y_mean = y_true.reshape(-1)
    else:
        pred_mean = predictions.flatten()
        y_mean = y_true.flatten() if y_true.ndim > 1 else y_true
    
    mse = mean_squared_error(y_mean, pred_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_mean, pred_mean)
    r2 = r2_score(y_mean, pred_mean)
    mape = np.mean(np.abs((y_mean - pred_mean) / (y_mean + 1e-8))) * 100
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def save_results(df_test, predictions, filename):
    """保存预测结果"""
    result_df = df_test.copy()
    result_df['Prediction'] = predictions[:, -1] if predictions.ndim > 1 else predictions
    result_df['Actual'] = result_df['Measure']
    result_df['Error'] = np.abs(result_df['Actual'] - result_df['Prediction'])
    result_df['Error_pct'] = result_df['Error'] / (result_df['Actual'] + 1e-8) * 100
    result_df = result_df.round(4)
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"      Saved: {filename}")
    return result_df


# ============================================================
# 模型注册表
# ============================================================

MODEL_REGISTRY = {
    # ===== 1. 表格模型 (Tabular) - 单步预测 =====
    'Tabular_FMLP': {
        'type': 'tabular',
        'model_class': FMLP,
        'params': {'input_dim': 53, 'hidden_dim': 128, 'dropout': 0.2, 'kernel_size': 7},
        'description': 'Feature-gated residual MLP',
        'input_type': '2D',
        'eval_mode': 'single_step',
    },
    
    # ===== 2. LTSF-Linear模型 - 多步预测 =====
    'LTSF_DLinear': {
        'type': 'ltsf',
        'model_class': DLinear_LTSF,
        'params': {'seq_len': 24, 'pred_len': 6, 'enc_in': 1},
        'description': 'DLinear (序列分解)',
        'input_type': '3D_single',
        'eval_mode': 'multi_step',
    },
    'LTSF_NLinear': {
        'type': 'ltsf',
        'model_class': NLinear_LTSF,
        'params': {'seq_len': 24, 'pred_len': 6, 'enc_in': 1},
        'description': 'NLinear (归一化线性)',
        'input_type': '3D_single',
        'eval_mode': 'multi_step',
    },
    'LTSF_Linear': {
        'type': 'ltsf',
        'model_class': Linear_LTSF,
        'params': {'seq_len': 24, 'pred_len': 6, 'enc_in': 1},
        'description': 'Linear (简单线性)',
        'input_type': '3D_single',
        'eval_mode': 'multi_step',
    },
    
    # ===== 3. FMLP-LTSF模型 - FMLP的时序版本 =====
    'FMLP_LTSF': {
        'type': 'fmlp_ltsf',
        'model_class': FMLP_LTSF,
        'params': {'seq_len': 24, 'pred_len': 6, 'enc_in': 53, 'hidden_dim': 64, 'dropout': 0.2, 'kernel_size': 7, 'individual': False, 'use_mlp': False, 'target_index': 0},
        'description': 'FMLP-LTSF with multivariate context',
        'input_type': '3D_multi',
        'eval_mode': 'multi_step',
    },
    'FMLP_LTSF_MLP': {
        'type': 'fmlp_ltsf',
        'model_class': FMLP_LTSF,
        'params': {'seq_len': 24, 'pred_len': 6, 'enc_in': 53, 'hidden_dim': 64, 'dropout': 0.2, 'kernel_size': 7, 'individual': False, 'use_mlp': True, 'target_index': 0},
        'description': 'FMLP-LTSF with nonlinear multivariate fusion',
        'input_type': '3D_multi',
        'eval_mode': 'multi_step',
    },
    
    # ===== 4. Transformer系列 =====
    'Transformer': {
        'type': 'transformer',
        'model_class': Transformer_LTSF,
        'params': {'seq_len': 24, 'label_len': 12, 'pred_len': 6, 'enc_in': 1, 'd_model': 64, 'n_heads': 2, 'e_layers': 2, 'd_layers': 1, 'd_ff': 128, 'dropout': 0.1},
        'description': 'Transformer (编码器-解码器)',
        'input_type': '3D_single',
        'eval_mode': 'multi_step',
    },
    'Informer': {
        'type': 'transformer',
        'model_class': Informer_LTSF,
        'params': {'seq_len': 24, 'label_len': 12, 'pred_len': 6, 'enc_in': 1, 'd_model': 64, 'n_heads': 2, 'e_layers': 2, 'd_layers': 1, 'd_ff': 128, 'dropout': 0.1, 'distil': True},
        'description': 'Informer (ProbAttention)',
        'input_type': '3D_single',
        'eval_mode': 'multi_step',
    },
    'Autoformer': {
        'type': 'transformer',
        'model_class': Autoformer_LTSF,
        'params': {'seq_len': 24, 'label_len': 12, 'pred_len': 6, 'enc_in': 1, 'd_model': 64, 'n_heads': 2, 'e_layers': 1, 'd_layers': 1, 'd_ff': 128, 'dropout': 0.1, 'moving_avg': 25},
        'description': 'Autoformer (自相关机制)',
        'input_type': '3D_single',
        'eval_mode': 'multi_step',
    },
}


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("统一模型训练脚本 - 绝缘子等值盐密预测")
    print("=" * 70)
    
    config = Config()
    
    # ===== 加载数据 =====
    print("\n[1] 加载数据...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"    原始数据: {len(df)} 行")
    
    # 数据过滤
    if config.FILTER_HUM:
        df = df[df['Hum'] < 80]
        print(f"    Hum<80过滤后: {len(df)} 行")
    if config.FILTER_NEGATIVE:
        df = df[df['Measure'] >= 0]
        print(f"    非负值过滤后: {len(df)} 行")
    
    # ===== 特征工程 =====
    print("\n[2] 特征工程...")
    feature_type = 'rich' if config.USE_RICH_FEATURES else 'basic'
    df = prepare_features(df, feature_type)
    feature_cols = get_feature_columns(feature_type)
    n_features = len(feature_cols)
    target_idx = feature_cols.index('Measure')
    print(f"    特征类型: {feature_type}")
    print(f"    特征数量: {n_features}")
    print(f"    目标历史通道: {target_idx} (Measure)")
    
    # ===== 划分数据集 =====
    print("\n[3] 划分数据集 (70/15/15)...")
    df_train, df_val, df_test = split_by_device_time(
        df,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.SEQ_LEN,
    )
    
    print(f"    训练集: {len(df_train)} 行")
    print(f"    验证集: {len(df_val)} 行 (含历史上下文)")
    print(f"    测试集: {len(df_test)} 行 (含历史上下文)")
    
    # ===== 创建序列数据 =====
    print("\n[4] 创建滑动窗口序列...")
    X_train, y_train, w_train, _ = create_sequences_by_device(df_train, feature_cols, config.SEQ_LEN, config.PRED_LEN)
    X_val, y_val, w_val, _ = create_sequences_by_device(df_val, feature_cols, config.SEQ_LEN, config.PRED_LEN)
    X_test, y_test, _, idx_test = create_sequences_by_device(df_test, feature_cols, config.SEQ_LEN, config.PRED_LEN)

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError(
            "At least one time split has no complete sequence. Reduce SEQ_LEN/PRED_LEN "
            "or provide more samples per device."
        )
    
    print(f"    训练序列: {X_train.shape}")
    print(f"    验证序列: {X_val.shape}")
    print(f"    测试序列: {X_test.shape}")
    
    # ===== 标准化 =====
    print("\n[5] 标准化...")
    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
    
    scaler_y = StandardScaler()
    y_train_flat = y_train.reshape(-1, config.PRED_LEN)
    scaler_y.fit(y_train_flat)
    y_train_scaled = scaler_y.transform(y_train_flat).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, config.PRED_LEN)).reshape(y_val.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, config.PRED_LEN)).reshape(y_test.shape)
    
    print(f"    设备: {config.DEVICE}")
    
    # ===== 训练模型 =====
    print("\n" + "=" * 70)
    print("[6] 训练模型")
    print("=" * 70)
    
    results_all = []
    timestamp = int(time.time())
    
    # 逐个训练模型
    for model_name, model_info in MODEL_REGISTRY.items():
        print(f"\n{'─'*60}")
        print(f"训练模型: {model_name}")
        print(f"描述: {model_info['description']}")
        print(f"{'─'*60}")
        
        try:
            model_type = model_info['type']
            
            # 创建模型
            if model_type == 'tabular':
                # 更新参数量
                params = model_info['params'].copy()
                params['input_dim'] = n_features
                model = model_info['model_class'](**params)
                
                # 数据加载器 (2D)
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_scaled[:, -1, :]),
                    torch.FloatTensor(y_train_scaled[:, -1])
                )
                valid_dataset = TensorDataset(
                    torch.FloatTensor(X_val_scaled[:, -1, :]),
                    torch.FloatTensor(y_val_scaled[:, -1])
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
                
                # 训练
                model, best_loss = train_model(model, train_loader, valid_loader, config.DEVICE, config, 'tabular')
                
                # 预测 (单步)
                model.eval()
                with torch.no_grad():
                    preds_scaled = model(torch.FloatTensor(X_test_scaled[:, -1, :])).numpy()
                
                # 评估
                preds_single = (
                    preds_scaled * scaler_y.scale_[-1] + scaler_y.mean_[-1]
                )
                metrics = evaluate_model(preds_single, y_test[:, -1], 'single_step')
                preds = np.tile(preds_single.reshape(-1, 1), (1, config.PRED_LEN))
                
            elif model_type == 'ltsf':
                # 单变量LTSF模型
                params = model_info['params'].copy()
                params['enc_in'] = 1
                model = model_info['model_class'](**params)
                
                # 数据加载器 (使用单变量Measure)
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_scaled[:, :, target_idx:target_idx + 1]),
                    torch.FloatTensor(y_train_scaled),
                    torch.FloatTensor(w_train)
                )
                valid_dataset = TensorDataset(
                    torch.FloatTensor(X_val_scaled[:, :, target_idx:target_idx + 1]),
                    torch.FloatTensor(y_val_scaled),
                    torch.FloatTensor(w_val)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
                
                # 训练
                model, best_loss = train_model(model, train_loader, valid_loader, config.DEVICE, config, 'ltsf')
                
                # 预测
                model.eval()
                with torch.no_grad():
                    preds_scaled = model(
                        torch.FloatTensor(X_test_scaled[:, :, target_idx:target_idx + 1])
                    ).cpu().numpy()
                
                preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, config.PRED_LEN)).reshape(preds_scaled.shape)
                metrics = evaluate_model(preds, y_test, 'multi_step')
                
            elif model_type == 'fmlp_ltsf':
                # FMLP-LTSF多变量模型
                params = model_info['params'].copy()
                params['enc_in'] = n_features
                params['target_index'] = target_idx
                params['seq_len'] = config.SEQ_LEN
                params['pred_len'] = config.PRED_LEN
                params['kernel_size'] = config.KERNEL_SIZE
                model = model_info['model_class'](**params)
                
                # 数据加载器
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_scaled),
                    torch.FloatTensor(y_train_scaled),
                    torch.FloatTensor(w_train)
                )
                valid_dataset = TensorDataset(
                    torch.FloatTensor(X_val_scaled),
                    torch.FloatTensor(y_val_scaled),
                    torch.FloatTensor(w_val)
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
                
                # 训练
                model, best_loss = train_model(model, train_loader, valid_loader, config.DEVICE, config, 'fmlp_ltsf')
                
                # 预测
                model.eval()
                with torch.no_grad():
                    preds_scaled = model(torch.FloatTensor(X_test_scaled)).cpu().numpy()
                
                preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, config.PRED_LEN)).reshape(preds_scaled.shape)
                metrics = evaluate_model(preds, y_test, 'multi_step')
                
            elif model_type == 'transformer':
                # Transformer系列
                params = model_info['params'].copy()
                params['enc_in'] = 1
                params['pred_len'] = config.PRED_LEN
                params['label_len'] = config.LABEL_LEN
                model = model_info['model_class'](**params)
                
                # 数据集
                train_dataset = TimeSeriesDataset(
                    X_train_scaled[:, :, target_idx:target_idx + 1].astype(np.float32),
                    y_train_scaled.astype(np.float32),
                    config.SEQ_LEN, config.LABEL_LEN, config.PRED_LEN
                )
                valid_dataset = TimeSeriesDataset(
                    X_val_scaled[:, :, target_idx:target_idx + 1].astype(np.float32),
                    y_val_scaled.astype(np.float32),
                    config.SEQ_LEN, config.LABEL_LEN, config.PRED_LEN
                )
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
                
                # 训练
                model, best_loss = train_model(model, train_loader, valid_loader, config.DEVICE, config, 'transformer')
                
                # 预测
                model.eval()
                preds = []
                with torch.no_grad():
                    for i in range(len(X_test_scaled) - config.SEQ_LEN - config.PRED_LEN + 1):
                        seq_x = X_test_scaled[
                            i:i + config.SEQ_LEN,
                            :,
                            target_idx:target_idx + 1,
                        ]
                        seq_x = torch.FloatTensor(seq_x).unsqueeze(0).to(config.DEVICE)
                        seq_x_mark = torch.zeros(1, config.SEQ_LEN, 4).to(config.DEVICE)
                        dec_inp = torch.zeros(1, config.PRED_LEN, 1).to(config.DEVICE)
                        dec_mark = torch.zeros(1, config.PRED_LEN, 4).to(config.DEVICE)
                        
                        output = model(seq_x, seq_x_mark, dec_inp, dec_mark)
                        pred = output.cpu().numpy()[0]
                        preds.append(pred)
                
                preds = np.array(preds)
                # 反标准化
                preds = scaler_y.inverse_transform(preds.reshape(-1, config.PRED_LEN)).reshape(preds.shape)
                # 对齐长度
                if len(preds) < len(y_test):
                    preds = np.pad(preds, ((0, len(y_test) - len(preds)), (0, 0)), mode='edge')
                metrics = evaluate_model(preds[:len(y_test)], y_test[:len(preds)], 'multi_step')
            
            # 记录结果
            result = {
                'Model': model_name,
                'Description': model_info['description'],
                'Type': model_info['type'],
                'Input_Type': model_info['input_type'],
                'Features': n_features,
                **metrics,
                'Best_Val_Loss': best_loss
            }
            results_all.append(result)
            
            print(f"\n    Results:")
            print(f"      MSE:  {metrics['MSE']:.6f}")
            print(f"      RMSE: {metrics['RMSE']:.6f}")
            print(f"      MAE:  {metrics['MAE']:.6f}")
            print(f"      R2:   {metrics['R2']:.6f}")
            print(f"      MAPE: {metrics['MAPE']:.2f}%")
            
            # 保存结果
            result_df = df.loc[idx_test[:len(preds)]].copy()
            save_results(result_df, preds, f'results_{model_name}_{timestamp}.csv')
            torch.save(model.state_dict(), f'{model_name}_model_{timestamp}.pth')
            print(f"      Model saved: {model_name}_model_{timestamp}.pth")
            
        except Exception as e:
            print(f"    Training failed: {str(e)[:100]}")
            results_all.append({
                'Model': model_name,
                'Description': model_info['description'],
                'Type': model_info['type'],
                'Input_Type': model_info['input_type'],
                'Features': n_features,
                'MSE': None, 'RMSE': None, 'MAE': None, 'R2': None, 'MAPE': None,
                'Best_Val_Loss': None
            })
    
    # ===== 结果汇总 =====
    print("\n" + "=" * 70)
    print("[7] 结果汇总")
    print("=" * 70)
    
    results_df = pd.DataFrame(results_all)
    results_df = results_df.sort_values('RMSE')
    
    print(f"\n{'Model':<25} {'Type':<12} {'Input':<8} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12} {'MAPE':<10}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        if row['MSE'] is not None:
            print(f"{row['Model']:<25} {row['Type']:<12} {row['Input_Type']:<8} "
                  f"{row['MSE']:<12.6f} {row['RMSE']:<12.6f} {row['MAE']:<12.6f} "
                  f"{row['R2']:<12.6f} {row['MAPE']:<10.2f}%")
        else:
            print(f"{row['Model']:<25} {row['Type']:<12} {row['Input_Type']:<8} {'Failed':<12}")
    
    # 保存汇总结果
    results_df.to_csv(f'unified_results_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"\nResults saved: unified_results_{timestamp}.csv")
    
    # 最佳模型
    best = results_df[results_df['MSE'].notna()].iloc[0]
    print(f"\nBest model (lowest RMSE): {best['Model']}")
    print(f"  RMSE: {best['RMSE']:.6f}")
    print(f"  R2:   {best['R2']:.6f}")
    print(f"  MAPE: {best['MAPE']:.2f}%")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
    return results_df


if __name__ == '__main__':
    results = main()
