"""
LTSF-Linear Models - Complete Training Script
训练所有模型: Autoformer, Informer, Transformer, DLinear, NLinear, Linear
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

# 导入LTSF-Linear层
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Transformer_EncDec import Decoder as TransformerDecoder, DecoderLayer as TransformerDecoderLayer, Encoder as TransformerEncoder, EncoderLayer as TransformerEncoderLayer, ConvLayer


# ============================================================
# 数据准备
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
# 模型定义
# ============================================================

class moving_avg(nn.Module):
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
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# DLinear
class DLinear_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


# NLinear
class NLinear_Model(nn.Module):
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


# Linear
class Linear_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)


# PositionalEmbedding
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

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# Transformer (from LTSF-Linear)
class Transformer_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.enc_embedding = DataEmbeddingCustom(configs.enc_in, configs.d_model, configs.d_in, configs.dropout)
        self.dec_embedding = DataEmbeddingCustom(configs.dec_in, configs.d_model, configs.d_in, configs.dropout)

        self.encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
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

        self.decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
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


# Informer (from LTSF-Linear)
class Informer_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.enc_embedding = DataEmbeddingCustom(configs.enc_in, configs.d_model, configs.d_in, configs.dropout)
        self.dec_embedding = DataEmbeddingCustom(configs.dec_in, configs.d_model, configs.d_in, configs.dropout)

        self.encoder = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.decoder = TransformerDecoder(
            [
                TransformerDecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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


# Autoformer (from LTSF-Linear)
class Autoformer_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]


# ============================================================
# 配置和数据集
# ============================================================

class ModelConfig:
    def __init__(self):
        self.seq_len = 24
        self.label_len = 12
        self.pred_len = 1
        self.d_model = 64
        self.n_heads = 2
        self.e_layers = 1
        self.d_layers = 1
        self.d_ff = 128
        self.dropout = 0.1
        self.activation = 'gelu'
        self.factor = 1
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.d_in = 4
        self.embed = 'timeF'
        self.freq = 'h'
        self.moving_avg = 25
        self.embed_type = 0
        self.output_attention = False
        self.distil = False


class TimeSeriesDataset(Dataset):
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

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), \
               torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark)


# ============================================================
# 训练和评估函数
# ============================================================

def train_model(model, train_loader, valid_loader, device, max_epochs=30, lr=0.001, seq_len=24, pred_len=1, label_len=12):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 7

    # 获取模型的label_len
    model_label_len = getattr(model, 'label_len', label_len)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            optimizer.zero_grad()
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :model_label_len, :], dec_inp], dim=1).float()

            if hasattr(model, 'enc_embedding'):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x)

            outputs = outputs[:, -pred_len:, :]
            batch_y_target = batch_y[:, -pred_len:, :].to(device)
            loss = criterion(outputs, batch_y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_x_mark = batch_x_mark.to(device)
                batch_y_mark = batch_y_mark.to(device)

                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :model_label_len, :], dec_inp], dim=1).float()

                if hasattr(model, 'enc_embedding'):
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = model(batch_x)

                outputs = outputs[:, -pred_len:, :]
                batch_y_target = batch_y[:, -pred_len:, :].to(device)
                loss = criterion(outputs, batch_y_target)
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


def predict_model(model, data_x, seq_len, pred_len, device, label_len=12):
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


def save_results(test_df, predictions, filename):
    result_df = test_df.copy()
    result_df['PredictResult'] = predictions
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')


def evaluate_and_print(predictions, y_test):
    mse = mean_squared_error(y_test['Measure'].values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test['Measure'].values, predictions)
    r2 = r2_score(y_test['Measure'].values, predictions)
    print(f"    MSE:  {mse:.6f}")
    print(f"    RMSE: {rmse:.6f}")
    print(f"    MAE:  {mae:.6f}")
    print(f"    R2:   {r2:.6f}")
    return mse, rmse, mae, r2


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("LTSF-Linear Models - Complete Training")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    df = pd.read_csv('combined_data.csv')
    print(f"    Original: {len(df)} rows")
    df = df[df['Hum'] < 80]
    df = df[df['Measure'] >= 0]
    print(f"    After filtering: {len(df)} rows")

    # 准备特征
    print("\n[2] Preparing features...")
    df_prepared = prepare_features(df)
    feature_cols = get_feature_columns()

    X = df_prepared[feature_cols].fillna(0)
    y = df_prepared[['Measure']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"    Training set: {len(X_train)} samples")
    print(f"    Test set: {len(X_test)} samples")

    test_df = df_prepared.iloc[X_test.index].copy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3] Using device: {device}")

    # 准备时序数据
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(y_test['Measure'].values.reshape(-1, 1)).flatten()
    data_x = target_scaled.reshape(-1, 1).astype(np.float32)
    data_y = target_scaled.reshape(-1, 1).astype(np.float32)

    seq_len = 24
    pred_len = 1
    label_len = 12

    dataset = TimeSeriesDataset(data_x, data_y, seq_len, label_len, pred_len)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建配置
    config = ModelConfig()
    config.seq_len = seq_len
    config.label_len = label_len
    config.pred_len = pred_len

    results_summary = []

    # ============================================================
    # 训练所有模型
    # ============================================================

    print("\n" + "=" * 60)
    print("Training Models")
    print("=" * 60)

    # DLinear
    print("\n[4] Training DLinear...")
    model = DLinear_Model(config)
    model = train_model(model, train_loader, test_loader, device, max_epochs=30)
    pred = predict_model(model, data_x, seq_len, pred_len, device)
    pred_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred)), pred)
    mse, rmse, mae, r2 = evaluate_and_print(pred_aligned, y_test)
    results_summary.append(('DLinear', mse, rmse, mae, r2))
    save_results(test_df, pred_aligned, 'results_ltsf_dlinear.csv')
    print("    Saved: results_ltsf_dlinear.csv")

    # NLinear
    print("\n[5] Training NLinear...")
    model = NLinear_Model(config)
    model = train_model(model, train_loader, test_loader, device, max_epochs=30)
    pred = predict_model(model, data_x, seq_len, pred_len, device)
    pred_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred)), pred)
    mse, rmse, mae, r2 = evaluate_and_print(pred_aligned, y_test)
    results_summary.append(('NLinear', mse, rmse, mae, r2))
    save_results(test_df, pred_aligned, 'results_ltsf_nlinear.csv')
    print("    Saved: results_ltsf_nlinear.csv")

    # Linear
    print("\n[6] Training Linear...")
    model = Linear_Model(config)
    model = train_model(model, train_loader, test_loader, device, max_epochs=30)
    pred = predict_model(model, data_x, seq_len, pred_len, device)
    pred_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred)), pred)
    mse, rmse, mae, r2 = evaluate_and_print(pred_aligned, y_test)
    results_summary.append(('Linear', mse, rmse, mae, r2))
    save_results(test_df, pred_aligned, 'results_ltsf_linear.csv')
    print("    Saved: results_ltsf_linear.csv")

    # Transformer
    print("\n[7] Training Transformer...")
    model = Transformer_Model(config)
    model = train_model(model, train_loader, test_loader, device, max_epochs=20)
    pred = predict_model(model, data_x, seq_len, pred_len, device)
    pred_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred)), pred)
    mse, rmse, mae, r2 = evaluate_and_print(pred_aligned, y_test)
    results_summary.append(('Transformer', mse, rmse, mae, r2))
    save_results(test_df, pred_aligned, 'results_ltsf_transformer.csv')
    print("    Saved: results_ltsf_transformer.csv")

    # Informer (简化版，使用FullAttention代替ProbAttention)
    print("\n[8] Training Informer (simplified)...")
    try:
        model = Informer_Model(config)
        model = train_model(model, train_loader, test_loader, device, max_epochs=15)
        pred = predict_model(model, data_x, seq_len, pred_len, device)
        pred_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred)), pred)
        mse, rmse, mae, r2 = evaluate_and_print(pred_aligned, y_test)
        results_summary.append(('Informer', mse, rmse, mae, r2))
        save_results(test_df, pred_aligned, 'results_ltsf_informer.csv')
        print("    Saved: results_ltsf_informer.csv")
    except Exception as e:
        print(f"    Informer failed: {str(e)[:100]}")
        results_summary.append(('Informer', None, None, None, None))

    # Autoformer
    print("\n[9] Training Autoformer...")
    try:
        model = Autoformer_Model(config)
        model = train_model(model, train_loader, test_loader, device, max_epochs=15)
        pred = predict_model(model, data_x, seq_len, pred_len, device)
        pred_aligned = np.interp(np.arange(len(y_test)), np.linspace(0, len(y_test)-1, len(pred)), pred)
        mse, rmse, mae, r2 = evaluate_and_print(pred_aligned, y_test)
        results_summary.append(('Autoformer', mse, rmse, mae, r2))
        save_results(test_df, pred_aligned, 'results_ltsf_autoformer.csv')
        print("    Saved: results_ltsf_autoformer.csv")
    except Exception as e:
        print(f"    Autoformer failed: {str(e)[:100]}")
        results_summary.append(('Autoformer', None, None, None, None))

    # ============================================================
    # 结果汇总
    # ============================================================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R2':<12}")
    print("-" * 60)
    for name, mse, rmse, mae, r2 in results_summary:
        if mse is not None:
            print(f"{name:<15} {mse:<12.6f} {rmse:<12.6f} {mae:<12.6f} {r2:<12.6f}")
        else:
            print(f"{name:<15} {'Failed':<12}")

    print("\n" + "=" * 60)
    print("All result files saved!")
    print("=" * 60)
    print("Files:")
    print("  - results_ltsf_dlinear.csv")
    print("  - results_ltsf_nlinear.csv")
    print("  - results_ltsf_linear.csv")
    print("  - results_ltsf_transformer.csv")
    print("  - results_ltsf_informer.csv")
    print("  - results_ltsf_autoformer.csv")


if __name__ == "__main__":
    main()
