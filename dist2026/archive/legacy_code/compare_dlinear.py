"""
DLinear模型对比脚本
对比 test.py 和 test_ltsf_38features.py 中的DLinear模型
"""
import pandas as pd
import numpy as np
import math
import os
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

# ============================================================
# 特征工程（两个脚本共用）
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
# Model 1: test.py 中的 DecompositionNLinear（从 ltsf_nlinear.py 复制）
# ============================================================

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
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


class DecompositionNLinear(nn.Module):
    """
    test.py 中的 DLinear 模型
    特点：2D输入，无真实时序建模
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2, kernel_size=25):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.decomposition = series_decomp(kernel_size)
        
        self.seasonal_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.trend_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
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
        # x: [batch, input_dim]  2D输入
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        x_reshaped = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        seasonal, trend = self.decomposition(x_reshaped)
        seasonal = seasonal.squeeze(1)  # [batch, input_dim]
        trend = trend.squeeze(1)
        
        seasonal_interaction = self.interaction_layer(seasonal)
        trend_interaction = self.interaction_layer(trend)
        seasonal = seasonal * seasonal_interaction
        trend = trend * trend_interaction
        
        seasonal_features = self.seasonal_layer(seasonal)
        trend_features = self.trend_layer(trend)
        
        combined_features = torch.cat([seasonal_features, trend_features], dim=1)
        y_pred = self.fusion_layer(combined_features)
        
        return y_pred.squeeze(-1)


# ============================================================
# Model 2: test_ltsf_38features.py 中的 DLinear_MultiFeature
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = torch.FloatTensor(data_x)
        self.data_y = torch.FloatTensor(data_y).unsqueeze(-1) if data_y.ndim == 1 else torch.FloatTensor(data_y)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]


class DLinear_MultiFeature(nn.Module):
    """
    test_ltsf_38features.py 中的 DLinear 模型
    特点：3D时序输入，滑动窗口
    """
    def __init__(self, n_features, seq_len=24, pred_len=1, hidden_dim=128, kernel_size=25):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.decomposition = series_decomp(kernel_size)
        
        self.seasonal_branch = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.trend_branch = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
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
        
        s_perm = seasonal.permute(0, 2, 1)
        s_out = self.seasonal_branch(s_perm)
        
        t_perm = trend.permute(0, 2, 1)
        t_out = self.trend_branch(t_perm)
        
        combined = torch.cat([s_out, t_out], dim=-1)
        combined = combined.view(combined.size(0), -1)
        output = self.output_layer(combined)
        
        return output.squeeze(-1)


# ============================================================
# 训练函数
# ============================================================

def train_model_v1(model, train_loader, valid_loader, device, max_epochs=100, lr=0.001, model_name="Model"):
    """训练2D输入模型 (test.py风格)"""
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 15

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

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{max_epochs}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_loss


def train_model_v2(model, train_loader, valid_loader, device, max_epochs=100, lr=0.001, model_name="Model"):
    """训练3D输入模型 (test_ltsf_38features.py风格)"""
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

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{max_epochs}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_loss


def predict_model(model, data_x, device, model_type):
    """预测"""
    model.eval()
    preds = []
    with torch.no_grad():
        if model_type == "v1":
            data_tensor = torch.FloatTensor(data_x).to(device)
            preds = model(data_tensor).cpu().numpy()
        else:  # v2: 3D input
            for i in range(len(data_x)):
                seq_x = torch.FloatTensor(data_x[i:i + 1]).to(device)
                output = model(seq_x)
                preds.append(output.cpu().numpy()[0])
            preds = np.array(preds)
    return preds


def evaluate_and_print(predictions, y_true):
    """评估模型"""
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
    print("DLinear Model Comparison: test.py vs test_ltsf_38features.py")
    print("=" * 70)

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
    n_features = len(feature_cols)
    print(f"    Number of features: {n_features}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    results = []

    # ============================================================
    # Model 1: test.py DLinear (2D Input)
    # ============================================================
    print("\n" + "=" * 70)
    print("[3] Model 1: test.py DLinear (2D Input - No Time Series)")
    print("=" * 70)
    
    X = df_prepared[feature_cols].fillna(0)
    y = df_prepared[['Measure']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"    X_train shape: {X_train_scaled.shape}")
    print(f"    X_test shape: {X_test_scaled.shape}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train['Measure'].values))
    valid_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test['Measure'].values))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = DecompositionNLinear(input_dim=n_features, hidden_dim=128, dropout=0.2, kernel_size=25)
    model, best_loss = train_model_v1(model, train_loader, valid_loader, device, max_epochs=100, model_name="test.py DLinear")
    
    pred = predict_model(model, X_test_scaled, device, "v1")
    mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test['Measure'].values)
    
    results.append({
        'Model': 'test.py DLinear',
        'Input': '2D [B, 38]',
        'TimeSeries': 'No',
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
    })

    # ============================================================
    # Model 2: test_ltsf_38features.py DLinear (3D Input with Time Windows)
    # ============================================================
    print("\n" + "=" * 70)
    print("[4] Model 2: test_ltsf_38features.py DLinear (3D Input - Time Windows)")
    print("=" * 70)
    
    seq_len = 24
    pred_len = 1
    
    imei_groups = df_prepared.groupby('IMEI')
    all_sequences = []
    
    for imei, group in imei_groups:
        group = group.sort_values('Timestamp').reset_index(drop=True)
        feats = group[feature_cols].fillna(0).values
        target = group['Measure'].values
        
        for i in range(len(group) - seq_len - pred_len + 1):
            seq_x = feats[i:i+seq_len]
            y_target = target[i+seq_len]
            all_sequences.append({'x': seq_x, 'y': y_target})
    
    print(f"    Total time series samples: {len(all_sequences)}")
    
    split_threshold = int(len(all_sequences) * 0.8)
    X_train_3d = np.array([s['x'] for s in all_sequences[:split_threshold]], dtype=np.float32)
    X_test_3d = np.array([s['x'] for s in all_sequences[split_threshold:]], dtype=np.float32)
    y_train_arr = np.array([s['y'] for s in all_sequences[:split_threshold]], dtype=np.float32)
    y_test_arr = np.array([s['y'] for s in all_sequences[split_threshold:]], dtype=np.float32)
    
    scaler_ts = StandardScaler()
    X_train_3d_flat = X_train_3d.reshape(-1, n_features)
    X_test_3d_flat = X_test_3d.reshape(-1, n_features)
    X_train_3d_scaled_flat = scaler_ts.fit_transform(X_train_3d_flat)
    X_test_3d_scaled_flat = scaler_ts.transform(X_test_3d_flat)
    X_train_3d = X_train_3d_scaled_flat.reshape(X_train_3d.shape).astype(np.float32)
    X_test_3d = X_test_3d_scaled_flat.reshape(X_test_3d.shape).astype(np.float32)
    
    print(f"    X_train_3d shape: {X_train_3d.shape}")
    print(f"    X_test_3d shape: {X_test_3d.shape}")
    
    train_dataset = TimeSeriesDataset(X_train_3d, y_train_arr)
    test_dataset = TimeSeriesDataset(X_test_3d, y_test_arr)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = DLinear_MultiFeature(n_features=n_features, seq_len=seq_len, pred_len=pred_len, hidden_dim=128)
    model, best_loss = train_model_v2(model, train_loader, valid_loader, device, max_epochs=100, model_name="test_ltsf_38features DLinear")
    
    pred = predict_model(model, X_test_3d, device, "v2")
    mse, rmse, mae, r2, mape = evaluate_and_print(pred, y_test_arr)
    
    results.append({
        'Model': 'test_ltsf_38features DLinear',
        'Input': '3D [B, 24, 38]',
        'TimeSeries': 'Yes (sliding window)',
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape
    })

    # ============================================================
    # 结果对比
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Input Format':<18} {'Time Series':<20}")
    print("-" * 70)
    for r in results:
        print(f"{r['Model']:<30} {r['Input']:<18} {r['TimeSeries']:<20}")
    
    print(f"\n{'Model':<30} {'MSE':<14} {'RMSE':<14} {'MAE':<14} {'R2':<12} {'MAPE':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['Model']:<30} {r['MSE']:<14.6f} {r['RMSE']:<14.6f} {r['MAE']:<14.6f} {r['R2']:<12.6f} {r['MAPE']:<10.2f}%")
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("""
    1. Input Format:
       - test.py:           2D [batch, 38 features] - flat feature vector
       - test_ltsf_38features: 3D [batch, 24 timesteps, 38 features] - time series window
    
    2. Time Series Modeling:
       - test.py:           NO - treats features as independent
       - test_ltsf_38features: YES - uses sliding window of 24 timesteps
    
    3. Data Split:
       - test.py:           Random split (may leak future info)
       - test_ltsf_38features: Sequential split (proper time series split)
    
    4. Model Architecture:
       - test.py:           DecompositionNLinear with interaction layers
       - test_ltsf_38features: DLinear_MultiFeature with seq processing
    """)
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('dlinear_comparison_results.csv', index=False)
    print("\nResults saved to: dlinear_comparison_results.csv")
    
    return results


if __name__ == "__main__":
    main()
