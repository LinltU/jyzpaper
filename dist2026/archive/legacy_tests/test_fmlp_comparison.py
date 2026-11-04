"""
对比测试 FMLP-LTSF 和 FMLP 表格模型
整合 test.py 中的丰富特征工程进行训练
"""
import pandas as pd
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy
import warnings
warnings.filterwarnings('ignore')

from model.fmlp_ltsf import FMLP_LTSF
from model.fmlp import FMLP


def absolute_humidity(T, RH):
    """相对湿度转换为绝对湿度"""
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH


def prepare_rich_features(df):
    """准备丰富的时序特征 (整合 test.py 中的特征工程)"""
    df = df.copy()

    # 按 IMEI 和时间排序
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['IMEI', 'Timestamp'], ascending=[True, True]).reset_index(drop=True)

    # ===== 时间特征 =====
    df['Hour'] = df['Timestamp'].dt.hour
    df['Hour_group'] = (df['Timestamp'].dt.hour // 3).astype(int)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month

    # 周期性特征编码
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Hour_group_sin'] = np.sin(2 * np.pi * df['Hour_group'] / 8)
    df['Hour_group_cos'] = np.cos(2 * np.pi * df['Hour_group'] / 8)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # 计算绝对湿度
    df['AH'] = df.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)

    # ===== 按设备分组计算时序特征 =====
    df['Measure_diff'] = df.groupby('IMEI')['Measure'].diff().fillna(0)
    df['Temp_diff'] = df.groupby('IMEI')['Temp'].diff().fillna(0)
    df['Hum_diff'] = df.groupby('IMEI')['Hum'].diff().fillna(0)

    # 移动平均 (每个设备单独计算)
    def calc_rolling_ma(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    df['Measure_MA'] = df.groupby('IMEI')['Measure'].transform(
        lambda x: calc_rolling_ma(x, window=5))
    df['Measure_MA_10'] = df.groupby('IMEI')['Measure'].transform(
        lambda x: calc_rolling_ma(x, window=10))

    # 滞后特征 (每个设备单独计算)
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

    # 高湿度标记
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

    # 综合湿温权重
    df['HumTemp_x_diff'] = df['HumTemp_weight_high'] * df['Measure_diff'].abs()
    df['HumTemp_x_MA'] = df['HumTemp_weight_high'] * df['Measure_MA']
    df['HumTemp_x_lag1'] = df['HumTemp_weight_high'] * df['Measure_lag1']

    # 保留数值精度
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def get_rich_feature_columns():
    """获取丰富特征列名"""
    return [
        # 时间特征
        'Hour', 'Hour_group', 'DayOfWeek', 'Month',
        'Hour_sin', 'Hour_cos', 'Hour_group_sin', 'Hour_group_cos',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos',
        # 基础环境特征
        'Temp', 'Hum', 'AH',
        # 时序特征
        'Measure_diff', 'Temp_diff', 'Hum_diff',
        'Measure_MA', 'Measure_MA_10', 'Measure_lag1', 'Measure_lag2',
        # 湿度区域特征
        'Hum_0_25', 'Hum_25_35', 'Hum_35_45', 'Hum_45_55', 'Hum_55_65',
        'Hum_65_75', 'Hum_75_85', 'Hum_85_100',
        'Hum_high', 'Hum_very_high',
        # 位置特征
        'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
        'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount',
        # 交互特征
        'Temp_Hum', 'Temp_AH', 'HumTemp_weight', 'HumTemp_weight_high',
        'Hum_high_x_Temp', 'Hum_high_x_AH', 'Hum_high_x_MA', 'Hum_high_x_lag1',
        'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1',
    ]


def create_sequences_by_device(df, feature_cols, seq_len, pred_len):
    """按设备分组创建滑动窗口序列，确保不跨越设备边界"""
    hum_idx = feature_cols.index('Hum')

    X_list, y_list, w_list, idx_list = [], [], [], []

    for imei, group in df.groupby('IMEI', sort=False):
        group = group.sort_values('Timestamp').reset_index(drop=True)
        data = group[feature_cols + ['Measure']].values
        n = len(data)

        for i in range(n - seq_len - pred_len + 1):
            X = data[i:i+seq_len, :-1]
            y = data[i+seq_len:i+seq_len+pred_len, -1]

            # 计算湿度权重
            avg_hum = np.mean(data[i:i+seq_len, hum_idx])
            if avg_hum >= 85:
                w = 3.0
            elif avg_hum >= 75:
                w = 2.5
            elif avg_hum >= 65:
                w = 2.0
            elif avg_hum >= 55:
                w = 1.5
            else:
                w = 1.0

            X_list.append(X)
            y_list.append(y)
            w_list.append(w)
            idx_list.append(group.index[i + seq_len + pred_len // 2])

    return np.array(X_list), np.array(y_list), np.array(w_list), np.array(idx_list)


def save_results(df_test, predictions, pred_mean, y_test, filename, model_name):
    """保存预测结果"""
    result_df = df_test.copy()
    result_df['Prediction'] = pred_mean
    result_df['Prediction_first'] = predictions[:, 0]
    result_df['Prediction_last'] = predictions[:, -1]
    result_df['Actual_mean'] = y_test.mean(axis=1)
    result_df['Actual_first'] = y_test[:, 0]
    result_df['Actual_last'] = y_test[:, -1]
    result_df['Error_mean'] = np.abs(result_df['Actual_mean'] - result_df['Prediction'])
    result_df['Error_first'] = np.abs(result_df['Actual_first'] - result_df['Prediction_first'])
    result_df['Error_last'] = np.abs(result_df['Actual_last'] - result_df['Prediction_last'])
    result_df['Error_pct_mean'] = np.where(
        result_df['Actual_mean'] != 0,
        result_df['Error_mean'] / result_df['Actual_mean'] * 100, 0)
    result_df['Hum_region'] = pd.cut(result_df['Hum'],
                                       bins=[0, 25, 35, 45, 55, 65, 75, 85, 100],
                                       labels=['0-25', '25-35', '35-45', '45-55', '55-65', '65-75', '75-85', '85+'])
    result_df = result_df.round(4)
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"保存结果: {filename}")
    return result_df


def main():
    print("=" * 70)
    print("FMLP-LTSF vs FMLP 表格模型对比测试 (丰富特征工程)")
    print("=" * 70)

    # ===== 加载数据 =====
    df = pd.read_csv('combined_data.csv')
    print(f"原始数据: {len(df)} 行")

    # ===== 特征工程 =====
    print("\n准备丰富时序特征...")
    df = prepare_rich_features(df)

    feature_cols = get_rich_feature_columns()
    n_features = len(feature_cols)
    print(f"特征数量: {n_features}")
    print(f"特征列表: {feature_cols}")

    # ===== 时序参数 =====
    seq_len = 24
    pred_len = 6
    print(f"时序参数: seq_len={seq_len}, pred_len={pred_len}")

    # ===== 划分数据集 =====
    n_samples = len(df)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train+n_val].copy()
    df_test = df.iloc[n_train+n_val:].copy()

    print(f"\n数据划分: 训练={len(df_train)}, 验证={len(df_val)}, 测试={len(df_test)}")

    # ===== 创建序列 =====
    print("\n创建滑动窗口序列（按设备分组）...")

    X_train, y_train, w_train, _ = create_sequences_by_device(df_train, feature_cols, seq_len, pred_len)
    X_val, y_val, w_val, _ = create_sequences_by_device(df_val, feature_cols, seq_len, pred_len)
    X_test, y_test, _, idx_test = create_sequences_by_device(df_test, feature_cols, seq_len, pred_len)

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    # ===== 标准化 =====
    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)

    scaler_y = StandardScaler()
    y_train_flat = y_train.reshape(-1, pred_len)
    scaler_y.fit(y_train_flat)
    y_train_scaled = scaler_y.transform(y_train_flat).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, pred_len)).reshape(y_val.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, pred_len)).reshape(y_test.shape)

    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    results_summary = []

    # ===== 1. 训练 FMLP_LTSF 模型 =====
    ltsf_models = [
        {'name': 'FMLP_LTSF_no_MLP', 'use_mlp': False},
        {'name': 'FMLP_LTSF_with_MLP', 'use_mlp': True}
    ]

    for model_config in ltsf_models:
        model_name = model_config['name']
        use_mlp = model_config['use_mlp']

        print(f"\n{'='*70}")
        print(f"开始训练: {model_name}")
        print(f"{'='*70}")

        torch.manual_seed(42 + hash(model_name) % 100)
        np.random.seed(42 + hash(model_name) % 100)

        model = FMLP_LTSF(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_features,
            hidden_dim=64,
            dropout=0.2,
            kernel_size=25,
            individual=False,
            use_mlp=use_mlp
        ).to(device)

        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 数据加载器
        X_tensor = torch.FloatTensor(X_train_scaled)
        y_tensor = torch.FloatTensor(y_train_scaled)
        w_tensor = torch.FloatTensor(w_train)
        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 验证集加载器
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val_scaled)
        w_val_tensor = torch.FloatTensor(w_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, w_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        max_epochs = 100

        print(f"开始训练 (共 {max_epochs} 轮, 早停patience=15)...")
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y, batch_w in loader:
                batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze(-1)
                loss = (criterion(outputs, batch_y) * batch_w).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)

            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y, batch_w in val_loader:
                    batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
                    outputs = model(batch_X).squeeze(-1)
                    val_loss += (criterion(outputs, batch_y) * batch_w).mean().item()
            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:3d}/{max_epochs}] Train: {avg_loss:.6f} | Val: {avg_val_loss:.6f}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 15:
                print(f"早停于 Epoch {epoch+1}")
                break

        model.load_state_dict(best_model_state)
        print(f"训练完成! 最佳验证损失: {best_loss:.6f}")

        # 评估
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, pred_len)).reshape(predictions_scaled.shape)
        pred_mean = predictions.mean(axis=1)
        y_mean = y_test.mean(axis=1)

        mse = mean_squared_error(y_mean, pred_mean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_mean, pred_mean)
        r2 = r2_score(y_mean, pred_mean)
        mape = np.mean(np.abs((y_mean - pred_mean) / (y_mean + 1e-8))) * 100

        print(f"\n{model_name} 测试集评估结果:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R2:   {r2:.6f}")
        print(f"  MAPE: {mape:.2f}%")

        results_summary.append({'Model': model_name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'N_Test': len(y_test)})

        result_df = save_results(df_test.iloc[idx_test].copy(), predictions, pred_mean, y_test, f'results_{model_name}.csv', model_name)
        torch.save(model.state_dict(), f'{model_name}_model.pth')
        print(f"保存模型: {model_name}_model.pth")

    # ===== 2. 训练 FMLP 表格模型 (完全按照 test.py 的训练方式) =====
    print(f"\n{'='*70}")
    print(f"开始训练: FMLP (test.py 训练方式)")
    print(f"{'='*70}")

    # 准备 FMLP 需要的表格数据 - 使用序列最后时间步
    X_train_fmlp = X_train_scaled[:, -1, :]
    y_train_fmlp = y_train_scaled[:, -1]
    X_val_fmlp = X_val_scaled[:, -1, :]
    y_val_fmlp = y_val_scaled[:, -1]
    X_test_fmlp = X_test_scaled[:, -1, :]

    # ===== 完全按照 test.py 的训练方式 =====

    torch.manual_seed(42)
    np.random.seed(42)

    fmlp_model = FMLP(
        input_dim=n_features,
        hidden_dim=128,  # 与 test.py 一致
        dropout=0.2,
        kernel_size=25
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in fmlp_model.parameters()):,}")

    # batch_size 与 test.py 一致
    batch_size = min(32, len(X_train_fmlp) // 2)
    batch_size = max(8, batch_size)
    print(f"batch_size: {batch_size}")

    # 数据加载器与 test.py 完全一致
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_fmlp),
        torch.FloatTensor(y_train_fmlp)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_val_fmlp),
        torch.FloatTensor(y_val_fmlp)  # 使用验证集的真实值
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False)

    # 优化器与 test.py 一致
    optimizer = optim.AdamW(fmlp_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # 没有权重
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    patience = 15
    patience_counter = 0
    best_epoch_loss = float('inf')
    best_model_state = None
    max_epochs = 200

    print(f"开始训练 (共 {max_epochs} 轮, 早停patience=15)...")
    print("-" * 60)

    for epoch in range(max_epochs):
        fmlp_model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = fmlp_model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)  # 没有权重
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fmlp_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # 验证
        fmlp_model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_X, val_y in valid_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = fmlp_model(val_X)
                val_loss += criterion(val_outputs.squeeze(), val_y).item()

        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{max_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_epoch_loss:
            best_epoch_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(fmlp_model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n早停于 Epoch {epoch+1}")
            break

    fmlp_model.load_state_dict(best_model_state)
    fmlp_model = fmlp_model.to('cpu')  # 与 test.py 一致

    print("-" * 60)
    print(f"训练完成! 最佳验证损失: {best_epoch_loss:.6f}")

    # 评估
    fmlp_model.eval()
    with torch.no_grad():
        predictions_scaled = fmlp_model(torch.FloatTensor(X_test_fmlp)).numpy()

    # 扩展为多步预测进行对比
    predictions_scaled = np.tile(predictions_scaled.reshape(-1, 1), (1, y_test.shape[1]))
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, pred_len)).reshape(predictions_scaled.shape)
    pred_mean = predictions.mean(axis=1)
    y_mean = y_test.mean(axis=1)

    mse = mean_squared_error(y_mean, pred_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_mean, pred_mean)
    r2 = r2_score(y_mean, pred_mean)
    mape = np.mean(np.abs((y_mean - pred_mean) / (y_mean + 1e-8))) * 100

    print(f"\nFMLP 测试集评估结果:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%")

    results_summary.append({'Model': 'FMLP', 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'N_Test': len(y_test)})

    import time
    timestamp = int(time.time())
    result_df = save_results(df_test.iloc[idx_test].copy(), predictions, pred_mean, y_test, f'results_FMLP_rich_{timestamp}.csv', 'FMLP')
    torch.save(fmlp_model.state_dict(), f'FMLP_rich_model_{timestamp}.pth')
    print(f"保存模型: FMLP_rich_model_{timestamp}.pth")

    # ===== 模型对比表 =====
    print("\n" + "=" * 70)
    print("FMLP 模型对比结果")
    print("=" * 70)

    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    summary_df.to_csv('fmlp_models_comparison_rich.csv', index=False, encoding='utf-8-sig')
    print(f"\n保存对比结果: fmlp_models_comparison_rich.csv")

    best_model = min(results_summary, key=lambda x: x['RMSE'])
    print(f"\n最佳模型 (RMSE最低): {best_model['Model']}")
    print(f"  RMSE: {best_model['RMSE']:.6f}")
    print(f"  MAE:  {best_model['MAE']:.6f}")
    print(f"  R2:   {best_model['R2']:.6f}")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")

    return results_summary


if __name__ == '__main__':
    results = main()
    print("\n" + "=" * 70)
    print("所有模型训练完成!")
    print("=" * 70)
