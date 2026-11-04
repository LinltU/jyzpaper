"""
Test script: train multiple models using combined_data.csv
支持 Transformer, FMLP, SVM 三种模型
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
import copy

# 导入模型
from model.Transformer import SimpleTransformer
from model.fmlp import FMLP


def absolute_humidity(T, RH):
    """相对湿度转换为绝对湿度"""
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH


def prepare_features(df):
    """准备特征，按设备分组计算时序特征"""
    df = df.copy()

    # 按 IMEI 和时间排序
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['IMEI', 'Timestamp'], ascending=[True, True]).reset_index(drop=True)

    # ===== 时间特征 =====
    # 小时分组 (每3小时一组: 0-3, 3-6, 6-9, 9-12, 12-15, 15-18, 18-21, 21-24)
    df['Hour_group'] = (df['Timestamp'].dt.hour // 3).astype(int)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month

    # 周期性特征编码 (使用sin/cos捕捉周期性)
    # 小时周期 (8个时段)
    df['Hour_group_sin'] = np.sin(2 * np.pi * df['Hour_group'] / 8)
    df['Hour_group_cos'] = np.cos(2 * np.pi * df['Hour_group'] / 8)
    # 星期周期 (7天)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    # 月份周期 (12个月)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # 计算绝对湿度
    df['AH'] = df.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)

    # ===== 按设备分组计算时序特征 =====
    # 分组计算差分、移动平均、滞后
    df['Measure_diff'] = df.groupby('IMEI')['Measure'].diff().fillna(0)
    df['Temp_diff'] = df.groupby('IMEI')['Temp'].diff().fillna(0)
    df['Hum_diff'] = df.groupby('IMEI')['Hum'].diff().fillna(0)

    # 移动平均 (每个设备单独计算)
    def calc_rolling_ma(series, window):
        return series.rolling(window=window, min_periods=1).mean()
    df['Measure_MA'] = df.groupby('IMEI')['Measure'].transform(
        lambda x: calc_rolling_ma(x, window=5))

    # 滞后特征 (每个设备单独计算)
    df['Measure_lag1'] = df.groupby('IMEI')['Measure'].shift(1).fillna(0)

    # ===== 交互特征 =====
    df['Temp_Hum_interact'] = df['Temp'] * df['Hum']
    df['Temp_AH_interact'] = df['Temp'] * df['AH']

    # 湿度区间特征（精细划分，补充原始湿度）
    df['Hum_0_35'] = np.where(df['Hum'] < 35, 1, 0)
    df['Hum_35_50'] = np.where((df['Hum'] >= 35) & (df['Hum'] < 50), 1, 0)
    df['Hum_50_60'] = np.where((df['Hum'] >= 50) & (df['Hum'] < 60), 1, 0)
    df['Hum_60_70'] = np.where((df['Hum'] >= 60) & (df['Hum'] < 70), 1, 0)
    df['Hum_70_80'] = np.where((df['Hum'] >= 70) & (df['Hum'] < 80), 1, 0)

    # 高湿度标记（统一阈值：60%）
    df['Hum_high'] = np.where(df['Hum'] >= 60, 1, 0)

    # 综合湿温权重：湿度高且温度低时权重更大
    df['HumTemp_weight'] = df['Hum'] / (df['Temp'] + 5)
    df['HumTemp_weight_high'] = np.where(df['HumTemp_weight'] > 8, 1, 0)

    # 湿温交互特征（统一使用HumTemp_weight_high）
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

    # 所有数值特征保留4位小数
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def get_feature_columns():
    """获取特征列名"""
    return [
        # 时间特征 (9个)
        'Hour_group', 'DayOfWeek', 'Month',
        'Hour_group_sin', 'Hour_group_cos',
        'DayOfWeek_sin', 'DayOfWeek_cos',
        'Month_sin', 'Month_cos',
        # 基础环境特征 (3个)
        'Temp', 'Hum', 'AH',
        # 时序特征 (5个)
        'Measure_diff', 'Temp_diff', 'Hum_diff', 'Measure_MA', 'Measure_lag1',
        # 交互特征 (10个)
        'Temp_Hum_interact', 'Temp_AH_interact',
        'Hum_0_35', 'Hum_35_50', 'Hum_50_60', 'Hum_60_70', 'Hum_70_80',
        'HumTemp_weight', 'HumTemp_weight_high',
        'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1',
        # 位置特征 (9个)
        'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
        'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount'
    ]


def train_transformer_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """训练 Transformer 模型"""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train['Measure'].values)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val['Measure'].values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test['Measure'].values)

    batch_size = min(32, len(X_train) // 2)
    batch_size = max(8, batch_size)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    input_dim = X_train.shape[1]
    model = SimpleTransformer(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        dropout=0.2,
        feed_forward_dim=256
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    patience = 15
    patience_counter = 0
    best_epoch_loss = float('inf')
    best_model_state = None
    max_epochs = 200

    print(f"\n开始训练 (共 {max_epochs} 轮)...")
    print("-" * 60)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
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
                val_loss += criterion(val_outputs, val_y).item()

        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{max_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_epoch_loss:
            best_epoch_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n早停于 Epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    model = model.to('cpu')

    print("-" * 60)
    print(f"训练完成! 最佳验证损失: {best_epoch_loss:.6f}")

    return model, "transformer"


def train_fmlp_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """训练 FMLP 模型"""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train['Measure'].values)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val['Measure'].values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test['Measure'].values)

    batch_size = min(32, len(X_train) // 2)
    batch_size = max(8, batch_size)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=False)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    input_dim = X_train.shape[1]
    model = FMLP(
        input_dim=input_dim,
        hidden_dim=128,
        dropout=0.2,
        kernel_size=25
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    patience = 15
    patience_counter = 0
    best_epoch_loss = float('inf')
    best_model_state = None
    max_epochs = 200

    print(f"\n开始训练 (共 {max_epochs} 轮)...")
    print("-" * 60)

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{max_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_epoch_loss:
            best_epoch_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n早停于 Epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    model = model.to('cpu')

    print("-" * 60)
    print(f"训练完成! 最佳验证损失: {best_epoch_loss:.6f}")

    return model, "fmlp"


def train_svm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """训练 SVM 模型"""
    print("\n开始训练 SVM 模型...")

    model = SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
    model.fit(X_train, y_train['Measure'].values)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test['Measure'].values, predictions)
    print(f"SVM 训练完成! MSE: {mse:.6f}")

    return model, "svm"


def predict_model(model, X_test, model_type):
    """使用模型进行预测"""
    if model_type == "svm":
        predictions = model.predict(X_test)
    else:
        X_test_tensor = torch.FloatTensor(X_test)
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).numpy()

    predictions = np.maximum(predictions, 0)
    return predictions


def evaluate_and_print(predictions, y_test):
    """评估模型并打印结果"""
    y_true = y_test['Measure'].values
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R^2:  {r2:.6f}")
    print("=" * 60)

    return mse, rmse, mae, r2


def main():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    df = pd.read_csv('combined_data.csv')
    print(f"Original: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")

    # 不进行数据过滤 (使用全部数据)
    df_filtered = df.copy()
    print(f"使用全部数据: {len(df_filtered)} rows")

    print("\nPreparing features...")
    df_prepared = prepare_features(df_filtered)

    # 选择特征列
    feature_cols = get_feature_columns()
    print(f"Number of features: {len(feature_cols)}")

    # 提取特征和目标
    X = df_prepared[feature_cols].fillna(0)
    y = df_prepared[['Measure']]

    # 划分训练/验证/测试集 (70:15:15) - 正确做法
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 保存训练/测试数据
    train_df = df_prepared.iloc[X_train.index].copy()
    val_df = df_prepared.iloc[X_val.index].copy()
    test_df = df_prepared.iloc[X_test.index].copy()

    train_df.to_csv('training.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv('validation.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('test.csv', index=False, encoding='utf-8-sig')
    print("\nSaved: training.csv, validation.csv, test.csv")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 定义模型列表
    models = [
        ("Transformer", train_transformer_model),
        ("FMLP", train_fmlp_model),
        ("SVM", train_svm_model),
    ]

    results_summary = []

    # 依次训练每个模型
    for model_name, train_func in models:
        print("\n" + "=" * 60)
        print(f"训练 {model_name} 模型")
        print("=" * 60)

        # 训练模型
        model, model_type = train_func(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)

        # 预测
        predictions = predict_model(model, X_test_scaled, model_type)

        # 评估
        mse, rmse, mae, r2 = evaluate_and_print(predictions, y_test)

        # 保存结果
        result_df = test_df.copy()
        result_df['PredictResult'] = predictions
        result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['PredictResult'])
        result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100

        # 预测结果保留4位小数
        result_df[['PredictResult', 'AbsoluteError', 'PercentageError']] = result_df[['PredictResult', 'AbsoluteError', 'PercentageError']].round(4)

        result_file = f'results_{model_type}.csv'
        result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved result: {result_file}")

        # 记录结果
        results_summary.append({
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })

    # 打印对比表
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    main()
