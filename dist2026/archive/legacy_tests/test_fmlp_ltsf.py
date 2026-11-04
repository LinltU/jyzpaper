"""
Test LTSF-FMLP: 时序预测版本的FMLP
参考LTSF-Linear的DLinear架构进行时序建模
使用滑动窗口进行未来预测
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


def absolute_humidity(T, RH):
    """相对湿度转换为绝对湿度"""
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH


def prepare_ltsf_features(df, seq_len=96):
    """
    准备LTSF时序特征
    按设备分组，构造滑动窗口数据
    """
    df = df.copy()
    
    # 按设备和时间排序
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['IMEI', 'Timestamp'], ascending=[True, True]).reset_index(drop=True)
    
    # ===== 时间特征 =====
    df['Hour'] = df['Timestamp'].dt.hour
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month'] = df['Timestamp'].dt.month
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # 绝对湿度
    df['AH'] = df.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)
    
    # ===== 湿度区域特征 (扩展湿度区间) =====
    # 细粒度湿度区间
    df['Hum_0_25'] = np.where(df['Hum'] < 25, 1, 0)
    df['Hum_25_35'] = np.where((df['Hum'] >= 25) & (df['Hum'] < 35), 1, 0)
    df['Hum_35_45'] = np.where((df['Hum'] >= 35) & (df['Hum'] < 45), 1, 0)
    df['Hum_45_55'] = np.where((df['Hum'] >= 45) & (df['Hum'] < 55), 1, 0)
    df['Hum_55_65'] = np.where((df['Hum'] >= 55) & (df['Hum'] < 65), 1, 0)
    df['Hum_65_75'] = np.where((df['Hum'] >= 65) & (df['Hum'] < 75), 1, 0)
    df['Hum_75_85'] = np.where((df['Hum'] >= 75) & (df['Hum'] < 85), 1, 0)
    df['Hum_85_100'] = np.where(df['Hum'] >= 85, 1, 0)
    
    # 高湿度权重标记 (湿度>=65%为高湿度)
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
    
    # 高湿度区域的额外权重
    df['Hum_high_x_Temp'] = df['Hum_high'] * df['Temp']
    df['Hum_high_x_AH'] = df['Hum_high'] * df['AH']
    
    # 保留数值精度
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    return df


def get_ltsf_feature_columns():
    """获取LTSF特征列名"""
    return [
        # 时间特征 (9个)
        'Hour', 'Hour_sin', 'Hour_cos',
        'DayOfWeek', 'DayOfWeek_sin', 'DayOfWeek_cos',
        'Month', 'Month_sin', 'Month_cos',
        # 环境特征 (3个)
        'Temp', 'Hum', 'AH',
        # 湿度区域特征 (11个)
        'Hum_0_25', 'Hum_25_35', 'Hum_35_45', 'Hum_45_55', 'Hum_55_65',
        'Hum_65_75', 'Hum_75_85', 'Hum_85_100',
        'Hum_high', 'Hum_very_high',
        # 位置特征 (9个)
        'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
        'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount',
        # 交互特征 (5个)
        'Temp_Hum', 'Temp_AH', 'HumTemp_weight',
        'Hum_high_x_Temp', 'Hum_high_x_AH',
    ]


def create_sequences_by_device(df, feature_cols, target_col, seq_len=96, pred_len=24, device_col='IMEI'):
    """
    按设备创建滑动窗口序列
    """
    all_X = []
    all_y = []
    all_weights = []
    all_indices = []
    
    for device in df[device_col].unique():
        device_df = df[df[device_col] == device].copy()
        
        # 获取特征和目标
        data = device_df[feature_cols + [target_col]].values
        n_samples = len(data)
        
        # 滑动窗口
        for i in range(n_samples - seq_len - pred_len + 1):
            X_seq = data[i:i+seq_len, :-1]  # 特征 (seq_len, n_features)
            y_seq = data[i+seq_len:i+seq_len+pred_len, -1]  # 目标 (pred_len,)
            
            # 计算样本权重：高湿度样本权重更高
            avg_hum = np.mean(data[i:i+seq_len, feature_cols.index('Hum')])
            humidity_weight = 1.0
            if avg_hum >= 85:
                humidity_weight = 3.0  # 极高湿度
            elif avg_hum >= 75:
                humidity_weight = 2.5  # 高湿度
            elif avg_hum >= 65:
                humidity_weight = 2.0  # 中高湿度
            elif avg_hum >= 55:
                humidity_weight = 1.5  # 中等湿度
            
            all_X.append(X_seq)
            all_y.append(y_seq)
            all_weights.append(humidity_weight)
            all_indices.append(i + seq_len + pred_len // 2)  # 存储预测区间中点的索引
    
    return np.array(all_X), np.array(all_y), np.array(all_weights), np.array(all_indices)


def train_fmlp_ltsf_model(X_train, y_train, weights_train, seq_len, pred_len, enc_in,
                          hidden_dim=64, epochs=50, batch_size=32, lr=0.001, device='cpu'):
    """训练FMLP-LTSF模型"""
    print(f"\n使用设备: {device}")
    print(f"模型参数: seq_len={seq_len}, pred_len={pred_len}, enc_in={enc_in}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模型 (简化版DLinear，无MLP增强)
    model = FMLP_LTSF(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        hidden_dim=hidden_dim,
        dropout=0.2,
        kernel_size=25,
        individual=False,
        use_mlp=False
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    w_tensor = torch.FloatTensor(weights_train)
    
    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器 (使用权重衰减)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    print(f"\n开始训练 (共 {epochs} 轮)...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        weighted_loss = 0
        
        for batch_X, batch_y, batch_w in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)  # [B, pred_len, 1]
            outputs = outputs.squeeze(-1)  # [B, pred_len]
            
            # 加权损失
            loss_unweighted = criterion(outputs, batch_y)
            loss = (loss_unweighted * batch_w).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_unweighted.item()
            weighted_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        avg_weighted_loss = weighted_loss / len(loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.6f} | Weighted Loss: {avg_weighted_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n早停于 Epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    print("-" * 60)
    print(f"训练完成! 最佳损失: {best_loss:.6f}")
    
    return model


def predict_fmlp_ltsf(model, X_test, device='cpu'):
    """预测"""
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = outputs.squeeze(-1).cpu().numpy()
    
    return predictions


def evaluate_predictions(predictions, y_true, threshold_percentile=65):
    """评估预测结果"""
    # 全部评估
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    # 高湿度区域评估
    hum_col_idx = 12  # Hum在特征列中的位置 (时间特征9 + 环境3 = 12)
    # 注意：预测时没有Hum，需要另外处理
    
    return mse, rmse, mae, r2


def save_results(df, predictions, indices, filename):
    """保存预测结果"""
    result_df = df.iloc[indices].copy()
    result_df['Prediction'] = predictions.mean(axis=1)  # 预测区间均值
    result_df['Prediction_first'] = predictions[:, 0]  # 第一个预测点
    result_df['Prediction_last'] = predictions[:, -1]    # 最后一个预测点
    result_df['AbsoluteError'] = np.abs(result_df['Measure'] - result_df['Prediction'])
    result_df['PercentageError'] = result_df['AbsoluteError'] / (result_df['Measure'] + 1e-8) * 100
    
    result_df = result_df.round(4)
    result_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"保存结果: {filename}")
    
    return result_df


def main():
    print("=" * 70)
    print("LTSF-FMLP 时序预测训练")
    print("=" * 70)
    
    # ===== 加载数据 =====
    df = pd.read_csv('combined_data.csv')
    print(f"原始数据: {len(df)} 行")
    
    # 不再过滤湿度和负值，保留所有数据
    print(f"保留所有数据 (不过滤 Hum<80, Measure>=0)")
    
    # ===== 特征工程 =====
    print("\n准备时序特征...")
    df = prepare_ltsf_features(df)
    
    feature_cols = get_ltsf_feature_columns()
    n_features = len(feature_cols)
    print(f"特征数量: {n_features}")
    print(f"特征列: {feature_cols}")
    
    # ===== 时序参数 =====
    seq_len = 96   # 输入序列长度
    pred_len = 24  # 预测长度
    print(f"\n时序参数: seq_len={seq_len}, pred_len={pred_len}")
    
    # ===== 划分训练/测试集 (按时间顺序，滑动窗口) =====
    # 数据量较小，使用较小的窗口
    seq_len = 24   # 输入序列长度 (减少)
    pred_len = 6   # 预测长度 (减少)
    print(f"\n时序参数: seq_len={seq_len}, pred_len={pred_len}")
    
    # 按时间顺序划分数据
    n_samples = len(df)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train+n_val].copy()
    df_test = df.iloc[n_train+n_val:].copy()
    
    print(f"\n数据划分: 训练={len(df_train)}, 验证={len(df_val)}, 测试={len(df_test)}")
    
    # ===== 创建序列 (整体滑动窗口) =====
    print("\n创建滑动窗口序列...")
    
    # 训练集序列
    train_data = df_train[feature_cols + ['Measure']].values
    n = len(train_data)
    
    X_train_list, y_train_list, w_train_list, idx_train_list = [], [], [], []
    hum_idx = feature_cols.index('Hum')
    
    for i in range(n - seq_len - pred_len + 1):
        X = train_data[i:i+seq_len, :-1]
        y = train_data[i+seq_len:i+seq_len+pred_len, -1]
        hum = np.mean(train_data[i:i+seq_len, hum_idx])
        
        w = 1.0
        if hum >= 85: w = 3.0
        elif hum >= 75: w = 2.5
        elif hum >= 65: w = 2.0
        elif hum >= 55: w = 1.5
        
        X_train_list.append(X)
        y_train_list.append(y)
        w_train_list.append(w)
        idx_train_list.append(i + seq_len + pred_len // 2)
    
    # 验证集序列
    val_data = df_val[feature_cols + ['Measure']].values
    n = len(val_data)
    
    X_val_list, y_val_list, w_val_list, idx_val_list = [], [], [], []
    
    for i in range(n - seq_len - pred_len + 1):
        X = val_data[i:i+seq_len, :-1]
        y = val_data[i+seq_len:i+seq_len+pred_len, -1]
        hum = np.mean(val_data[i:i+seq_len, hum_idx])
        
        w = 1.0
        if hum >= 85: w = 3.0
        elif hum >= 75: w = 2.5
        elif hum >= 65: w = 2.0
        elif hum >= 55: w = 1.5
        
        X_val_list.append(X)
        y_val_list.append(y)
        w_val_list.append(w)
        idx_val_list.append(len(df_train) + i + seq_len + pred_len // 2)
    
    # 测试集序列
    test_data = df_test[feature_cols + ['Measure']].values
    n = len(test_data)
    
    X_test_list, y_test_list, idx_test_list = [], [], []
    
    for i in range(n - seq_len - pred_len + 1):
        X = test_data[i:i+seq_len, :-1]
        y = test_data[i+seq_len:i+seq_len+pred_len, -1]
        
        X_test_list.append(X)
        y_test_list.append(y)
        idx_test_list.append(len(df_train) + len(df_val) + i + seq_len + pred_len // 2)
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    w_train = np.array(w_train_list)
    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # ===== 标准化 =====
    # 对每个时间步、每个特征分别标准化
    scaler_X = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    scaler_X.fit(X_train_flat)
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape)
    
    # 标准化y (按每个pred_len位置)
    scaler_y = StandardScaler()
    y_train_flat = y_train.reshape(-1, pred_len)
    scaler_y.fit(y_train_flat)
    y_train_scaled = scaler_y.transform(y_train_flat).reshape(y_train.shape)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, pred_len)).reshape(y_val.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, pred_len)).reshape(y_test.shape)
    
    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # ===== 训练模型 =====
    model = train_fmlp_ltsf_model(
        X_train_scaled, y_train_scaled, w_train,
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=n_features,
        hidden_dim=64,
        epochs=50,
        batch_size=32,
        lr=0.001,
        device=device
    )
    
    # ===== 预测 =====
    print("\n" + "=" * 60)
    print("预测测试集...")
    
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()
    
    # 反标准化
    predictions = scaler_y.inverse_transform(
        predictions_scaled.reshape(-1, pred_len)
    ).reshape(predictions_scaled.shape)
    
    # ===== 评估 =====
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    
    # 计算每个pred_len位置的误差
    for i in [0, pred_len//2, pred_len-1]:
        mse = mean_squared_error(y_test[:, i], predictions[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[:, i], predictions[:, i])
        r2 = r2_score(y_test[:, i], predictions[:, i])
        print(f"\n预测步长 {i+1}/{pred_len}:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R2:   {r2:.6f}")
    
    # 整体评估 (取预测区间的均值)
    pred_mean = predictions.mean(axis=1)
    mse = mean_squared_error(y_test.mean(axis=1), pred_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.mean(axis=1), pred_mean)
    r2 = r2_score(y_test.mean(axis=1), pred_mean)
    
    print(f"\n整体评估 (预测均值):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.6f}")
    
    # ===== 保存结果 =====
    # 找到对应的原始数据索引
    test_indices = idx_test_list
    
    # 创建结果DataFrame
    result_df = df.iloc[test_indices].copy()
    result_df['Prediction'] = pred_mean
    result_df['Prediction_first'] = predictions[:, 0]
    result_df['Prediction_last'] = predictions[:, -1]
    result_df['Actual_mean'] = y_test.mean(axis=1)
    result_df['Actual_first'] = y_test[:, 0]
    result_df['Actual_last'] = y_test[:, -1]
    result_df['Error_mean'] = np.abs(result_df['Actual_mean'] - result_df['Prediction'])
    result_df['Error_first'] = np.abs(result_df['Actual_first'] - result_df['Prediction_first'])
    result_df['Error_last'] = np.abs(result_df['Actual_last'] - result_df['Prediction_last'])
    
    # 添加湿度区域标记
    result_df['Hum_region'] = pd.cut(result_df['Hum'], 
                                       bins=[0, 25, 35, 45, 55, 65, 75, 85, 100],
                                       labels=['0-25', '25-35', '35-45', '45-55', '55-65', '65-75', '75-85', '85+'])
    
    result_df = result_df.round(4)
    result_df.to_csv('results_fmlp_ltsf.csv', index=False, encoding='utf-8-sig')
    print(f"\n保存结果: results_fmlp_ltsf.csv")
    
    # ===== 按湿度区域统计 =====
    print("\n" + "=" * 60)
    print("按湿度区域统计误差")
    print("=" * 60)
    
    hum_stats = result_df.groupby('Hum_region').agg({
        'Error_mean': ['mean', 'std', 'count'],
        'Hum': 'mean'
    }).round(4)
    print(hum_stats)
    
    # ===== 保存模型 =====
    torch.save(model.state_dict(), 'fmlp_ltsf_model.pth')
    print(f"\n保存模型: fmlp_ltsf_model.pth")
    
    # ===== 保存训练信息 =====
    train_info = {
        'seq_len': seq_len,
        'pred_len': pred_len,
        'n_features': n_features,
        'feature_cols': feature_cols,
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0],
        'overall_mse': mse,
        'overall_rmse': rmse,
        'overall_mae': mae,
        'overall_r2': r2
    }
    
    return train_info


if __name__ == '__main__':
    info = main()
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n输出文件:")
    print(f"  - results_fmlp_ltsf.csv: 预测结果")
    print(f"  - fmlp_ltsf_model.pth: 模型权重")
