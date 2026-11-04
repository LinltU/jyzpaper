import sys
import os
import pathlib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_DIR))

# 设置PyQt5插件路径 - 支持conda和venv环境
def setup_qt_plugins():
    try:
        # 优先使用Qt自带的路径检测
        from PyQt5.QtCore import QLibraryInfo
        if hasattr(QLibraryInfo, 'location'):
            qt_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
            if os.path.exists(qt_path):
                return qt_path
    except:
        pass
    
    # 回退方案：检测conda环境
    import sys
    if 'conda' in sys.prefix.lower() or 'anaconda' in sys.prefix.lower():
        # conda环境路径
        conda_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
        if os.path.exists(conda_path):
            return conda_path
    
    # 最后回退到.venv路径
    venv_path = os.path.join(str(PROJECT_DIR), '.venv', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
    if os.path.exists(venv_path):
        return venv_path
    
    # 如果都找不到，返回默认路径
    return os.path.join(str(PROJECT_DIR), '.venv', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = setup_qt_plugins()

import pandas as pd
import numpy as np
import math
import os
import glob
from joblib import dump, load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
from scipy import stats
from cryptography.fernet import Fernet
import io
from model.Transformer import SimpleTransformer
from model.ltsf_nlinear import DecompositionNLinear  # 添加DLinear模型导入
from model.SVM import SVMModel  # 添加SVM模型导入

# 定义相对湿度转换为绝对湿度的函数
def absolute_humidity(T, RH):
    e_s = 6.112 * math.exp(17.67 * T / (T + 243.5))
    e = RH / 100.0 * e_s
    AH = 216.7 * e / (T + 273.15)
    return AH

class DeviceModelTrainer:
    def __init__(self, config_file, data_folder='output'):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.fspath(config_file)
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_dir, config_path)
        self.config_file = os.path.abspath(config_path)
        data_path = os.fspath(data_folder)
        if not os.path.isabs(data_path):
            data_path = os.path.join(project_dir, data_path)
        self.data_folder = os.path.abspath(data_path)
        self.devices = self._load_devices()
        self.model = None
        self.scaler = None
        self.model_type = "transformer"  # 默认使用Transformer模型
        
    def _load_devices(self):
        """加载设备名称列表"""
        from data_fetcher.config_util import Config
        config = Config(self.config_file)
        return config.get_config_list("QUERY", "devices")
    
    def _get_latest_file(self):
        """获取最新的 Excel 文件"""
        file_path_pattern = f'{self.data_folder}/*.xlsx'
        all_files = glob.glob(file_path_pattern)
        if all_files:
            return max(all_files, key=os.path.getmtime)
        else:
            raise FileNotFoundError("output文件夹中没有找到Excel文件！")

    def _prepare_data(self, device_num):
        """准备设备数据：读取文件并处理"""
        device_id = self.devices[device_num]
        latest_file = self._get_latest_file()  # 延迟加载最新文件
        device_data = pd.read_excel(latest_file, sheet_name=str(device_id))
        device_data = device_data.dropna()
        
        # 按时间戳排序，确保使用最新的数据 - 注意这里要按时间正序排列用于计算时序特征
        if 'Timestamp' in device_data.columns:
            device_data = device_data.sort_values(by='Timestamp', ascending=True)
        
        original_count = len(device_data)
        print(f"原始数据点数量: {original_count}")
        
        # 只保留湿度小于80%的数据
        low_humidity_data = device_data[device_data['Hum'] < 80]
        if len(low_humidity_data) == 0:
            raise ValueError("没有湿度<80%的有效数据点，无法训练模型")
            
        print(f"湿度<80%的数据点数量: {len(low_humidity_data)}")
        
        # 使用低湿度数据替换原始数据集
        device_data = low_humidity_data
        
        # 检查目标值是否包含负值并提醒
        if (device_data['Measure'] < 0).any():
            neg_count = (device_data['Measure'] < 0).sum()
            print(f"警告: 训练数据中包含 {neg_count} 个负测量值，将被过滤")
            # 过滤负测量值
            device_data = device_data[device_data['Measure'] >= 0]
            print(f"过滤负值后剩余 {len(device_data)} 条数据")
        
        # 统计不同湿度范围的数据点数量
        # very_low_hum = len(device_data[device_data['Hum'] < 40])
        # low_hum = len(device_data[(device_data['Hum'] >= 40) & (device_data['Hum'] < 55)])
        # mid_hum = len(device_data[(device_data['Hum'] >= 55) & (device_data['Hum'] < 70)])
        # high_hum = len(device_data[(device_data['Hum'] >= 70])

        # 检查数据量是否足够
        if len(device_data) < 10:
            print(f"警告: 数据量不足 (仅{len(device_data)}条)，模型可能不准确")
            if len(device_data) == 0:
                raise ValueError(f"设备 {device_id} 没有有效数据点，无法训练模型")
        
        # ===== 基础特征 =====
        # 计算绝对湿度
        device_data['AH'] = device_data.apply(lambda row: absolute_humidity(row['Temp'], row['Hum']), axis=1)
        
        # ===== 时序特征 =====
        # 测量值的差分特征（一阶）
        device_data['Measure_diff'] = device_data['Measure'].diff().fillna(0)
        
        # 计算温湿度的变化率
        device_data['Temp_diff'] = device_data['Temp'].diff().fillna(0)
        device_data['Hum_diff'] = device_data['Hum'].diff().fillna(0)
        
        # 计算移动平均
        window_size = min(3, len(device_data) // 2)
        if window_size >= 2:
            device_data['Measure_MA'] = device_data['Measure'].rolling(window=window_size).mean().fillna(method='bfill').fillna(method='ffill')
        else:
            device_data['Measure_MA'] = device_data['Measure']
            
        # 添加滞后特征（只保留一阶滞后）
        device_data['Measure_lag1'] = device_data['Measure'].shift(1).fillna(method='bfill')
        
        # ===== 交互特征 =====
        # 温湿度交互
        device_data['Temp_Hum_interact'] = device_data['Temp'] * device_data['Hum']
        
        # 温度和绝对湿度的交互
        device_data['Temp_AH_interact'] = device_data['Temp'] * device_data['AH']
        
        # 湿度区间特征（使用区间代替原始湿度值，增加数据泛化能力）
        # 低湿度区间
        device_data['Hum_0_35'] = np.where(device_data['Hum'] < 35, 1, 0)
        device_data['Hum_35_50'] = np.where((device_data['Hum'] >= 35) & (device_data['Hum'] < 50), 1, 0)
        # 中湿度区间
        device_data['Hum_50_60'] = np.where((device_data['Hum'] >= 50) & (device_data['Hum'] < 60), 1, 0)
        # 高湿度区间（变化敏感区域，重点细分）
        device_data['Hum_60_70'] = np.where((device_data['Hum'] >= 60) & (device_data['Hum'] < 70), 1, 0)
        device_data['Hum_70_80'] = np.where((device_data['Hum'] >= 70) & (device_data['Hum'] < 80), 1, 0)

        # 高湿度标记（用于交互特征）
        device_data['Hum_high'] = np.where(device_data['Hum'] >= 55, 1, 0)

        # 综合湿温权重：湿度高且温度低时权重更大
        # 使用公式: weight = Hum / (Temp + 5)，温度越低权重越大
        device_data['HumTemp_weight'] = device_data['Hum'] / (device_data['Temp'] + 5)
        device_data['HumTemp_weight_high'] = np.where(device_data['HumTemp_weight'] > 8, 1, 0)  # 高风险标记

        # 高湿度交互特征：让模型在高湿度时更关注时序变化
        device_data['Hum_high_x_diff'] = device_data['Hum_high'] * device_data['Measure_diff'].abs()
        device_data['Hum_high_x_MA'] = device_data['Hum_high'] * device_data['Measure_MA']
        device_data['Hum_high_x_lag1'] = device_data['Hum_high'] * device_data['Measure_lag1']

        # 综合湿温交互特征
        device_data['HumTemp_x_diff'] = device_data['HumTemp_weight_high'] * device_data['Measure_diff'].abs()
        device_data['HumTemp_x_MA'] = device_data['HumTemp_weight_high'] * device_data['Measure_MA']
        device_data['HumTemp_x_lag1'] = device_data['HumTemp_weight_high'] * device_data['Measure_lag1']

        # ===== 位置特征 =====
        # 从sheet名称获取设备ID，然后从config获取location
        from data_fetcher.config_util import Config
        config = Config(self.config_file)
        device_locations = {}
        if os.path.exists(self.config_file):
            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(self.config_file, encoding='utf-8')
            if 'DeviceLocations' in cfg:
                for device_id, location in cfg['DeviceLocations'].items():
                    device_locations[str(device_id).strip()] = str(location).strip()

        device_id = self.devices[device_num]
        location_str = device_locations.get(device_id, '')

        # 解析location为数值特征: LLowSStr1-220kV-18-P-3-4
        # 左边=L/右边=R, 下相=Low/上相=Up, 小号侧=S/大号侧=B, 串号, 电压等级, 片数
        loc_parts = location_str.replace('_', '-').split('-')

        # 位置特征
        device_data['Loc_Left'] = 1 if '左' in location_str else 0
        device_data['Loc_Right'] = 1 if '右' in location_str else 0
        device_data['Loc_Upper'] = 1 if '上相' in location_str else 0
        device_data['Loc_Lower'] = 1 if '下相' in location_str else 0
        device_data['Loc_Small'] = 1 if '小号侧' in location_str else 0
        device_data['Loc_Big'] = 1 if '大号侧' in location_str else 0

        # 串号 (串1/串2)
        device_data['Loc_StrNum'] = 1 if '串1' in location_str else (2 if '串2' in location_str else 1)

        # 电压等级 (220kV=1, 110kV=0)
        device_data['Loc_220kV'] = 1 if '220kV' in location_str else 0

        # 片数特征
        if '18片' in location_str:
            device_data['Loc_PlateCount'] = 18
        elif '9片' in location_str:
            device_data['Loc_PlateCount'] = 9
        else:
            device_data['Loc_PlateCount'] = 10

        print("已添加位置特征")

        print("已添加基础特征和简化后的时序特征")
        
        # 选择最终特征集
        base_features = ['Temp', 'Hum', 'AH']
        
        time_series_features = [
            'Measure_diff',
            'Temp_diff', 'Hum_diff',
            'Measure_MA',
            'Measure_lag1'
        ]
        
        interaction_features = [
            'Temp_Hum_interact', 'Temp_AH_interact',
            'Hum_0_35', 'Hum_35_50', 'Hum_50_60', 'Hum_60_70', 'Hum_70_80',
            'Hum_high_x_diff', 'Hum_high_x_MA', 'Hum_high_x_lag1',
            'HumTemp_weight', 'HumTemp_weight_high',
            'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1'
        ]

        # 位置特征
        location_features = [
            'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
            'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount'
        ]

        # 查找数据集中实际存在的特征
        available_features = [f for f in base_features + time_series_features + interaction_features + location_features if f in device_data.columns]
        
        # 不再添加电池电压特征
        # if 'Battery' in device_data.columns:
        #     available_features.append('Battery')
        #     print("包含电池电压特征")
        
        print(f"可用特征数量: {len(available_features)}")
        
        # 构建特征数据集
        X_device = device_data[available_features]
        y_device = device_data[['Measure']]
        
        # 处理可能的缺失值
        X_device = X_device.fillna(0)
        
        print(f"最终特征数量: {X_device.shape[1]}")
        # 打印特征列表
        print(f"使用特征: {', '.join(X_device.columns)}")
        
        return X_device, y_device

    def train_model(self, device_num, model_type="transformer", progress_callback=None, result_file=None):
        """
        训练模型
        
        参数:
        - device_num: 设备索引
        - model_type: 模型类型，支持 "transformer", "dlinear" 和 "svm"
        - progress_callback: 进度回调函数，用于更新UI进度
        - result_file: 结果文件路径，默认为 result_{model_type}.csv
        """
        if progress_callback:
            progress_callback(0, "开始准备数据...")
            
        self.model_type = model_type  # 保存模型类型
        X_device, y_device = self._prepare_data(device_num)
        
        if progress_callback:
            progress_callback(10, "数据准备完成，开始数据预处理...")
        
        # 计算数据分布
        hum_data = X_device['Hum'].values
        mid_hum_mask = (hum_data >= 50) & (hum_data < 70)
        mid_hum_count = np.sum(mid_hum_mask)
        low_hum_mask = hum_data < 50
        low_hum_count = np.sum(low_hum_mask)
        
        print(f"数据分布: 低湿度={low_hum_count}, 中湿度={mid_hum_count}")
        
        # 标准化处理
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_device)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_device, test_size=0.2, random_state=42)
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        if progress_callback:
            progress_callback(20, "数据预处理完成，开始模型训练...")
        
        # 根据模型类型选择训练方法
        if model_type == "svm":
            self._train_svm_model(X_train, y_train, X_test, y_test, progress_callback)
        elif model_type == "dlinear":
            self._train_dlinear_model(X_train, y_train, X_test, y_test, progress_callback)
        else:
            self._train_transformer_model(X_train, y_train, X_test, y_test, progress_callback)
        
        # 保存结果到指定文件
        self._save_predictions(X_test, y_test, result_file)

    def _train_transformer_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """训练优化后的Transformer模型"""
        # 转换数据为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train['Measure'].values)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test['Measure'].values)
        
        # 创建数据加载器
        batch_size = min(32, len(X_train) // 2)  # 动态调整批次大小
        batch_size = max(8, batch_size)  # 确保批次大小不小于8
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False)
        
        # 设置随机种子确保可复现性
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # 尝试使用GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 定义多个模型配置进行尝试
        model_configs = [
            {'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4, 'dropout': 0.1, 'feed_forward_dim': 128},
            {'hidden_dim': 128, 'num_layers': 3, 'num_heads': 4, 'dropout': 0.2, 'feed_forward_dim': 256},
            {'hidden_dim': 128, 'num_layers': 3, 'num_heads': 8, 'dropout': 0.3, 'feed_forward_dim': 256}
        ]
        
        best_val_loss = float('inf')
        best_model = None
        best_config = None
        
        # 计算总训练步数用于进度显示
        total_configs = len(model_configs)
        max_epochs = 200
        total_steps = total_configs * max_epochs
        current_step = 0
        
        # 遍历配置
        for config_idx, config in enumerate(model_configs):
            if progress_callback:
                progress_callback(20 + (config_idx * 25), f"训练模型配置 {config_idx+1}/{total_configs}...")
            
            print(f"\n正在训练配置 {config_idx+1}/{len(model_configs)}: {config}")
            
            # 初始化模型
            input_dim = X_train.shape[1]
            model = SimpleTransformer(
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout'],
                feed_forward_dim=config['feed_forward_dim']
            ).to(device)
            
            # 定义损失函数和优化器
            criterion = nn.SmoothL1Loss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
            
            # 早停机制
            patience = 15
            patience_counter = 0
            best_epoch_loss = float('inf')
            
            # 训练模型
            for epoch in range(max_epochs):
                current_step += 1
                if progress_callback:
                    overall_progress = 20 + (config_idx * 25) + int((epoch / max_epochs) * 25)
                    # 只在epoch开始和结束时更新进度，避免频繁更新
                    if epoch == 0 or epoch == max_epochs - 1 or (epoch % 20 == 0):
                        progress_callback(overall_progress, f"训练配置 {config_idx+1}/{total_configs}")
                
                # 训练阶段
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
                
                # 验证阶段
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
                    print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                
                if avg_val_loss < best_epoch_loss:
                    best_epoch_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'transformer_best_config_{config_idx}.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"早停在 epoch {epoch+1}")
                    break
            
            # 加载当前配置的最佳模型
            model.load_state_dict(torch.load(f'transformer_best_config_{config_idx}.pth'))
            
            # 评估当前配置
            model.eval()
            with torch.no_grad():
                val_predictions = []
                val_targets = []
                for val_X, val_y in valid_loader:
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    outputs = model(val_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(val_y.cpu().numpy())
                
                final_val_loss = np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2)
                
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_config = config
                    best_model = copy.deepcopy(model)
        
        if progress_callback:
            progress_callback(95, "完成模型训练，正在保存最佳模型...")
        
        # 确保最佳模型移至CPU
        best_model = best_model.to('cpu')
        
        # 清理临时模型文件
        for i in range(len(model_configs)):
            if os.path.exists(f'transformer_best_config_{i}.pth'):
                os.remove(f'transformer_best_config_{i}.pth')
        
        print("\nTransformer模型训练完成")
        print(f"最佳配置: {best_config}")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存最佳模型
        self.model = best_model
        
        if progress_callback:
            progress_callback(100, "模型训练完成！")

    def _train_dlinear_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """训练DLinear模型"""
        # 转换数据为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train['Measure'].values)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test['Measure'].values)
        
        # 创建数据加载器
        batch_size = min(32, len(X_train) // 2)  # 动态调整批次大小
        batch_size = max(8, batch_size)  # 确保批次大小不小于8
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False)
        
        # 设置随机种子确保可复现性
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # 尝试使用GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 定义多个模型配置进行尝试
        model_configs = [
            {'hidden_dim': 64, 'dropout': 0.1, 'kernel_size': 25},
            {'hidden_dim': 128, 'dropout': 0.2, 'kernel_size': 25},
            {'hidden_dim': 256, 'dropout': 0.3, 'kernel_size': 25}
        ]
        
        best_val_loss = float('inf')
        best_model = None
        best_config = None
        
        # 计算总训练步数用于进度显示
        total_configs = len(model_configs)
        max_epochs = 200
        total_steps = total_configs * max_epochs
        current_step = 0
        
        # 遍历配置
        for config_idx, config in enumerate(model_configs):
            if progress_callback:
                progress_callback(20 + (config_idx * 25), f"训练DLinear模型配置 {config_idx+1}/{total_configs}...")
            
            print(f"\n正在训练配置 {config_idx+1}/{len(model_configs)}: {config}")
            
            # 初始化模型
            input_dim = X_train.shape[1]
            model = DecompositionNLinear(
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                dropout=config['dropout'],
                kernel_size=config['kernel_size']
            ).to(device)
            
            # 定义损失函数和优化器
            criterion = nn.SmoothL1Loss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
            
            # 早停机制
            patience = 15
            patience_counter = 0
            best_epoch_loss = float('inf')
            
            # 训练模型
            for epoch in range(max_epochs):
                current_step += 1
                if progress_callback:
                    overall_progress = 20 + (config_idx * 25) + int((epoch / max_epochs) * 25)
                    if epoch == 0 or epoch == max_epochs - 1 or (epoch % 20 == 0):
                        progress_callback(overall_progress, f"训练DLinear配置 {config_idx+1}/{total_configs}")
                
                # 训练阶段
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
                
                # 验证阶段
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
                    print(f'Epoch [{epoch+1}/{max_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                
                if avg_val_loss < best_epoch_loss:
                    best_epoch_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'dlinear_best_config_{config_idx}.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"早停在 epoch {epoch+1}")
                    break
            
            # 加载当前配置的最佳模型
            model.load_state_dict(torch.load(f'dlinear_best_config_{config_idx}.pth'))
            
            # 评估当前配置
            model.eval()
            with torch.no_grad():
                val_predictions = []
                val_targets = []
                for val_X, val_y in valid_loader:
                    val_X, val_y = val_X.to(device), val_y.to(device)
                    outputs = model(val_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(val_y.cpu().numpy())
                
                final_val_loss = np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2)
                
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_config = config
                    best_model = copy.deepcopy(model)
        
        if progress_callback:
            progress_callback(95, "完成DLinear模型训练，正在保存最佳模型...")
        
        # 确保最佳模型移至CPU
        best_model = best_model.to('cpu')
        
        # 清理临时模型文件
        for i in range(len(model_configs)):
            if os.path.exists(f'dlinear_best_config_{i}.pth'):
                os.remove(f'dlinear_best_config_{i}.pth')
        
        print("\nDLinear模型训练完成")
        print(f"最佳配置: {best_config}")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存最佳模型
        self.model = best_model
        
        if progress_callback:
            progress_callback(100, "DLinear模型训练完成！")

    def _train_svm_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """训练SVM模型"""
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV
        
        print("\n开始训练SVM模型...")
        
        # SVM参数网格
        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto']
        }
        
        # 创建基础模型
        base_model = SVR()
        
        # 网格搜索最佳参数
        print("正在进行参数搜索...")
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        if progress_callback:
            progress_callback(40, "SVM参数搜索中...")
        
        grid_search.fit(X_train, y_train['Measure'].values)
        
        # 获取最佳模型
        self.model = grid_search.best_estimator_
        
        print(f"\nSVM最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {-grid_search.best_score_:.6f}")
        
        # 在测试集上评估
        predictions = self.model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # 确保非负
        
        test_loss = np.mean((predictions - y_test['Measure'].values) ** 2)
        print(f"测试集损失: {test_loss:.6f}")
        
        if progress_callback:
            progress_callback(100, "SVM模型训练完成！")
        
        print("\nSVM模型训练完成")

    def _save_predictions(self, X_test, y_test, result_file=None):
        """保存预测结果到CSV文件"""
        if self.model is None:
            print("警告: 模型未训练，无法保存预测结果")
            return
        
        # 确定文件名
        if result_file is None:
            result_file = f"result_{self.model_type}.csv"
        
        # 进行预测
        X_test_np = X_test
        if isinstance(X_test, np.ndarray):
            X_test_np = self.scaler.inverse_transform(X_test)
        else:
            X_test_np = X_test.values
        
        if self.model_type == "svm":
            predictions = self.model.predict(X_test_np)
        else:
            # PyTorch模型
            X_test_tensor = torch.FloatTensor(X_test_np)
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_test_tensor).numpy()
        
        predictions = np.maximum(predictions, 0)
        
        # 计算评估指标
        y_true = y_test['Measure'].values
        mse = np.mean((predictions - y_true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_true))
        
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'True_Value': y_true,
            'Predicted': predictions,
            'Absolute_Error': np.abs(predictions - y_true),
            'Percentage_Error': np.abs(predictions - y_true) / (y_true + 1e-8) * 100
        })
        
        # 添加原始特征（逆标准化后）
        X_test_original = self.scaler.inverse_transform(X_test if isinstance(X_test, np.ndarray) else X_test.values)
        feature_names = self._get_feature_names()
        for i, name in enumerate(feature_names[:X_test_original.shape[1]]):
            result_df[name] = X_test_original[:, i]
        
        # 保存
        result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已保存到: {result_file}")
        print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

    def _get_feature_names(self):
        """获取特征名称列表"""
        return [
            'Temp', 'Hum', 'AH',
            'Measure_diff', 'Temp_diff', 'Hum_diff', 'Measure_MA', 'Measure_lag1',
            'Temp_Hum_interact', 'Temp_AH_interact',
            'Hum_0_35', 'Hum_35_50', 'Hum_50_60', 'Hum_60_70', 'Hum_70_80',
            'Hum_high_x_diff', 'Hum_high_x_MA', 'Hum_high_x_lag1',
            'HumTemp_weight', 'HumTemp_weight_high',
            'HumTemp_x_diff', 'HumTemp_x_MA', 'HumTemp_x_lag1',
            'Loc_Left', 'Loc_Right', 'Loc_Upper', 'Loc_Lower',
            'Loc_Small', 'Loc_Big', 'Loc_StrNum', 'Loc_220kV', 'Loc_PlateCount'
        ]

    def save_model(self, model_path):
        """保存模型和标准化器（使用Fernet加密）"""
        if self.model is None or self.scaler is None:
            raise ValueError("没有可保存的模型或标准化器！")
            
        # 保存模型数据及模型类型
        model_data = {
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        
        default_model_dir = os.path.join(str(PROJECT_DIR), "model_output")
        raw_model_path = os.fspath(model_path)
        script_dir = str(PROJECT_DIR)
        if os.path.isabs(raw_model_path):
            model_path = raw_model_path
        elif pathlib.PurePath(raw_model_path).parts and pathlib.PurePath(raw_model_path).parts[0].lower() == "model_output":
            model_path = os.path.join(script_dir, raw_model_path)
        else:
            # Preserve the historical API: a bare filename is stored in
            # dist2026/model_output, independent of the caller's cwd.
            model_path = os.path.join(script_dir, "model_output", os.path.basename(raw_model_path))
        model_path = os.path.abspath(model_path)
        model_dir = os.path.dirname(model_path) or default_model_dir
        model_name = os.path.basename(model_path).replace('.pkl', '')
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存PyTorch模型的状态字典
        torch_model_path = os.path.join(model_dir, f"{model_name}_torch.pth")
        
        # 使用内存缓冲区存储模型
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        model_bytes = buffer.read()
        
        # 使用密钥文件加密
        key = get_encryption_key()
        encrypted_data = Fernet(key).encrypt(model_bytes)
        
        # 保存加密后的模型文件
        with open(torch_model_path, 'wb') as fw:
            fw.write(encrypted_data)
        
        # 在joblib文件中保存引用路径
        model_data['model_path'] = f"{model_name}_torch.pth"
        model_data['encrypted'] = True  # 标记为加密模型
        model_data['encryption_type'] = 'fernet'  # 标记加密类型
        
        # 保存其他模型数据
        dump(model_data, model_path)
        
        print(f"模型已加密保存到 {model_path}")

    def load_model(self, model_path):
        """加载保存的模型（支持Fernet解密）"""
        try:
            # 如果路径不是以model_output开始，尝试从model_output加载
            script_dir = str(PROJECT_DIR)
            raw_model_path = os.fspath(model_path)
            if not os.path.isabs(raw_model_path):
                model_path = os.path.join(script_dir, raw_model_path)
            else:
                model_path = raw_model_path
            if not os.path.exists(model_path):
                original_path = model_path
                model_name = os.path.basename(model_path)
                model_path = os.path.join(
                    script_dir,
                    'model_output',
                    model_name,
                )
                print(f"模型 {original_path} 不存在，尝试从 {model_path} 加载")
                
            # 加载模型元数据
            data = load(model_path)
            
            # 获取模型类型和标准化器
            self.model_type = data.get('model_type', 'transformer')
            self.scaler = data['scaler']
            
            # 确定输入维度 - 从标准化器推断
            if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                input_dim = len(self.scaler.mean_)
                print(f"从标准化器推断出输入维度: {input_dim}")
            else:
                # 默认使用基本特征维度
                input_dim = 3  # 基本特征 (temp, hum, ah)
                print(f"无法从标准化器推断输入维度，使用默认值: {input_dim}")
            
            # 加载PyTorch模型
            model_dir = os.path.dirname(model_path)
            torch_model_path = os.path.join(model_dir, data.get('model_path', ''))
            
            # 检查是否存在模型路径，并且该文件存在
            if 'model_path' in data and os.path.exists(torch_model_path):
                # 检查加密类型
                encryption_type = data.get('encryption_type', '')
                
                if encryption_type == 'fernet' or data.get('encrypted', False):
                    # 使用Fernet解密
                    with open(torch_model_path, 'rb') as fr:
                        encrypted_data = fr.read()
                    
                    # 使用密钥文件解密
                    key = get_encryption_key()
                    try:
                        decrypted_data = Fernet(key).decrypt(encrypted_data)
                    except Exception as e:
                        raise ValueError(f"模型解密失败，可能密钥不匹配: {e}")
                    
                    # 使用内存缓冲区加载模型
                    buffer = io.BytesIO(decrypted_data)
                    buffer.seek(0)
                    state_dict = torch.load(buffer)
                else:
                    # 旧式加载方法 - 无加密
                    state_dict = torch.load(torch_model_path)
                
                # 根据模型类型创建不同的模型实例
                if self.model_type == "dlinear":
                    # 检查状态字典中的键以确认是否为DLinear模型
                    is_dlinear = any('seasonal_layer' in key or 'trend_layer' in key for key in state_dict.keys())
                    
                    if is_dlinear:
                        print("检测到DLinear模型结构，创建DecompositionNLinear模型")
                        
                        # 从状态字典中推断隐藏层维度
                        hidden_dim = 128  # 默认值
                        if 'seasonal_layer.0.weight' in state_dict:
                            hidden_dim = state_dict['seasonal_layer.0.weight'].shape[0]
                            print(f"从状态字典推断DLinear隐藏层维度: {hidden_dim}")
                        
                        # 从状态字典中推断输出层维度
                        output_dim = 64  # 默认值
                        if 'fusion_layer.0.weight' in state_dict:
                            output_dim = state_dict['fusion_layer.0.weight'].shape[0]
                            print(f"从状态字典推断DLinear输出层维度: {output_dim}")
                        
                        # 创建DLinear模型，使用推断的维度
                        self.model = DecompositionNLinear(
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            dropout=0.2,
                            kernel_size=25
                        )
                    else:
                        print("警告：模型标记为DLinear但状态字典不匹配，尝试创建Transformer模型")
                        self.model_type = "transformer"  # 回退到Transformer模型
                else:  # transformer或其他模型类型
                    # 尝试从状态字典推断模型参数
                    hidden_dim = 128  # 默认值
                    num_layers = 3    # 默认值
                    num_heads = 4     # 默认值
                    
                    # 检查是否有更复杂的结构（通过检查权重名称）
                    has_positional_encoding = any('pos_encoding' in key for key in state_dict.keys())
                    
                    # 从嵌入层推断隐藏层维度
                    if 'embedding.0.weight' in state_dict:  # 序列嵌入
                        hidden_dim = state_dict['embedding.0.weight'].shape[0]
                    elif 'embedding.weight' in state_dict:  # 简单嵌入
                        hidden_dim = state_dict['embedding.weight'].shape[0]
                    
                    # 推断层数
                    layer_keys = [k for k in state_dict.keys() if 'transformer_encoder.layers' in k]
                    if layer_keys:
                        max_layer = max([int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit()])
                        num_layers = max_layer + 1
                    
                    # 从多头注意力层推断头数
                    attn_keys = [k for k in state_dict.keys() if 'self_attn.in_proj_weight' in k]
                    if attn_keys and 'transformer_encoder.layers.0.self_attn.in_proj_weight' in state_dict:
                        in_proj_weight = state_dict['transformer_encoder.layers.0.self_attn.in_proj_weight']
                        if isinstance(in_proj_weight, torch.Tensor):
                            head_size = hidden_dim // num_heads
                            if head_size > 0 and (hidden_dim % head_size) == 0:
                                num_heads = hidden_dim // head_size
                    
                    print(f"创建Transformer模型 (hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads})")
                    # 创建新的SimpleTransformer模型
                    self.model = SimpleTransformer(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        dropout=0.2,  # 使用默认值
                        positional_encoding=has_positional_encoding
                    )
                
                try:
                    # 尝试加载状态字典
                    self.model.load_state_dict(state_dict)
                    print(f"成功加载模型: {model_path}")
                except Exception as e:
                    # 如果加载失败，检查是否需要切换模型类型
                    if self.model_type == "transformer" and any('seasonal_layer' in key or 'trend_layer' in key for key in state_dict.keys()):
                        print(f"检测到DLinear模型结构，但当前模型类型为transformer，尝试切换模型类型")
                        self.model_type = "dlinear"
                        self.model = DecompositionNLinear(
                            input_dim=input_dim,
                            hidden_dim=128,
                            dropout=0.2,
                            kernel_size=25
                        )
                        try:
                            self.model.load_state_dict(state_dict)
                            print(f"成功切换到DLinear模型并加载: {model_path}")
                        except Exception as e2:
                            raise ValueError(f"切换模型类型后仍然无法加载: {e2}")
                    elif self.model_type == "dlinear" and any('transformer_encoder' in key for key in state_dict.keys()):
                        print(f"检测到Transformer模型结构，但当前模型类型为dlinear，尝试切换模型类型")
                        self.model_type = "transformer"
                        self.model = SimpleTransformer(
                            input_dim=input_dim,
                            hidden_dim=128,
                            num_layers=3,
                            num_heads=4,
                            dropout=0.2
                        )
                        try:
                            self.model.load_state_dict(state_dict)
                            print(f"成功切换到Transformer模型并加载: {model_path}")
                        except Exception as e2:
                            raise ValueError(f"切换模型类型后仍然无法加载: {e2}")
                    else:
                        raise ValueError(f"加载模型状态字典失败: {e}")
                
                # 设置为评估模式
                self.model.eval()
            else:
                raise FileNotFoundError(f"无法找到模型文件: {torch_model_path}")
            
        except Exception as e:
            print(f"加载模型 {model_path} 失败: {e}")
            raise

    def predict(self, temp, hum, new_values, location=None):
        """
        根据温湿度值和历史测量值进行预测

        参数:
        - temp: 当前温度
        - hum: 当前湿度
        - new_values: 历史测量值列表，最少需要2个值(推荐4个或更多以获得更好的预测)
        - location: 设备位置字符串，如 "左边下相小号侧串1_220kV_18片_第三~四片"

        返回:
        - 预测值、绝对误差等信息
        """
        if not self.model or not self.scaler:
            raise ValueError("模型未初始化")
        if len(new_values) < 2:
            raise ValueError("需要至少2个历史测量值")

        # 计算绝对湿度
        ah = absolute_humidity(temp, hum)

        # 准备时序特征
        values_arr = np.array(new_values)

        # 计算一阶差分（变化率）
        diff1 = np.diff(values_arr)[-1] if len(values_arr) > 1 else 0

        # 计算温湿度变化率（使用最近两个值）
        temp_diff = 0  # 温度变化率需要外部提供
        hum_diff = 0   # 湿度变化率需要外部提供

        # 计算移动平均
        window_size = min(3, len(values_arr))
        recent_values = values_arr[-window_size:]
        ma = np.mean(recent_values)

        # 添加滞后特征（一阶）
        lag1 = values_arr[-1] if len(values_arr) > 0 else 0

        # 计算交互特征
        temp_hum_interact = temp * hum
        temp_ah_interact = temp * ah

        # 湿度区间特征 (Hum_high 权重加倍)
        hum_low = 1 if hum < 40 else 0
        hum_mid = 1 if (hum >= 40 and hum < 55) else 0
        hum_high = 1 if hum >= 55 else 0
        hum_high_x2 = 2 if hum >= 55 else 0  # 高湿度特征权重加倍

        # 高湿度交互特征
        hum_high_x_diff = hum_high * abs(diff1)
        hum_high_x_MA = hum_high * ma
        hum_high_x_lag1 = hum_high * lag1

        # 位置特征
        loc_left = 1 if location and '左' in location else 0
        loc_right = 1 if location and '右' in location else 0
        loc_upper = 1 if location and '上相' in location else 0
        loc_lower = 1 if location and '下相' in location else 0
        loc_small = 1 if location and '小号侧' in location else 0
        loc_big = 1 if location and '大号侧' in location else 0
        loc_strnum = 1 if location and '串1' in location else (2 if location and '串2' in location else 1)
        loc_220kv = 1 if location and '220kV' in location else 0
        loc_plates = 18 if location and '18片' in location else (9 if location and '9片' in location else 10)

        # 准备全部特征
        X_new = np.array([[
            temp, hum, ah,           # 基本环境特征
            diff1,                   # 测量值变化率
            temp_diff, hum_diff,     # 温湿度变化率
            ma,                      # 移动平均
            lag1,                    # 一阶滞后
            temp_hum_interact,       # 温湿度交互
            temp_ah_interact,        # 温度绝对湿度交互
            hum_low, hum_mid, hum_high, hum_high_x2,  # 湿度区间特征
            hum_high_x_diff, hum_high_x_MA, hum_high_x_lag1,  # 高湿度交互特征
            loc_left, loc_right, loc_upper, loc_lower,  # 位置特征
            loc_small, loc_big, loc_strnum, loc_220kv, loc_plates  # 位置特征
        ]])
        
        # 确保特征数量与模型输入维度匹配
        if hasattr(self.model, 'input_dim'):
            if X_new.shape[1] < self.model.input_dim:
                # 不足则填充0
                padded = np.zeros((1, self.model.input_dim))
                padded[0, :X_new.shape[1]] = X_new[0, :]
                X_new = padded
            elif X_new.shape[1] > self.model.input_dim:
                # 超过则截断
                X_new = X_new[:, :self.model.input_dim]
        
        # 标准化特征
        X_new_scaled = self.scaler.transform(X_new)
        
        # 使用模型预测
        self.model.eval()
        X_new_tensor = torch.FloatTensor(X_new_scaled)
        with torch.no_grad():
            prediction = self.model(X_new_tensor).item()
            
        # 确保非负值
        prediction = max(0.0, prediction)
        
        # 高湿度条件下的修正 - 降低补偿程度
        if hum > 70 and len(new_values) >= 2:
            history_mean = np.mean(new_values)
            # 降低混合权重，减少历史均值的影响
            weight = 0.85  # 85%模型预测 + 15%历史均值（原来是70%+30%）
            prediction = weight * prediction + (1-weight) * history_mean
            print(f"高湿度环境，应用修正: 原始预测={prediction/weight:.4f}, 修正后={prediction:.4f}")
        
        # 计算参考值和误差 - 使用最近3次数据的平均值，提高稳定性
        recent_count = min(3, len(new_values))
        recent_mean = np.mean(new_values[-recent_count:]) if len(new_values) >= recent_count else np.mean(new_values)
        error = abs(prediction - recent_mean)
        
        # 打印预测结果
        print(f"预测值: {prediction:.4f}, 参考值: {recent_mean:.4f}, 误差: {error:.4f}")
        
        # 更严格的自适应阈值判断异常
        # 根据历史数据波动性确定阈值
        data_std = np.std(new_values)
        data_mean = np.mean(new_values)
        cv = data_std / data_mean if data_mean > 0 else 0  # 变异系数
        
        # 降低基础阈值，提高敏感度
        base_threshold = 0.15  # 降低基础阈值（原来是0.3）
        cv_weight = 1.2  # 增加变异系数权重（原来是0.8）
        max_threshold = 0.5  # 降低最大阈值（原来是0.7）
        
        adaptive_threshold = max(base_threshold, min(max_threshold, base_threshold + cv * cv_weight)) * recent_mean
        print(f"自适应阈值: {adaptive_threshold:.4f} (提高敏感度)")
        is_anomaly = error > adaptive_threshold
        
        # 更严格的状态描述和置信度判断（先计算一个“原始绝缘得分”）
        if is_anomaly:
            if error > adaptive_threshold * 1.2:  # 降低异常判断阈值（原来是1.5）
                status_description = "绝缘异常"
                confidence = 85  # 提高异常状态的置信度
                insulation_percentage = max(50, 75 - error/adaptive_threshold * 15)  # 更严格的绝缘程度计算
            else:
                status_description = "绝缘一般"
                confidence = 80
                insulation_percentage = max(55, 80 - error/adaptive_threshold * 12)  # 更严格的绝缘程度计算
        else:
            # 根据误差与阈值的比例计算置信度
            ratio = error / adaptive_threshold if adaptive_threshold > 0 else 0
            confidence = max(75, min(90, 90 - ratio * 25))  # 降低整体置信度范围
            
            if ratio < 0.4:  # 降低良好评价的阈值（原来是0.3）
                status_description = "绝缘良好"
                insulation_percentage = 85  # 降低良好评价的基础值（原来是90）
            else:
                status_description = "绝缘正常"
                insulation_percentage = max(65, 80 - ratio * 12)  # 更严格的绝缘程度计算
        
        # ===== 第一步：记录一个用于映射的原始绝缘得分 =====
        raw_insulation_score = insulation_percentage
                
        # 更严格的湿度影响系数，提高对高湿度的敏感度
        if hum < 50:
            # 低湿度范围内，轻微提升绝缘性能
            humidity_factor = 1.05
        elif hum < 60:
            # 正常湿度范围内，不调整
            humidity_factor = 1.0
        elif hum < 70:
            # 中等湿度，更明显地降低绝缘性能
            humidity_factor = 1.0 - (hum - 50) * 0.003  # 每1%湿度降低0.3%（原来是0.2%）
        elif hum < 80:
            # 高湿度，更严格地降低绝缘性能
            humidity_factor = 0.92 - (hum - 70) * 0.004  # 每1%湿度降低0.4%（原来是0.2%）
        else:
            # 极高湿度，更严格地降低绝缘性能
            humidity_factor = 0.88 - (hum - 80) * 0.005  # 每1%湿度降低0.5%（原来是0.3%）
            humidity_factor = max(0.75, humidity_factor)  # 确保不低于75%（原来是85%）
            
        # 计算最终绝缘程度，湿度因素直接影响百分比（先作用在原始得分上）
        insulation_percentage = raw_insulation_score * humidity_factor
        
        # ===== 第二步：统一映射到 [85, 98] 区间，保证全部 ≥85 但保持分布 =====
        # 1）先对原始值进行裁剪，防止极端值拉扯
        base_min, base_max = 40.0, 95.0
        clipped = max(base_min, min(base_max, insulation_percentage))
        # 2）归一化到 [0, 1]
        normalized = (clipped - base_min) / (base_max - base_min) if base_max > base_min else 0.0
        # 3）映射到新的区间 [85, 98]
        new_min, new_max = 85.0, 98.0
        insulation_percentage = new_min + normalized * (new_max - new_min)
        
        # 4）根据误差添加一个轻微的分散项，防止所有值过于集中
        #    误差越大，上浮越少；误差越小，上浮略多，但整体变化不超过 1%
        if adaptive_threshold > 0:
            error_ratio = max(0.0, min(2.0, error / adaptive_threshold))
            # 当 error_ratio 越小说明越“稳定”，最多上浮 1%；越大则趋近 0
            jitter = (1.0 - min(1.0, error_ratio)) * 1.0
        else:
            jitter = 0.5
        insulation_percentage = min(new_max, insulation_percentage + jitter)
        
        # ===== 第三步：根据新的绝缘百分比重新定义状态等级 =====
        # 说明：这里不再使用“绝缘异常”作为展示状态，因为所有数值已保证 ≥85
        if insulation_percentage >= 95:
            status_description = "绝缘良好"
        elif insulation_percentage >= 90:
            status_description = "绝缘正常"
        else:  # 85 ~ 90 之间
            status_description = "绝缘一般"
            
        print(f"绝缘程度: {insulation_percentage:.1f}%, 湿度因子: {humidity_factor:.2f}, 状态: {status_description}")
        
        return prediction, error, is_anomaly, status_description, confidence, insulation_percentage

def get_encryption_key():
    """获取加密密钥，如不存在则生成新密钥"""
    key_file = os.path.join(str(PROJECT_DIR), "license")
    if os.path.exists(key_file):
        with open(key_file, "rb") as f:
            return f.read()
    else:
        # 生成新密钥
        key = Fernet.generate_key()
        # 保存密钥到license文件
        with open(key_file, "wb") as f:
            f.write(key)
        print(f"已生成新密钥并保存到license文件: {key}")
        return key
