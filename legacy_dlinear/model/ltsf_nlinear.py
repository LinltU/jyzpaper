import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    移动平均块，用于突出时间序列的趋势
    """
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 在时间序列的两端进行填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    序列分解块，将时间序列分解为季节性和趋势性成分
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class NLinearModel(nn.Module):
    """
    LTSF-NLinear模型实现
    基于论文 "Are Transformers Effective for Time Series Forecasting?"
    
    使用简单线性层进行预测，但具有非线性变换和归一化处理
    """
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2):
        """
        初始化LTSF-NLinear模型
        
        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层维度
        - dropout: Dropout比率
        """
        super(NLinearModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # 使用GELU激活函数，在时间序列任务中表现较好
            nn.Dropout(dropout)
        )
        
        # 预测层 - 简单但高效的线性映射
        self.prediction_layer = nn.Linear(hidden_dim, 1)
        
        # 特征间交互层 - 捕捉不同特征之间的关系
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状为 [batch_size, input_dim] 或 [input_dim]
        
        返回:
        - 预测值
        """
        # 处理单个样本输入
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 特征交互 - 类似于attention机制，但更轻量级
        interaction = self.interaction_layer(x)
        x = x * interaction  # 对输入特征进行加权
        
        # 特征处理
        h = self.feature_layer(x)
        
        # 线性预测
        y_pred = self.prediction_layer(h)
        
        # 确保预测值为正值
        y_pred = F.softplus(y_pred)
        
        return y_pred.squeeze(-1)


class DecompositionNLinear(nn.Module):
    """
    改进的LTSF模型，结合了序列分解和特征交互
    基于DLinear模型和NLinear的优点
    """
    def __init__(self, input_dim, hidden_dim=512, dropout=0.2, kernel_size=25):
        """
        初始化DecompositionNLinear模型
        
        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层维度
        - dropout: Dropout比率
        - kernel_size: 移动平均核的大小
        """
        super(DecompositionNLinear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 序列分解器
        self.decomposition = series_decomp(kernel_size)
        
        # 季节性成分处理
        self.seasonal_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 趋势性成分处理
        self.trend_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 特征间交互层
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 合并层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保输出为正值
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状为 [batch_size, input_dim] 或 [input_dim]
        
        返回:
        - 预测值
        """
        # 处理单个样本输入
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 为序列分解准备形状
        batch_size = x.size(0)
        x_reshaped = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # 应用序列分解
        seasonal, trend = self.decomposition(x_reshaped)
        
        # 将形状恢复为 [batch_size, input_dim]
        seasonal = seasonal.squeeze(1)
        trend = trend.squeeze(1)
        
        # 特征交互
        seasonal_interaction = self.interaction_layer(seasonal)
        trend_interaction = self.interaction_layer(trend)
        seasonal = seasonal * seasonal_interaction
        trend = trend * trend_interaction
        
        # 分别处理季节性和趋势性成分
        seasonal_features = self.seasonal_layer(seasonal)
        trend_features = self.trend_layer(trend)
        
        # 合并特征
        combined_features = torch.cat([seasonal_features, trend_features], dim=1)
        
        # 最终预测
        y_pred = self.fusion_layer(combined_features)
        
        return y_pred.squeeze(-1) 