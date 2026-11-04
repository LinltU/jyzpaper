"""
SVM回归模型
用于绝缘子泄漏电流预测
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle


class SVMModel:
    """SVM回归模型包装类"""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        初始化SVM模型
        
        Args:
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C: 正则化参数
            epsilon: epsilon-tube宽度
            gamma: RBF核的gamma参数 ('scale', 'auto' 或数值)
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        """
        训练SVM模型
        
        Args:
            X: 特征数组 (n_samples, n_features)
            y: 目标值数组 (n_samples,)
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建并训练模型
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            cache_size=500
        )
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """
        使用训练好的模型进行预测
        
        Args:
            X: 特征数组 (n_samples, n_features)
            
        Returns:
            predictions: 预测值数组
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测
        predictions = self.model.predict(X_scaled)
        
        # 确保非负
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_params(self):
        """获取模型参数"""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma
        }
    
    def save(self, filepath):
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'params': self.get_params()
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(**data['params'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        return instance
