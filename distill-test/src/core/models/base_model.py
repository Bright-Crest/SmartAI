from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call
from utils.config import config

class BaseModel(ABC):
    def __init__(self, model_type, model_name, model_params):
        self.model_type = model_type  # 'teacher' 或 'student'
        self.model_name = model_name
        self.model_params = model_params
        self.model = None
    
    @abstractmethod
    def build(self):
        """构建模型"""
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """模型预测"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """预测概率"""
        pass
    
    @log_function_call
    def save(self, filename=None):
        """保存模型"""
        if filename is None:
            filename = f"{self.model_type}_{self.model_name}.joblib"
        
        save_path = config.models_dir / filename
        try:
            joblib.dump(self.model, save_path)
            logger.info(f"模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")
            raise
    
    @log_function_call
    def load(self, filename=None):
        """加载模型"""
        if filename is None:
            filename = f"{self.model_type}_{self.model_name}.joblib"
        
        load_path = config.models_dir / filename
        try:
            self.model = joblib.load(load_path)
            logger.info(f"模型已从 {load_path} 加载")
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
    
    def get_model(self):
        """获取底层模型实例"""
        return self.model
    
    @abstractmethod
    def get_model_size(self):
        """获取模型大小"""
        pass