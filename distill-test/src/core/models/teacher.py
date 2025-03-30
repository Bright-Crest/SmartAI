import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call
from .model_registry import ModelRegistry
from .base_model import BaseModel

class TeacherModel(BaseModel):
    @log_function_call
    def build(self):
        """构建教师模型"""
        try:
            # 使用模型注册表获取模型类
            model_class = ModelRegistry.get_model_class("teacher", self.model_name)
            
            # 实例化模型
            # 对PyTorch模型进行特殊处理
            if self.model_name.startswith("pytorch_"):
                # 如果是PyTorch模型，使用model_type和model_name参数初始化
                self.model = model_class(
                    model_type=self.model_type,
                    model_name=self.model_name,
                    model_params=self.model_params
                )
            else:
                # 非PyTorch模型，直接使用参数字典初始化
                self.model = model_class(**self.model_params)
            
            logger.info(f"已创建教师模型 {self.model_name}")
            return self
        except Exception as e:
            logger.error(f"创建教师模型失败: {str(e)}")
            raise
    
    @log_function_call
    def train(self, X, y, **kwargs):
        """训练教师模型"""
        if self.model is None:
            self.build()
        
        eval_set = kwargs.pop('eval_set', None)
        
        # 移除模型fit方法不支持的参数
        # XGBoost的fit方法不支持random_state，这个参数应该在模型初始化时使用
        for param in ['random_state']:
            kwargs.pop(param, None)
        
        try:
            # 针对PyTorch模型的特殊处理
            if self.model_name.startswith("pytorch_"):
                # PyTorch模型已经在类中实现了自己的训练逻辑
                self.model.train(X, y, **kwargs)
            elif eval_set:
                if hasattr(self.model, 'fit') and 'eval_set' in self.model.fit.__code__.co_varnames:
                    self.model.fit(X, y, eval_set=eval_set, **kwargs)
                else:
                    logger.warning(f"模型 {self.model_name} 不支持eval_set参数，将忽略验证集")
                    self.model.fit(X, y, **kwargs)
            else:
                self.model.fit(X, y, **kwargs)
            logger.info("教师模型训练完成")
        except Exception as e:
            logger.error(f"教师模型训练失败: {str(e)}")
            raise
    
    @log_function_call
    def predict(self, X):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    @log_function_call
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        try:
            probs = self.model.predict_proba(X)
            
            # 确保返回的是二维概率数组
            if probs.ndim == 1:
                # 如果是一维数组（通常是二分类的正类概率），转换为二维
                probs = np.column_stack((1.0 - probs, probs))
                
            return probs
        except (AttributeError, NotImplementedError):
            # 如果模型不支持概率预测，则使用独热编码
            logger.warning("模型不支持概率预测，将使用独热编码")
            y_pred = self.predict(X)
            n_classes = len(np.unique(y_pred))
            return np.eye(n_classes)[y_pred]
    
    @log_function_call
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        try:
            # 尝试不同方法获取特征重要性
            if hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            elif hasattr(self.model, 'feature_importance'):
                return self.model.feature_importance()
            elif hasattr(self.model, 'coef_'):
                return np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
            else:
                logger.warning(f"模型 {self.model_name} 不支持特征重要性")
                return None
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            return None
    
    @log_function_call
    def get_model_size(self):
        """获取模型大小"""
        if self.model is None:
            return 0
        
        try:
            # 根据不同模型类型获取大小
            if self.model_name == "xgboost":
                return len(self.model.get_booster().get_dump())
            elif self.model_name == "random_forest":
                return len(self.model.estimators_)
            elif self.model_name == "lightgbm":
                return self.model.num_trees()
            elif self.model_name == "catboost":
                return len(self.model.get_tree_count())
            elif hasattr(self.model, 'coef_'):
                return np.prod(self.model.coef_.shape)
            else:
                # 尝试使用通用方法估计大小
                import sys
                return sys.getsizeof(self.model) / 1024  # 大小（KB）
        except Exception as e:
            logger.warning(f"获取模型大小失败: {str(e)}")
            return 0