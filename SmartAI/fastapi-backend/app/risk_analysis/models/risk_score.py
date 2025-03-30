"""
风险评分模型
使用XGBoost实现区块链地址风险评分
"""

import os
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskScoreModel:
    """
    使用XGBoost的区块链地址风险评分模型
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化风险评分模型
        
        参数:
            model_path: 模型文件路径，如果为None则创建新模型
        """
        self.model = None
        self.feature_columns = [
            'eth_balance', 'token_count', 'has_usdt', 'has_usdc', 
            'total_token_value_usd', 'transaction_count', 
            'unique_interaction_addresses', 'avg_transaction_value',
            'max_transaction_value', 'days_since_first_tx', 
            'days_since_last_tx', 'outgoing_tx_ratio',
            'contract_interaction_count', 'defi_interaction_count',
            'high_risk_interaction_count'
        ]
        
        # 尝试加载现有模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"已加载模型: {model_path}")
        else:
            logger.info("初始化新的XGBoost风险评分模型")
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        训练风险评分模型
        
        参数:
            data: 训练数据
            test_size: 测试集比例
            random_state: 随机种子
            
        返回:
            包含训练性能指标的字典
        """
        logger.info("开始训练风险评分模型")
        
        X = data[self.feature_columns]
        y = data['risk_label']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 设置XGBoost参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        
        # 训练模型
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 计算性能指标
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob)
        }
        
        logger.info(f"模型训练完成，性能指标: {metrics}")
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        预测地址风险评分
        
        参数:
            features: 特征数据框
            
        返回:
            风险评分和特征重要性
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        # 确保输入包含所需特征
        missing_features = [col for col in self.feature_columns if col not in features.columns]
        if missing_features:
            raise ValueError(f"输入特征缺少以下列: {missing_features}")
        
        # 获取模型所需特征
        X = features[self.feature_columns]
        
        # 预测概率
        risk_prob = self.model.predict_proba(X)[0, 1]
        
        # 计算特征重要性
        feature_importance = {}
        for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
            feature_importance[feature] = float(importance)
        
        # 按重要性排序
        feature_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda item: item[1], 
            reverse=True
        ))
        
        return float(risk_prob * 100), feature_importance
    
    def save_model(self, model_path: str):
        """
        保存模型到文件
        
        参数:
            model_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        directory = os.path.dirname(model_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        joblib.dump(self.model, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path: str):
        """
        从文件加载模型
        
        参数:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"已从{model_path}加载模型") 