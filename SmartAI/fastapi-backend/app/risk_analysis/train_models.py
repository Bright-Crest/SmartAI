"""
风险分析模型训练脚本
用于训练风险评分和用户聚类模型
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# 设置项目根目录导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# 导入风险分析模块
from app.risk_analysis.models.risk_score import RiskScoreModel
from app.risk_analysis.models.user_clustering import UserClusteringModel
from app.risk_analysis.data.feature_extractor import FeatureExtractor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 模型保存目录
MODELS_DIR = os.path.join(current_dir, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train_risk_score_model():
    """训练风险评分模型"""
    logger.info("开始训练风险评分模型...")
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor()
    
    # 获取训练数据
    training_data = feature_extractor.get_training_data()
    logger.info(f"已获取训练数据，样本数量: {len(training_data)}")
    
    # 创建并训练模型
    model = RiskScoreModel()
    metrics = model.train(training_data)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"risk_score_model_{timestamp}.joblib")
    model.save_model(model_path)
    
    logger.info(f"风险评分模型训练完成，性能指标: {metrics}")
    logger.info(f"模型已保存到: {model_path}")
    
    # 创建最新模型的软链接
    latest_path = os.path.join(MODELS_DIR, "risk_score_model_latest.joblib")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(model_path, latest_path)
    
    return model, metrics

def train_user_clustering_model():
    """训练用户聚类模型"""
    logger.info("开始训练用户聚类模型...")
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor()
    
    # 获取训练数据
    training_data = feature_extractor.get_training_data()
    logger.info(f"已获取训练数据，样本数量: {len(training_data)}")
    
    # 创建并训练模型
    model = UserClusteringModel(n_clusters=4)
    metrics = model.train(training_data)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"user_clustering_model_{timestamp}.joblib")
    model.save_model(model_path)
    
    logger.info(f"用户聚类模型训练完成，性能指标: {metrics}")
    logger.info(f"模型已保存到: {model_path}")
    
    # 创建最新模型的软链接
    latest_path = os.path.join(MODELS_DIR, "user_clustering_model_latest.joblib")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(model_path, latest_path)
    
    return model, metrics

if __name__ == "__main__":
    logger.info("=== 开始风险分析模型训练 ===")
    
    try:
        # 训练风险评分模型
        risk_model, risk_metrics = train_risk_score_model()
        
        # 训练用户聚类模型
        cluster_model, cluster_metrics = train_user_clustering_model()
        
        logger.info("=== 模型训练完成 ===")
        
    except Exception as e:
        logger.error(f"模型训练出错: {str(e)}")
        raise 