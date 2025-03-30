"""
用户聚类模型
使用K-Means实现区块链地址用户画像分类
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserClusteringModel:
    """
    使用K-Means的区块链地址用户画像分类模型
    """
    
    def __init__(self, model_path: Optional[str] = None, n_clusters: int = 4):
        """
        初始化用户聚类模型
        
        参数:
            model_path: 模型文件路径，如果为None则创建新模型
            n_clusters: 聚类数量
        """
        self.model = None
        self.scaler = StandardScaler()
        self.n_clusters = n_clusters
        self.feature_columns = [
            'eth_balance', 'token_count', 'total_token_value_usd', 
            'transaction_count', 'unique_interaction_addresses', 
            'avg_transaction_value', 'days_since_first_tx', 
            'outgoing_tx_ratio', 'contract_interaction_count',
            'defi_interaction_count'
        ]
        
        # 聚类标签含义
        self.cluster_names = {
            0: "低活跃小额用户",
            1: "活跃交易用户",
            2: "DeFi重度用户",
            3: "大额持币者"
        }
        
        # 尝试加载现有模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"已加载模型: {model_path}")
        else:
            logger.info(f"初始化新的K-Means聚类模型，聚类数: {n_clusters}")
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练聚类模型
        
        参数:
            data: 训练数据
            
        返回:
            包含训练性能指标的字典
        """
        logger.info("开始训练聚类模型")
        
        # 获取特征
        X = data[self.feature_columns]
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建并训练K-Means模型
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        
        self.model.fit(X_scaled)
        
        # 计算聚类指标
        labels = self.model.labels_
        inertia = self.model.inertia_
        
        # 计算每个聚类的样本数
        cluster_counts = {}
        for i in range(self.n_clusters):
            cluster_counts[i] = np.sum(labels == i)
        
        # 计算PCA用于可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        metrics = {
            "inertia": inertia,
            "cluster_counts": cluster_counts,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
        }
        
        logger.info(f"聚类模型训练完成，聚类数: {self.n_clusters}")
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        预测地址所属聚类
        
        参数:
            features: 特征数据框
            
        返回:
            聚类结果和解释
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        
        # 确保输入包含所需特征
        missing_features = [col for col in self.feature_columns if col not in features.columns]
        if missing_features:
            raise ValueError(f"输入特征缺少以下列: {missing_features}")
        
        # 获取模型所需特征
        X = features[self.feature_columns]
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测聚类
        cluster = int(self.model.predict(X_scaled)[0])
        
        # 计算到聚类中心的距离
        centroid_distances = {}
        for i in range(self.n_clusters):
            dist = np.linalg.norm(X_scaled - self.model.cluster_centers_[i])
            centroid_distances[i] = float(dist)
        
        # 解释聚类结果
        cluster_name = self.cluster_names.get(cluster, f"用户群体{cluster}")
        
        result = {
            "cluster": cluster,
            "cluster_name": cluster_name,
            "centroid_distances": centroid_distances,
            "cluster_description": self._get_cluster_description(cluster, X.iloc[0])
        }
        
        return result
    
    def _get_cluster_description(self, cluster: int, features: pd.Series) -> str:
        """
        获取聚类的描述
        
        参数:
            cluster: 聚类编号
            features: 特征值
            
        返回:
            聚类描述
        """
        # 根据聚类编号生成描述
        descriptions = {
            0: "该地址表现为低活跃小额用户，交易频率较低，持有资产较少，可能是普通持币者或新用户。",
            1: "该地址表现为活跃交易用户，有较高的交易频率和多样化的交易对象，但交易金额中等。",
            2: "该地址表现为DeFi重度用户，大量与合约交互，高比例的DeFi操作，活跃于去中心化金融生态。",
            3: "该地址表现为大额持币者，拥有较高的ETH和代币余额，交易频率中等但单笔交易金额较大。"
        }
        
        # 添加基于特征的具体描述
        additional_desc = ""
        
        if features['eth_balance'] > 10:
            additional_desc += "持有大量ETH，"
        
        if features['token_count'] > 5:
            additional_desc += "持有多种代币，"
        
        if features['transaction_count'] > 50:
            additional_desc += "交易非常活跃，"
        
        if features['defi_interaction_count'] > 10:
            additional_desc += "频繁使用DeFi服务，"
        
        # 组合描述
        base_desc = descriptions.get(cluster, f"该地址属于用户群体{cluster}，显示出独特的使用模式。")
        if additional_desc:
            return base_desc + " 具体特征包括：" + additional_desc[:-1] + "。"
        else:
            return base_desc
    
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
        
        # 保存模型和归一化器
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "n_clusters": self.n_clusters,
            "feature_columns": self.feature_columns,
            "cluster_names": self.cluster_names
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"聚类模型已保存到: {model_path}")
    
    def load_model(self, model_path: str):
        """
        从文件加载模型
        
        参数:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型和归一化器
        model_data = joblib.load(model_path)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.n_clusters = model_data["n_clusters"]
        self.feature_columns = model_data["feature_columns"]
        self.cluster_names = model_data["cluster_names"]
        
        logger.info(f"已从{model_path}加载聚类模型") 