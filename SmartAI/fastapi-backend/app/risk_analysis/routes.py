"""
风险分析API路由模块
提供风险评分和用户画像分类的API端点
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# 导入风险分析模块
from app.risk_analysis.models.risk_score import RiskScoreModel
from app.risk_analysis.models.user_clustering import UserClusteringModel
from app.risk_analysis.utils.risk_explainer import RiskExplainer
from app.risk_analysis.data.feature_extractor import FeatureExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型目录
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# 创建路由
router = APIRouter(
    prefix="/risk",
    tags=["risk"],
    responses={404: {"description": "未找到"}},
)

# 定义请求和响应模型
class AddressRequest(BaseModel):
    """地址请求模型"""
    address: str = Field(..., description="以太坊地址")


class RiskScoreResponse(BaseModel):
    """风险评分响应模型"""
    address: str = Field(..., description="以太坊地址")
    risk_score: float = Field(..., description="风险评分 (0-100)")
    risk_level: str = Field(..., description="风险等级")
    risk_description: str = Field(..., description="风险描述")
    risk_explanation: str = Field(..., description="风险解释")
    risk_factors: List[Dict[str, Any]] = Field([], description="风险因素")
    attention_points: List[str] = Field([], description="需要关注的点")
    features: Dict[str, Any] = Field({}, description="提取的特征")


class UserClusterResponse(BaseModel):
    """用户聚类响应模型"""
    address: str = Field(..., description="以太坊地址")
    cluster: int = Field(..., description="聚类ID")
    cluster_name: str = Field(..., description="聚类名称")
    cluster_description: str = Field(..., description="聚类描述")
    features: Dict[str, Any] = Field({}, description="提取的特征")


class FullAnalysisResponse(BaseModel):
    """完整分析响应模型"""
    address: str = Field(..., description="以太坊地址")
    risk_analysis: RiskScoreResponse
    user_profile: UserClusterResponse


# 初始化模型
risk_model = None
cluster_model = None
risk_explainer = RiskExplainer()
feature_extractor = FeatureExtractor()

def load_models():
    """加载最新的模型"""
    global risk_model, cluster_model
    
    try:
        # 加载风险评分模型
        risk_model_path = os.path.join(MODELS_DIR, "risk_score_model_latest.joblib")
        if os.path.exists(risk_model_path):
            risk_model = RiskScoreModel(model_path=risk_model_path)
            logger.info("已加载风险评分模型")
        else:
            # 创建新模型
            risk_model = RiskScoreModel()
            logger.warning("未找到预训练风险评分模型，将使用新模型")
        
        # 加载用户聚类模型
        cluster_model_path = os.path.join(MODELS_DIR, "user_clustering_model_latest.joblib")
        if os.path.exists(cluster_model_path):
            cluster_model = UserClusteringModel(model_path=cluster_model_path)
            logger.info("已加载用户聚类模型")
        else:
            # 创建新模型
            cluster_model = UserClusteringModel()
            logger.warning("未找到预训练用户聚类模型，将使用新模型")
            
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        # 创建新模型作为备份
        risk_model = RiskScoreModel()
        cluster_model = UserClusteringModel()


@router.get("/score/{address}", response_model=RiskScoreResponse)
async def get_risk_score(address: str):
    """
    获取地址的风险评分
    
    参数:
        address: 以太坊地址
        
    返回:
        风险评分信息
    """
    try:
        # 确保模型已加载
        if risk_model is None:
            load_models()
        
        # 提取特征
        features = feature_extractor.extract_features(address)
        
        # 创建数据框
        features_df = pd.DataFrame([features])
        if 'address' in features_df.columns:
            features_df.set_index('address', inplace=True)
        
        # 预测风险评分
        risk_score, feature_importance = risk_model.predict(features_df)
        
        # 解释风险评分
        risk_explanation = risk_explainer.explain_risk(features, risk_score)
        
        # 构建响应
        response = {
            "address": address,
            "risk_score": risk_score,
            "risk_level": risk_explanation["risk_level"]["level"],
            "risk_description": risk_explanation["risk_level"]["description"],
            "risk_explanation": risk_explanation["complete_explanation"],
            "risk_factors": risk_explanation["risk_factors"],
            "attention_points": risk_explanation["attention_points"],
            "features": features
        }
        
        return response
        
    except Exception as e:
        logger.error(f"获取风险评分时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/profile/{address}", response_model=UserClusterResponse)
async def get_user_profile(address: str):
    """
    获取地址的用户画像
    
    参数:
        address: 以太坊地址
        
    返回:
        用户画像信息
    """
    try:
        # 确保模型已加载
        if cluster_model is None:
            load_models()
        
        # 提取特征
        features = feature_extractor.extract_features(address)
        
        # 创建数据框
        features_df = pd.DataFrame([features])
        if 'address' in features_df.columns:
            features_df.set_index('address', inplace=True)
        
        # 预测聚类
        cluster_result = cluster_model.predict(features_df)
        
        # 构建响应
        response = {
            "address": address,
            "cluster": cluster_result["cluster"],
            "cluster_name": cluster_result["cluster_name"],
            "cluster_description": cluster_result["cluster_description"],
            "features": features
        }
        
        return response
        
    except Exception as e:
        logger.error(f"获取用户画像时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@router.get("/analyze/{address}", response_model=FullAnalysisResponse)
async def get_full_analysis(address: str):
    """
    获取地址的完整风险分析
    
    参数:
        address: 以太坊地址
        
    返回:
        完整的风险分析信息
    """
    try:
        # 获取风险评分
        risk_analysis = await get_risk_score(address)
        
        # 获取用户画像
        user_profile = await get_user_profile(address)
        
        # 构建响应
        response = {
            "address": address,
            "risk_analysis": risk_analysis,
            "user_profile": user_profile
        }
        
        return response
        
    except Exception as e:
        logger.error(f"获取完整分析时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}") 