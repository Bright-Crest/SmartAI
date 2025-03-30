#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import tempfile
import json
from omegaconf import OmegaConf

# 添加AI模块路径到系统路径
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
ai_dir = os.path.join(project_dir, "AI")
sys.path.append(project_dir)
sys.path.append(ai_dir)

# 导入AI模块 - 尝试处理可能的导入路径问题
try:
    from AI.inference.inference import SmartMoneyInference
    from AI.utils.train_utils import get_device, load_checkpoint
    from AI.models.smartmoney_model import SmartMoneyModel
except ImportError:
    # 尝试相对导入
    from SmartAI.AI.inference.inference import SmartMoneyInference
    from SmartAI.AI.utils.train_utils import get_device, load_checkpoint
    from SmartAI.AI.models.smartmoney_model import SmartMoneyModel

from app.core.config import settings
from app.schemas.ai_model import (
    AddressEmbeddingRequest, 
    SimilarityRequest, 
    AddressTypeRequest,
    EmbeddingResponse, 
    SimilarityResponse, 
    AddressTypeResponse,
    ModelInfoResponse
)

# 创建路由器
router = APIRouter(
    prefix="/ai_model",
    tags=["AI模型"],
    responses={404: {"description": "未找到"}}
)

# 配置日志
logger = logging.getLogger(__name__)

# 模型配置常量
MODEL_PATH = os.path.join(ai_dir, "checkpoints/best_model.pth")
CONFIG_PATH = os.path.join(ai_dir, "configs/inference_config.yaml")

# 全局模型实例（延迟加载）
_model_instance = None

def get_model_instance():
    """获取或创建模型实例"""
    global _model_instance
    
    if _model_instance is None:
        logger.info("初始化智能模型...")
        try:
            # 加载配置
            if not os.path.exists(CONFIG_PATH):
                raise FileNotFoundError(f"找不到配置文件: {CONFIG_PATH}")
            
            config = OmegaConf.load(CONFIG_PATH)
            
            # 设置正确的模型路径
            model_config = config.get("model", {})
            model_config["path"] = MODEL_PATH
            
            # 确保使用适当的设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 创建模型实例
            _model_instance = SmartMoneyInference(
                model_path=MODEL_PATH,
                config=config,
                device=device
            )
            
            logger.info("智能模型初始化完成")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"模型初始化失败: {str(e)}"
            )
    
    return _model_instance


@router.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """获取模型信息"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404, 
            detail=f"模型文件不存在: {MODEL_PATH}"
        )
    
    # 获取模型文件信息
    model_stats = os.stat(MODEL_PATH)
    model_size = model_stats.st_size / (1024 * 1024)  # 转换为MB
    
    # 加载配置文件获取模型参数
    try:
        config = OmegaConf.load(CONFIG_PATH)
        model_config = config.get("model", {})
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        model_config = {}
    
    # 返回模型信息
    return ModelInfoResponse(
        model_path=MODEL_PATH,
        model_size_mb=round(model_size, 2),
        model_config=model_config,
        last_modified=model_stats.st_mtime,
        status="available"
    )


@router.post("/encode_address", response_model=EmbeddingResponse)
async def encode_address(request: AddressEmbeddingRequest):
    """对地址进行编码"""
    model = get_model_instance()
    
    try:
        # 检查图文件是否存在
        if not os.path.exists(request.graph_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"找不到图文件: {request.graph_file_path}"
            )
        
        # 使用模型编码地址
        embedding = model.encode_single_address(request.graph_file_path)
        
        # 分析嵌入（如果需要）
        analysis = None
        if request.analyze:
            address_id = os.path.basename(request.graph_file_path).split('.')[0]
            embeddings_dict = {address_id: embedding}
            analysis = model.analyze_embeddings(embeddings_dict)
        
        # 将嵌入转换为列表
        embedding_list = embedding.flatten().tolist()
        
        return EmbeddingResponse(
            embedding=embedding_list,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"地址编码失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"地址编码失败: {str(e)}"
        )


@router.post("/upload_and_encode", response_model=EmbeddingResponse)
async def upload_and_encode(
    file: UploadFile = File(...),
    analyze: bool = Form(False)
):
    """上传并编码图文件"""
    model = get_model_instance()
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # 使用模型编码地址
        embedding = model.encode_single_address(temp_path)
        
        # 分析嵌入（如果需要）
        analysis = None
        if analyze:
            address_id = file.filename.split('.')[0]
            embeddings_dict = {address_id: embedding}
            analysis = model.analyze_embeddings(embeddings_dict)
        
        # 将嵌入转换为列表
        embedding_list = embedding.flatten().tolist()
        
        # 删除临时文件
        os.unlink(temp_path)
        
        return EmbeddingResponse(
            embedding=embedding_list,
            analysis=analysis
        )
    except Exception as e:
        # 确保临时文件被删除
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            
        logger.error(f"上传并编码失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"上传并编码失败: {str(e)}"
        )


@router.post("/find_similar", response_model=SimilarityResponse)
async def find_similar(request: SimilarityRequest, background_tasks: BackgroundTasks):
    """查找相似地址"""
    model = get_model_instance()
    
    try:
        # 加载嵌入数据库
        reference_file = os.path.join(ai_dir, "results/embeddings.npz")
        if not os.path.exists(reference_file):
            raise HTTPException(
                status_code=404,
                detail=f"找不到参考嵌入文件: {reference_file}"
            )
        
        # 加载嵌入
        embeddings_data = np.load(reference_file, allow_pickle=True)
        address_embeddings = {str(k): v for k, v in embeddings_data.items()}
        
        # 获取查询地址的嵌入
        if request.address_id not in address_embeddings:
            raise HTTPException(
                status_code=404,
                detail=f"找不到地址ID: {request.address_id}"
            )
            
        query_embedding = address_embeddings[request.address_id]
        
        # 查找相似地址
        similar_addresses = model.find_similar_addresses(
            query_embedding,
            address_embeddings,
            top_k=request.top_k
        )
        
        # 格式化结果
        result = []
        for addr_id, similarity in similar_addresses:
            result.append({
                "address_id": addr_id,
                "similarity": float(similarity)
            })
        
        return SimilarityResponse(similar_addresses=result)
    except Exception as e:
        logger.error(f"查找相似地址失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"查找相似地址失败: {str(e)}"
        )


@router.post("/predict_type", response_model=AddressTypeResponse)
async def predict_address_type(request: AddressTypeRequest):
    """预测地址类型"""
    model = get_model_instance()
    
    try:
        # 检查图文件是否存在
        if not os.path.exists(request.graph_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"找不到图文件: {request.graph_file_path}"
            )
        
        # 标签映射
        label_map = {
            0: "普通用户",
            1: "交易所",
            2: "矿池",
            3: "智能资金",
            4: "套利者",
            5: "攻击者"
        }
        
        # 使用模型预测地址类型
        prediction = model.predict_address_type(
            request.graph_file_path,
            label_map=label_map
        )
        
        # 格式化响应
        return AddressTypeResponse(
            address_type=prediction["predicted_label"],
            confidence=prediction["confidence"],
            type_probabilities=prediction["probabilities"]
        )
    except Exception as e:
        logger.error(f"地址类型预测失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"地址类型预测失败: {str(e)}"
        ) 