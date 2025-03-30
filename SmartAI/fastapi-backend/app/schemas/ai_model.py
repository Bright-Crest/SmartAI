#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class AddressEmbeddingRequest(BaseModel):
    """地址嵌入请求模型"""
    graph_file_path: str = Field(..., description="图文件路径")
    analyze: Optional[bool] = Field(False, description="是否分析嵌入")

    class Config:
        schema_extra = {
            "example": {
                "graph_file_path": "/path/to/address_graph.pt",
                "analyze": False
            }
        }

class SimilarityRequest(BaseModel):
    """相似度请求模型"""
    address_id: str = Field(..., description="地址ID")
    top_k: Optional[int] = Field(10, description="返回的相似地址数量")

    class Config:
        schema_extra = {
            "example": {
                "address_id": "0x1234567890abcdef1234567890abcdef12345678",
                "top_k": 10
            }
        }

class AddressTypeRequest(BaseModel):
    """地址类型预测请求模型"""
    graph_file_path: str = Field(..., description="图文件路径")

    class Config:
        schema_extra = {
            "example": {
                "graph_file_path": "/path/to/address_graph.pt"
            }
        }

class EmbeddingResponse(BaseModel):
    """嵌入响应模型"""
    embedding: List[float] = Field(..., description="嵌入向量")
    analysis: Optional[Dict[str, Any]] = Field(None, description="嵌入分析结果")

class SimilarityResponse(BaseModel):
    """相似度响应模型"""
    similar_addresses: List[Dict[str, Any]] = Field(..., description="相似地址列表")

class AddressTypeResponse(BaseModel):
    """地址类型预测响应模型"""
    address_type: str = Field(..., description="预测的地址类型")
    confidence: float = Field(..., description="预测的置信度")
    type_probabilities: Dict[str, float] = Field(..., description="各类型的概率")

class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    model_path: str = Field(..., description="模型路径")
    model_size_mb: float = Field(..., description="模型大小(MB)")
    model_config: Dict[str, Any] = Field(..., description="模型配置")
    last_modified: float = Field(..., description="最后修改时间戳")
    status: str = Field(..., description="模型状态") 