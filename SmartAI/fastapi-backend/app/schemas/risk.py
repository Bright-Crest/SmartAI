from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class UserCategory(str, Enum):
    """用户类别枚举"""
    NORMAL = "普通用户"
    HIGH_FREQUENCY = "高频交易者"
    HIGH_RISK = "高风险用户"
    UNKNOWN = "未知类型"

class RiskReason(BaseModel):
    """风险原因说明"""
    description: str = Field(..., description="风险原因描述")
    severity: float = Field(..., ge=0, le=1, description="严重程度 (0-1)")
    type: str = Field(..., description="风险类型")

class RiskRequest(BaseModel):
    """风险分析请求"""
    address: str = Field(..., description="要分析的区块链地址")
    include_transactions: bool = Field(False, description="是否包含交易详情")
    
    class Config:
        schema_extra = {
            "example": {
                "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                "include_transactions": True
            }
        }

class RiskScoreResponse(BaseModel):
    """风险评分响应"""
    address: str = Field(..., description="分析的区块链地址")
    risk_score: int = Field(..., ge=0, le=100, description="风险评分(0-100)")
    risk_level: RiskLevel = Field(..., description="风险等级")
    risk_reasons: List[str] = Field([], description="风险原因列表")
    user_category: UserCategory = Field(..., description="用户画像类别")
    details: Optional[Dict[str, Any]] = Field(None, description="详细分析数据") 
    
    class Config:
        schema_extra = {
            "example": {
                "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                "risk_score": 78,
                "risk_level": "high",
                "risk_reasons": [
                    "⚠️ 该地址与3个已知诈骗地址交互",
                    "🚨 该地址70%交易涉及DEX",
                    "🔴 资金来源60%来自Tornado Cash"
                ],
                "user_category": "高风险用户",
                "details": None
            }
        } 