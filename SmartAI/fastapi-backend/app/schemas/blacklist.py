from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import datetime

class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Transaction(BaseModel):
    """交易信息"""
    to: str = Field(..., description="接收方地址")
    value: str = Field(..., description="交易金额")
    risk: RiskLevel = Field(..., description="风险等级") 
    timestamp: Optional[datetime.datetime] = Field(None, description="交易时间")
    tx_hash: Optional[str] = Field(None, description="交易哈希")
    
    class Config:
        schema_extra = {
            "example": {
                "to": "0xdef2",
                "value": "1.5 ETH",
                "risk": "medium",
                "timestamp": "2023-03-15T14:30:45Z",
                "tx_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            }
        }

class BlacklistResponse(BaseModel):
    """黑名单追踪响应"""
    address: str = Field(..., description="分析的黑名单地址")
    is_blacklisted: bool = Field(..., description="是否在已知黑名单中")
    risk_score: Optional[int] = Field(None, ge=0, le=100, description="风险评分(0-100)")
    recent_transactions: List[Transaction] = Field([], description="最近交易列表") 
    
    class Config:
        schema_extra = {
            "example": {
                "address": "0x5678",
                "is_blacklisted": True,
                "risk_score": 90,
                "recent_transactions": [
                    {"to": "0xabc1", "value": "3 ETH", "risk": "high", "timestamp": "2023-03-15T10:30:45Z"},
                    {"to": "0xdef2", "value": "1.5 ETH", "risk": "medium", "timestamp": "2023-03-14T21:45:12Z"}
                ]
            }
        } 