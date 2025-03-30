from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .transaction import RiskLevel, TransactionType

class AddressType(str, Enum):
    """地址类型枚举"""
    EOA = "eoa"  # 外部拥有账户
    CONTRACT = "contract"  # 智能合约
    EXCHANGE = "exchange"  # 交易所
    DEFI = "defi"  # DeFi协议
    MIXER = "mixer"  # 混币器
    SCAMMER = "scammer"  # 诈骗者
    UNKNOWN = "unknown"  # 未知

class AddressTag(BaseModel):
    """地址标签模型"""
    name: str = Field(..., description="标签名称")
    type: str = Field(..., description="标签类型")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    created_at: datetime = Field(..., description="创建时间")
    source: Optional[str] = Field(None, description="数据来源")

class AddressBalance(BaseModel):
    """地址余额模型"""
    token: str = Field(..., description="代币名称")
    symbol: str = Field(..., description="代币符号")
    amount: str = Field(..., description="余额数量")
    value_usd: Optional[float] = Field(None, description="美元价值")
    last_updated: Optional[datetime] = Field(None, description="最后更新时间")

class InteractionStats(BaseModel):
    """交互统计模型"""
    address: str = Field(..., description="交互地址")
    count: int = Field(..., description="交互次数")
    volume: str = Field(..., description="交互总量")
    last_interaction: datetime = Field(..., description="最后交互时间")
    types: Dict[str, int] = Field(..., description="交互类型统计")
    risk_level: RiskLevel = Field(..., description="风险等级")

class AddressActivity(BaseModel):
    """地址活动模型"""
    date: datetime = Field(..., description="活动日期")
    tx_count: int = Field(..., description="交易数量")
    volume: str = Field(..., description="交易量")
    activities: Dict[TransactionType, int] = Field(..., description="活动类型统计")

class AddressRequest(BaseModel):
    """地址分析请求"""
    address: str = Field(..., description="要分析的区块链地址")
    
    class Config:
        schema_extra = {
            "example": {
                "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F"
            }
        }

class AddressResponse(BaseModel):
    """地址分析响应"""
    address: str = Field(..., description="区块链地址")
    type: AddressType = Field(..., description="地址类型")
    tags: List[AddressTag] = Field([], description="地址标签")
    risk_score: int = Field(..., ge=0, le=100, description="风险评分")
    risk_level: RiskLevel = Field(..., description="风险等级")
    risk_factors: List[str] = Field([], description="风险因素")
    first_tx_time: Optional[datetime] = Field(None, description="首次交易时间")
    last_tx_time: Optional[datetime] = Field(None, description="最后交易时间")
    total_tx_count: int = Field(..., description="总交易数量")
    balances: List[AddressBalance] = Field([], description="地址余额")
    top_interactions: List[InteractionStats] = Field([], description="主要交互地址")
    activity_history: List[AddressActivity] = Field([], description="活动历史")
    
    class Config:
        schema_extra = {
            "example": {
                "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                "type": "eoa",
                "tags": [
                    {
                        "name": "Binance User",
                        "type": "entity",
                        "confidence": 0.85,
                        "created_at": "2023-03-10T14:25:16Z",
                        "source": "chain_analysis"
                    }
                ],
                "risk_score": 35,
                "risk_level": "medium",
                "risk_factors": [
                    "曾与高风险地址交互",
                    "频繁使用混币服务"
                ],
                "first_tx_time": "2020-05-10T08:30:42Z",
                "last_tx_time": "2023-06-15T16:42:35Z",
                "total_tx_count": 287,
                "balances": [
                    {
                        "token": "Ethereum",
                        "symbol": "ETH",
                        "amount": "5.721",
                        "value_usd": 10532.64,
                        "last_updated": "2023-06-20T12:00:00Z"
                    },
                    {
                        "token": "USD Coin",
                        "symbol": "USDC",
                        "amount": "2500.00",
                        "value_usd": 2500.00,
                        "last_updated": "2023-06-20T12:00:00Z"
                    }
                ],
                "top_interactions": [
                    {
                        "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                        "count": 42,
                        "volume": "15,000 USDT",
                        "last_interaction": "2023-05-20T10:15:30Z",
                        "types": {"transfer": 40, "swap": 2},
                        "risk_level": "low"
                    }
                ],
                "activity_history": [
                    {
                        "date": "2023-06-01T00:00:00Z",
                        "tx_count": 12,
                        "volume": "3.5 ETH",
                        "activities": {"transfer": 8, "swap": 4}
                    }
                ]
            }
        } 