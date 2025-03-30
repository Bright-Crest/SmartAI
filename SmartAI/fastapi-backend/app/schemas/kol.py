from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class Investment(BaseModel):
    """DeFi投资记录"""
    protocol_name: str = Field(..., description="协议名称")
    amount: str = Field(..., description="投资金额")
    days_ago: int = Field(..., description="多少天前")
    timestamp: datetime = Field(..., description="交易时间戳")
    tx_hash: str = Field(..., description="交易哈希")
    tx_type: str = Field(..., description="交易类型")
    
    class Config:
        schema_extra = {
            "example": {
                "protocol_name": "Uniswap V3",
                "amount": "5.2 ETH",
                "days_ago": 3,
                "timestamp": "2023-06-15T10:30:25Z",
                "tx_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tx_type": "Swap"
            }
        }

class KolResponse(BaseModel):
    """KOL追踪响应"""
    wallet_address: str = Field(..., description="KOL钱包地址")
    is_known_kol: bool = Field(..., description="是否是已知KOL")
    influence_score: Optional[int] = Field(None, ge=0, le=100, description="影响力评分")
    recent_investments: List[Investment] = Field([], description="近期投资")
    active_defi_protocols: List[Dict[str, Any]] = Field([], description="活跃DeFi协议")
    
    class Config:
        schema_extra = {
            "example": {
                "wallet_address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                "is_known_kol": True,
                "influence_score": 85,
                "recent_investments": [
                    {
                        "protocol_name": "Uniswap V3",
                        "amount": "5.2 ETH",
                        "days_ago": 3,
                        "timestamp": "2023-06-15T10:30:25Z",
                        "tx_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "tx_type": "Swap"
                    },
                    {
                        "protocol_name": "Aave",
                        "amount": "10,000 USDC",
                        "days_ago": 5,
                        "timestamp": "2023-06-13T08:15:42Z",
                        "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                        "tx_type": "Deposit"
                    }
                ],
                "active_defi_protocols": [
                    {
                        "name": "Uniswap",
                        "interactions": 28,
                        "last_interaction": "2023-06-15T10:30:25Z",
                        "total_volume": "42.5 ETH"
                    },
                    {
                        "name": "Aave",
                        "interactions": 15,
                        "last_interaction": "2023-06-13T08:15:42Z",
                        "total_volume": "25,000 USDC"
                    },
                    {
                        "name": "Curve",
                        "interactions": 10,
                        "last_interaction": "2023-06-10T14:22:30Z",
                        "total_volume": "15.8 ETH"
                    }
                ]
            }
        } 