from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    """交易类型枚举"""
    TRANSFER = "transfer"
    SWAP = "swap"
    LIQUIDITY = "liquidity"
    MINT = "mint"
    BURN = "burn"
    SMART_CONTRACT = "smart_contract"
    OTHER = "other"

class TransactionStatus(str, Enum):
    """交易状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"

class RiskLevel(str, Enum):
    """风险等级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

class Transaction(BaseModel):
    """交易数据模型"""
    txHash: str = Field(..., description="交易哈希")
    blockNumber: int = Field(..., description="区块高度")
    timestamp: datetime = Field(..., description="交易时间戳")
    from_address: str = Field(..., description="发送地址")
    to_address: str = Field(..., description="接收地址")
    value: str = Field(..., description="交易金额")
    gas: str = Field(..., description="Gas用量")
    gasPrice: str = Field(..., description="Gas价格")
    status: TransactionStatus = Field(..., description="交易状态")
    type: TransactionType = Field(TransactionType.OTHER, description="交易类型")
    riskLevel: RiskLevel = Field(RiskLevel.NONE, description="风险等级")
    riskDetail: Optional[str] = Field(None, description="风险详情")
    token: Optional[str] = Field(None, description="代币名称")
    method: Optional[str] = Field(None, description="调用方法")
    metadata: Optional[Dict[str, Any]] = Field(None, description="附加元数据")

class TransactionRequest(BaseModel):
    """交易分析请求"""
    address: str = Field(..., description="要查询的地址")
    startDate: Optional[datetime] = Field(None, description="开始日期")
    endDate: Optional[datetime] = Field(None, description="结束日期")
    page: int = Field(1, ge=1, description="页码")
    limit: int = Field(10, ge=1, le=100, description="每页数量")
    type: Optional[List[TransactionType]] = Field(None, description="交易类型过滤")
    riskLevel: Optional[List[RiskLevel]] = Field(None, description="风险等级过滤")
    
    class Config:
        schema_extra = {
            "example": {
                "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                "startDate": "2023-01-01T00:00:00Z",
                "endDate": "2023-01-31T23:59:59Z",
                "page": 1,
                "limit": 20,
                "type": ["transfer", "swap"],
                "riskLevel": ["high", "medium"]
            }
        }

class TransactionResponse(BaseModel):
    """交易分析响应"""
    transactions: List[Transaction] = Field(..., description="交易列表")
    total: int = Field(..., description="总交易数")
    page: int = Field(..., description="当前页码")
    limit: int = Field(..., description="每页数量")
    totalPages: int = Field(..., description="总页数")
    summary: Optional[Dict[str, Any]] = Field(None, description="交易摘要统计")
    
    class Config:
        schema_extra = {
            "example": {
                "transactions": [
                    {
                        "txHash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "blockNumber": 12345678,
                        "timestamp": "2023-01-15T12:30:45Z",
                        "from_address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                        "to_address": "0xAbC456DEf789gHi012jKl345mNo678pQr901sTu2",
                        "value": "1.5 ETH",
                        "gas": "21000",
                        "gasPrice": "50 Gwei",
                        "status": "success",
                        "type": "transfer",
                        "riskLevel": "medium",
                        "riskDetail": "交易对象曾与高风险地址交互",
                        "token": "ETH",
                        "method": "transfer"
                    }
                ],
                "total": 156,
                "page": 1,
                "limit": 20,
                "totalPages": 8,
                "summary": {
                    "totalVolume": "125.7 ETH",
                    "highRiskCount": 5,
                    "mostFrequentInteraction": "0xAbC456DEf789gHi012jKl345mNo678pQr901sTu2",
                    "transferCount": 87,
                    "swapCount": 42
                }
            }
        } 