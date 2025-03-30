from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class TimeRange(str, Enum):
    """时间范围枚举"""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"

class NetworkMetric(BaseModel):
    """网络指标模型"""
    name: str = Field(..., description="指标名称")
    value: Union[int, float, str] = Field(..., description="指标值")
    change: Optional[float] = Field(None, description="变化百分比")
    trend: Optional[str] = Field(None, description="趋势方向 (up/down/stable)")

class TimeSeriesPoint(BaseModel):
    """时间序列数据点"""
    timestamp: datetime = Field(..., description="时间戳")
    value: float = Field(..., description="数值")

class TimeSeriesData(BaseModel):
    """时间序列数据模型"""
    name: str = Field(..., description="数据名称")
    data: List[TimeSeriesPoint] = Field(..., description="数据点列表")

class RiskDistribution(BaseModel):
    """风险分布模型"""
    high: int = Field(..., description="高风险交易数量")
    medium: int = Field(..., description="中等风险交易数量")
    low: int = Field(..., description="低风险交易数量")
    none: int = Field(..., description="无风险交易数量")

class PopularEntity(BaseModel):
    """热门实体模型"""
    address: str = Field(..., description="地址")
    name: Optional[str] = Field(None, description="名称")
    type: str = Field(..., description="类型")
    interaction_count: int = Field(..., description="交互次数")
    volume: str = Field(..., description="交易量")

class DashboardRequest(BaseModel):
    """仪表盘数据请求"""
    time_range: TimeRange = Field(TimeRange.WEEK, description="时间范围")
    network: Optional[str] = Field(None, description="区块链网络")
    
    class Config:
        schema_extra = {
            "example": {
                "time_range": "week", 
                "network": "ethereum"
            }
        }

class DashboardResponse(BaseModel):
    """仪表盘数据响应"""
    key_metrics: List[NetworkMetric] = Field(..., description="关键指标")
    transaction_volume: TimeSeriesData = Field(..., description="交易量时间序列")
    active_addresses: TimeSeriesData = Field(..., description="活跃地址时间序列")
    risk_distribution: RiskDistribution = Field(..., description="风险分布")
    top_exchanges: List[PopularEntity] = Field(..., description="热门交易所")
    top_defi_protocols: List[PopularEntity] = Field(..., description="热门DeFi协议")
    recent_high_risk: List[Dict[str, Any]] = Field(..., description="最近高风险交易")
    
    class Config:
        schema_extra = {
            "example": {
                "key_metrics": [
                    {"name": "总交易量", "value": "125.7 ETH", "change": 5.2, "trend": "up"},
                    {"name": "活跃地址数", "value": 1250, "change": -2.1, "trend": "down"},
                    {"name": "平均交易额", "value": "0.35 ETH", "change": 1.5, "trend": "up"},
                    {"name": "高风险交易比例", "value": "12%", "change": -0.5, "trend": "down"}
                ],
                "transaction_volume": {
                    "name": "每日交易量",
                    "data": [
                        {"timestamp": "2023-06-01T00:00:00Z", "value": 58.4},
                        {"timestamp": "2023-06-02T00:00:00Z", "value": 62.1},
                        {"timestamp": "2023-06-03T00:00:00Z", "value": 57.9}
                    ]
                },
                "active_addresses": {
                    "name": "每日活跃地址",
                    "data": [
                        {"timestamp": "2023-06-01T00:00:00Z", "value": 570},
                        {"timestamp": "2023-06-02T00:00:00Z", "value": 620},
                        {"timestamp": "2023-06-03T00:00:00Z", "value": 590}
                    ]
                },
                "risk_distribution": {
                    "high": 125,
                    "medium": 350,
                    "low": 1200,
                    "none": 2500
                },
                "top_exchanges": [
                    {
                        "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                        "name": "Binance",
                        "type": "exchange",
                        "interaction_count": 3250,
                        "volume": "15,000 ETH"
                    },
                    {
                        "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                        "name": "Coinbase",
                        "type": "exchange",
                        "interaction_count": 2840,
                        "volume": "12,300 ETH"
                    }
                ],
                "top_defi_protocols": [
                    {
                        "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                        "name": "Uniswap V2",
                        "type": "defi",
                        "interaction_count": 5420,
                        "volume": "32,500 ETH"
                    },
                    {
                        "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                        "name": "Aave",
                        "type": "defi",
                        "interaction_count": 3150,
                        "volume": "18,900 ETH"
                    }
                ],
                "recent_high_risk": [
                    {
                        "txHash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "from": "0x123abc...",
                        "to": "0x456def...",
                        "value": "10.5 ETH",
                        "risk_factors": ["交易对象是已知混币器"]
                    },
                    {
                        "txHash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                        "from": "0x789ghi...",
                        "to": "0xjkl012...",
                        "value": "5.2 ETH",
                        "risk_factors": ["交易对象是已知诈骗地址"]
                    }
                ]
            }
        } 