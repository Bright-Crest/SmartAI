from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class NodeType(str, Enum):
    """节点类型枚举"""
    USER = "user"
    EXCHANGE = "exchange"
    CONTRACT = "contract"
    MIXER = "mixer"
    SCAMMER = "scammer"
    UNKNOWN = "unknown"

class Node(BaseModel):
    """网络节点"""
    id: str = Field(..., description="节点ID（通常是地址）")
    type: NodeType = Field(NodeType.UNKNOWN, description="节点类型")
    value: Optional[float] = Field(None, description="节点值（通常表示交易量）")
    label: Optional[str] = Field(None, description="节点标签")
    riskScore: Optional[int] = Field(None, ge=0, le=100, description="风险评分")
    metadata: Optional[Dict[str, Any]] = Field(None, description="其他元数据")

class Link(BaseModel):
    """网络连接"""
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    value: Optional[float] = Field(None, description="连接值（通常表示交易量）")
    id: Optional[str] = Field(None, description="连接ID")
    type: Optional[str] = Field(None, description="连接类型")
    metadata: Optional[Dict[str, Any]] = Field(None, description="其他元数据")

class NetworkRequest(BaseModel):
    """交易网络请求"""
    addresses: List[str] = Field(..., min_items=1, description="要分析的区块链地址列表")
    depth: int = Field(1, ge=1, le=3, description="网络深度")
    max_nodes: int = Field(50, ge=10, le=200, description="最大节点数量")
    include_exchanges: bool = Field(True, description="是否包含交易所地址")
    filter_by_type: Optional[List[NodeType]] = Field(None, description="按节点类型筛选")
    
    class Config:
        schema_extra = {
            "example": {
                "addresses": ["0x71C7656EC7ab88b098defB751B7401B5f6d8976F"],
                "depth": 2,
                "max_nodes": 100,
                "include_exchanges": True,
                "filter_by_type": ["user", "scammer"]
            }
        }

class NetworkResponse(BaseModel):
    """交易网络响应"""
    nodes: List[Node] = Field(..., description="网络节点列表")
    links: List[Link] = Field(..., description="网络连接列表")
    central_address: str = Field(..., description="中心地址")
    stats: Optional[Dict[str, Any]] = Field(None, description="网络统计数据")
    
    class Config:
        schema_extra = {
            "example": {
                "nodes": [
                    {"id": "0x1234", "type": "user", "value": 10},
                    {"id": "0x5678", "type": "exchange", "value": 5},
                    {"id": "0x9999", "type": "scammer", "value": 2, "riskScore": 90}
                ],
                "links": [
                    {"source": "0x1234", "target": "0x5678", "value": 2},
                    {"source": "0x1234", "target": "0x9999", "value": 5}
                ],
                "central_address": "0x1234",
                "stats": {
                    "total_volume": "7.5 ETH",
                    "high_risk_interactions": 1,
                    "total_interactions": 2
                }
            }
        } 