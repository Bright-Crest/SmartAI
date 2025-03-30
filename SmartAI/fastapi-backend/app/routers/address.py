from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging

from app.schemas.address import AddressRequest, AddressResponse, AddressTag, AddressBalance, InteractionStats, AddressActivity, AddressType
from app.blockchain.address_analyzer import analyze_address, get_address_transactions, check_address_type
from app.core.dependencies import cache

# 创建路由器
router = APIRouter(
    prefix="/address",
    tags=["address"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=AddressResponse)
async def analyze_address_endpoint(request: AddressRequest):
    """
    分析区块链地址
    
    - **address**: 要分析的区块链地址
    """
    try:
        logger.info(f"分析地址: {request.address}")
        
        # 调用区块链分析服务
        result = await analyze_address(request.address)
        
        # 获取地址类型
        address_type = await check_address_type(request.address)
        
        # 构建响应
        response = AddressResponse(
            address=request.address,
            type=AddressType(address_type),
            tags=[
                AddressTag(
                    name="Exchange User" if "exchange" in result["risk_factors"] else "Regular User",
                    type="category",
                    confidence=0.8,
                    created_at=result["recent_transactions"][0]["timestamp"] if result["recent_transactions"] else None,
                    source="internal"
                )
            ],
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            risk_factors=result["risk_factors"],
            first_tx_time=result["recent_transactions"][-1]["timestamp"] if result["recent_transactions"] else None,
            last_tx_time=result["recent_transactions"][0]["timestamp"] if result["recent_transactions"] else None,
            total_tx_count=result["transaction_count"],
            balances=[
                AddressBalance(
                    token="Ethereum",
                    symbol="ETH",
                    amount=str(result["balance_eth"]),
                    value_usd=result["balance_eth"] * 3000,  # 简化处理，实际应查询价格
                    last_updated=None
                )
            ],
            top_interactions=[
                InteractionStats(
                    address=interaction["address"],
                    count=interaction["frequency"],
                    volume="1.75 ETH",
                    last_interaction=result["recent_transactions"][0]["timestamp"],
                    types={"transfer": 2, "swap": 1},
                    risk_level="medium"
                ) for interaction in result["top_interactions"]
            ],
            activity_history=[
                AddressActivity(
                    date=result["recent_transactions"][0]["timestamp"],
                    tx_count=3,
                    volume="2.5 ETH",
                    activities={"transfer": 2, "swap": 1}
                )
            ]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"地址分析出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"地址分析出错: {str(e)}")

@router.get("/tag/{address}", response_model=List[AddressTag])
@cache(ttl=3600)
async def get_address_tags(address: str):
    """
    获取区块链地址标签
    
    - **address**: 区块链地址
    """
    try:
        logger.info(f"获取地址标签: {address}")
        
        # 检查地址类型
        address_type = await check_address_type(address)
        
        # 构建标签
        tags = []
        if address_type == "contract":
            tags.append(
                AddressTag(
                    name="Smart Contract",
                    type="category",
                    confidence=0.95,
                    created_at=None,
                    source="blockchain"
                )
            )
        elif address_type == "exchange":
            tags.append(
                AddressTag(
                    name="Exchange",
                    type="category",
                    confidence=0.9,
                    created_at=None,
                    source="internal"
                )
            )
        else:
            tags.append(
                AddressTag(
                    name="Regular User",
                    type="category",
                    confidence=0.7,
                    created_at=None,
                    source="internal"
                )
            )
        
        return tags
        
    except Exception as e:
        logger.error(f"获取地址标签出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取地址标签出错: {str(e)}")

@router.get("/balance/{address}", response_model=List[AddressBalance])
@cache(ttl=300)  # 5分钟缓存
async def get_address_balance(address: str):
    """
    获取区块链地址余额
    
    - **address**: 区块链地址
    """
    try:
        logger.info(f"获取地址余额: {address}")
        
        from app.blockchain.eth_client import get_eth_client
        eth_client = get_eth_client()
        
        # 获取ETH余额
        balance = await eth_client.get_balance_async(address)
        
        # 模拟一些代币余额
        tokens = [
            {
                "token": "Ethereum",
                "symbol": "ETH",
                "amount": str(balance),
                "value_usd": balance * 3000,  # 简化处理，实际应查询价格
                "last_updated": None
            },
            {
                "token": "USD Coin",
                "symbol": "USDC",
                "amount": "1000.00",
                "value_usd": 1000.0,
                "last_updated": None
            },
            {
                "token": "Tether",
                "symbol": "USDT",
                "amount": "500.00",
                "value_usd": 500.0,
                "last_updated": None
            }
        ]
        
        # 构建余额响应
        balances = [
            AddressBalance(
                token=token["token"],
                symbol=token["symbol"],
                amount=token["amount"],
                value_usd=token["value_usd"],
                last_updated=token["last_updated"]
            ) for token in tokens
        ]
        
        return balances
        
    except Exception as e:
        logger.error(f"获取地址余额出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取地址余额出错: {str(e)}") 