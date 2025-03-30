from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging

from app.schemas.kol import Investment, KolResponse

# 创建路由器
router = APIRouter(
    prefix="/kol",
    tags=["kol"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/track/{address}", response_model=KolResponse)
async def track_kol(
    address: str,
    days: int = Query(30, ge=1, le=365, description="追踪的天数")
):
    """
    追踪KOL（关键意见领袖）的活动
    
    - **address**: 要追踪的钱包地址
    - **days**: 要追踪的天数，默认为30天，最小1天，最大365天
    """
    try:
        logger.info(f"追踪KOL活动: 地址={address}, 天数={days}")
        
        # TODO: 实现真实的KOL追踪逻辑
        # 这里仅返回模拟数据作为演示
        
        # 设置日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 模拟投资数据
        investments = [
            Investment(
                protocol_name="Uniswap V3",
                amount="5.2 ETH",
                days_ago=3,
                timestamp=datetime.now() - timedelta(days=3, hours=5, minutes=30),
                tx_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                tx_type="Swap"
            ),
            Investment(
                protocol_name="Aave",
                amount="10,000 USDC",
                days_ago=5,
                timestamp=datetime.now() - timedelta(days=5, hours=8, minutes=15),
                tx_hash="0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                tx_type="Deposit"
            ),
            Investment(
                protocol_name="Curve",
                amount="3.8 ETH",
                days_ago=8,
                timestamp=datetime.now() - timedelta(days=8, hours=12, minutes=45),
                tx_hash="0x9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba",
                tx_type="Swap"
            ),
            Investment(
                protocol_name="MakerDAO",
                amount="15 ETH",
                days_ago=12,
                timestamp=datetime.now() - timedelta(days=12, hours=3, minutes=20),
                tx_hash="0xfedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                tx_type="Collateral"
            )
        ]
        
        # 模拟活跃协议数据
        active_protocols = [
            {
                "name": "Uniswap",
                "interactions": 28,
                "last_interaction": (datetime.now() - timedelta(days=3, hours=5, minutes=30)).isoformat(),
                "total_volume": "42.5 ETH"
            },
            {
                "name": "Aave",
                "interactions": 15,
                "last_interaction": (datetime.now() - timedelta(days=5, hours=8, minutes=15)).isoformat(),
                "total_volume": "25,000 USDC"
            },
            {
                "name": "Curve",
                "interactions": 10,
                "last_interaction": (datetime.now() - timedelta(days=8, hours=12, minutes=45)).isoformat(),
                "total_volume": "15.8 ETH"
            },
            {
                "name": "MakerDAO",
                "interactions": 7,
                "last_interaction": (datetime.now() - timedelta(days=12, hours=3, minutes=20)).isoformat(),
                "total_volume": "18.2 ETH"
            },
            {
                "name": "Compound",
                "interactions": 5,
                "last_interaction": (datetime.now() - timedelta(days=18, hours=10, minutes=10)).isoformat(),
                "total_volume": "8.5 ETH"
            }
        ]
        
        # 创建响应
        response = KolResponse(
            wallet_address=address,
            is_known_kol=True,
            influence_score=85,
            recent_investments=investments,
            active_defi_protocols=active_protocols
        )
        
        return response
        
    except Exception as e:
        logger.error(f"追踪KOL活动出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"追踪KOL活动出错: {str(e)}")

@router.get("/list", response_model=List[Dict])
async def list_known_kols(
    limit: int = Query(10, ge=1, le=100, description="返回的KOL数量"),
    offset: int = Query(0, ge=0, description="分页偏移量")
):
    """
    获取已知的KOL列表
    
    - **limit**: 返回的KOL数量，默认为10
    - **offset**: 分页偏移量，默认为0
    """
    try:
        logger.info(f"获取已知的KOL列表: limit={limit}, offset={offset}")
        
        # TODO: 实现真实的KOL列表获取逻辑
        # 这里仅返回模拟数据作为演示
        kols = [
            {
                "address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F",
                "name": "0xSassal",
                "influence_score": 92,
                "favorite_protocols": ["Uniswap", "MakerDAO", "Aave"],
                "last_activity": (datetime.now() - timedelta(days=1, hours=5)).isoformat()
            },
            {
                "address": "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
                "name": "Vitalik.eth",
                "influence_score": 98,
                "favorite_protocols": ["Ethereum", "Optimism", "Uniswap"],
                "last_activity": (datetime.now() - timedelta(hours=12)).isoformat()
            },
            {
                "address": "0x9E8f2D30e4fA5C05c1d3B3554A8aC1E320C54e8B",
                "name": "DefiGod",
                "influence_score": 85,
                "favorite_protocols": ["Curve", "Yearn", "Convex"],
                "last_activity": (datetime.now() - timedelta(days=2, hours=8)).isoformat()
            },
            {
                "address": "0x4862733B5FdDFd35f35ea8CCf08F5045e57388B3",
                "name": "CryptoNinja",
                "influence_score": 78,
                "favorite_protocols": ["Synthetix", "dYdX", "GMX"],
                "last_activity": (datetime.now() - timedelta(days=3, hours=2)).isoformat()
            },
            {
                "address": "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326",
                "name": "WhaleAlert",
                "influence_score": 90,
                "favorite_protocols": ["Aave", "Compound", "Balancer"],
                "last_activity": (datetime.now() - timedelta(days=1, hours=18)).isoformat()
            }
        ]
        
        # 应用分页
        paginated_kols = kols[offset:offset+limit]
        
        return paginated_kols
        
    except Exception as e:
        logger.error(f"获取已知的KOL列表出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取已知的KOL列表出错: {str(e)}")

@router.get("/activities", response_model=Dict)
async def get_kol_activities(
    days: int = Query(7, ge=1, le=90, description="要查询的天数")
):
    """
    获取KOL活动摘要
    
    - **days**: 要查询的天数，默认为7天
    """
    try:
        logger.info(f"获取KOL活动摘要: days={days}")
        
        # TODO: 实现真实的KOL活动摘要获取逻辑
        # 这里仅返回模拟数据作为演示
        
        # 设置日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 创建响应数据
        activities = {
            "time_range": f"{start_date.date().isoformat()} to {end_date.date().isoformat()}",
            "total_kol_txs": 1250,
            "unique_kols_active": 85,
            "top_protocols": [
                {"name": "Uniswap", "activity_count": 320, "total_volume": "450 ETH"},
                {"name": "Aave", "activity_count": 280, "total_volume": "380 ETH"},
                {"name": "Curve", "activity_count": 210, "total_volume": "290 ETH"}
            ],
            "most_active_kols": [
                {"address": "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B", "name": "Vitalik.eth", "tx_count": 35},
                {"address": "0x71C7656EC7ab88b098defB751B7401B5f6d8976F", "name": "0xSassal", "tx_count": 28},
                {"address": "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326", "name": "WhaleAlert", "tx_count": 22}
            ],
            "trending_tokens": [
                {"symbol": "ETH", "kol_buys": 320, "kol_sells": 180},
                {"symbol": "USDC", "kol_buys": 280, "kol_sells": 210},
                {"symbol": "ARB", "kol_buys": 240, "kol_sells": 120},
            ],
            "daily_activity": [
                {"date": (end_date - timedelta(days=i)).date().isoformat(), "tx_count": hash(f"day{i}") % 200 + 100}
                for i in range(days)
            ]
        }
        
        return activities
        
    except Exception as e:
        logger.error(f"获取KOL活动摘要出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取KOL活动摘要出错: {str(e)}") 