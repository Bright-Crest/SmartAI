from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.schemas.transaction import (
    TransactionRequest,
    TransactionResponse,
    Transaction,
    TransactionStatus,
    TransactionType,
    RiskLevel
)
from app.blockchain.transaction_analyzer import analyze_transaction, get_transaction_path
from app.core.dependencies import cache

# 创建路由器
router = APIRouter(
    prefix="/transactions",
    tags=["transactions"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post("/", response_model=TransactionResponse)
async def get_transactions(request: TransactionRequest):
    """
    获取区块链地址的交易列表
    
    - **address**: 要查询的地址
    - **startDate**: 开始日期 (可选)
    - **endDate**: 结束日期 (可选)
    - **page**: 页码 (默认: 1)
    - **limit**: 每页数量 (默认: 10)
    - **type**: 交易类型过滤 (可选)
    - **riskLevel**: 风险等级过滤 (可选)
    """
    try:
        logger.info(f"获取地址交易: {request.address}")
        
        # 设置日期范围默认值
        if not request.startDate:
            request.startDate = datetime.now() - timedelta(days=30)
        if not request.endDate:
            request.endDate = datetime.now()
            
        # 在实际应用中，应使用Web3库或区块链API获取真实交易数据
        # 这里使用模拟数据
        
        # 模拟生成交易数据
        all_transactions = []
        for i in range(100):  # 模拟100笔交易
            tx_type = list(TransactionType)[i % len(TransactionType)]
            risk_level = list(RiskLevel)[i % len(RiskLevel)]
            
            days_ago = i % 30  # 最近30天内
            tx_date = datetime.now() - timedelta(days=days_ago)
            
            if (tx_date >= request.startDate and tx_date <= request.endDate):
                tx = Transaction(
                    txHash=f"0x{i:064x}",
                    blockNumber=10000000 + i,
                    timestamp=tx_date,
                    from_address=request.address if i % 2 == 0 else f"0x{(i*2):040x}",
                    to_address=f"0x{(i*2):040x}" if i % 2 == 0 else request.address,
                    value=f"{(i+1) * 0.05:.2f} ETH",
                    gas="21000",
                    gasPrice="50 Gwei",
                    status=TransactionStatus.SUCCESS,
                    type=tx_type,
                    riskLevel=risk_level,
                    method="transfer" if tx_type == TransactionType.TRANSFER else tx_type.value
                )
                
                # 应用过滤器
                include_tx = True
                if request.type and tx.type not in request.type:
                    include_tx = False
                if request.riskLevel and tx.riskLevel not in request.riskLevel:
                    include_tx = False
                    
                if include_tx:
                    all_transactions.append(tx)
        
        # 计算分页信息
        total = len(all_transactions)
        start_idx = (request.page - 1) * request.limit
        end_idx = start_idx + request.limit
        transactions = all_transactions[start_idx:end_idx]
        
        # 计算汇总信息
        transfer_count = sum(1 for tx in all_transactions if tx.type == TransactionType.TRANSFER)
        swap_count = sum(1 for tx in all_transactions if tx.type == TransactionType.SWAP)
        high_risk_count = sum(1 for tx in all_transactions if tx.riskLevel == RiskLevel.HIGH)
        
        total_volume = 0
        for tx in all_transactions:
            try:
                # 简单处理，提取数值部分
                vol = float(tx.value.split()[0])
                total_volume += vol
            except:
                pass
                
        # 最常交互的地址
        interaction_counts = {}
        for tx in all_transactions:
            other_addr = tx.to_address if tx.from_address == request.address else tx.from_address
            interaction_counts[other_addr] = interaction_counts.get(other_addr, 0) + 1
            
        most_frequent = max(interaction_counts.items(), key=lambda x: x[1])[0] if interaction_counts else None
        
        # 构建响应
        response = TransactionResponse(
            transactions=transactions,
            total=total,
            page=request.page,
            limit=request.limit,
            totalPages=(total + request.limit - 1) // request.limit,
            summary={
                "totalVolume": f"{total_volume:.2f} ETH",
                "highRiskCount": high_risk_count,
                "mostFrequentInteraction": most_frequent,
                "transferCount": transfer_count,
                "swapCount": swap_count
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"获取交易列表出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取交易列表出错: {str(e)}")

@router.get("/{tx_hash}", response_model=Transaction)
@cache(ttl=3600)  # 缓存一小时
async def get_transaction_details(tx_hash: str):
    """
    获取交易详情
    
    - **tx_hash**: 交易哈希
    """
    try:
        logger.info(f"获取交易详情: {tx_hash}")
        
        # 使用Web3库分析交易
        result = await analyze_transaction(tx_hash)
        
        # 从分析结果构建交易对象
        tx = Transaction(
            txHash=result["tx_hash"],
            blockNumber=result["block_number"],
            timestamp=datetime.now() - timedelta(minutes=30),  # 模拟时间
            from_address=result["from"],
            to_address=result["to"],
            value=f"{float(result['value']) / 10**18:.18f} ETH" if result["value"] else "0 ETH",
            gas=str(result["gas_used"]),
            gasPrice="50 Gwei",  # 简化处理
            status=TransactionStatus.SUCCESS if result["status"] == "success" else TransactionStatus.FAILED,
            type=TransactionType.TRANSFER,  # 默认类型，实际应根据交易内容判断
            riskLevel=result["risk_level"],
            riskDetail=", ".join(result["risk_factors"]) if result["risk_factors"] else None,
            token="ETH",
            method="transfer"  # 默认方法，实际应根据交易内容判断
        )
        
        return tx
        
    except Exception as e:
        logger.error(f"获取交易详情出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取交易详情出错: {str(e)}")

@router.get("/risk/{tx_hash}", response_model=Dict[str, Any])
@cache(ttl=3600)  # 缓存一小时
async def get_transaction_risk(tx_hash: str):
    """
    分析交易风险
    
    - **tx_hash**: 交易哈希
    """
    try:
        logger.info(f"分析交易风险: {tx_hash}")
        
        # 使用Web3库分析交易
        result = await analyze_transaction(tx_hash)
        
        # 获取交易路径
        path = await get_transaction_path(tx_hash)
        
        # 构建风险分析响应
        risk_analysis = {
            "transaction": {
                "hash": result["tx_hash"],
                "from": result["from"],
                "to": result["to"],
                "value": f"{float(result['value']) / 10**18:.18f} ETH" if result["value"] else "0 ETH",
                "status": result["status"]
            },
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "risk_factors": result["risk_factors"],
            "flow_path": path,
            "suggestions": []
        }
        
        # 根据风险因素添加建议
        if "大额交易" in result["risk_factors"]:
            risk_analysis["suggestions"].append("请验证这笔大额交易是否为授权交易")
        if "与混币器交互" in result["risk_factors"]:
            risk_analysis["suggestions"].append("注意：与混币器的交互可能违反某些司法管辖区的法规")
        if "与受制裁地址交互" in result["risk_factors"]:
            risk_analysis["suggestions"].append("警告：该地址已被列入国际制裁名单，交互可能违法")
        if "异常交易模式" in result["risk_factors"]:
            risk_analysis["suggestions"].append("建议检查相关账户是否存在可疑活动")
        
        return risk_analysis
        
    except Exception as e:
        logger.error(f"分析交易风险出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析交易风险出错: {str(e)}") 