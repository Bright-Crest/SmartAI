import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta

from app.blockchain.eth_client import get_eth_client
from app.core.dependencies import calculate_risk_score, determine_risk_level

logger = logging.getLogger(__name__)

async def analyze_transaction(tx_hash: str) -> Dict[str, Any]:
    """
    分析区块链交易
    
    Args:
        tx_hash: 交易哈希
        
    Returns:
        Dict: 分析结果
    """
    eth_client = get_eth_client()
    
    try:
        # 获取交易详情
        tx_details = await eth_client.get_transaction_async(tx_hash)
        
        # 风险评分计算
        risk_factors = {
            "high_value": float(tx_details.get("value", 0)) > 10 * 10**18,  # 超过10 ETH
            "exchange_interaction": False,
            "mixer_interaction": False,
            "unusual_pattern": False,
            "sanctioned_address": False
        }
        
        # 检查接收方是否为已知高风险地址
        known_risky_addresses = {
            "0x05e0b5b40b7b66098c2161a5ee11c5740a3a7c45": "mixer",
            "0xba214c1c1928a32bffe790263e38b4af9bfcd659": "sanctioned"
        }
        
        to_address = tx_details.get("to", "").lower()
        if to_address in known_risky_addresses:
            address_type = known_risky_addresses[to_address]
            if address_type == "mixer":
                risk_factors["mixer_interaction"] = True
            elif address_type == "sanctioned":
                risk_factors["sanctioned_address"] = True
        
        # 检查是否为异常交易模式（例如：分散为多笔小额交易）
        # 这需要查询相关地址的历史交易，这里简化处理
        risk_factors["unusual_pattern"] = tx_hash.endswith("a") or tx_hash.endswith("f")
        
        # 计算风险评分
        risk_score = calculate_risk_score(risk_factors)
        risk_level = determine_risk_level(risk_score)
        
        # 构建分析结果
        result = {
            "tx_hash": tx_hash,
            "block_number": tx_details.get("blockNumber"),
            "from": tx_details.get("from"),
            "to": tx_details.get("to"),
            "value": tx_details.get("value"),
            "gas_used": tx_details.get("receipt", {}).get("gasUsed"),
            "status": "success" if tx_details.get("receipt", {}).get("status") == 1 else "failed",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": []
        }
        
        # 添加风险因素
        if risk_factors["high_value"]:
            result["risk_factors"].append("大额交易")
        if risk_factors["mixer_interaction"]:
            result["risk_factors"].append("与混币器交互")
        if risk_factors["sanctioned_address"]:
            result["risk_factors"].append("与受制裁地址交互")
        if risk_factors["unusual_pattern"]:
            result["risk_factors"].append("异常交易模式")
        
        return result
        
    except Exception as e:
        logger.error(f"分析交易时出错: {str(e)}")
        raise

async def get_transaction_path(tx_hash: str) -> List[Dict[str, Any]]:
    """
    获取交易路径（资金流向）
    
    Args:
        tx_hash: 交易哈希
        
    Returns:
        List[Dict]: 交易路径（节点和链接）
    """
    eth_client = get_eth_client()
    
    try:
        # 获取交易详情
        tx_details = await eth_client.get_transaction_async(tx_hash)
        
        # 模拟交易路径
        from_address = tx_details.get("from")
        to_address = tx_details.get("to")
        value = tx_details.get("value")
        
        # 简化处理，实际应该分析内部交易和事件
        path = [
            {
                "type": "node",
                "id": from_address,
                "label": from_address[:6] + "..." + from_address[-4:],
                "node_type": "address"
            },
            {
                "type": "link",
                "source": from_address,
                "target": to_address,
                "value": value,
                "label": f"{float(value) / 10**18:.4f} ETH"
            },
            {
                "type": "node",
                "id": to_address,
                "label": to_address[:6] + "..." + to_address[-4:],
                "node_type": "address"
            }
        ]
        
        # 如果是与合约交互，可能有内部交易
        if tx_details.get("receipt", {}).get("logs", []):
            # 模拟一个内部交易
            internal_to = "0x" + tx_hash[-40:]  # 简化处理，使用哈希后缀
            internal_value = float(value) / 2 if value else 0
            
            path.extend([
                {
                    "type": "link",
                    "source": to_address,
                    "target": internal_to,
                    "value": internal_value,
                    "label": f"{float(internal_value) / 10**18:.4f} ETH"
                },
                {
                    "type": "node",
                    "id": internal_to,
                    "label": internal_to[:6] + "..." + internal_to[-4:],
                    "node_type": "address"
                }
            ])
        
        return path
        
    except Exception as e:
        logger.error(f"获取交易路径出错: {str(e)}")
        raise

async def get_transaction_volume_by_date_range(start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """
    获取指定日期范围内的交易量
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        List[Dict]: 每日交易量数据
    """
    # 模拟数据，实际应查询区块链数据
    days = (end_date - start_date).days + 1
    result = []
    
    # 生成每日数据
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        
        # 模拟交易量，根据工作日/周末有所变化
        is_weekend = current_date.weekday() >= 5
        base_volume = 100000 if is_weekend else 150000
        variation = (hash(current_date.isoformat()) % 50000) - 25000
        
        volume = max(10000, base_volume + variation)
        
        result.append({
            "date": current_date,
            "volume": volume,
            "tx_count": int(volume / 100),
            "avg_value": volume / (volume / 100)
        })
    
    return result 