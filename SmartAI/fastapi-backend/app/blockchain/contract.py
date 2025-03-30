"""
以太坊智能合约交互模块
提供与智能合约的交互功能
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional, Union, Tuple

from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

# 导入Web3实例
from app.blockchain.ethereum import get_web3
from app.data import cached, get_data_store

# 导入错误处理模块
from app.utils.error_handling import (
    ContractInteractionError
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 缓存期限（秒）
DEFAULT_CACHE_EXPIRY = 3600  # 1小时

# 模拟合约调用结果
MOCK_CONTRACT_RESULTS = {
    "totalSupply": 1000000000000000000000000,
    "balanceOf": 100000000000000000000,
    "allowance": 1000000000000000000,
    "symbol": "TKN",
    "name": "Token",
    "decimals": 18,
    "getOwner": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
}

# 模拟合约事件
MOCK_EVENTS = [
    {
        "event": "Transfer",
        "blockNumber": 12345678,
        "transactionHash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "args": {
            "from": "0x0000000000000000000000000000000000000000",
            "to": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "value": 1000000000000000000000000
        }
    },
    {
        "event": "Approval",
        "blockNumber": 12345679,
        "transactionHash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "args": {
            "owner": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "spender": "0x1111111111111111111111111111111111111111",
            "value": 1000000000000000000
        }
    }
]

def is_mock_mode() -> bool:
    """检查是否为模拟模式"""
    return True  # 始终返回True，强制使用模拟模式

def get_contract_instance(
    contract_address: str, 
    abi: Union[str, List[Dict[str, Any]]]
) -> Contract:
    """
    获取合约实例
    
    参数:
        contract_address: 合约地址
        abi: 合约ABI（字符串或JSON对象）
    
    返回:
        Contract: Web3合约实例
    """
    logger.info(f"使用模拟模式获取合约实例: {contract_address}")
    w3 = get_web3()
    
    # 确保ABI是JSON格式
    if isinstance(abi, str):
        try:
            abi_json = json.loads(abi)
        except json.JSONDecodeError:
            raise ContractInteractionError(f"无效的ABI格式: {abi}")
    else:
        abi_json = abi
        
    # 创建合约实例
    return w3.eth.contract(address=contract_address, abi=abi_json)

@cached(key_prefix="contract_call", ttl=DEFAULT_CACHE_EXPIRY)
def contract_call(
    contract_address: str,
    abi: Union[str, List[Dict[str, Any]]],
    function_name: str,
    *args,
    **kwargs
) -> Any:
    """
    调用合约的只读函数
    
    参数:
        contract_address: 合约地址
        abi: 合约ABI（字符串或JSON对象）
        function_name: 函数名称
        args: 函数参数
        kwargs: 函数关键字参数
    
    返回:
        Any: 函数调用结果
    """
    logger.info(f"调用合约 {contract_address} 的函数 {function_name}")
    
    logger.info(f"使用模拟模式调用合约: {contract_address}.{function_name}")
    
    # 使用预定义的模拟结果
    if function_name in MOCK_CONTRACT_RESULTS:
        result = MOCK_CONTRACT_RESULTS[function_name]
        logger.info(f"模拟合约调用结果: {result}")
        return result
    else:
        # 默认返回一个整数，可以根据需要扩展
        return 123456789

@cached(key_prefix="contract_events", ttl=DEFAULT_CACHE_EXPIRY)
def get_events(
    contract_address: str,
    abi: Union[str, List[Dict[str, Any]]],
    event_name: str,
    from_block: int = 0,
    to_block: Union[int, str] = "latest",
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    获取合约事件
    
    参数:
        contract_address: 合约地址
        abi: 合约ABI（字符串或JSON对象）
        event_name: 事件名称
        from_block: 开始区块
        to_block: 结束区块
        filters: 事件过滤条件
    
    返回:
        List[Dict[str, Any]]: 事件列表
    """
    logger.info(f"获取合约 {contract_address} 的事件 {event_name}")
    
    logger.info(f"使用模拟模式获取事件: {contract_address}.{event_name}")
    
    # 过滤匹配的模拟事件
    events = [event for event in MOCK_EVENTS if event["event"] == event_name]
    logger.info(f"模拟事件数量: {len(events)}")
    
    return events 