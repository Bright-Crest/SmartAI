"""
以太坊区块链接口模块
提供以太坊区块链数据访问接口
"""

import os
import logging
import json
import time
from typing import Dict, Any, List, Optional
from hexbytes import HexBytes
from dotenv import load_dotenv

# Web3库
from web3 import Web3
from web3.exceptions import BlockNotFound

# 导入数据存储模块
from app.data import cached, get_data_store, DEFAULT_CACHE_EXPIRY

# 导入错误处理模块
from app.utils.error_handling import (
    InvalidAddressError,
    InvalidTransactionError,
    BlockchainConnectionError
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模拟模式配置
MOCK_MODE = True  # 强制开启模拟模式
logger.info("区块链API运行在强制模拟模式")

# 区块链连接配置
WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "http://localhost:8545")
INFURA_URL = os.getenv("INFURA_URL", "")
ALCHEMY_URL = os.getenv("ALCHEMY_URL", "")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

# 缓存期限配置（秒）
BALANCE_CACHE_EXPIRY = int(os.getenv("BALANCE_CACHE_EXPIRY", str(DEFAULT_CACHE_EXPIRY)))
TX_CACHE_EXPIRY = int(os.getenv("TX_CACHE_EXPIRY", str(DEFAULT_CACHE_EXPIRY)))
TOKEN_CACHE_EXPIRY = int(os.getenv("TOKEN_CACHE_EXPIRY", str(DEFAULT_CACHE_EXPIRY)))

# 模拟地址余额
MOCK_BALANCES = {
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e": 1000000000000000000,  # 1 ETH
    "0x1111111111111111111111111111111111111111": 2000000000000000000,  # 2 ETH
    "0x2222222222222222222222222222222222222222": 5000000000000000000   # 5 ETH
}

# 模拟交易收据
MOCK_RECEIPTS = {
    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef": {
        "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "blockNumber": 12345678,
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
        "status": 1  # 交易成功
    },
    "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890": {
        "hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "blockNumber": 12345679,
        "from": "0x2222222222222222222222222222222222222222",
        "to": "0x1111111111111111111111111111111111111111",
        "status": 1  # 交易成功
    }
}

# 模拟代币余额
MOCK_TOKEN_BALANCES = {
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e": [
        {
            "contract_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "symbol": "USDT",
            "name": "Tether USD",
            "decimals": 6,
            "balance": "1000000000"  # 1000 USDT
        },
        {
            "contract_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "symbol": "USDC",
            "name": "USD Coin",
            "decimals": 6,
            "balance": "2000000000"  # 2000 USDC
        }
    ]
}

# 模拟交易历史
MOCK_TRANSACTIONS = {
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e": [
        {
            "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "blockNumber": 12345678,
            "from": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "to": "0x1111111111111111111111111111111111111111",
            "value": 1000000000000000000,  # 1 ETH
            "timestamp": int(time.time()) - 86400  # 一天前
        },
        {
            "hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "blockNumber": 12345679,
            "from": "0x1111111111111111111111111111111111111111",
            "to": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "value": 500000000000000000,  # 0.5 ETH
            "timestamp": int(time.time()) - 43200  # 12小时前
        }
    ]
}

def is_mock_mode() -> bool:
    """检查是否为模拟模式"""
    return True  # 始终返回True，强制使用模拟模式

def get_web3() -> Web3:
    """
    获取Web3连接实例
    
    返回:
        Web3: Web3连接实例
    """
    logger.info("使用模拟Web3实例")
    # 使用本地节点URL创建一个实例，但不会真正连接
    return Web3(Web3.HTTPProvider("http://localhost:8545"))

def validate_address(address: str) -> str:
    """
    验证以太坊地址格式
    
    参数:
        address: 待验证的以太坊地址
        
    返回:
        校验和格式的地址
        
    抛出:
        InvalidAddressError: 如果地址格式无效
    """
    if not address.startswith("0x") or len(address) != 42:
        raise InvalidAddressError(f"无效的以太坊地址格式: {address}")
    
    # 在模拟模式下，直接返回地址
    return address

@cached(key_prefix="balance", ttl=BALANCE_CACHE_EXPIRY)
def get_balance(address: str) -> Dict[str, Any]:
    """
    获取以太坊地址的ETH余额
    
    参数:
        address: 以太坊地址
        
    返回:
        包含余额信息的字典
    """
    logger.info(f"获取地址余额: {address}")
    
    # 验证地址
    checksum_address = validate_address(address)
    
    logger.info(f"使用模拟数据获取地址余额: {checksum_address}")
    
    # 获取模拟余额或默认值
    balance_wei = MOCK_BALANCES.get(checksum_address, 1000000000000000000)  # 默认1 ETH
    balance_eth = float(balance_wei) / 10**18
    
    return {
        "address": checksum_address,
        "balance_wei": balance_wei,
        "balance_eth": balance_eth,
        "block_number": 12345678
    }

@cached(key_prefix="transaction", ttl=TX_CACHE_EXPIRY)
def get_transaction_receipt(tx_hash: str) -> Dict[str, Any]:
    """
    获取交易收据信息
    
    参数:
        tx_hash: 交易哈希
        
    返回:
        包含交易收据信息的字典
    """
    logger.info(f"获取交易收据: {tx_hash}")
    
    # 检查交易哈希格式
    if not tx_hash.startswith("0x") or len(tx_hash) != 66:
        raise InvalidTransactionError(f"无效的交易哈希格式: {tx_hash}")
    
    logger.info(f"使用模拟数据获取交易收据: {tx_hash}")
    
    # 获取模拟收据或返回默认模拟数据
    receipt = MOCK_RECEIPTS.get(tx_hash, {
        "hash": tx_hash,
        "blockNumber": 12345678,
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
        "status": 1  # 交易成功
    })
    
    return receipt

@cached(key_prefix="tokens", ttl=TOKEN_CACHE_EXPIRY)
def get_token_balances(address: str) -> List[Dict[str, Any]]:
    """
    获取地址持有的ERC20代币余额
    
    参数:
        address: 以太坊地址
        
    返回:
        代币余额列表
    """
    logger.info(f"获取代币余额: {address}")
    
    # 验证地址
    checksum_address = validate_address(address)
    
    logger.info(f"使用模拟数据获取代币余额: {checksum_address}")
    
    # 获取模拟代币余额或默认值
    tokens = MOCK_TOKEN_BALANCES.get(checksum_address, [
        {
            "contract_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "symbol": "USDT",
            "name": "Tether USD",
            "decimals": 6,
            "balance": "1000000000"  # 1000 USDT
        }
    ])
    
    return tokens

@cached(key_prefix="transactions", ttl=TX_CACHE_EXPIRY)
def get_address_transactions(
    address: str,
    start_block: int = 0,
    end_block: int = 99999999,
    page: int = 1,
    offset: int = 10
) -> List[Dict[str, Any]]:
    """
    获取地址的交易历史
    
    参数:
        address: 以太坊地址
        start_block: 开始区块
        end_block: 结束区块
        page: 页码
        offset: 每页记录数
        
    返回:
        交易列表
    """
    logger.info(f"获取地址交易历史: {address}")
    
    # 验证地址
    checksum_address = validate_address(address)
    
    # 计算分页参数
    start_idx = (page - 1) * offset
    end_idx = start_idx + offset
    
    logger.info(f"使用模拟数据获取地址交易历史: {checksum_address}")
    
    # 获取模拟交易历史或默认值
    transactions = MOCK_TRANSACTIONS.get(checksum_address, [])
    
    # 过滤区块范围
    filtered_txs = [
        tx for tx in transactions
        if tx["blockNumber"] >= start_block and tx["blockNumber"] <= end_block
    ]
    
    # 应用分页
    paged_txs = filtered_txs[start_idx:end_idx]
    
    return paged_txs 