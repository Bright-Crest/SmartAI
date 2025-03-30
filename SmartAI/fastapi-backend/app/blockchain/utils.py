"""
区块链工具模块 - 提供区块链通用工具函数
"""

import re
from typing import Optional, Union, Dict, Any, List, Tuple
from eth_utils import is_address, to_checksum_address, is_hex_address
from hexbytes import HexBytes
import json
import logging

logger = logging.getLogger(__name__)

def is_valid_eth_address(address: str) -> bool:
    """
    检查地址是否为有效的以太坊地址
    
    Args:
        address: 要检查的地址
        
    Returns:
        bool: 是否为有效地址
    """
    if not address:
        return False
    
    # 移除前缀空格和'0x'前缀
    clean_address = address.strip().lower()
    if clean_address.startswith('0x'):
        clean_address = clean_address[2:]
    
    # 检查长度和格式
    if len(clean_address) != 40:
        return False
    
    # 检查是否只包含有效的十六进制字符
    hex_pattern = re.compile(r'^[0-9a-f]+$')
    return bool(hex_pattern.match(clean_address))

def normalize_address(address: str) -> str:
    """
    规范化以太坊地址格式
    
    Args:
        address: 要规范化的地址
        
    Returns:
        str: 规范化后的地址
    """
    if not address:
        return ""
        
    try:
        # 移除空格
        clean_address = address.strip()
        
        # 添加0x前缀（如果没有）
        if not clean_address.startswith('0x'):
            clean_address = '0x' + clean_address
            
        # 转换为校验和地址
        return to_checksum_address(clean_address)
    except Exception as e:
        logger.error(f"地址规范化错误: {str(e)}")
        return address

def format_tx_value(value: Union[int, str], decimals: int = 18) -> str:
    """
    格式化交易值为可读形式
    
    Args:
        value: 交易值（整数或字符串）
        decimals: 代币小数位数
        
    Returns:
        str: 格式化后的值
    """
    try:
        # 将value转换为整数
        if isinstance(value, str):
            if value.startswith('0x'):
                value = int(value, 16)
            else:
                value = int(value)
                
        # 将Wei转换为ETH
        eth_value = value / (10 ** decimals)
        
        # 格式化输出
        if eth_value >= 1:
            return f"{eth_value:.4f}"
        elif eth_value >= 0.0001:
            return f"{eth_value:.6f}"
        else:
            return f"{eth_value:.18f}"
    except Exception as e:
        logger.error(f"格式化交易值错误: {str(e)}")
        return str(value)

def bytes_to_hex(data: Union[bytes, HexBytes, str]) -> str:
    """
    将字节数据转换为十六进制字符串
    
    Args:
        data: 字节数据或HexBytes对象
        
    Returns:
        str: 十六进制字符串
    """
    if isinstance(data, (bytes, HexBytes)):
        return "0x" + data.hex()
    elif isinstance(data, str):
        if data.startswith("0x"):
            return data
        else:
            return "0x" + data
    else:
        return str(data)

def hex_to_int(hex_value: str) -> int:
    """
    将十六进制字符串转换为整数
    
    Args:
        hex_value: 十六进制字符串
        
    Returns:
        int: 整数值
    """
    if isinstance(hex_value, str) and hex_value.startswith("0x"):
        return int(hex_value, 16)
    elif isinstance(hex_value, int):
        return hex_value
    else:
        try:
            return int(hex_value)
        except (ValueError, TypeError):
            return 0

def serialize_web3_response(response: Any) -> Any:
    """
    序列化Web3响应对象为可JSON序列化的数据结构
    
    Args:
        response: Web3响应对象
        
    Returns:
        Any: 可JSON序列化的数据
    """
    if hasattr(response, "to_dict"):
        response = response.to_dict()
        
    if isinstance(response, (list, tuple)):
        return [serialize_web3_response(item) for item in response]
    elif isinstance(response, dict):
        return {k: serialize_web3_response(v) for k, v in response.items()}
    elif isinstance(response, HexBytes):
        return bytes_to_hex(response)
    elif isinstance(response, bytes):
        return bytes_to_hex(response)
    else:
        return response

def get_known_address_label(address: str) -> Optional[Dict[str, str]]:
    """
    获取已知地址的标签信息
    
    Args:
        address: 以太坊地址
        
    Returns:
        Optional[Dict]: 地址标签信息或None
    """
    # 这里应该查询数据库或API获取地址标签
    # 下面是一些预定义的地址示例
    normalized_address = address.lower()
    
    known_addresses = {
        "0xdac17f958d2ee523a2206206994597c13d831ec7": {
            "name": "Tether Treasury",
            "type": "exchange",
            "risk": "low"
        },
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
            "name": "USD Coin",
            "type": "stablecoin",
            "risk": "low"
        },
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": {
            "name": "Uniswap V2: Router",
            "type": "dex",
            "risk": "low"
        },
        "0x05e0b5b40b7b66098c2161a5ee11c5740a3a7c45": {
            "name": "Tornado Cash",
            "type": "mixer",
            "risk": "high"
        },
        "0xba214c1c1928a32bffe790263e38b4af9bfcd659": {
            "name": "伊朗制裁地址",
            "type": "sanctioned",
            "risk": "high"
        }
    }
    
    return known_addresses.get(normalized_address)

def get_token_metadata(token_address: str) -> Dict[str, Any]:
    """
    获取代币元数据
    
    Args:
        token_address: 代币合约地址
        
    Returns:
        Dict: 代币元数据
    """
    # 这里应该查询数据库或API获取代币元数据
    # 下面是一些预定义的代币示例
    normalized_address = token_address.lower()
    
    token_metadata = {
        "0xdac17f958d2ee523a2206206994597c13d831ec7": {
            "name": "Tether USD",
            "symbol": "USDT",
            "decimals": 6,
            "logo": "https://etherscan.io/token/images/tether_32.png"
        },
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": {
            "name": "USD Coin",
            "symbol": "USDC",
            "decimals": 6,
            "logo": "https://etherscan.io/token/images/centre-usdc_28.png"
        },
        "0x6b175474e89094c44da98b954eedeac495271d0f": {
            "name": "Dai Stablecoin",
            "symbol": "DAI",
            "decimals": 18,
            "logo": "https://etherscan.io/token/images/mcdDai_32.png"
        }
    }
    
    # 默认元数据
    default_metadata = {
        "name": "未知代币",
        "symbol": "???",
        "decimals": 18,
        "logo": None
    }
    
    return token_metadata.get(normalized_address, default_metadata) 