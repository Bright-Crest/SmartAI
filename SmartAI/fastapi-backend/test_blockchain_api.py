#!/usr/bin/env python
"""
区块链API测试脚本
用于测试区块链API的各个端点
"""

import argparse
import json
import logging
import sys
import requests
from typing import Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 测试配置
DEFAULT_BASE_URL = "http://localhost:8001"
TEST_ADDRESS = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
TEST_TX_HASH = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"

# ERC20代币ABI样例
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试区块链API端点")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="API基础URL")
    parser.add_argument("--skip-contract", action="store_true", help="跳过合约调用测试")
    parser.add_argument("--address", default=TEST_ADDRESS, help="测试用的以太坊地址")
    parser.add_argument("--tx", default=TEST_TX_HASH, help="测试用的交易哈希")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细输出")
    return parser.parse_args()

def test_root(base_url: str) -> bool:
    """测试根端点"""
    url = f"{base_url}/"
    
    try:
        logger.info("测试根端点...")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"API状态: {data.get('status')}")
        logger.info(f"API版本: {data.get('api_version')}")
        
        return True
    except Exception as e:
        logger.error(f"测试根端点失败: {str(e)}")
        return False

def test_health(base_url: str) -> bool:
    """测试健康检查端点"""
    url = f"{base_url}/health"
    
    try:
        logger.info("测试健康检查端点...")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"健康状态: {data.get('status')}")
        
        return True
    except Exception as e:
        logger.error(f"测试健康检查端点失败: {str(e)}")
        return False

def test_address(base_url: str, address: str) -> bool:
    """测试地址查询端点"""
    url = f"{base_url}/api/v1/blockchain/address/{address}"
    
    try:
        logger.info(f"测试地址查询端点: {address}...")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"地址: {data.get('address')}")
        logger.info(f"ETH余额: {data.get('balance_eth')} ETH ({data.get('balance_wei')} Wei)")
        logger.info(f"区块号: {data.get('block_number')}")
        
        return True
    except Exception as e:
        logger.error(f"测试地址查询端点失败: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"响应: {e.response.text}")
        return False

def test_transaction(base_url: str, tx_hash: str) -> bool:
    """测试交易查询端点"""
    url = f"{base_url}/api/v1/blockchain/transaction/{tx_hash}"
    
    try:
        logger.info(f"测试交易查询端点: {tx_hash}...")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not response or not data:
            logger.error("请求返回空响应")
            return False
        
        logger.info(f"区块号: {data.get('blockNumber')}")
        logger.info(f"发送方: {data.get('from_address')}")
        logger.info(f"接收方: {data.get('to')}")
        logger.info(f"状态: {data.get('status')}")
        
        return True
    except Exception as e:
        logger.error(f"测试交易查询端点失败: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"响应: {e.response.text}")
        return False

def test_contract_call(base_url: str, address: str) -> bool:
    """测试合约调用端点"""
    url = f"{base_url}/api/v1/blockchain/contract/call"
    
    payload = {
        "contract_address": address,
        "function_name": "balanceOf",
        "abi": ERC20_ABI,
        "args": [address],
        "kwargs": {}
    }
    
    try:
        logger.info(f"测试合约调用端点: {address}, 函数: balanceOf...")
        logger.info(f"请求数据: {json.dumps(payload)}")
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"合约调用结果: {data.get('result')}")
        
        return True
    except Exception as e:
        logger.error(f"测试合约调用端点失败: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"响应: {e.response.text}")
        return False

def test_token_balances(base_url: str, address: str) -> bool:
    """测试代币余额查询端点"""
    url = f"{base_url}/api/v1/blockchain/address/{address}/tokens"
    
    try:
        logger.info(f"测试代币余额查询端点: {address}...")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        tokens = data.get("tokens", [])
        logger.info(f"代币数量: {len(tokens)}")
        
        for idx, token in enumerate(tokens[:3], 1):  # 只显示前3个
            logger.info(f"代币 {idx}: {token.get('symbol', '未知')} ({token.get('name', '未知')}) - 余额: {token.get('tokenBalance', '0')}")
        
        return True
    except Exception as e:
        logger.error(f"测试代币余额查询端点失败: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"响应: {e.response.text}")
        return False

def test_address_transactions(base_url: str, address: str) -> bool:
    """测试地址交易记录查询端点"""
    url = f"{base_url}/api/v1/blockchain/address/{address}/transactions"
    
    try:
        logger.info(f"测试地址交易记录查询端点: {address}...")
        response = requests.get(url, params={"page": 1, "offset": 5})
        response.raise_for_status()
        data = response.json()
        
        transactions = data.get("transactions", [])
        logger.info(f"交易数量: {len(transactions)}")
        
        for idx, tx in enumerate(transactions[:3], 1):  # 只显示前3个
            logger.info(f"交易 {idx}: Hash={tx.get('hash', '未知')} - 值: {tx.get('value', '0')} Wei - 状态: {tx.get('txreceipt_status', '未知')}")
        
        return True
    except Exception as e:
        logger.error(f"测试地址交易记录查询端点失败: {str(e)}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"响应: {e.response.text}")
        return False

def main():
    """主函数"""
    args = parse_args()
    
    base_url = args.url
    logger.info(f"基础URL: {base_url}")
    
    # 测试结果
    results = []
    
    # 测试各个端点
    results.append(("根端点", test_root(base_url)))
    results.append(("健康检查端点", test_health(base_url)))
    results.append(("地址查询端点", test_address(base_url, args.address)))
    results.append(("交易查询端点", test_transaction(base_url, args.tx)))
    results.append(("代币余额查询端点", test_token_balances(base_url, args.address)))
    results.append(("地址交易记录查询端点", test_address_transactions(base_url, args.address)))
    
    # 如果未跳过合约测试，则测试合约调用端点
    if not args.skip_contract:
        results.append(("合约调用端点", test_contract_call(base_url, args.address)))
    
    # 输出总结
    logger.info("\n总结:")
    success_count = 0
    for name, result in results:
        status = "成功" if result else "失败"
        logger.info(f"{name}: {status}")
        if result:
            success_count += 1
    
    logger.info(f"\n总计: {success_count}/{len(results)} 测试成功")
    
    # 如果有任何测试失败，返回非零退出码
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main()) 