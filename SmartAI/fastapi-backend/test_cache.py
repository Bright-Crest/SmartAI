#!/usr/bin/env python
"""
区块链API缓存功能测试脚本
"""

import requests
import time
import json
import sys
import os

# 配置API基础URL
BASE_URL = "http://localhost:8001"

def test_address_cache():
    """测试地址查询缓存功能"""
    address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    
    print(f"\n=== 测试地址余额缓存 ===")
    
    # 第一次查询，应该从区块链获取
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/address/{address}")
    first_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第一次查询结果: {data}")
        print(f"第一次查询用时: {first_query_time:.4f}秒")
    else:
        print(f"查询失败: {response.status_code} - {response.text}")
        return False
    
    # 获取缓存统计信息
    response = requests.get(f"{BASE_URL}/cache/stats")
    if response.status_code == 200:
        print(f"查询前缓存统计: {response.json()}")
    
    # 第二次查询，应该从缓存获取
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/address/{address}")
    second_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第二次查询结果: {data}")
        print(f"第二次查询用时: {second_query_time:.4f}秒")
        print(f"缓存加速比: {first_query_time / second_query_time:.2f}倍")
    else:
        print(f"查询失败: {response.status_code} - {response.text}")
        return False
    
    # 获取缓存统计信息
    response = requests.get(f"{BASE_URL}/cache/stats")
    if response.status_code == 200:
        print(f"查询后缓存统计: {response.json()}")
    
    return True

def test_transaction_cache():
    """测试交易查询缓存功能"""
    tx_hash = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    
    print(f"\n=== 测试交易查询缓存 ===")
    
    # 第一次查询，应该从区块链获取
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/transaction/{tx_hash}")
    first_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第一次查询结果: {data}")
        print(f"第一次查询用时: {first_query_time:.4f}秒")
    else:
        print(f"查询失败: {response.status_code} - {response.text}")
        return False
    
    # 第二次查询，应该从缓存获取
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/transaction/{tx_hash}")
    second_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第二次查询结果: {data}")
        print(f"第二次查询用时: {second_query_time:.4f}秒")
        print(f"缓存加速比: {first_query_time / second_query_time:.2f}倍")
    else:
        print(f"查询失败: {response.status_code} - {response.text}")
        return False
    
    return True

def test_tokens_cache():
    """测试代币余额查询缓存功能"""
    address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    
    print(f"\n=== 测试代币余额缓存 ===")
    
    # 第一次查询，应该从区块链获取
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/address/{address}/tokens")
    first_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第一次查询结果: 找到{len(data)}个代币")
        print(f"第一次查询用时: {first_query_time:.4f}秒")
    else:
        print(f"查询失败: {response.status_code} - {response.text}")
        return False
    
    # 第二次查询，应该从缓存获取
    start_time = time.time()
    response = requests.get(f"{BASE_URL}/address/{address}/tokens")
    second_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第二次查询结果: 找到{len(data)}个代币")
        print(f"第二次查询用时: {second_query_time:.4f}秒")
        print(f"缓存加速比: {first_query_time / second_query_time:.2f}倍")
    else:
        print(f"查询失败: {response.status_code} - {response.text}")
        return False
    
    return True

def test_contract_call_cache():
    """测试合约调用缓存功能"""
    print(f"\n=== 测试合约调用缓存 ===")
    
    # 简单的ERC20合约ABI
    abi = [
        {
            "constant": True,
            "inputs": [],
            "name": "symbol",
            "outputs": [{"name": "", "type": "string"}],
            "payable": False,
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    contract_request = {
        "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7", # USDT合约地址
        "abi": abi,
        "function_name": "symbol",
        "args": [],
        "kwargs": {}
    }
    
    # 第一次调用，应该从区块链获取
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/contract/call", json=contract_request)
    first_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第一次调用结果: {data}")
        print(f"第一次调用用时: {first_query_time:.4f}秒")
    else:
        print(f"调用失败: {response.status_code} - {response.text}")
        return False
    
    # 第二次调用，应该从缓存获取
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/contract/call", json=contract_request)
    second_query_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print(f"第二次调用结果: {data}")
        print(f"第二次调用用时: {second_query_time:.4f}秒")
        print(f"缓存加速比: {first_query_time / second_query_time:.2f}倍")
    else:
        print(f"调用失败: {response.status_code} - {response.text}")
        return False
    
    return True

def test_cache_clear():
    """测试缓存清理功能"""
    print(f"\n=== 测试缓存清理 ===")
    
    # 获取当前缓存统计
    response = requests.get(f"{BASE_URL}/cache/stats")
    if response.status_code == 200:
        before_stats = response.json()
        print(f"清理前缓存统计: {before_stats}")
    else:
        print(f"获取缓存统计失败: {response.status_code} - {response.text}")
        return False
    
    # 清理过期缓存
    response = requests.post(f"{BASE_URL}/cache/clear", json={"expired_only": True})
    if response.status_code == 200:
        print(f"清理过期缓存结果: {response.json()}")
    else:
        print(f"清理缓存失败: {response.status_code} - {response.text}")
        return False
    
    # 获取清理后的缓存统计
    response = requests.get(f"{BASE_URL}/cache/stats")
    if response.status_code == 200:
        after_stats = response.json()
        print(f"清理后缓存统计: {after_stats}")
    else:
        print(f"获取缓存统计失败: {response.status_code} - {response.text}")
        return False
    
    # 清理所有缓存
    response = requests.post(f"{BASE_URL}/cache/clear", json={"expired_only": False})
    if response.status_code == 200:
        print(f"清理所有缓存结果: {response.json()}")
    else:
        print(f"清理缓存失败: {response.status_code} - {response.text}")
        return False
    
    # 获取清理后的缓存统计
    response = requests.get(f"{BASE_URL}/cache/stats")
    if response.status_code == 200:
        final_stats = response.json()
        print(f"清理所有缓存后统计: {final_stats}")
    else:
        print(f"获取缓存统计失败: {response.status_code} - {response.text}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试区块链API缓存功能...")
    
    # 检查API是否正常运行
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print(f"API服务未就绪，状态码: {response.status_code}")
            return False
        print(f"API服务就绪: {response.json()}")
    except requests.exceptions.ConnectionError:
        print(f"无法连接到API服务，请确认服务已启动在 {BASE_URL}")
        return False
    
    # 检查区块链连接
    try:
        response = requests.get(f"{BASE_URL}/blockchain")
        if response.status_code != 200:
            print(f"区块链连接未就绪，状态码: {response.status_code}")
            return False
        print(f"区块链连接就绪: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.ConnectionError:
        print(f"无法连接到区块链服务")
        return False
    
    # 运行测试
    tests = [
        test_address_cache,
        test_transaction_cache,
        test_tokens_cache,
        test_contract_call_cache,
        test_cache_clear
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"测试{test.__name__}时出错: {str(e)}")
            results.append((test.__name__, False))
    
    # 打印测试结果汇总
    print("\n=== 测试结果汇总 ===")
    success_count = sum(1 for _, result in results if result)
    print(f"总测试数: {len(tests)}")
    print(f"成功测试数: {success_count}")
    print(f"失败测试数: {len(tests) - success_count}")
    
    for name, result in results:
        status = "成功" if result else "失败"
        print(f"- {name}: {status}")
    
    return all(result for _, result in results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 