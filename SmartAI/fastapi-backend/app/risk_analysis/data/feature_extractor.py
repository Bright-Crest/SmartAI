"""
特征提取模块
从区块链交易数据中提取用于风险分析的特征
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    从区块链交易数据中提取特征的工具类
    """
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        """
        初始化特征提取器
        
        参数:
            api_url: 区块链API的URL
        """
        self.api_url = api_url
        logger.info(f"初始化特征提取器，API URL: {api_url}")
    
    def get_address_data(self, address: str) -> Dict[str, Any]:
        """
        获取地址相关数据
        
        参数:
            address: 以太坊地址
            
        返回:
            包含地址数据的字典
        """
        try:
            # 获取ETH余额
            balance_url = f"{self.api_url}/address/{address}"
            balance_resp = requests.get(balance_url)
            balance_data = balance_resp.json() if balance_resp.status_code == 200 else {}
            
            # 获取代币余额
            tokens_url = f"{self.api_url}/address/{address}/tokens"
            tokens_resp = requests.get(tokens_url)
            tokens_data = tokens_resp.json() if tokens_resp.status_code == 200 else []
            
            # 合并数据
            return {
                "address": address,
                "eth_balance": float(balance_data.get("balance_eth", 0)),
                "tokens": tokens_data
            }
            
        except Exception as e:
            logger.error(f"获取地址数据时出错: {str(e)}")
            # 返回默认数据
            return {
                "address": address,
                "eth_balance": 0,
                "tokens": []
            }
    
    def extract_features(self, address: str) -> Dict[str, Any]:
        """
        提取地址的风险评估特征
        
        参数:
            address: 以太坊地址
            
        返回:
            特征字典
        """
        # 获取地址数据
        address_data = self.get_address_data(address)
        
        # 提取基本特征
        features = {
            "address": address,
            "eth_balance": address_data.get("eth_balance", 0),
            "token_count": len(address_data.get("tokens", [])),
            "has_usdt": 0,
            "has_usdc": 0,
            "total_token_value_usd": 0
        }
        
        # 分析代币
        for token in address_data.get("tokens", []):
            symbol = token.get("symbol", "").upper()
            balance = float(token.get("balance", "0")) / (10 ** token.get("decimals", 0))
            
            # 检查是否持有特定代币
            if symbol == "USDT":
                features["has_usdt"] = 1
                features["total_token_value_usd"] += balance
            elif symbol == "USDC":
                features["has_usdc"] = 1
                features["total_token_value_usd"] += balance
        
        # 模拟其他特征（在实际应用中，这些应该从真实交易历史计算得出）
        features.update({
            "transaction_count": 10,  # 模拟交易数量
            "unique_interaction_addresses": 5,  # 模拟互动地址数量
            "avg_transaction_value": 0.5,  # 模拟平均交易价值（ETH）
            "max_transaction_value": 2.0,  # 模拟最大交易价值（ETH）
            "days_since_first_tx": 100,  # 模拟首次交易至今天数
            "days_since_last_tx": 5,  # 模拟最近交易至今天数
            "outgoing_tx_ratio": 0.6,  # 模拟出账交易比例
            "contract_interaction_count": 3,  # 模拟合约交互次数
            "defi_interaction_count": 2,  # 模拟DeFi交互次数
            "high_risk_interaction_count": 0  # 模拟高风险地址交互次数
        })
        
        return features
    
    def get_features_dataframe(self, addresses: List[str]) -> pd.DataFrame:
        """
        获取多个地址的特征数据框
        
        参数:
            addresses: 以太坊地址列表
            
        返回:
            特征数据框
        """
        features_list = []
        
        for address in addresses:
            features = self.extract_features(address)
            features_list.append(features)
        
        # 创建数据框
        df = pd.DataFrame(features_list)
        
        # 设置地址为索引
        if 'address' in df.columns:
            df.set_index('address', inplace=True)
        
        return df
    
    def get_training_data(self) -> pd.DataFrame:
        """
        获取用于训练的数据
        
        返回:
            特征数据框和标签
        """
        # 模拟地址列表
        addresses = [
            "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            # 添加更多模拟地址以构建训练集
            "0x3333333333333333333333333333333333333333",
            "0x4444444444444444444444444444444444444444",
            "0x5555555555555555555555555555555555555555",
            "0x6666666666666666666666666666666666666666",
            "0x7777777777777777777777777777777777777777",
            "0x8888888888888888888888888888888888888888",
            "0x9999999999999999999999999999999999999999"
        ]
        
        # 获取特征
        features_df = self.get_features_dataframe(addresses)
        
        # 添加模拟风险标签（0=低风险，1=高风险）
        # 在实际应用中，这些标签应来自实际的标记数据
        risk_labels = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], index=features_df.index)
        features_df['risk_label'] = risk_labels
        
        return features_df 