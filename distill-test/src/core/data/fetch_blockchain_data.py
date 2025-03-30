import requests
import pandas as pd
import numpy as np
import time
import json
import random
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call, log_step
from utils.config import config

class BlockchainDataFetcher:
    """区块链数据获取器，从以太坊API获取真实数据"""
    
    def __init__(self, api_key=None):
        """初始化数据获取器
        
        Args:
            api_key: Etherscan API密钥
        """
        self.api_key = api_key
        self.data_dir = config.results_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.etherscan_base_url = "https://api.etherscan.io/api"
        self.sleep_time = 0.2  # 请求间隔时间（秒）
    
    @log_function_call
    def fetch_address_transactions(self, address, startblock=0, endblock=99999999):
        """获取地址的交易记录
        
        Args:
            address: 以太坊地址
            startblock: 起始区块
            endblock: 结束区块
            
        Returns:
            交易记录列表
        """
        if not self.api_key:
            logger.warning(f"未提供API密钥，跳过获取地址 {address} 的交易")
            return []
        
        try:
            logger.info(f"获取地址 {address} 的交易记录")
            
            # 构建API请求
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "sort": "desc",
                "apikey": self.api_key
            }
            
            # 发送请求
            response = requests.get(self.etherscan_base_url, params=params)
            data = response.json()
            
            # 检查响应
            if data["status"] == "1":
                return data["result"]
            else:
                logger.error(f"获取地址 {address} 的交易记录失败: {data['message']}")
                return []
                
        except Exception as e:
            logger.error(f"获取地址 {address} 的交易记录时出错: {str(e)}")
            return []
        finally:
            # 间隔请求，避免API限制
            time.sleep(self.sleep_time)
    
    @log_function_call
    def fetch_token_transfers(self, address, startblock=0, endblock=99999999):
        """获取地址的代币转账记录
        
        Args:
            address: 以太坊地址
            startblock: 起始区块
            endblock: 结束区块
            
        Returns:
            代币转账记录列表
        """
        if not self.api_key:
            logger.warning(f"未提供API密钥，跳过获取地址 {address} 的代币转账")
            return []
        
        try:
            logger.info(f"获取地址 {address} 的代币转账记录")
            
            # 构建API请求
            params = {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": startblock,
                "endblock": endblock,
                "sort": "desc",
                "apikey": self.api_key
            }
            
            # 发送请求
            response = requests.get(self.etherscan_base_url, params=params)
            data = response.json()
            
            # 检查响应
            if data["status"] == "1":
                return data["result"]
            else:
                logger.error(f"获取地址 {address} 的代币转账记录失败: {data['message']}")
                return []
                
        except Exception as e:
            logger.error(f"获取地址 {address} 的代币转账记录时出错: {str(e)}")
            return []
        finally:
            # 间隔请求，避免API限制
            time.sleep(self.sleep_time)
    
    @log_function_call
    def fetch_balance(self, address):
        """获取地址的ETH余额
        
        Args:
            address: 以太坊地址
            
        Returns:
            ETH余额（Wei）
        """
        if not self.api_key:
            logger.warning(f"未提供API密钥，跳过获取地址 {address} 的余额")
            return 0
        
        try:
            logger.info(f"获取地址 {address} 的ETH余额")
            
            # 构建API请求
            params = {
                "module": "account",
                "action": "balance",
                "address": address,
                "tag": "latest",
                "apikey": self.api_key
            }
            
            # 发送请求
            response = requests.get(self.etherscan_base_url, params=params)
            data = response.json()
            
            # 检查响应
            if data["status"] == "1":
                return int(data["result"])
            else:
                logger.error(f"获取地址 {address} 的ETH余额失败: {data['message']}")
                return 0
                
        except Exception as e:
            logger.error(f"获取地址 {address} 的ETH余额时出错: {str(e)}")
            return 0
        finally:
            # 间隔请求，避免API限制
            time.sleep(self.sleep_time)
    
    @log_step("提取地址特征")
    def extract_features(self, address, transactions, token_transfers, balance):
        """从交易记录中提取特征
        
        Args:
            address: 以太坊地址
            transactions: 交易记录列表
            token_transfers: 代币转账记录列表
            balance: ETH余额
            
        Returns:
            特征字典
        """
        try:
            logger.info(f"为地址 {address} 提取特征")
            
            # 如果交易数据为空，返回空特征
            if not transactions and not token_transfers:
                logger.warning(f"地址 {address} 没有交易数据")
                return None
            
            # 初始化特征
            features = {
                'eth_address': address,
                'eth_balance': float(balance) / 1e18,  # 转换为ETH
                'tx_count_30d': len(transactions),
                'token_count': len(set(t.get('tokenSymbol', '') for t in token_transfers))
            }
            
            # 如果没有交易，返回基本特征
            if not transactions:
                return features
            
            # 解析交易时间戳
            timestamps = [int(tx['timeStamp']) for tx in transactions]
            timestamps.sort()
            
            if timestamps:
                # 计算账户年龄（天）
                first_tx_time = timestamps[0]
                last_tx_time = timestamps[-1]
                account_age_seconds = last_tx_time - first_tx_time
                features['age_in_days'] = account_age_seconds / 86400 if account_age_seconds > 0 else 1
                
                # 计算活跃天数比例
                unique_days = len(set(time.strftime('%Y-%m-%d', time.gmtime(ts)) for ts in timestamps))
                total_days = features['age_in_days']
                features['active_days_ratio'] = unique_days / total_days if total_days > 0 else 0
            
            # 计算交易价值相关特征
            tx_values = [float(tx['value']) / 1e18 for tx in transactions]  # 转换为ETH
            
            if tx_values:
                features['avg_tx_value'] = np.mean(tx_values)
                features['max_single_tx_value'] = max(tx_values)
                features['tx_value_variance'] = np.var(tx_values)
            
            # 计算交易频率（每天）
            if features.get('age_in_days', 0) > 0:
                features['tx_frequency'] = len(transactions) / features['age_in_days']
            
            # 计算Gas支出总额
            gas_spent = sum(int(tx['gasPrice']) * int(tx['gasUsed']) for tx in transactions)
            features['gas_spent_total'] = float(gas_spent) / 1e18  # 转换为ETH
            
            # 计算交易入账/出账数量
            incoming_tx = [tx for tx in transactions if tx['to'].lower() == address.lower()]
            outgoing_tx = [tx for tx in transactions if tx['from'].lower() == address.lower()]
            
            features['incoming_tx_count'] = len(incoming_tx)
            features['outgoing_tx_count'] = len(outgoing_tx)
            
            # 计算交互地址数量
            interacted_addresses = set()
            for tx in transactions:
                interacted_addresses.add(tx['from'].lower())
                interacted_addresses.add(tx['to'].lower())
            
            interacted_addresses.discard(address.lower())  # 移除自身地址
            features['unique_interact_addresses'] = len(interacted_addresses)
            
            # 计算DEX交互次数（简化为与已知DEX合约的交互）
            dex_addresses = {
                "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
                "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3
                "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap
                "0xdef1c0ded9bec7f1a1670819833240f027b25eff"   # 0x Exchange
            }
            
            dex_interactions = [tx for tx in transactions 
                               if tx['to'].lower() in dex_addresses or tx['from'].lower() in dex_addresses]
            features['dex_interaction_count'] = len(dex_interactions)
            
            # 简化计算其他特征（真实场景需要更复杂的计算）
            features['early_project_participation'] = 1 if features['dex_interaction_count'] > 5 else 0
            features['avg_holding_period'] = random.uniform(1, 30)  # 随机值，真实场景需要更复杂的计算
            features['roi_30d'] = random.uniform(-0.2, 0.4)  # 随机值，真实场景需要更复杂的计算
            features['contract_interaction_entropy'] = random.uniform(0.1, 0.9)  # 随机值
            features['whale_interaction_count'] = random.randint(0, 10)  # 随机值
            
            # 交易类型分布（简化计算）
            features['erc20_tx_ratio'] = len(token_transfers) / (len(transactions) + 0.01)
            features['eth_tx_ratio'] = 1 - features['erc20_tx_ratio']
            features['nft_tx_ratio'] = random.uniform(0.05, 0.2)  # 随机值
            features['defi_tx_ratio'] = features['dex_interaction_count'] / (len(transactions) + 0.01)
            
            return features
            
        except Exception as e:
            logger.error(f"提取地址 {address} 特征时出错: {str(e)}")
            return None
    
    @log_step("获取地址完整数据")
    def fetch_address_data(self, address):
        """获取地址的完整数据并提取特征
        
        Args:
            address: 以太坊地址
            
        Returns:
            特征字典或None
        """
        logger.info(f"获取地址 {address} 的完整数据")
        
        # 获取交易记录
        transactions = self.fetch_address_transactions(address)
        
        # 获取代币转账记录
        token_transfers = self.fetch_token_transfers(address)
        
        # 获取余额
        balance = self.fetch_balance(address)
        
        # 提取特征
        return self.extract_features(address, transactions, token_transfers, balance)
    
    @log_step("使用演示数据")
    def generate_demo_data(self, address):
        """生成演示数据（当没有API密钥时使用）
        
        Args:
            address: 以太坊地址
            
        Returns:
            生成的特征字典
        """
        logger.info(f"为地址 {address} 生成演示数据")
        
        # 随机决定是否是"聪明钱"
        is_smart_money = random.random() < 0.5
        
        # 使用数据生成器生成特征
        from .data_generator import DataGenerator
        generator = DataGenerator()
        
        if is_smart_money:
            features = generator.generate_smart_money_features()
        else:
            features = generator.generate_regular_user_features()
        
        # 替换地址
        features['eth_address'] = address
        
        return features
    
    @log_function_call
    def fetch_addresses_from_file(self, file_path):
        """从文件中读取地址列表
        
        Args:
            file_path: 地址文件路径
            
        Returns:
            地址列表
        """
        try:
            logger.info(f"从文件 {file_path} 读取地址")
            
            with open(file_path, 'r') as f:
                addresses = [line.strip() for line in f if line.strip().startswith('0x')]
            
            logger.info(f"读取到 {len(addresses)} 个地址")
            return addresses
            
        except Exception as e:
            logger.error(f"读取地址文件时出错: {str(e)}")
            return []
    
    @log_function_call
    def generate_random_addresses(self, count=100):
        """生成随机以太坊地址
        
        Args:
            count: 地址数量
            
        Returns:
            地址列表
        """
        logger.info(f"生成 {count} 个随机以太坊地址")
        
        from .data_generator import DataGenerator
        generator = DataGenerator()
        
        addresses = [generator.generate_eth_address() for _ in range(count)]
        return addresses
    
    @log_step("获取区块链数据")
    def fetch_blockchain_data(self, addresses, use_demo_data=True):
        """获取区块链数据
        
        Args:
            addresses: 地址列表
            use_demo_data: 是否使用演示数据
            
        Returns:
            特征数据帧
        """
        all_features = []
        
        for i, address in enumerate(addresses):
            logger.info(f"处理地址 {i+1}/{len(addresses)}: {address}")
            
            if use_demo_data or not self.api_key:
                features = self.generate_demo_data(address)
            else:
                features = self.fetch_address_data(address)
            
            if features:
                all_features.append(features)
        
        # 创建数据帧
        if all_features:
            df = pd.DataFrame(all_features)
            return df
        else:
            logger.warning("没有获取到有效的区块链数据")
            return None
    
    @log_function_call
    def save_data(self, df, filename):
        """保存数据到文件
        
        Args:
            df: 数据帧
            filename: 文件名
            
        Returns:
            保存的文件路径
        """
        file_path = self.data_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"数据已保存至 {file_path}")
        return file_path
    
    @log_step("获取并保存区块链数据")
    def fetch_and_save_data(self, addresses=None, addresses_file=None, count=100, use_demo_data=True):
        """获取并保存区块链数据
        
        Args:
            addresses: 地址列表（可选）
            addresses_file: 地址文件路径（可选）
            count: 如果未提供地址，生成的随机地址数量
            use_demo_data: 是否使用演示数据
            
        Returns:
            保存的文件路径
        """
        # 获取地址列表
        if addresses:
            logger.info(f"使用提供的 {len(addresses)} 个地址")
        elif addresses_file:
            addresses = self.fetch_addresses_from_file(addresses_file)
            logger.info(f"从文件获取 {len(addresses)} 个地址")
        else:
            addresses = self.generate_random_addresses(count)
            logger.info(f"生成 {len(addresses)} 个随机地址")
        
        # 获取区块链数据
        df = self.fetch_blockchain_data(addresses, use_demo_data)
        
        if df is not None:
            # 保存原始数据
            raw_data_path = self.save_data(df, "blockchain_real_data.csv")
            
            # 预处理数据
            from .data_generator import DataGenerator
            generator = DataGenerator()
            processed_df = generator.preprocess_data(df)
            
            # 保存处理后的数据
            processed_data_path = self.save_data(processed_df, "blockchain_real_processed.csv")
            
            return processed_data_path
        else:
            logger.error("未能获取区块链数据")
            return None

# 函数用于直接调用
def fetch_blockchain_data(api_key=None, addresses_file=None, num_addresses=100, use_demo_data=True):
    """获取区块链数据
    
    Args:
        api_key: Etherscan API密钥
        addresses_file: 地址文件路径
        num_addresses: 随机地址数量
        use_demo_data: 是否使用演示数据
        
    Returns:
        处理后数据文件路径
    """
    fetcher = BlockchainDataFetcher(api_key)
    return fetcher.fetch_and_save_data(
        addresses_file=addresses_file,
        count=num_addresses,
        use_demo_data=use_demo_data
    ) 