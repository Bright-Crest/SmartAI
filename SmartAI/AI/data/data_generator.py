import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import torch
from torch_geometric.data import Data
from datetime import datetime, timedelta
import json
from tqdm import tqdm

class BlockchainDataGenerator:
    """
    区块链交易数据生成器，用于生成模拟的区块链交易网络图数据
    - 模拟多种类型的地址（普通用户、交易所、聪明钱、机器人等）
    - 生成符合真实区块链交易特征的交易图
    - 支持生成地址的时序行为序列
    """
    
    def __init__(self, 
                 num_addresses=1000, 
                 num_contracts=20, 
                 num_days=30,
                 smart_money_ratio=0.05,
                 exchange_ratio=0.02,
                 bot_ratio=0.10):
        """
        初始化数据生成器
        
        参数:
        ----
        num_addresses: 生成的总地址数
        num_contracts: 生成的智能合约数量
        num_days: 模拟的天数
        smart_money_ratio: 聪明钱地址比例
        exchange_ratio: 交易所地址比例
        bot_ratio: 机器人地址比例
        """
        self.num_addresses = num_addresses
        self.num_contracts = num_contracts
        self.num_days = num_days
        
        # 创建保存数据的目录
        os.makedirs('data/graphs', exist_ok=True)
        
        # 定义地址类型比例
        self.address_types = {
            'normal_user': 1 - smart_money_ratio - exchange_ratio - bot_ratio,
            'smart_money': smart_money_ratio,
            'exchange': exchange_ratio,
            'bot': bot_ratio
        }
        
        # 定义不同类型地址的行为模式参数
        self.behavior_params = {
            'normal_user': {
                'activity_level': (0.1, 0.3),  # 活跃度范围 (低, 高)
                'transaction_size': (0.01, 2),  # 交易规模范围 (低, 高)
                'gas_range': (21000, 100000),  # gas使用范围
                'contract_affinity': 0.2,  # 与合约交互的概率
                'pattern_strength': 0.1  # 行为模式的规律性
            },
            'smart_money': {
                'activity_level': (0.3, 0.7),
                'transaction_size': (1, 50),
                'gas_range': (50000, 500000),
                'contract_affinity': 0.8,
                'pattern_strength': 0.8,
                'arbitrage_prob': 0.6,  # 套利行为概率
                'new_token_prob': 0.4,  # 参与新代币活动概率
                'cyclical_behavior': 0.7  # 周期性行为概率
            },
            'exchange': {
                'activity_level': (0.7, 0.9),
                'transaction_size': (0.1, 100),
                'gas_range': (21000, 150000),
                'contract_affinity': 0.3,
                'pattern_strength': 0.5
            },
            'bot': {
                'activity_level': (0.5, 0.9),
                'transaction_size': (0.1, 5),
                'gas_range': (21000, 300000),
                'contract_affinity': 0.9,
                'pattern_strength': 0.95,
                'high_frequency': True
            }
        }
        
        # 生成地址
        self.addresses = self._generate_addresses()
        
        # 生成合约
        self.contracts = self._generate_contracts()
        
        # 生成起始时间
        self.start_date = datetime(2023, 1, 1)
        
        # 初始化交易图
        self.transaction_graph = nx.DiGraph()
        
    def _generate_addresses(self):
        """生成模拟地址及其类型"""
        addresses = {}
        address_id = 0
        
        # 为每种类型生成相应数量的地址
        for addr_type, ratio in self.address_types.items():
            count = int(self.num_addresses * ratio)
            for _ in range(count):
                addr = f"0x{random.getrandbits(160):040x}"
                addresses[addr] = {
                    'type': addr_type,
                    'id': address_id,
                    'balance': np.random.lognormal(2, 1.5)  # 初始余额
                }
                address_id += 1
                
        return addresses
    
    def _generate_contracts(self):
        """生成模拟智能合约"""
        contracts = {}
        
        # 合约类型及其功能
        contract_types = [
            'DEX', 'Lending', 'NFT', 'Gaming', 'Staking', 
            'Yield Farming', 'Bridge', 'DAO', 'Insurance'
        ]
        
        for i in range(self.num_contracts):
            contract_type = random.choice(contract_types)
            addr = f"0x{random.getrandbits(160):040x}"
            contracts[addr] = {
                'type': contract_type,
                'id': i,
                'creation_block': np.random.randint(1, 10000),
                'popularity': np.random.beta(2, 5)  # 合约受欢迎程度
            }
            
        return contracts
    
    def _generate_transaction_batch(self, day_index):
        """为指定的一天生成批量交易"""
        current_date = self.start_date + timedelta(days=day_index)
        
        # 每日基本交易量浮动
        daily_tx_count = int(np.random.normal(5000, 1000))
        transactions = []
        
        for _ in range(daily_tx_count):
            # 随机时间点（当天内）
            tx_time = current_date + timedelta(
                seconds=random.randint(0, 86399)
            )
            timestamp = int(tx_time.timestamp())
            
            # 随机选择发送方地址
            sender_addr = random.choice(list(self.addresses.keys()))
            sender_type = self.addresses[sender_addr]['type']
            sender_params = self.behavior_params[sender_type]
            
            # 确定是否与合约交互
            interact_with_contract = random.random() < sender_params['contract_affinity']
            
            if interact_with_contract and self.contracts:
                # 与合约交互
                receiver_addr = random.choice(list(self.contracts.keys()))
                contract_type = self.contracts[receiver_addr]['type']
                
                # 如果是聪明钱地址，增加与特定类型合约交互的概率
                if sender_type == 'smart_money':
                    # 聪明钱倾向于与DEX、Yield Farming等高收益合约交互
                    if contract_type in ['DEX', 'Yield Farming', 'Lending'] and random.random() < 0.7:
                        pass
                    else:
                        # 重新选择合约
                        for _ in range(3):  # 尝试3次
                            potential_contract = random.choice(list(self.contracts.keys()))
                            if self.contracts[potential_contract]['type'] in ['DEX', 'Yield Farming', 'Lending']:
                                receiver_addr = potential_contract
                                break
                
                # 交易特征
                tx_value = np.random.uniform(*sender_params['transaction_size'])
                gas_used = np.random.randint(*sender_params['gas_range'])
                gas_price = np.random.lognormal(3, 0.5)
                
                # 合约特有特征
                method_id = f"0x{random.getrandbits(32):08x}"
                is_successful = random.random() < 0.95  # 95%成功率
                
            else:
                # 普通地址间交易
                remaining_addresses = [a for a in self.addresses.keys() if a != sender_addr]
                receiver_addr = random.choice(remaining_addresses)
                
                # 聪明钱对聪明钱的交易概率增加
                if sender_type == 'smart_money' and random.random() < 0.3:
                    # 倾向于与其他聪明钱交易
                    smart_money_addrs = [
                        a for a in self.addresses.keys() 
                        if self.addresses[a]['type'] == 'smart_money' and a != sender_addr
                    ]
                    if smart_money_addrs:
                        receiver_addr = random.choice(smart_money_addrs)
                
                # 交易特征
                tx_value = np.random.uniform(*sender_params['transaction_size'])
                gas_used = np.random.randint(*sender_params['gas_range'])
                gas_price = np.random.lognormal(3, 0.5)
                
                method_id = None
                is_successful = True
            
            # 添加周期性模式（对聪明钱）
            if sender_type == 'smart_money' and random.random() < sender_params.get('cyclical_behavior', 0):
                # 使交易值遵循某种周期模式
                hour_of_day = tx_time.hour
                # 早上和晚上的交易值更高
                if hour_of_day < 6 or hour_of_day > 20:
                    tx_value *= 1.5
                
                # 星期几影响
                day_of_week = tx_time.weekday()
                if day_of_week in [0, 4]:  # 周一和周五
                    tx_value *= 1.3
            
            # 构造交易
            tx = {
                'hash': f"0x{random.getrandbits(256):064x}",
                'from': sender_addr,
                'to': receiver_addr,
                'value': tx_value,
                'gas': gas_used,
                'gas_price': gas_price,
                'timestamp': timestamp,
                'block_number': int(timestamp / 15) + 10000000,
                'method_id': method_id,
                'is_contract_interaction': interact_with_contract,
                'is_successful': is_successful
            }
            
            transactions.append(tx)
            
            # 更新网络图
            if not self.transaction_graph.has_edge(sender_addr, receiver_addr):
                self.transaction_graph.add_edge(sender_addr, receiver_addr, transactions=[])
            
            self.transaction_graph[sender_addr][receiver_addr]['transactions'].append(tx)
        
        return transactions
    
    def generate_data(self):
        """生成完整的模拟区块链数据并保存为PyTorch Geometric数据"""
        print("正在生成模拟区块链交易数据...")
        
        # 生成每天的交易数据
        all_transactions = []
        for day in tqdm(range(self.num_days)):
            daily_transactions = self._generate_transaction_batch(day)
            all_transactions.extend(daily_transactions)
        
        # 为每个地址创建一个图
        for address in tqdm(self.addresses.keys(), desc="生成地址行为图"):
            self._save_address_graph(address)
        
        # 保存元数据
        self._save_metadata()
        
        print(f"数据生成完成! 共生成 {len(all_transactions)} 笔交易")
        print(f"聪明钱地址数量: {sum(1 for a in self.addresses.values() if a['type'] == 'smart_money')}")
        
        return all_transactions
    
    def _save_address_graph(self, center_address):
        """为指定地址创建行为图并保存为.pt文件"""
        # 获取与该地址相关的所有交易（包括发出和接收）
        related_nodes = set()
        edges = []
        edge_attrs = []
        timestamps = []
        
        # 先收集该地址发出的交易
        for _, to_addr, data in self.transaction_graph.out_edges(center_address, data=True):
            for tx in data['transactions']:
                related_nodes.add(to_addr)
                edges.append([self.addresses[center_address]['id'], 
                             self.addresses.get(to_addr, {'id': len(self.addresses) + self.contracts.get(to_addr, {'id': 0})['id']})['id']])
                
                edge_attr = [
                    tx['value'],  
                    tx['gas'] * tx['gas_price'] / 1e9,  # gas费（单位：ETH）
                    1 if tx['is_contract_interaction'] else 0,  # 是否合约交互
                    1.0  # 边权重（默认为1）
                ]
                edge_attrs.append(edge_attr)
                timestamps.append(tx['timestamp'])
        
        # 再收集该地址接收的交易
        for from_addr, _, data in self.transaction_graph.in_edges(center_address, data=True):
            for tx in data['transactions']:
                related_nodes.add(from_addr)
                edges.append([self.addresses.get(from_addr, {'id': len(self.addresses) + self.contracts.get(from_addr, {'id': 0})['id']})['id'],
                             self.addresses[center_address]['id']])
                
                edge_attr = [
                    tx['value'],
                    tx['gas'] * tx['gas_price'] / 1e9,
                    1 if tx['is_contract_interaction'] else 0,
                    1.0
                ]
                edge_attrs.append(edge_attr)
                timestamps.append(tx['timestamp'])
        
        # 如果没有交易，跳过
        if not edges:
            return
        
        # 构建节点特征矩阵（这里简单使用one-hot编码表示节点类型）
        node_ids = {self.addresses[center_address]['id']}
        for edge in edges:
            node_ids.add(edge[0])
            node_ids.add(edge[1])
        
        node_ids = sorted(list(node_ids))
        num_nodes = len(node_ids)
        
        # 4种节点类型：普通用户、聪明钱、交易所、机器人
        node_features = torch.zeros((num_nodes, 4))
        
        # 为中心地址设置特征
        center_idx = node_ids.index(self.addresses[center_address]['id'])
        node_type = self.addresses[center_address]['type']
        if node_type == 'normal_user':
            node_features[center_idx, 0] = 1.0
        elif node_type == 'smart_money':
            node_features[center_idx, 1] = 1.0
        elif node_type == 'exchange':
            node_features[center_idx, 2] = 1.0
        elif node_type == 'bot':
            node_features[center_idx, 3] = 1.0
        
        # 为其他节点设置特征（简化起见，随机赋值）
        for i, node_id in enumerate(node_ids):
            if i != center_idx:
                node_type_idx = np.random.choice(4, p=[0.7, 0.1, 0.1, 0.1])
                node_features[i, node_type_idx] = 1.0
        
        # 转换为PyTorch Geometric格式
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        timestamps = torch.tensor(timestamps, dtype=torch.long)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            timestamps=timestamps,
            address_id=center_address
        )
        
        # 保存为.pt文件
        torch.save(data, f'data/graphs/{center_address}.pt')
    
    def _save_metadata(self):
        """保存地址和合约元数据"""
        # 地址元数据
        address_metadata = {addr: {
            'type': info['type'],
            'id': info['id'],
            'is_smart_money': info['type'] == 'smart_money'
        } for addr, info in self.addresses.items()}
        
        # 合约元数据
        contract_metadata = {addr: {
            'type': info['type'],
            'id': info['id'],
            'popularity': float(info['popularity'])
        } for addr, info in self.contracts.items()}
        
        # 保存为JSON文件
        with open('data/address_metadata.json', 'w') as f:
            json.dump(address_metadata, f, indent=2)
            
        with open('data/contract_metadata.json', 'w') as f:
            json.dump(contract_metadata, f, indent=2)

# 示例使用
if __name__ == "__main__":
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 初始化生成器（小规模示例）
    generator = BlockchainDataGenerator(
        num_addresses=500,
        num_contracts=10,
        num_days=15,
        smart_money_ratio=0.08
    )
    
    # 生成数据
    transactions = generator.generate_data()
    
    # 简单统计
    print(f"总地址数量: {len(generator.addresses)}")
    print(f"总合约数量: {len(generator.contracts)}")
    
    # 检查生成的数据文件
    graph_files = os.listdir('data/graphs')
    print(f"生成的图数据文件数量: {len(graph_files)}")
    
    # 加载一个示例图文件进行验证
    if graph_files:
        sample_file = f"data/graphs/{graph_files[0]}"
        sample_data = torch.load(sample_file)
        print("\n示例图数据统计:")
        print(f"节点数量: {sample_data.x.size(0)}")
        print(f"边数量: {sample_data.edge_index.size(1)}")
        print(f"边特征维度: {sample_data.edge_attr.size(1)}")
        print(f"时间戳数量: {sample_data.timestamps.size(0)}") 