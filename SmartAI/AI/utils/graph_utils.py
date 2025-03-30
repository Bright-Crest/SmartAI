import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from torch_geometric.data import Data
import networkx as nx
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_transaction_data(file_path: str) -> pd.DataFrame:
    """
    加载交易数据文件
    
    参数:
        file_path (str): 交易数据文件路径，支持csv, parquet
        
    返回:
        DataFrame: 包含交易数据的DataFrame
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    # 检查必要列是否存在
    required_columns = ['from_address', 'to_address', 'timestamp', 'value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据文件缺少必要列: {missing_columns}")
    
    logger.info(f"加载了{len(df)}条交易记录")
    return df


def extract_address_features(df: pd.DataFrame, address: str) -> Dict[str, Any]:
    """
    提取地址的特征
    
    参数:
        df (DataFrame): 交易数据
        address (str): 目标地址
        
    返回:
        Dict: 地址特征字典
    """
    # 提取该地址的所有出入交易
    outgoing = df[df['from_address'] == address]
    incoming = df[df['to_address'] == address]
    
    # 计算基本特征
    total_outgoing = len(outgoing)
    total_incoming = len(incoming)
    total_outgoing_value = outgoing['value'].sum() if len(outgoing) > 0 else 0
    total_incoming_value = incoming['value'].sum() if len(incoming) > 0 else 0
    
    # 计算交易频率
    if total_outgoing + total_incoming > 0:
        timestamps = pd.concat([outgoing['timestamp'], incoming['timestamp']]).values
        timestamps = np.sort(timestamps)
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            avg_time_diff = np.mean(time_diffs)
        else:
            avg_time_diff = 0
    else:
        avg_time_diff = 0
    
    # 计算交易值的统计量
    if total_outgoing > 0:
        outgoing_values = outgoing['value'].values
        outgoing_mean = np.mean(outgoing_values)
        outgoing_std = np.std(outgoing_values)
        outgoing_min = np.min(outgoing_values)
        outgoing_max = np.max(outgoing_values)
    else:
        outgoing_mean = outgoing_std = outgoing_min = outgoing_max = 0
    
    if total_incoming > 0:
        incoming_values = incoming['value'].values
        incoming_mean = np.mean(incoming_values)
        incoming_std = np.std(incoming_values)
        incoming_min = np.min(incoming_values)
        incoming_max = np.max(incoming_values)
    else:
        incoming_mean = incoming_std = incoming_min = incoming_max = 0
    
    # 计算交易对象多样性
    outgoing_receivers = outgoing['to_address'].nunique() if len(outgoing) > 0 else 0
    incoming_senders = incoming['from_address'].nunique() if len(incoming) > 0 else 0
    
    # 返回特征字典
    return {
        'total_outgoing': total_outgoing,
        'total_incoming': total_incoming,
        'total_outgoing_value': total_outgoing_value,
        'total_incoming_value': total_incoming_value,
        'avg_time_diff': avg_time_diff,
        'outgoing_mean': outgoing_mean,
        'outgoing_std': outgoing_std,
        'outgoing_min': outgoing_min,
        'outgoing_max': outgoing_max,
        'incoming_mean': incoming_mean,
        'incoming_std': incoming_std,
        'incoming_min': incoming_min,
        'incoming_max': incoming_max,
        'outgoing_receivers': outgoing_receivers,
        'incoming_senders': incoming_senders
    }


def extract_edge_features(transaction: pd.Series) -> List[float]:
    """
    提取交易边特征
    
    参数:
        transaction (Series): 交易记录
        
    返回:
        List[float]: 边特征列表
    """
    # 基本特征
    value = transaction['value']
    
    # 如果有gas相关信息
    gas_price = transaction.get('gas_price', 0)
    gas_used = transaction.get('gas_used', 0)
    gas_fee = gas_price * gas_used if gas_price and gas_used else 0
    
    # 如果有合约交互信息
    is_contract_call = int(transaction.get('is_contract_call', False))
    
    # 返回特征列表
    return [value, gas_fee, is_contract_call, 0]  # 最后一个位置可用于补充其他特征


def build_address_graph(
    df: pd.DataFrame, 
    target_address: str, 
    max_neighbors: int = 100,
    max_transactions: int = 1000
) -> Data:
    """
    为目标地址构建行为图
    
    参数:
        df (DataFrame): 交易数据
        target_address (str): 目标地址
        max_neighbors (int): 最大邻居数量
        max_transactions (int): 最大交易数量
        
    返回:
        Data: PyG格式的图数据
    """
    # 过滤与目标地址相关的交易
    related_txs = df[(df['from_address'] == target_address) | (df['to_address'] == target_address)]
    
    # 如果交易太多，随机抽样
    if len(related_txs) > max_transactions:
        related_txs = related_txs.sample(max_transactions, random_state=42)
    
    # 提取相关地址（节点）
    related_addresses = set()
    related_addresses.add(target_address)
    related_addresses.update(related_txs['from_address'].values)
    related_addresses.update(related_txs['to_address'].values)
    
    # 如果地址太多，保留最活跃的地址
    if len(related_addresses) > max_neighbors + 1:  # +1是为了保证目标地址在内
        # 计算每个地址的交易次数
        address_counts = {}
        for addr in related_addresses:
            if addr == target_address:  # 确保目标地址保留
                address_counts[addr] = float('inf')
                continue
            count = ((related_txs['from_address'] == addr) | (related_txs['to_address'] == addr)).sum()
            address_counts[addr] = count
        
        # 选择最活跃的地址
        sorted_addresses = sorted(address_counts.items(), key=lambda x: x[1], reverse=True)
        related_addresses = set([addr for addr, _ in sorted_addresses[:max_neighbors + 1]])
    
    # 为每个地址分配索引
    address_to_idx = {addr: i for i, addr in enumerate(related_addresses)}
    
    # 构建边列表和边特征
    edge_index = []
    edge_attr = []
    timestamps = []
    
    for _, tx in related_txs.iterrows():
        from_addr = tx['from_address']
        to_addr = tx['to_address']
        
        # 如果某个地址被过滤掉了，跳过这条交易
        if from_addr not in address_to_idx or to_addr not in address_to_idx:
            continue
        
        from_idx = address_to_idx[from_addr]
        to_idx = address_to_idx[to_addr]
        
        # 添加边
        edge_index.append([from_idx, to_idx])
        
        # 提取边特征
        edge_features = extract_edge_features(tx)
        edge_attr.append(edge_features)
        
        # 添加时间戳
        timestamps.append(tx['timestamp'])
    
    # 提取节点特征
    node_features = []
    for addr in related_addresses:
        features = extract_address_features(df, addr)
        
        # 将特征字典转换为列表
        feature_list = [
            features['total_outgoing'],
            features['total_incoming'],
            features['total_outgoing_value'],
            features['total_incoming_value'],
            features['avg_time_diff'],
            features['outgoing_mean'],
            features['outgoing_std'],
            features['incoming_mean'],
            features['incoming_std'],
            features['outgoing_receivers'],
            features['incoming_senders']
        ]
        
        node_features.append(feature_list)
    
    # 转换为张量
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    timestamps = torch.tensor(timestamps, dtype=torch.float)
    
    # 创建PyG数据对象
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        timestamps=timestamps,
        address_id=target_address
    )
    
    return data


def process_transaction_data(
    input_file: str,
    output_dir: str,
    max_addresses: Optional[int] = None,
    max_neighbors: int = 100,
    max_transactions: int = 1000
) -> None:
    """
    处理交易数据，为每个地址构建行为图并保存
    
    参数:
        input_file (str): 输入交易数据文件路径
        output_dir (str): 输出图文件目录
        max_addresses (int, 可选): 最大处理地址数量，None表示处理所有地址
        max_neighbors (int): 每个地址图的最大邻居数量
        max_transactions (int): 每个地址图的最大交易数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载交易数据
    df = load_transaction_data(input_file)
    
    # 获取所有唯一地址
    unique_addresses = set(df['from_address'].values) | set(df['to_address'].values)
    logger.info(f"找到{len(unique_addresses)}个唯一地址")
    
    # 选择要处理的地址
    if max_addresses is not None and max_addresses < len(unique_addresses):
        # 选择交易最活跃的地址
        address_counts = {}
        for addr in tqdm(unique_addresses, desc="计算地址活跃度"):
            count = ((df['from_address'] == addr) | (df['to_address'] == addr)).sum()
            address_counts[addr] = count
        
        sorted_addresses = sorted(address_counts.items(), key=lambda x: x[1], reverse=True)
        addresses_to_process = [addr for addr, _ in sorted_addresses[:max_addresses]]
    else:
        addresses_to_process = list(unique_addresses)
    
    logger.info(f"将处理{len(addresses_to_process)}个地址")
    
    # 为每个地址构建图并保存
    for i, address in enumerate(tqdm(addresses_to_process, desc="构建地址图")):
        try:
            # 构建地址图
            graph = build_address_graph(
                df, 
                address, 
                max_neighbors=max_neighbors,
                max_transactions=max_transactions
            )
            
            # 保存图
            output_file = os.path.join(output_dir, f"{address}.pt")
            torch.save(graph, output_file)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理{i + 1}/{len(addresses_to_process)}个地址")
                
        except Exception as e:
            logger.error(f"处理地址{address}时出错: {str(e)}")
    
    logger.info(f"完成所有地址处理，共{len(addresses_to_process)}个地址的图已保存到{output_dir}")


def visualize_address_graph(data: Data, output_file: Optional[str] = None) -> nx.Graph:
    """
    可视化地址行为图
    
    参数:
        data (Data): PyG格式的图数据
        output_file (str, 可选): 输出图像文件路径
        
    返回:
        nx.Graph: NetworkX图对象
    """
    try:
        import matplotlib.pyplot as plt
        
        # 将PyG数据转换为NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(data.x.size(0)):
            G.add_node(i, features=data.x[i].numpy())
        
        # 添加边
        edge_index = data.edge_index.t().numpy()
        edge_attr = data.edge_attr.numpy() if hasattr(data, "edge_attr") else None
        timestamps = data.timestamps.numpy() if hasattr(data, "timestamps") else None
        
        for i in range(edge_index.shape[0]):
            source, target = edge_index[i]
            
            edge_data = {}
            if edge_attr is not None:
                edge_data['attr'] = edge_attr[i]
            
            if timestamps is not None:
                edge_data['timestamp'] = timestamps[i]
            
            G.add_edge(source, target, **edge_data)
        
        # 绘制图
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=300, alpha=0.7)
        
        # 绘制边，颜色基于值大小
        if edge_attr is not None:
            edge_values = [attr[0] for _, _, attr in G.edges(data='attr')]
            if len(edge_values) > 0:
                edge_values = np.array(edge_values)
                normalized_values = (edge_values - edge_values.min()) / (edge_values.max() - edge_values.min() + 1e-6)
                edge_colors = plt.cm.viridis(normalized_values)
                
                nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color=edge_colors)
            else:
                nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)
        else:
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.6)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title(f"地址行为图 - {data.address_id}")
        plt.axis('off')
        
        # 保存或显示图像
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return G
    
    except ImportError:
        logger.warning("未安装matplotlib，无法进行可视化")
        
        # 仍然返回NetworkX图对象
        G = nx.DiGraph()
        
        # 添加节点
        for i in range(data.x.size(0)):
            G.add_node(i, features=data.x[i].numpy())
        
        # 添加边
        edge_index = data.edge_index.t().numpy()
        edge_attr = data.edge_attr.numpy() if hasattr(data, "edge_attr") else None
        timestamps = data.timestamps.numpy() if hasattr(data, "timestamps") else None
        
        for i in range(edge_index.shape[0]):
            source, target = edge_index[i]
            
            edge_data = {}
            if edge_attr is not None:
                edge_data['attr'] = edge_attr[i]
            
            if timestamps is not None:
                edge_data['timestamp'] = timestamps[i]
            
            G.add_edge(source, target, **edge_data)
        
        return G


if __name__ == "__main__":
    import argparse
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理交易数据，构建地址行为图')
    parser.add_argument('--input', type=str, required=True, help='输入交易数据文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出图文件目录')
    parser.add_argument('--max_addresses', type=int, default=None, help='最大处理地址数量')
    parser.add_argument('--max_neighbors', type=int, default=100, help='每个地址图的最大邻居数量')
    parser.add_argument('--max_transactions', type=int, default=1000, help='每个地址图的最大交易数量')
    
    args = parser.parse_args()
    
    # 处理交易数据
    process_transaction_data(
        args.input,
        args.output,
        max_addresses=args.max_addresses,
        max_neighbors=args.max_neighbors,
        max_transactions=args.max_transactions
    ) 