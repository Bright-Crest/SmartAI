import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datetime import datetime
from data_generator import BlockchainDataGenerator


def plot_transaction_statistics(transactions):
    """绘制交易数据的基本统计图表"""
    # 创建图表保存目录
    os.makedirs('data/plots', exist_ok=True)
    
    # 转换为DataFrame以便于分析
    df = pd.DataFrame(transactions)
    
    # 1. 交易价值分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['value'].clip(upper=df['value'].quantile(0.95)), bins=50, kde=True)
    plt.title('交易价值分布 (排除最高5%)')
    plt.xlabel('交易价值')
    plt.ylabel('频率')
    plt.savefig('data/plots/transaction_value_dist.png')
    plt.close()
    
    # 2. 每日交易量时间序列
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    daily_counts = df.groupby('date').size()
    
    plt.figure(figsize=(12, 6))
    daily_counts.plot()
    plt.title('每日交易量')
    plt.xlabel('日期')
    plt.ylabel('交易数量')
    plt.savefig('data/plots/daily_transaction_volume.png')
    plt.close()
    
    # 3. 合约交互与非合约交互的交易比例
    contract_counts = df['is_contract_interaction'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(contract_counts, labels=['普通交易', '合约交互'] if 0 in contract_counts.index else ['合约交互', '普通交易'],
            autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title('合约交互与普通交易比例')
    plt.savefig('data/plots/contract_interaction_ratio.png')
    plt.close()
    
    return df

def analyze_address_behavior(generator, df):
    """分析不同类型地址的行为模式"""
    # 获取地址类型信息
    address_types = {}
    for addr, info in generator.addresses.items():
        address_types[addr] = info['type']
    
    # 添加发送方地址类型到交易数据
    df['sender_type'] = df['from'].map(address_types)
    
    # 按地址类型分组分析
    type_stats = df.groupby('sender_type').agg({
        'value': ['mean', 'median', 'std', 'count'],
        'gas': ['mean', 'median'],
        'is_contract_interaction': 'mean'  # 合约交互比例
    })
    
    # 打印统计信息
    print("\n不同类型地址的行为统计:")
    print(type_stats)
    
    # 绘制每种类型地址的交易价值箱型图
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='sender_type', y='value', data=df[df['value'] < df['value'].quantile(0.95)])
    plt.title('不同类型地址的交易价值分布')
    plt.xlabel('地址类型')
    plt.ylabel('交易价值')
    plt.savefig('data/plots/value_by_address_type.png')
    plt.close()
    
    # 绘制每种类型地址的合约交互比例
    contract_ratio = df.groupby('sender_type')['is_contract_interaction'].mean()
    plt.figure(figsize=(10, 6))
    contract_ratio.plot(kind='bar')
    plt.title('不同类型地址的合约交互比例')
    plt.xlabel('地址类型')
    plt.ylabel('合约交互比例')
    plt.savefig('data/plots/contract_ratio_by_type.png')
    plt.close()
    
    return type_stats

def verify_graph_data():
    """验证生成的图数据结构"""
    graph_files = os.listdir('data/graphs')
    if not graph_files:
        return
    
    # 随机选择10个样本图文件进行检查
    import random
    sample_size = min(10, len(graph_files))
    sample_files = random.sample(graph_files, sample_size)
    
    # 统计图的基本信息
    graph_stats = []
    for filename in sample_files:
        filepath = f"data/graphs/{filename}"
        data = torch.load(filepath)
        
        stats = {
            'address': data.address_id,
            'nodes': data.x.size(0),
            'edges': data.edge_index.size(1),
            'timestamps': len(data.timestamps),
            'time_span': (max(data.timestamps.numpy()) - min(data.timestamps.numpy())) / 86400  # 天数
        }
        graph_stats.append(stats)
    
    # 转换为DataFrame并打印
    stats_df = pd.DataFrame(graph_stats)
    print("\n样本图数据统计:")
    print(stats_df.describe())
    
    # 绘制节点数和边数的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(stats_df['nodes'], stats_df['edges'])
    plt.title('图数据节点数与边数关系')
    plt.xlabel('节点数')
    plt.ylabel('边数')
    plt.savefig('data/plots/nodes_vs_edges.png')
    plt.close()
    
    return stats_df

def main():
    """主函数：生成数据并进行分析"""
    print("开始生成区块链交易模拟数据...")
    
    # 创建数据和图表目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/plots', exist_ok=True)
    
    # 生成模拟数据
    generator = BlockchainDataGenerator(
        num_addresses=1000,
        num_contracts=20,
        num_days=30,
        smart_money_ratio=0.08,
        exchange_ratio=0.03,
        bot_ratio=0.12
    )
    
    start_time = datetime.now()
    transactions = generator.generate_data()
    end_time = datetime.now()
    
    print(f"数据生成完成，耗时: {(end_time - start_time).total_seconds():.2f} 秒")
    print(f"生成的交易总数: {len(transactions)}")
    
    # 分析交易数据
    print("开始分析交易数据...")
    df = plot_transaction_statistics(transactions)
    
    # 分析地址行为
    type_stats = analyze_address_behavior(generator, df)
    
    # 验证图数据
    graph_stats = verify_graph_data()
    
    print("\n数据分析完成，图表已保存至 data/plots/ 目录")
    
    # 保存一些基本统计数据为CSV
    df.describe().to_csv('data/transaction_stats.csv')
    
    # 返回生成的数据以便进一步分析
    return {
        'generator': generator,
        'transactions': transactions,
        'df': df,
        'type_stats': type_stats,
        'graph_stats': graph_stats
    }

if __name__ == "__main__":
    results = main() 