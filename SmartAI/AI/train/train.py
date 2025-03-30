#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import hydra
import random
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import datetime
from torch.utils.data import DataLoader

# 添加项目根目录到系统路径，确保可以导入其他模块
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_dir)

from smartmoney_pipeline.utils.train_utils import (
    setup_logging, 
    set_seed, 
    get_device, 
    create_directories,
    print_config
)
from smartmoney_pipeline.utils.dataset import (
    AddressGraphDataset, 
    TimeWindowGraphDataset, 
    GraphAugmentation,
    SequenceAugmentation,
    create_dataloader,
    collate_time_windows,
    collate_address_graphs
)
from smartmoney_pipeline.models.smartmoney_model import SmartMoneyModel
from smartmoney_pipeline.train.trainer import Trainer

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    训练SmartMoney模型的主函数
    
    参数:
        cfg (DictConfig): Hydra配置对象
    """
    # 获取当前时间，用于日志和输出文件命名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    output_dir = os.path.join(cfg.output.base_dir, f"{cfg.project_name}_{timestamp}")
    cfg.output.log_dir = os.path.join(output_dir, "logs")
    cfg.output.model_dir = os.path.join(output_dir, "models")
    cfg.output.tensorboard_dir = os.path.join(output_dir, "tensorboard")
    cfg.output.results_dir = os.path.join(output_dir, "results")
    cfg.output.plots_dir = os.path.join(output_dir, "plots")
    
    # 创建所需目录
    create_directories(cfg)
    
    # 设置日志
    setup_logging(cfg.output.log_dir)
    
    # 打印配置信息
    logger.info(f"运行训练任务: {cfg.project_name}")
    logger.info(f"输出目录: {output_dir}")
    print_config(cfg)
    
    # 保存完整配置
    config_save_path = os.path.join(cfg.output.log_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"配置已保存到 {config_save_path}")
    
    # 设置随机种子
    if cfg.seed is None:
        cfg.seed = random.randint(0, 100000000)
    set_seed(cfg.seed)
    
    # 获取设备
    device = get_device(cfg.device)
    logger.info(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset, test_dataset = create_datasets(cfg)
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_time_windows if isinstance(train_dataset, TimeWindowGraphDataset) else collate_address_graphs
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_time_windows if isinstance(val_dataset, TimeWindowGraphDataset) else collate_address_graphs
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = create_dataloader(
            test_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_time_windows if isinstance(test_dataset, TimeWindowGraphDataset) else collate_address_graphs
        )
    
    # 创建模型
    model = create_model(cfg)
    logger.info(f"创建模型: {cfg.model.name}")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # 开始训练
    logger.info("开始训练...")
    train_results = trainer.train()
    
    # 打印训练结果
    logger.info("训练完成!")
    logger.info(f"训练轮数: {train_results['epochs_trained']}")
    logger.info(f"最佳验证损失: {train_results['best_val_loss']:.6f} (轮数 {train_results['best_epoch']})")
    logger.info(f"总训练时间: {train_results['total_time']:.2f}秒")
    
    # 保存训练结果摘要
    summary_path = os.path.join(cfg.output.results_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"项目名称: {cfg.project_name}\n")
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"训练轮数: {train_results['epochs_trained']}\n")
        f.write(f"最佳验证损失: {train_results['best_val_loss']:.6f} (轮数 {train_results['best_epoch']})\n")
        f.write(f"总训练时间: {train_results['total_time']:.2f}秒\n")
        f.write(f"模型保存路径: {os.path.join(cfg.output.model_dir, 'best_model.pth')}\n")
    
    logger.info(f"训练摘要已保存到 {summary_path}")


def create_datasets(cfg: DictConfig):
    """
    创建训练、验证和测试数据集
    
    参数:
        cfg (DictConfig): 配置对象
        
    返回:
        Tuple: 训练、验证和测试数据集
    """
    dataset_type = cfg.data.dataset_type.lower()
    
    # 数据增强
    if dataset_type == "address_graph":
        # 地址图数据集的增强
        augmentation = GraphAugmentation(
            edge_drop_rate=cfg.data.augmentation.edge_drop_rate,
            feat_mask_rate=cfg.data.augmentation.feat_mask_rate,
            time_mask_prob=cfg.data.augmentation.time_mask_prob
        )
        
        dataset = AddressGraphDataset(
            root_dir=cfg.data.graphs_dir,
            pre_transform=None,
            transform=augmentation,
            mode='train',
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio
        )
        
        train_dataset = dataset.get_subset('train')
        val_dataset = dataset.get_subset('val')
        test_dataset = dataset.get_subset('test')
        
    elif dataset_type == "time_window":
        # 时间窗口图数据集的增强
        augmentation = SequenceAugmentation(
            seq_mask_rate=cfg.data.augmentation.seq_mask_rate,
            time_stretch_factor=cfg.data.augmentation.time_stretch_factor,
            random_crop=cfg.data.augmentation.random_crop
        )
        
        dataset = TimeWindowGraphDataset(
            root_dir=cfg.data.graphs_dir,
            time_window=cfg.data.time_window,
            max_windows=cfg.data.max_time_windows,
            transform=augmentation,
            mode='train',
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio
        )
        
        train_dataset = dataset.get_subset('train')
        val_dataset = dataset.get_subset('val')
        test_dataset = dataset.get_subset('test')
    
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    logger.info(f"创建 {dataset_type} 数据集:")
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_model(cfg: DictConfig) -> SmartMoneyModel:
    """
    创建SmartMoney模型
    
    参数:
        cfg (DictConfig): 配置对象
        
    返回:
        SmartMoneyModel: 模型实例
    """
    # 获取数据集示例来确定特征维度
    sample_path = find_sample_graph(cfg.data.graphs_dir)
    if sample_path is None:
        # 如果找不到示例，则使用配置中的默认值
        node_feature_dim = cfg.model.node_dim
        edge_feature_dim = cfg.model.edge_dim
        # 假设节点数量为配置中的值或默认值10000
        num_nodes = cfg.model.num_nodes if hasattr(cfg.model, 'num_nodes') else 10000
        logger.warning(f"找不到示例图，使用配置中的默认特征维度: node_feature_dim={node_feature_dim}, edge_feature_dim={edge_feature_dim}, num_nodes={num_nodes}")
    else:
        # 从示例中获取特征维度
        sample_data = torch.load(sample_path)
        # 确定节点数量
        if hasattr(sample_data, 'x') and sample_data.x is not None:
            num_nodes = sample_data.x.size(0)
            node_feature_dim = sample_data.x.size(1)
        else:
            num_nodes = cfg.model.num_nodes if hasattr(cfg.model, 'num_nodes') else 10000
            node_feature_dim = cfg.model.node_dim
            logger.warning(f"示例图中无节点特征，使用默认值: node_feature_dim={node_feature_dim}, num_nodes={num_nodes}")
        
        if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None:
            edge_feature_dim = sample_data.edge_attr.size(1)
        else:
            edge_feature_dim = cfg.model.edge_dim
            logger.warning(f"示例图中无边特征，使用默认值: edge_feature_dim={edge_feature_dim}")
    
    # 将模型配置转换为字典
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    
    # 创建SmartMoney模型
    model = SmartMoneyModel(
        config=model_config,
        num_nodes=num_nodes,
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim
    )
    
    return model


def find_sample_graph(graphs_dir: str):
    """
    在图目录中找到一个示例图文件
    
    参数:
        graphs_dir (str): 图目录
        
    返回:
        str 或 None: 找到的示例图路径，如果没有找到则返回None
    """
    # 检查目录是否存在
    if not os.path.exists(graphs_dir):
        logger.warning(f"图目录不存在: {graphs_dir}")
        return None
    
    # 寻找.pt文件
    for root, _, files in os.walk(graphs_dir):
        for file in files:
            if file.endswith('.pt'):
                sample_path = os.path.join(root, file)
                logger.info(f"找到示例图: {sample_path}")
                return sample_path
    
    logger.warning(f"在 {graphs_dir} 中找不到.pt图文件")
    return None


if __name__ == "__main__":
    main() 