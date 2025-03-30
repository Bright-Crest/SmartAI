#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import hydra
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Union, Tuple
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 添加项目根目录到系统路径，确保可以导入其他模块
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_dir)

from smartmoney_pipeline.evaluation.metrics import (
    evaluate_embeddings_retrieval,
    evaluate_embeddings_clustering,
    interclass_distance,
    silhouette_score,
    classification_metrics
)
from smartmoney_pipeline.utils.train_utils import (
    setup_logging,
    set_seed
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="evaluation_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    评估模型性能的主函数
    
    参数:
        cfg (DictConfig): Hydra配置对象
    """
    # 创建输出目录
    os.makedirs(cfg.output.log_dir, exist_ok=True)
    os.makedirs(cfg.output.plots_dir, exist_ok=True)
    os.makedirs(cfg.output.results_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(cfg.output.log_dir)
    
    # 设置随机种子
    if cfg.seed is None:
        cfg.seed = random.randint(0, 100000000)
    set_seed(cfg.seed)
    
    # 打印配置
    logger.info(f"运行评估任务: {cfg.evaluation.name}")
    logger.info(f"嵌入文件: {cfg.evaluation.embeddings_file}")
    logger.info(f"标签文件: {cfg.evaluation.labels_file}")
    
    # 保存配置
    config_path = os.path.join(cfg.output.log_dir, "evaluation_config.yaml")
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    # 加载嵌入和标签
    embeddings, labels, ids = load_data(
        cfg.evaluation.embeddings_file,
        cfg.evaluation.labels_file
    )
    
    # 打印数据集大小
    logger.info(f"加载了 {len(embeddings)} 个嵌入向量，维度为 {embeddings.shape[1]}")
    logger.info(f"标签数: {len(labels)}")
    
    # 检验嵌入的维度
    is_valid, validation_msg = validate_embeddings(embeddings)
    if not is_valid:
        logger.error(f"嵌入向量验证失败: {validation_msg}")
        return
    
    # 打印标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_distribution = {
        label: count 
        for label, count in zip(unique_labels, counts)
    }
    logger.info(f"标签分布: {label_distribution}")
    
    # 可视化嵌入
    visualize_embeddings(
        embeddings, 
        labels, 
        ids,
        output_path=os.path.join(cfg.output.plots_dir, "embeddings_tsne.png"),
        title=cfg.evaluation.name,
        label_map=cfg.evaluation.label_map
    )
    
    # 执行评估
    results = evaluate_embeddings(embeddings, labels, cfg)
    
    # 保存结果
    save_results(results, os.path.join(cfg.output.results_dir, "evaluation_results.json"))
    
    # 打印主要结果
    logger.info("评估结果摘要:")
    for metric_name, metric_value in results.items():
        if isinstance(metric_value, dict):
            logger.info(f"{metric_name}:")
            for k, v in metric_value.items():
                logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value:.4f}")
    
    logger.info(f"评估完成，结果已保存到 {cfg.output.results_dir}")


def load_data(
    embeddings_file: str, 
    labels_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    加载嵌入向量和标签
    
    参数:
        embeddings_file (str): 嵌入向量文件路径
        labels_file (str, 可选): 标签文件路径
        
    返回:
        Tuple: (嵌入向量, 标签, ID列表)
    """
    # 检查文件是否存在
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"找不到嵌入文件: {embeddings_file}")
    
    # 加载嵌入
    embeddings_data = np.load(embeddings_file, allow_pickle=True)
    
    # 解析嵌入数据
    if isinstance(embeddings_data, np.ndarray) and embeddings_data.ndim == 2:
        # 简单的嵌入数组
        embeddings = embeddings_data
        ids = [str(i) for i in range(len(embeddings))]
    elif isinstance(embeddings_data, np.lib.npyio.NpzFile):
        # .npz格式，包含嵌入和可能的IDs
        if 'embeddings' in embeddings_data:
            embeddings = embeddings_data['embeddings']
        else:
            # 尝试找到嵌入数组
            for key in embeddings_data.keys():
                if embeddings_data[key].ndim == 2:
                    embeddings = embeddings_data[key]
                    break
            else:
                raise ValueError(f"在文件中找不到嵌入向量: {embeddings_file}")
        
        # 提取IDs
        if 'ids' in embeddings_data:
            ids = embeddings_data['ids']
            if isinstance(ids[0], np.ndarray):
                ids = [str(id[0]) for id in ids]
            else:
                ids = [str(id) for id in ids]
        else:
            ids = [str(i) for i in range(len(embeddings))]
    else:
        raise ValueError(f"不支持的嵌入文件格式: {embeddings_file}")
    
    # 加载标签
    if labels_file and os.path.exists(labels_file):
        # 根据文件扩展名确定加载方式
        if labels_file.endswith('.csv'):
            # 加载CSV标签文件
            df = pd.read_csv(labels_file)
            
            # 检查ID列和标签列
            id_col = 'id'
            label_col = 'label'
            
            if id_col not in df.columns:
                # 尝试其他可能的ID列名
                for col in ['address_id', 'address', 'addr_id', 'ID']:
                    if col in df.columns:
                        id_col = col
                        break
                else:
                    logger.warning(f"在标签文件中找不到ID列，使用第一列作为ID")
                    id_col = df.columns[0]
            
            if label_col not in df.columns:
                # 尝试其他可能的标签列名
                for col in ['category', 'class', 'type', 'classification']:
                    if col in df.columns:
                        label_col = col
                        break
                else:
                    logger.warning(f"在标签文件中找不到标签列，使用最后一列作为标签")
                    label_col = df.columns[-1]
            
            # 创建ID到标签的映射
            id_to_label = dict(zip(df[id_col].astype(str), df[label_col]))
            
            # 将标签映射到嵌入ID
            labels = np.array([id_to_label.get(id, -1) for id in ids])
            
            # 移除无标签的样本
            valid_mask = labels != -1
            if valid_mask.sum() < len(labels):
                logger.warning(f"过滤 {len(labels) - valid_mask.sum()} 个无标签的样本")
                embeddings = embeddings[valid_mask]
                labels = labels[valid_mask]
                ids = [id for i, id in enumerate(ids) if valid_mask[i]]
            
        elif labels_file.endswith('.npy') or labels_file.endswith('.npz'):
            # 加载NumPy格式的标签
            labels_data = np.load(labels_file, allow_pickle=True)
            
            if isinstance(labels_data, np.lib.npyio.NpzFile):
                # .npz格式
                if 'labels' in labels_data:
                    labels = labels_data['labels']
                elif 'classes' in labels_data:
                    labels = labels_data['classes']
                else:
                    raise ValueError(f"在标签文件中找不到标签数组: {labels_file}")
            else:
                # 直接的NumPy数组
                labels = labels_data
            
            # 确保标签和嵌入数量匹配
            if len(labels) != len(embeddings):
                raise ValueError(f"标签数量 ({len(labels)}) 与嵌入数量 ({len(embeddings)}) 不匹配")
        
        else:
            raise ValueError(f"不支持的标签文件格式: {labels_file}")
    else:
        # 如果没有标签文件，使用全零标签
        logger.warning("没有提供标签文件，将使用全零标签")
        labels = np.zeros(len(embeddings), dtype=int)
    
    # 标签编码
    if not np.issubdtype(labels.dtype, np.number):
        logger.info("将非数字标签转换为数值标签")
        le = LabelEncoder()
        labels = le.fit_transform(labels)
    
    return embeddings, labels, ids


def validate_embeddings(embeddings: np.ndarray) -> Tuple[bool, str]:
    """
    验证嵌入向量的有效性
    
    参数:
        embeddings (np.ndarray): 嵌入向量
        
    返回:
        Tuple: (是否有效, 错误消息)
    """
    # 检查是否为空
    if embeddings.size == 0:
        return False, "嵌入向量为空"
    
    # 检查NaN和无穷大
    if np.isnan(embeddings).any():
        return False, "嵌入向量包含NaN值"
    
    if np.isinf(embeddings).any():
        return False, "嵌入向量包含无穷大值"
    
    # 检查是否全零
    if np.all(embeddings == 0):
        return False, "嵌入向量全为零"
    
    return True, "嵌入向量有效"


def visualize_embeddings(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    ids: List[str],
    output_path: str,
    title: str = "嵌入向量可视化",
    label_map: Optional[Dict[str, str]] = None
) -> None:
    """
    可视化嵌入向量
    
    参数:
        embeddings (np.ndarray): 嵌入向量
        labels (np.ndarray): 标签
        ids (List[str]): ID列表
        output_path (str): 输出文件路径
        title (str): 图像标题
        label_map (Dict, 可选): 标签ID到名称的映射
    """
    # 使用t-SNE降维到2D
    logger.info("使用t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 准备标签名映射
    if label_map:
        # 转换为整数键，因为标签已编码为整数
        int_label_map = {}
        for k, v in label_map.items():
            try:
                int_label_map[int(k)] = v
            except (ValueError, TypeError):
                # 如果键不能转换为整数，则跳过
                continue
        
        label_names = [int_label_map.get(label, str(label)) for label in labels]
    else:
        label_names = [str(label) for label in labels]
    
    # 为每个唯一标签分配颜色
    unique_labels = np.unique(labels)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制散点图，按标签着色
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # 添加图例
    if len(unique_labels) <= 20:
        # 只在类别数量合理时添加图例
        legend_labels = [f"{label} ({label_names[np.where(labels==label)[0][0]]}" + 
                        f": {np.sum(labels==label)})" for label in unique_labels]
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, 
                  title="类别 (标签: 数量)", loc="best", fontsize=9)
    
    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel("t-SNE维度1")
    plt.ylabel("t-SNE维度2")
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"嵌入可视化已保存到 {output_path}")


def evaluate_embeddings(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    cfg: DictConfig
) -> Dict[str, Any]:
    """
    评估嵌入向量的质量
    
    参数:
        embeddings (np.ndarray): 嵌入向量
        labels (np.ndarray): 标签
        cfg (DictConfig): 配置
        
    返回:
        Dict: 评估结果
    """
    results = {}
    
    # 检索评估
    if cfg.evaluation.tasks.retrieval:
        logger.info("执行检索评估...")
        retrieval_metrics = evaluate_embeddings_retrieval(embeddings, labels)
        results['retrieval'] = retrieval_metrics
        
        logger.info(f"检索评估结果: P@1={retrieval_metrics['p@1']:.4f}, " + 
                   f"P@5={retrieval_metrics['p@5']:.4f}, " + 
                   f"P@10={retrieval_metrics['p@10']:.4f}, " + 
                   f"MAP={retrieval_metrics['map']:.4f}")
    
    # 聚类评估
    if cfg.evaluation.tasks.clustering:
        logger.info("执行聚类评估...")
        clustering_metrics = evaluate_embeddings_clustering(
            embeddings, 
            labels, 
            n_clusters=cfg.evaluation.n_clusters
        )
        results['clustering'] = clustering_metrics
        
        logger.info(f"聚类评估结果: NMI={clustering_metrics['nmi']:.4f}, " + 
                   f"ARI={clustering_metrics['ari']:.4f}")
    
    # 计算类内和类间距离
    if cfg.evaluation.tasks.distances:
        logger.info("计算类内和类间距离...")
        distance_metrics = interclass_distance(embeddings, labels)
        results['distances'] = distance_metrics
        
        logger.info(f"距离评估结果: 类内距离={distance_metrics['intra_class_distance']:.4f}, " + 
                   f"类间距离={distance_metrics['inter_class_distance']:.4f}, " + 
                   f"比率={distance_metrics['distance_ratio']:.4f}")
    
    # 计算轮廓系数
    if cfg.evaluation.tasks.silhouette:
        logger.info("计算轮廓系数...")
        try:
            sil_score = silhouette_score(embeddings, labels)
            results['silhouette_score'] = sil_score
            logger.info(f"轮廓系数: {sil_score:.4f}")
        except Exception as e:
            logger.error(f"计算轮廓系数失败: {str(e)}")
            results['silhouette_score'] = float('nan')
    
    # 分类评估
    if cfg.evaluation.tasks.classification:
        logger.info("执行分类评估...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, 
            labels, 
            test_size=0.3, 
            random_state=42,
            stratify=labels
        )
        
        # 使用不同的分类器进行评估
        classification_results = {}
        
        # 线性SVM
        if 'svm' in cfg.evaluation.classifiers:
            try:
                from sklearn.svm import LinearSVC
                from sklearn.calibration import CalibratedClassifierCV
                
                logger.info("使用线性SVM分类器...")
                
                # 使用CalibratedClassifierCV包装LinearSVC，使其可以输出概率
                model = CalibratedClassifierCV(
                    LinearSVC(random_state=42), 
                    cv=5
                )
                
                # 训练并预测
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 计算指标
                metrics = classification_metrics(y_test, y_pred)
                classification_results['svm'] = metrics
                
                logger.info(f"SVM分类结果: 准确率={metrics['accuracy']:.4f}, " + 
                           f"F1分数={metrics['f1']:.4f}")
                
            except ImportError:
                logger.error("无法导入LinearSVC，跳过SVM分类评估")
        
        # 随机森林
        if 'random_forest' in cfg.evaluation.classifiers:
            try:
                from sklearn.ensemble import RandomForestClassifier
                
                logger.info("使用随机森林分类器...")
                
                # 训练并预测
                model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 计算指标
                metrics = classification_metrics(y_test, y_pred)
                classification_results['random_forest'] = metrics
                
                logger.info(f"随机森林分类结果: 准确率={metrics['accuracy']:.4f}, " + 
                           f"F1分数={metrics['f1']:.4f}")
                
            except ImportError:
                logger.error("无法导入RandomForestClassifier，跳过随机森林分类评估")
        
        # 神经网络
        if 'neural_network' in cfg.evaluation.classifiers:
            try:
                from sklearn.neural_network import MLPClassifier
                
                logger.info("使用神经网络分类器...")
                
                # 训练并预测
                model = MLPClassifier(
                    hidden_layer_sizes=(100,), 
                    max_iter=300, 
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 计算指标
                metrics = classification_metrics(y_test, y_pred)
                classification_results['neural_network'] = metrics
                
                logger.info(f"神经网络分类结果: 准确率={metrics['accuracy']:.4f}, " + 
                           f"F1分数={metrics['f1']:.4f}")
                
            except ImportError:
                logger.error("无法导入MLPClassifier，跳过神经网络分类评估")
        
        results['classification'] = classification_results
    
    return results


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    保存评估结果
    
    参数:
        results (Dict): 评估结果
        output_path (str): 输出文件路径
    """
    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将numpy数组转换为普通Python类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    # 转换结果
    converted_results = convert_numpy(results)
    
    # 保存为JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    logger.info(f"评估结果已保存到 {output_path}")


if __name__ == "__main__":
    main() 