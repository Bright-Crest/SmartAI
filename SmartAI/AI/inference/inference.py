#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import random
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Union, Tuple

# 添加项目根目录到系统路径，确保可以导入其他模块
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_dir)

from smartmoney_pipeline.utils.train_utils import (
    setup_logging, 
    set_seed, 
    get_device,
    load_checkpoint
)
from smartmoney_pipeline.models.smartmoney_model import SmartMoneyModel
from smartmoney_pipeline.utils.dataset import (
    AddressGraphDataset, 
    TimeWindowGraphDataset,
    create_dataloader,
    collate_time_windows,
    collate_address_graphs
)

logger = logging.getLogger(__name__)


class SmartMoneyInference:
    """
    SmartMoney模型推理类
    """
    def __init__(
        self, 
        model_path: str, 
        config: DictConfig, 
        device: Optional[str] = None
    ):
        """
        初始化推理类
        
        参数:
            model_path (str): 模型路径
            config (DictConfig): 配置
            device (str, 可选): 设备
        """
        self.config = config
        
        # 设置设备
        self.device = get_device(device if device is not None else config.device)
        logger.info(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载模型权重
        self._load_model(model_path)
        
        # 设置为评估模式
        self.model.eval()
        logger.info("模型已加载并设置为评估模式")
    
    def _create_model(self) -> SmartMoneyModel:
        """
        创建模型
        
        返回:
            SmartMoneyModel: 模型
        """
        # 将模型配置转换为字典
        model_config = OmegaConf.to_container(self.config.model, resolve=True)
        
        # 假设默认节点数量
        num_nodes = getattr(self.config.model, 'num_nodes', 10000)
        
        # 从配置中获取模型参数
        model = SmartMoneyModel(
            config=model_config,
            num_nodes=num_nodes,
            node_feature_dim=self.config.model.node_dim,
            edge_feature_dim=self.config.model.edge_dim
        )
        
        return model.to(self.device)
    
    def _load_model(self, model_path: str) -> None:
        """
        加载模型权重
        
        参数:
            model_path (str): 模型路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 加载检查点
        checkpoint = load_checkpoint(
            self.model, 
            model_path, 
            device=self.device
        )
        
        logger.info(f"从 {model_path} 加载模型")
        if 'epoch' in checkpoint:
            logger.info(f"模型轮数: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            logger.info(f"模型损失: {checkpoint['loss']:.6f}")
    
    def encode_single_address(
        self, 
        graph_file: str
    ) -> np.ndarray:
        """
        对单个地址图进行编码
        
        参数:
            graph_file (str): 地址图文件路径
            
        返回:
            np.ndarray: 编码后的特征向量
        """
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"找不到图文件: {graph_file}")
        
        # 加载图数据
        try:
            graph_data = torch.load(graph_file, weights_only=False)
        except TypeError:
            graph_data = torch.load(graph_file)
        
        # 将数据移动到设备
        if isinstance(graph_data, tuple):
            graph_data = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in graph_data)
        else:
            graph_data = graph_data.to(self.device)
        
        # 推理
        with torch.no_grad():
            # 获取编码
            if self.config.data.dataset_type.lower() == "address_graph":
                embedding = self.model.encode_graph(graph_data)
            else:
                embedding = self.model.encode_sequence(graph_data)
        
        return embedding.cpu().numpy()
    
    def batch_encode(
        self, 
        dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        批量编码地址
        
        参数:
            dataset: 数据集
            batch_size (int): 批大小
            num_workers (int): 工作线程数
            
        返回:
            Dict: 地址ID到编码的映射
        """
        # 创建数据加载器
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_time_windows if isinstance(dataset, TimeWindowGraphDataset) else collate_address_graphs
        )
        
        # 存储结果
        embeddings = []
        address_ids = []
        
        # 批量处理
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="编码地址"):
                # 处理批次数据
                if isinstance(dataset, TimeWindowGraphDataset):
                    # 时间窗口数据集
                    data_batch, ids = batch
                    data_batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in data_batch)
                    batch_embeddings = self.model.encode_sequence(data_batch)
                else:
                    # 普通地址图数据集
                    if isinstance(batch, tuple) and len(batch) == 2:
                        data_batch, ids = batch
                    else:
                        data_batch = batch
                        ids = [i for i in range(len(batch))]
                    
                    if isinstance(data_batch, list):
                        # 单独处理每个图
                        batch_embeddings = []
                        for graph in data_batch:
                            graph = graph.to(self.device)
                            embedding = self.model.encode_graph(graph)
                            batch_embeddings.append(embedding)
                        batch_embeddings = torch.cat(batch_embeddings, dim=0)
                    else:
                        # 将数据移动到设备
                        data_batch = data_batch.to(self.device)
                        # 获取编码
                        batch_embeddings = self.model.encode_graph(data_batch)
                
                # 保存结果
                embeddings.append(batch_embeddings.cpu().numpy())
                address_ids.extend(ids)
        
        # 合并结果
        all_embeddings = np.concatenate(embeddings, axis=0)
        
        # 创建地址ID到编码的映射
        result = {
            addr_id: embedding 
            for addr_id, embedding in zip(address_ids, all_embeddings)
        }
        
        logger.info(f"完成 {len(result)} 个地址的编码")
        
        return result
    
    def find_similar_addresses(
        self, 
        query_embedding: np.ndarray, 
        address_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        查找与查询地址相似的地址
        
        参数:
            query_embedding (np.ndarray): 查询地址的编码
            address_embeddings (Dict): 地址ID到编码的映射
            top_k (int): 返回的相似地址数量
            
        返回:
            List[Tuple]: 相似地址列表，每个元素为(地址ID, 相似度)
        """
        results = []
        
        # 计算余弦相似度
        for addr_id, embedding in address_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((addr_id, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        return results[:top_k]
    
    def _cosine_similarity(
        self, 
        a: np.ndarray, 
        b: np.ndarray
    ) -> float:
        """
        计算余弦相似度
        
        参数:
            a (np.ndarray): 向量a
            b (np.ndarray): 向量b
            
        返回:
            float: 余弦相似度
        """
        # 归一化
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        # 避免除零
        if a_norm == 0 or b_norm == 0:
            return 0
            
        # 计算余弦相似度
        return np.dot(a, b) / (a_norm * b_norm)
    
    def analyze_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray],
        labels: Optional[Dict[str, int]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析嵌入向量
        
        参数:
            embeddings (Dict): 地址ID到编码的映射
            labels (Dict, 可选): 地址ID到标签的映射
            output_dir (str, 可选): 输出目录
            
        返回:
            Dict: 分析结果
        """
        import sklearn.cluster as cluster
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # 提取嵌入和ID
        ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[id] for id in ids])
        
        # 降维
        logger.info("使用t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embedding_matrix)
        
        # 聚类
        logger.info("使用KMeans进行聚类...")
        kmeans = cluster.KMeans(n_clusters=min(10, len(embeddings)), random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        
        # 可视化
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # 绘制t-SNE图
            plt.figure(figsize=(12, 10))
            
            if labels is not None:
                # 使用真实标签
                label_values = [labels.get(id, -1) for id in ids]
                scatter = plt.scatter(
                    embeddings_2d[:, 0], 
                    embeddings_2d[:, 1], 
                    c=label_values, 
                    cmap='viridis', 
                    alpha=0.7
                )
                plt.colorbar(scatter, label='标签')
                plt.title('地址嵌入t-SNE可视化 (真实标签)')
            else:
                # 使用聚类标签
                scatter = plt.scatter(
                    embeddings_2d[:, 0], 
                    embeddings_2d[:, 1], 
                    c=cluster_labels, 
                    cmap='viridis', 
                    alpha=0.7
                )
                plt.colorbar(scatter, label='聚类')
                plt.title('地址嵌入t-SNE可视化 (聚类)')
            
            plt.savefig(os.path.join(output_dir, 'embeddings_tsne.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存嵌入向量
            np.savez(
                os.path.join(output_dir, 'embeddings.npz'),
                ids=ids,
                embeddings=embedding_matrix,
                clusters=cluster_labels
            )
            
            logger.info(f"结果已保存到 {output_dir}")
        
        # 返回分析结果
        return {
            'ids': ids,
            'embeddings_2d': embeddings_2d,
            'cluster_labels': cluster_labels
        }
    
    def predict_address_type(
        self, 
        graph_file: str,
        label_map: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        预测地址类型（如果模型支持）
        
        参数:
            graph_file (str): 地址图文件路径
            label_map (Dict, 可选): 标签ID到名称的映射
            
        返回:
            Dict: 预测结果
        """
        # 检查模型是否支持分类
        if not hasattr(self.model, 'classify'):
            logger.warning("当前模型不支持分类预测")
            return {'error': '当前模型不支持分类预测'}
        
        # 编码地址
        embedding = self.encode_single_address(graph_file)
        embedding_tensor = torch.tensor(embedding).to(self.device)
        
        # 预测
        with torch.no_grad():
            logits = self.model.classify(embedding_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # 获取预测类别
            pred_class = torch.argmax(probs, dim=1).cpu().numpy()[0]
            pred_prob = probs[0, pred_class].cpu().item()
        
        # 构造结果
        result = {
            'class_id': int(pred_class),
            'confidence': float(pred_prob),
            'embedding': embedding
        }
        
        # 添加类别名称（如果有）
        if label_map is not None and pred_class in label_map:
            result['class_name'] = label_map[pred_class]
        
        return result


@hydra.main(config_path="../configs", config_name="inference_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    推理脚本的入口点
    
    参数:
        cfg (DictConfig): Hydra配置对象
    """
    # 设置日志
    os.makedirs(cfg.output.log_dir, exist_ok=True)
    setup_logging(cfg.output.log_dir)
    
    # 设置随机种子
    if cfg.seed is None:
        cfg.seed = random.randint(0, 100000000)
    set_seed(cfg.seed)
    
    # 初始化推理类
    inferencer = SmartMoneyInference(
        model_path=cfg.model.path,
        config=cfg,
        device=cfg.device
    )
    
    # 检查操作模式
    if cfg.mode == 'single':
        # 单个地址推理
        if not os.path.exists(cfg.inference.input_file):
            logger.error(f"找不到输入文件: {cfg.inference.input_file}")
            return
        
        try:
            # 编码地址
            embedding = inferencer.encode_single_address(cfg.inference.input_file)
            
            # 保存结果
            os.makedirs(os.path.dirname(cfg.output.output_file), exist_ok=True)
            np.save(cfg.output.output_file, embedding)
            
            logger.info(f"嵌入已保存到 {cfg.output.output_file}")
            
            # 打印维度信息
            logger.info(f"嵌入维度: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"推理出错: {str(e)}")
            
    elif cfg.mode == 'batch':
        # 批量推理
        try:
            # 创建数据集
            if cfg.data.dataset_type.lower() == "address_graph":
                dataset = AddressGraphDataset(
                    root_dir=cfg.data.graphs_dir,
                    pre_transform=None,
                    transform=None,
                    mode='all'
                )
            else:
                dataset = TimeWindowGraphDataset(
                    root_dir=cfg.data.graphs_dir,
                    time_window=cfg.data.time_window,
                    max_windows=cfg.data.max_time_windows,
                    transform=None,
                    mode='all'
                )
            
            logger.info(f"数据集大小: {len(dataset)}")
            
            # 批量编码
            embeddings = inferencer.batch_encode(
                dataset,
                batch_size=cfg.inference.batch_size,
                num_workers=cfg.data.num_workers
            )
            
            # 保存结果
            os.makedirs(os.path.dirname(cfg.output.output_file), exist_ok=True)
            np.savez(
                cfg.output.output_file,
                ids=list(embeddings.keys()),
                embeddings=np.array(list(embeddings.values()))
            )
            
            logger.info(f"嵌入已保存到 {cfg.output.output_file}")
            
            # 分析嵌入
            if cfg.inference.analyze_embeddings:
                # 创建输出目录
                analysis_dir = os.path.join(os.path.dirname(cfg.output.output_file), 'analysis')
                
                # 分析嵌入
                inferencer.analyze_embeddings(
                    embeddings,
                    output_dir=analysis_dir
                )
                
        except Exception as e:
            logger.error(f"批量推理出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    elif cfg.mode == 'similarity':
        # 相似度计算
        if not os.path.exists(cfg.inference.input_file):
            logger.error(f"找不到输入文件: {cfg.inference.input_file}")
            return
            
        if not os.path.exists(cfg.inference.reference_file):
            logger.error(f"找不到参考文件: {cfg.inference.reference_file}")
            return
            
        try:
            # 加载查询地址
            query_embedding = inferencer.encode_single_address(cfg.inference.input_file)
            
            # 加载参考嵌入
            ref_data = np.load(cfg.inference.reference_file, allow_pickle=True)
            
            # 检查文件格式
            if 'ids' in ref_data and 'embeddings' in ref_data:
                # .npz格式
                ids = ref_data['ids']
                embeddings = ref_data['embeddings']
                ref_embeddings = {id: emb for id, emb in zip(ids, embeddings)}
            else:
                # 单个.npy格式
                ref_embeddings = {'reference': ref_data}
            
            # 查找相似地址
            similar_addresses = inferencer.find_similar_addresses(
                query_embedding,
                ref_embeddings,
                top_k=cfg.inference.top_k
            )
            
            # 打印结果
            logger.info(f"找到 {len(similar_addresses)} 个相似地址:")
            for i, (addr_id, similarity) in enumerate(similar_addresses):
                logger.info(f"{i+1}. 地址ID: {addr_id}, 相似度: {similarity:.4f}")
            
            # 保存结果
            os.makedirs(os.path.dirname(cfg.output.output_file), exist_ok=True)
            with open(cfg.output.output_file, 'w') as f:
                f.write("地址ID,相似度\n")
                for addr_id, similarity in similar_addresses:
                    f.write(f"{addr_id},{similarity:.6f}\n")
                    
            logger.info(f"相似度结果已保存到 {cfg.output.output_file}")
            
        except Exception as e:
            logger.error(f"相似度计算出错: {str(e)}")
            
    else:
        logger.error(f"不支持的模式: {cfg.mode}")


if __name__ == "__main__":
    main() 