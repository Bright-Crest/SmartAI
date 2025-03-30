import os
import sys
import random
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig
from torch_geometric.data import Batch

# 添加项目根目录到系统路径，确保可以导入其他模块
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_dir)

from smartmoney_pipeline.utils.train_utils import (
    AverageMeter, 
    EarlyStopping, 
    save_checkpoint, 
    load_checkpoint, 
    plot_learning_curves,
    init_tensorboard,
    set_seed,
    get_device,
    create_optimizer,
    create_scheduler
)
from smartmoney_pipeline.models.contrastive_loss import NTXentLoss, ClusterLoss, SwAVLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    模型训练类，实现对比学习模型训练
    """
    def __init__(
        self, 
        config: DictConfig, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ):
        """
        初始化训练器
        
        参数:
            config (DictConfig): 训练配置
            model (nn.Module): 模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, 可选): 验证数据加载器
            test_loader (DataLoader, 可选): 测试数据加载器
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 设置随机种子
        if config.seed is None:
            config.seed = random.randint(0, 100000000)
        set_seed(config.seed)
        
        # 设置设备
        self.device = get_device(config.device)
        self.model = self.model.to(self.device)
        
        # 创建优化器和调度器
        self.optimizer = create_optimizer(model, config.train.optimizer)
        self.scheduler = create_scheduler(
            self.optimizer, 
            config.train.lr_scheduler, 
            config.train.epochs
        )
        
        # 设置损失函数
        self.loss_fn = self._create_loss_function()
        
        # 创建TensorBoard写入器
        self.tb_writer = init_tensorboard(config.output.tensorboard_dir)
        
        # 创建早停对象
        self.early_stopping = EarlyStopping(
            patience=config.train.early_stopping.patience,
            delta=config.train.early_stopping.delta,
            mode=config.train.early_stopping.mode,
            save_path=os.path.join(config.output.model_dir, 'best_model.pth')
        )
        
        # 初始化训练指标
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        
        # 检查是否有检查点可恢复
        if config.train.checkpoint_path and os.path.exists(config.train.checkpoint_path):
            self._resume_from_checkpoint(config.train.checkpoint_path)
    
    def _create_loss_function(self) -> nn.Module:
        """
        创建损失函数
        
        返回:
            nn.Module: 损失函数
        """
        loss_name = self.config.train.loss.name.lower()
        
        if loss_name == 'ntxent':
            # NT-Xent 损失 (SimCLR)
            return NTXentLoss(
                temperature=self.config.train.loss.temperature,
                use_cosine_similarity=self.config.train.loss.use_cosine_similarity
            )
        elif loss_name == 'cluster':
            # 聚类损失
            return ClusterLoss(
                num_clusters=self.config.train.loss.num_clusters,
                feature_dim=self.config.model.output_dim,
                temperature=self.config.train.loss.temperature
            )
        elif loss_name == 'swav':
            # SwAV 损失
            return SwAVLoss(
                num_prototypes=self.config.train.loss.num_prototypes,
                feature_dim=self.config.model.output_dim,
                temperature=self.config.train.loss.temperature,
                sinkhorn_iterations=self.config.train.loss.sinkhorn_iterations
            )
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")
    
    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        从检查点恢复训练
        
        参数:
            checkpoint_path (str): 检查点路径
        """
        checkpoint = load_checkpoint(
            self.model, 
            checkpoint_path, 
            self.optimizer, 
            self.scheduler, 
            self.device
        )
        
        self.epochs_trained = checkpoint.get('epoch', 0)
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"恢复训练自轮数 {self.epochs_trained}")
    
    def train(self) -> Dict[str, Any]:
        """
        训练模型
        
        返回:
            Dict: 包含训练结果的字典
        """
        logger.info("开始训练模型...")
        start_time = time.time()
        
        for epoch in range(self.epochs_trained, self.config.train.epochs):
            # 训练一个轮次
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate_epoch(epoch)
                self.val_losses.append(val_loss)
                
                # 检查早停
                if self.early_stopping(val_loss, self.model, epoch):
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        logger.info(f"轮数 {epoch+1}: 验证损失改善至 {val_loss:.6f}")
                
                if self.early_stopping.early_stop:
                    logger.info("提前停止训练")
                    break
            
            # 保存检查点
            if (epoch + 1) % self.config.train.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.output.model_dir, 
                    f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    val_loss if val_loss is not None else train_loss,
                    checkpoint_path,
                    {
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'best_val_loss': self.best_val_loss
                    }
                )
            
            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # 记录当前轮次的学习率
            if self.tb_writer is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('learning_rate', current_lr, epoch)
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成! 总共耗时: {total_time:.2f}秒")
        
        # 绘制学习曲线
        if self.val_loader is not None and len(self.val_losses) > 0:
            plot_path = os.path.join(self.config.output.plots_dir, "learning_curves.png")
            plot_learning_curves(self.train_losses, self.val_losses, save_path=plot_path)
        
        # 测试最佳模型
        if self.test_loader is not None:
            logger.info("加载最佳模型进行测试...")
            best_model_path = self.early_stopping.save_path
            if os.path.exists(best_model_path):
                checkpoint = load_checkpoint(self.model, best_model_path, device=self.device)
                self._test_model()
            else:
                logger.warning("找不到最佳模型，使用当前模型进行测试")
                self._test_model()
        
        # 关闭TensorBoard写入器
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        # 返回训练结果
        return {
            'epochs_trained': self.epochs_trained + epoch + 1 - self.epochs_trained,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': self.early_stopping.best_epoch + 1,
            'total_time': total_time
        }
    
    def _train_epoch(self, epoch: int) -> float:
        """
        训练一个轮次
        
        参数:
            epoch (int): 当前轮次
            
        返回:
            float: 训练损失
        """
        self.model.train()
        losses = AverageMeter('Loss')
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"轮数 {epoch+1}/{self.config.train.epochs}", 
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 处理批次数据
            view1, view2, metadata = self._process_batch(batch)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 正向传播
            output1 = self.model(view1)
            output2 = self.model(view2)
            
            # 获取投影
            proj1 = output1["projection"]
            proj2 = output2["projection"]
            
            # 计算损失
            loss = self.loss_fn(proj1, proj2)
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪
            if self.config.train.grad_clip.enabled:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.train.grad_clip.max_norm
                )
            
            self.optimizer.step()
            
            # 更新损失
            batch_size = view1[0].size(0) if isinstance(view1, tuple) else view1.size(0) if hasattr(view1, 'size') else len(view1)
            losses.update(loss.item(), batch_size)
            
            # 更新进度条
            progress_bar.set_postfix({'loss': losses.avg})
            
            # 记录TensorBoard
            if self.tb_writer is not None and batch_idx % self.config.train.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.tb_writer.add_scalar('train/loss', losses.val, step)
        
        # 打印轮次结果
        logger.info(f"轮数 {epoch+1}/{self.config.train.epochs} - 训练损失: {losses.avg:.6f}")
        
        # 记录TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('train/epoch_loss', losses.avg, epoch)
        
        return losses.avg
    
    def _validate_epoch(self, epoch: int) -> float:
        """
        验证一个轮次
        
        参数:
            epoch (int): 当前轮次
            
        返回:
            float: 验证损失
        """
        self.model.eval()
        losses = AverageMeter('Loss')
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, 
                desc=f"验证轮数 {epoch+1}", 
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # 处理批次数据
                view1, view2, metadata = self._process_batch(batch)
                
                # 正向传播
                output1 = self.model(view1)
                output2 = self.model(view2)
                
                # 获取投影
                proj1 = output1["projection"]
                proj2 = output2["projection"]
                
                # 计算损失
                loss = self.loss_fn(proj1, proj2)
                
                # 更新损失
                batch_size = view1[0].size(0) if isinstance(view1, tuple) else view1.size(0) if hasattr(view1, 'size') else len(view1)
                losses.update(loss.item(), batch_size)
                
                # 更新进度条
                progress_bar.set_postfix({'loss': losses.avg})
        
        # 打印验证结果
        logger.info(f"轮数 {epoch+1}/{self.config.train.epochs} - 验证损失: {losses.avg:.6f}")
        
        # 记录TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('validation/loss', losses.avg, epoch)
        
        return losses.avg
    
    def _test_model(self) -> Dict[str, float]:
        """
        测试模型
        
        返回:
            Dict: 测试结果
        """
        self.model.eval()
        losses = AverageMeter('Loss')
        
        # 保存所有表示向量以进行可视化和分析
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="测试", leave=False)
            
            for batch_idx, batch in enumerate(progress_bar):
                # 处理批次数据
                view1, view2, metadata = self._process_batch(batch)
                
                # 获取标签（如果有）
                if 'label' in metadata:
                    labels = metadata['label'].to(self.device)
                else:
                    batch_size = view1[0].size(0) if isinstance(view1, tuple) else view1.size(0) if hasattr(view1, 'size') else len(view1)
                    labels = torch.zeros(batch_size)
                
                # 正向传播
                output1 = self.model(view1)
                output2 = self.model(view2)
                
                # 获取投影
                proj1 = output1["projection"]
                proj2 = output2["projection"]
                
                # 保存表示向量和标签
                all_embeddings.append(proj1.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                # 计算损失
                loss = self.loss_fn(proj1, proj2)
                
                # 更新损失
                batch_size = view1[0].size(0) if isinstance(view1, tuple) else view1.size(0) if hasattr(view1, 'size') else len(view1)
                losses.update(loss.item(), batch_size)
                
                # 更新进度条
                progress_bar.set_postfix({'loss': losses.avg})
        
        # 打印测试结果
        logger.info(f"测试损失: {losses.avg:.6f}")
        
        # 保存嵌入向量用于可视化
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        embeddings_path = os.path.join(self.config.output.results_dir, "embeddings.npz")
        np.savez(
            embeddings_path, 
            embeddings=all_embeddings, 
            labels=all_labels
        )
        logger.info(f"嵌入向量已保存到 {embeddings_path}")
        
        # 尝试可视化嵌入（如果配置启用）
        if self.config.output.visualize_embeddings:
            try:
                self._visualize_embeddings(all_embeddings, all_labels)
            except ImportError:
                logger.warning("无法可视化嵌入向量，缺少必要的库")
        
        return {'test_loss': losses.avg}
    
    def _process_batch(self, batch) -> Tuple:
        """
        处理批次数据
        
        参数:
            batch: 数据批次
            
        返回:
            Tuple: 包含处理后的视图和元数据
        """
        # 检查批次格式
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            view1, view2, metadata = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            # 假设批次只包含两个视图和没有元数据
            view1, view2 = batch
            metadata = {}
        else:
            # 只有一个图，我们需要通过数据增强创建第二个视图
            view1 = batch
            view2 = self._apply_augmentation(batch)
            metadata = {}
        
        # 将数据移动到适当的设备
        if isinstance(view1, (list, tuple)):
            view1 = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in view1)
            view2 = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in view2)
        else:
            # 处理PyG的Batch对象或其他张量
            view1 = self._to_device(view1)
            view2 = self._to_device(view2)
        
        return view1, view2, metadata
        
    def _to_device(self, data):
        """
        将数据移动到指定设备
        
        参数:
            data: 输入数据（可以是Batch对象或Tensor）
            
        返回:
            移动到设备的数据
        """
        if hasattr(data, 'to'):
            # 如果是PyG的Batch对象或Tensor，使用to方法
            return data.to(self.device)
        elif isinstance(data, dict):
            # 如果是字典，递归处理每个值
            return {k: self._to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            # 如果是列表或元组，递归处理每个元素
            return type(data)(self._to_device(x) for x in data)
        else:
            # 其他类型不处理
            return data
            
    def _apply_augmentation(self, data):
        """
        对输入数据应用增强
        
        参数:
            data: 输入数据（可以是Batch对象或其他格式）
            
        返回:
            增强后的数据
        """
        # 检查是否有配置的增强转换
        if not hasattr(self, 'augmentation') or self.augmentation is None:
            # 如果没有配置增强，返回原始数据的副本
            if hasattr(data, 'clone'):
                # 对于Tensor和某些PyG对象
                return data.clone()
            else:
                # 对于其他类型，尝试深拷贝
                import copy
                return copy.deepcopy(data)
                
        # 应用配置的增强
        if hasattr(data, 'edge_index') and hasattr(data, 'x'):
            # 如果是PyG的Data或Batch对象
            augmented_data = self.augmentation(data)
            return augmented_data
        else:
            # 对于其他格式，保持原有逻辑
            logger.warning("未知数据格式，无法应用增强")
            return data
    
    def _visualize_embeddings(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        可视化嵌入向量
        
        参数:
            embeddings (np.ndarray): 嵌入向量
            labels (np.ndarray): 标签
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=self.config.seed)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # 绘制降维后的嵌入
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=labels, 
                cmap='viridis', 
                alpha=0.7
            )
            plt.colorbar(scatter, label='类别')
            plt.title('t-SNE可视化嵌入向量')
            
            # 保存图像
            save_path = os.path.join(self.config.output.plots_dir, "embeddings_tsne.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"嵌入向量可视化已保存到 {save_path}")
            
        except ImportError:
            logger.warning("可视化嵌入向量需要sklearn和matplotlib库") 