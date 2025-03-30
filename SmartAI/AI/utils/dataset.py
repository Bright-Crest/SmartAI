import os
import glob
import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable, Union
from torch_geometric.data import Data, Dataset, Batch
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class AddressGraphDataset(Dataset):
    """
    地址行为图数据集，加载.pt格式的地址图文件
    """
    def __init__(
        self, 
        root_dir: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        mode: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        参数:
            root_dir (str): 数据根目录，包含保存的地址图文件(.pt格式)
            transform (callable, 可选): 数据转换函数，应用于每次获取数据项时
            pre_transform (callable, 可选): 预处理函数，应用于所有数据一次
            mode (str): 数据集模式，可选"train", "val", "test"
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            test_ratio (float): 测试集比例
            seed (int): 随机种子，用于数据划分
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 加载并划分数据
        self._process()
        
    def _process(self):
        """
        处理数据集，查找所有.pt文件并划分数据集
        """
        # 查找所有地址图文件
        all_files = glob.glob(os.path.join(self.root_dir, "*.pt"))
        
        if len(all_files) == 0:
            raise ValueError(f"在{self.root_dir}目录下未找到任何.pt文件")
        
        logger.info(f"找到{len(all_files)}个地址图文件")
        
        # 随机打乱并划分数据集
        random.seed(self.seed)
        random.shuffle(all_files)
        
        train_size = int(len(all_files) * self.train_ratio)
        val_size = int(len(all_files) * self.val_ratio)
        
        if self.mode == "train":
            self.file_list = all_files[:train_size]
        elif self.mode == "val":
            self.file_list = all_files[train_size:train_size + val_size]
        elif self.mode == "test":
            self.file_list = all_files[train_size + val_size:]
        else:
            raise ValueError(f"不支持的数据集模式: {self.mode}")
        
        logger.info(f"{self.mode}集大小: {len(self.file_list)}")
        
        # 收集唯一地址ID
        self.address_ids = []
        for file_path in self.file_list:
            address_id = os.path.basename(file_path).split('.')[0]
            self.address_ids.append(address_id)
        
        # 如果有预处理函数，应用它
        if self.pre_transform:
            self._apply_pre_transform()
    
    def _apply_pre_transform(self):
        """
        应用预处理函数到所有数据
        """
        for i, file_path in enumerate(self.file_list):
            try:
                data = torch.load(file_path, weights_only=False)
            except TypeError:
                data = torch.load(file_path)
                
            data = self.pre_transform(data)
            
            try:
                torch.save(data, file_path, _use_new_zipfile_serialization=True)
            except TypeError:
                torch.save(data, file_path)
                
            if i % 100 == 0:
                logger.info(f"预处理进度: {i+1}/{len(self.file_list)}")
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Data:
        """
        获取指定索引的数据项
        
        参数:
            idx (int): 数据索引
            
        返回:
            Data: PyG数据对象，包含图结构
        """
        file_path = self.file_list[idx]
        try:
            data = torch.load(file_path, weights_only=False)
        except TypeError:
            # 兼容旧版PyTorch
            data = torch.load(file_path)
        
        # 应用转换
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def get_address_id(self, idx: int) -> str:
        """
        获取指定索引的地址ID
        
        参数:
            idx (int): 数据索引
            
        返回:
            str: 地址ID
        """
        return self.address_ids[idx]
        
    def get_subset(self, mode: str) -> 'AddressGraphDataset':
        """
        获取指定模式的子集
        
        参数:
            mode (str): 数据集模式，可选"train", "val", "test"
            
        返回:
            AddressGraphDataset: 子集数据集
        """
        # 因为当前实现中已经按照mode划分了数据，所以如果请求的mode与当前mode相同，直接返回self
        if mode == self.mode:
            return self
        
        # 否则，创建一个新的数据集实例，使用相同的配置但不同的mode
        return AddressGraphDataset(
            root_dir=self.root_dir,
            transform=self.transform,
            pre_transform=self.pre_transform,
            mode=mode,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )


class TimeWindowGraphDataset(Dataset):
    """
    时间窗口图数据集，将每个地址的交易划分为不同时间窗口的图序列
    """
    def __init__(
        self, 
        root_dir: str,
        time_window: int = 3600,  # 秒为单位的时间窗口大小
        max_windows: int = 24,    # 最大窗口数
        transform: Optional[Callable] = None,
        mode: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        参数:
            root_dir (str): 数据根目录，包含保存的地址图文件(.pt格式)
            time_window (int): 时间窗口大小（秒）
            max_windows (int): 最大窗口数
            transform (callable, 可选): 数据转换函数
            mode (str): 数据集模式，可选"train", "val", "test"
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            test_ratio (float): 测试集比例
            seed (int): 随机种子，用于数据划分
        """
        self.root_dir = root_dir
        self.time_window = time_window
        self.max_windows = max_windows
        self.transform = transform
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 加载并划分数据
        self._process()
    
    def _process(self):
        """
        处理数据集，查找所有.pt文件并划分数据集
        """
        # 查找所有地址图文件
        all_files = glob.glob(os.path.join(self.root_dir, "*.pt"))
        
        if len(all_files) == 0:
            raise ValueError(f"在{self.root_dir}目录下未找到任何.pt文件")
        
        logger.info(f"找到{len(all_files)}个地址图文件")
        
        # 随机打乱并划分数据集
        random.seed(self.seed)
        random.shuffle(all_files)
        
        train_size = int(len(all_files) * self.train_ratio)
        val_size = int(len(all_files) * self.val_ratio)
        
        if self.mode == "train":
            self.file_list = all_files[:train_size]
        elif self.mode == "val":
            self.file_list = all_files[train_size:train_size + val_size]
        elif self.mode == "test":
            self.file_list = all_files[train_size + val_size:]
        else:
            raise ValueError(f"不支持的数据集模式: {self.mode}")
        
        logger.info(f"{self.mode}集大小: {len(self.file_list)}")
        
        # 收集唯一地址ID
        self.address_ids = []
        for file_path in self.file_list:
            address_id = os.path.basename(file_path).split('.')[0]
            self.address_ids.append(address_id)
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> List[Data]:
        """
        获取指定索引的数据项，将单个地址图划分为时间窗口序列
        
        参数:
            idx (int): 数据索引
            
        返回:
            List[Data]: 时间窗口序列中的图列表
        """
        file_path = self.file_list[idx]
        try:
            data = torch.load(file_path, weights_only=False)
        except TypeError:
            # 兼容旧版PyTorch
            data = torch.load(file_path)
        
        # 按时间戳排序边
        timestamps = data.timestamps
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # 如果没有时间戳，返回原始数据
        if not hasattr(data, "timestamps") or len(timestamps) == 0:
            if self.transform:
                data = self.transform(data)
            return [data]
        
        # 确保边索引在有效范围内
        if data.x.size(0) > 0:
            num_nodes = data.x.size(0)
            valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            
            if not torch.all(valid_edges):
                edge_index = edge_index[:, valid_edges]
                edge_attr = edge_attr[valid_edges]
                timestamps = timestamps[valid_edges]
                
                # 如果过滤后没有边，返回只有节点的图
                if edge_index.size(1) == 0:
                    if self.transform:
                        data = self.transform(data)
                    return [data]
        
        # 按时间戳排序
        sorted_indices = torch.argsort(timestamps)
        timestamps = timestamps[sorted_indices]
        edge_index = edge_index[:, sorted_indices]
        edge_attr = edge_attr[sorted_indices]
        
        # 计算时间范围
        min_time = timestamps.min().item()
        max_time = timestamps.max().item()
        time_span = max_time - min_time
        
        # 确定时间窗口数
        if time_span <= 0:
            num_windows = 1
        else:
            num_windows = min(int(np.ceil(time_span / self.time_window)), self.max_windows)
        
        # 创建时间窗口序列
        window_graphs = []
        for i in range(num_windows):
            start_time = min_time + i * self.time_window
            end_time = start_time + self.time_window
            
            # 找出该时间窗口内的边
            mask = (timestamps >= start_time) & (timestamps < end_time)
            window_edge_index = edge_index[:, mask]
            window_edge_attr = edge_attr[mask]
            window_timestamps = timestamps[mask]
            
            # 如果该窗口没有边，继续下一个窗口
            if window_edge_index.size(1) == 0:
                continue
            
            # 创建时间窗口图
            window_data = Data(
                x=data.x,
                edge_index=window_edge_index,
                edge_attr=window_edge_attr,
                timestamps=window_timestamps,
                address_id=data.address_id
            )
            
            # 应用转换
            if self.transform:
                window_data = self.transform(window_data)
            
            window_graphs.append(window_data)
        
        # 如果没有窗口，返回原始数据
        if len(window_graphs) == 0:
            if self.transform:
                data = self.transform(data)
            return [data]
        
        return window_graphs
    
    def get_address_id(self, idx: int) -> str:
        """
        获取指定索引的地址ID
        
        参数:
            idx (int): 数据索引
            
        返回:
            str: 地址ID
        """
        return self.address_ids[idx]
        
    def get_subset(self, mode: str) -> 'TimeWindowGraphDataset':
        """
        获取指定模式的子集
        
        参数:
            mode (str): 数据集模式，可选"train", "val", "test"
            
        返回:
            TimeWindowGraphDataset: 子集数据集
        """
        # 因为当前实现中已经按照mode划分了数据，所以如果请求的mode与当前mode相同，直接返回self
        if mode == self.mode:
            return self
        
        # 否则，创建一个新的数据集实例，使用相同的配置但不同的mode
        return TimeWindowGraphDataset(
            root_dir=self.root_dir,
            time_window=self.time_window,
            max_windows=self.max_windows,
            transform=self.transform,
            mode=mode,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed
        )


class GraphAugmentation:
    """
    图增强类，用于对比学习的数据增强
    """
    def __init__(
        self,
        edge_drop_rate: float = 0.1,
        feat_mask_rate: float = 0.1,
        time_mask_prob: float = 0.05
    ):
        """
        参数:
            edge_drop_rate (float): 边丢弃率
            feat_mask_rate (float): 特征掩码率
            time_mask_prob (float): 时间掩码概率
        """
        self.edge_drop_rate = edge_drop_rate
        self.feat_mask_rate = feat_mask_rate
        self.time_mask_prob = time_mask_prob
    
    def __call__(self, data: Data) -> Data:
        """
        对输入图进行增强
        
        参数:
            data (Data): 输入图数据
            
        返回:
            Data: 增强后的图数据
        """
        # 复制数据，避免修改原始数据
        augmented_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if hasattr(data, "edge_attr") else None,
            timestamps=data.timestamps.clone() if hasattr(data, "timestamps") else None,
            address_id=data.address_id if hasattr(data, "address_id") else None
        )
        
        # 确保边索引不超出节点数量范围
        if augmented_data.edge_index.size(1) > 0 and augmented_data.x.size(0) > 0:
            num_nodes = augmented_data.x.size(0)
            # 过滤掉超出范围的边
            valid_edges = (augmented_data.edge_index[0] < num_nodes) & (augmented_data.edge_index[1] < num_nodes)
            
            if not torch.all(valid_edges):
                # 如果有无效边，保留有效的边
                augmented_data.edge_index = augmented_data.edge_index[:, valid_edges]
                
                if hasattr(augmented_data, "edge_attr") and augmented_data.edge_attr is not None:
                    augmented_data.edge_attr = augmented_data.edge_attr[valid_edges]
                
                if hasattr(augmented_data, "timestamps") and augmented_data.timestamps is not None:
                    augmented_data.timestamps = augmented_data.timestamps[valid_edges]
        
        # 边丢弃
        if self.edge_drop_rate > 0 and augmented_data.edge_index.size(1) > 0:
            num_edges = augmented_data.edge_index.size(1)
            num_edges_to_keep = max(1, int(num_edges * (1 - self.edge_drop_rate)))  # 至少保留一条边
            perm = torch.randperm(num_edges)
            keep_indices = perm[:num_edges_to_keep]
            
            augmented_data.edge_index = augmented_data.edge_index[:, keep_indices]
            
            if hasattr(augmented_data, "edge_attr") and augmented_data.edge_attr is not None:
                augmented_data.edge_attr = augmented_data.edge_attr[keep_indices]
            
            if hasattr(augmented_data, "timestamps") and augmented_data.timestamps is not None:
                augmented_data.timestamps = augmented_data.timestamps[keep_indices]
        
        # 节点特征掩码
        if self.feat_mask_rate > 0 and augmented_data.x.size(0) > 0:
            x = augmented_data.x
            num_nodes, feat_dim = x.size()
            
            # 对每个节点的特征进行掩码
            mask = torch.rand(num_nodes, feat_dim) > self.feat_mask_rate
            augmented_data.x = x * mask.float()
        
        # 时间戳增强
        if (self.time_mask_prob > 0 and 
            hasattr(augmented_data, "timestamps") and 
            augmented_data.timestamps is not None and
            augmented_data.timestamps.size(0) > 0):
            
            timestamps = augmented_data.timestamps.float()  # 将时间戳转换为浮点型
            
            # 计算时间戳的相对标准差，避免绝对值过大
            # 使用归一化的标准差，避免时间戳绝对值影响
            time_range = timestamps.max() - timestamps.min()
            if time_range > 0:
                # 使用时间范围的百分比作为噪声尺度，避免绝对标准差过大
                noise_scale = time_range * 0.01  # 使用时间范围的1%作为基准尺度
                noise = (torch.rand_like(timestamps) * 2 - 1) * self.time_mask_prob * noise_scale
            else:
                # 如果所有时间戳相同，使用很小的噪声
                noise = torch.zeros_like(timestamps)
                
            augmented_data.timestamps = (timestamps + noise).long()  # 转回长整型
        
        return augmented_data


class SequenceAugmentation:
    """
    序列增强类，用于时间序列的数据增强
    """
    def __init__(
        self,
        seq_mask_rate: float = 0.1,
        time_stretch_factor: float = 0.1,
        random_crop: bool = True,
        crop_ratio: float = 0.8
    ):
        """
        参数:
            seq_mask_rate (float): 序列元素掩码率
            time_stretch_factor (float): 时间伸缩因子
            random_crop (bool): 是否随机裁剪
            crop_ratio (float): 裁剪保留比例
        """
        self.seq_mask_rate = seq_mask_rate
        self.time_stretch_factor = time_stretch_factor
        self.random_crop = random_crop
        self.crop_ratio = crop_ratio
    
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        对输入序列进行增强
        
        参数:
            sequence (Tensor): 输入序列 [seq_len, feat_dim]
            
        返回:
            Tensor: 增强后的序列 [seq_len, feat_dim]
        """
        # 复制序列，避免修改原始数据
        augmented_seq = sequence.clone()
        seq_len, feat_dim = augmented_seq.size()
        
        # 序列元素掩码
        if self.seq_mask_rate > 0:
            mask = torch.rand(seq_len, feat_dim) > self.seq_mask_rate
            augmented_seq = augmented_seq * mask.float()
        
        # 时间伸缩（通过插值实现）
        if self.time_stretch_factor > 0:
            stretch_factor = 1.0 + (torch.rand(1) * 2 - 1) * self.time_stretch_factor
            new_len = int(seq_len * stretch_factor)
            
            if new_len != seq_len and new_len > 0:
                # 使用线性插值调整序列长度
                indices = torch.linspace(0, seq_len - 1, new_len)
                augmented_seq = torch.nn.functional.interpolate(
                    augmented_seq.unsqueeze(0).unsqueeze(0),
                    size=(new_len, feat_dim),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        
        # 随机裁剪
        if self.random_crop and seq_len > 1:
            crop_size = max(1, int(seq_len * self.crop_ratio))
            start_idx = random.randint(0, seq_len - crop_size) if seq_len > crop_size else 0
            augmented_seq = augmented_seq[start_idx:start_idx + crop_size]
        
        return augmented_seq


def collate_address_graphs(batch: List[Data]) -> Batch:
    """
    自定义收集函数，用于地址图数据的批处理
    
    参数:
        batch (List[Data]): 批数据，每个元素是一个PyG数据对象
        
    返回:
        Batch: PyG批对象
    """
    # 确保每个图的边索引不超出节点数量范围
    for data in batch:
        if data.edge_index.size(1) > 0 and data.x.size(0) > 0:
            num_nodes = data.x.size(0)
            valid_edges = (data.edge_index[0] < num_nodes) & (data.edge_index[1] < num_nodes)
            
            if not torch.all(valid_edges):
                data.edge_index = data.edge_index[:, valid_edges]
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[valid_edges]
                if hasattr(data, "timestamps") and data.timestamps is not None:
                    data.timestamps = data.timestamps[valid_edges]
    
    # 使用PyG的Batch.from_data_list函数将图列表转换为批
    return Batch.from_data_list(batch)


def collate_time_windows(batch: List[List[Data]]) -> Tuple[List[List[Tuple]], List[str]]:
    """
    自定义收集函数，用于时间窗口序列数据的批处理
    
    参数:
        batch (List[List[Data]]): 批数据，每个元素是一个时间窗口序列
        
    返回:
        Tuple: (图序列列表, 地址ID列表)
    """
    # 处理空批次
    if len(batch) == 0:
        return [], []
    
    # 提取地址ID
    address_ids = [windows[0].address_id if hasattr(windows[0], "address_id") else "unknown" 
                  for windows in batch]
    
    # 转换每个时间窗口为元组 (x, edge_index, edge_attr, timestamps)
    graph_sequences = []
    for windows in batch:
        sequence = []
        for data in windows:
            # 确保所有必要属性都存在
            x = data.x if hasattr(data, "x") else torch.zeros(0)
            edge_index = data.edge_index if hasattr(data, "edge_index") else torch.zeros((2, 0), dtype=torch.long)
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else torch.zeros((0, 0))
            timestamps = data.timestamps if hasattr(data, "timestamps") else torch.zeros(0)
            
            # 确保边索引在有效范围内
            if edge_index.size(1) > 0 and x.size(0) > 0:
                num_nodes = x.size(0)
                valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                
                if not torch.all(valid_edges):
                    edge_index = edge_index[:, valid_edges]
                    if edge_attr.size(0) > 0:
                        edge_attr = edge_attr[valid_edges]
                    if timestamps.size(0) > 0:
                        timestamps = timestamps[valid_edges]
            
            sequence.append((x, edge_index, edge_attr, timestamps))
        
        graph_sequences.append(sequence)
    
    return graph_sequences, address_ids


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    创建数据加载器
    
    参数:
        dataset (Dataset): 数据集
        batch_size (int): 批大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载工作进程数
        collate_fn (callable, 可选): 自定义收集函数
        
    返回:
        DataLoader: 数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) 