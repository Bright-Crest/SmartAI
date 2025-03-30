import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class TimeEncoder(nn.Module):
    """
    时间编码器，将时间戳转换为连续表示
    """
    def __init__(self, time_dim):
        """
        参数:
            time_dim (int): 时间编码的维度
        """
        super(TimeEncoder, self).__init__()
        
        self.time_dim = time_dim
        
        # 时间嵌入的参数
        self.basis_freq = nn.Parameter(torch.randn(time_dim))
        self.phase = nn.Parameter(torch.randn(time_dim))
        
    def forward(self, timestamps):
        """
        前向传播
        
        参数:
            timestamps (Tensor): 时间戳 [batch_size]
            
        返回:
            Tensor: 时间编码 [batch_size, time_dim]
        """
        # 归一化时间戳
        timestamps = timestamps.unsqueeze(1)  # [batch_size, 1]
        
        # 计算时间编码
        time_encoding = torch.cos(timestamps * self.basis_freq + self.phase)
        
        return time_encoding


class MemoryUpdater(nn.Module):
    """
    内存更新模块，基于GRU
    """
    def __init__(self, memory_dim, time_dim, hidden_dim, message_dim):
        """
        参数:
            memory_dim (int): 内存维度
            time_dim (int): 时间编码维度
            hidden_dim (int): GRU隐藏层维度
            message_dim (int): 消息维度
        """
        super(MemoryUpdater, self).__init__()
        
        self.memory_dim = memory_dim
        
        # GRU更新单元
        self.gru = nn.GRUCell(
            input_size=message_dim + time_dim,
            hidden_size=memory_dim
        )
        
        # 输入转换层
        self.input_transform = nn.Linear(message_dim + time_dim, hidden_dim)
        
    def forward(self, memory, messages, timestamps, time_encoder):
        """
        前向传播
        
        参数:
            memory (Tensor): 当前内存状态 [batch_size, memory_dim]
            messages (Tensor): 消息 [batch_size, message_dim]
            timestamps (Tensor): 时间戳 [batch_size]
            time_encoder (TimeEncoder): 时间编码器
            
        返回:
            Tensor: 更新后的内存 [batch_size, memory_dim]
        """
        # 编码时间
        time_encoding = time_encoder(timestamps)
        
        # 连接消息和时间编码
        inputs = torch.cat([messages, time_encoding], dim=1)
        
        # 更新内存
        updated_memory = self.gru(inputs, memory)
        
        return updated_memory


class TGNMemoryWrapper(nn.Module):
    """
    TGN内存管理器，管理地址的历史状态
    """
    def __init__(self, num_nodes, message_dim, memory_dim, time_dim, edge_dim=4):
        """
        参数:
            num_nodes (int): 节点数量
            message_dim (int): 消息维度
            memory_dim (int): 内存维度
            time_dim (int): 时间编码维度
            edge_dim (int): 边特征维度
        """
        super(TGNMemoryWrapper, self).__init__()
        
        self.num_nodes = num_nodes
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.edge_dim = edge_dim
        
        # 初始化内存，每个节点有一个内存状态
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim))
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        
        # 时间编码器
        self.time_encoder = TimeEncoder(time_dim)
        
        # 内存更新器
        self.updater = MemoryUpdater(
            memory_dim=memory_dim,
            time_dim=time_dim,
            hidden_dim=memory_dim,
            message_dim=message_dim
        )
        
        # 消息计算器
        self.message_function = nn.Linear(memory_dim * 2 + edge_dim, message_dim)
    
    def get_memory(self, node_idxs=None):
        """
        获取指定节点的内存
        
        参数:
            node_idxs (Tensor, 可选): 节点索引 [batch_size]
            
        返回:
            Tensor: 内存状态 [batch_size, memory_dim] 或 [num_nodes, memory_dim]
        """
        if node_idxs is None:
            return self.memory
        else:
            return self.memory[node_idxs]
    
    def update_memory(self, source_nodes, target_nodes, timestamps, edge_features):
        """
        更新内存
        
        参数:
            source_nodes (Tensor): 源节点索引 [batch_size]
            target_nodes (Tensor): 目标节点索引 [batch_size]
            timestamps (Tensor): 时间戳 [batch_size]
            edge_features (Tensor): 边特征 [batch_size, feature_dim]
        """
        # 确保索引在有效范围内
        valid_indices = (source_nodes < self.num_nodes) & (target_nodes < self.num_nodes)
        if not torch.all(valid_indices):
            # 过滤无效索引
            source_nodes = source_nodes[valid_indices]
            target_nodes = target_nodes[valid_indices]
            timestamps = timestamps[valid_indices]
            edge_features = edge_features[valid_indices]
            
            # 如果所有索引都无效，就不更新内存
            if len(source_nodes) == 0:
                return
        
        # 确保边特征维度正确
        expected_edge_dim = self.edge_dim
        if edge_features.size(1) != expected_edge_dim:
            # 边特征维度不匹配，需要调整
            if edge_features.size(1) > expected_edge_dim:
                # 如果边特征维度过大，截断
                edge_features = edge_features[:, :expected_edge_dim]
            else:
                # 如果边特征维度过小，填充
                padding = torch.zeros(edge_features.size(0), expected_edge_dim - edge_features.size(1), device=edge_features.device)
                edge_features = torch.cat([edge_features, padding], dim=1)
        
        # 获取源节点和目标节点的内存
        source_memory = self.memory[source_nodes]
        target_memory = self.memory[target_nodes]
        
        # 计算消息
        message_inputs = torch.cat([source_memory, target_memory, edge_features], dim=1)
        messages = F.relu(self.message_function(message_inputs))
        
        # 更新源节点和目标节点的内存（使用非原位操作）
        source_updated = self.updater(
            source_memory, messages, timestamps, self.time_encoder
        )
        target_updated = self.updater(
            target_memory, messages, timestamps, self.time_encoder
        )
        
        # 使用克隆创建内存的副本以避免原位操作
        new_memory = self.memory.clone()
        
        # 使用scatter_操作而不是直接索引赋值
        for i, idx in enumerate(source_nodes):
            new_memory[idx] = source_updated[i]
        
        for i, idx in enumerate(target_nodes):
            new_memory[idx] = target_updated[i]
        
        # 更新内存参数
        self.memory.data.copy_(new_memory)
        
        # 更新最后更新时间，使用data属性进行原位操作
        new_last_update = self.last_update.clone()
        for i, idx in enumerate(source_nodes):
            new_last_update[idx] = timestamps[i]
        for i, idx in enumerate(target_nodes):
            new_last_update[idx] = timestamps[i]
        self.last_update.data.copy_(new_last_update)
    
    def reset_memory(self):
        """
        重置所有内存
        """
        # 使用data属性进行原位操作，避免自动微分跟踪
        self.memory.data.zero_()
        self.last_update.data.zero_()


class TGNWrapper(nn.Module):
    """
    TGN模型包装器
    """
    def __init__(
        self,
        num_nodes,
        in_channels,
        hidden_channels,
        out_channels,
        memory_dim,
        time_dim,
        edge_dim=4,
        num_layers=2
    ):
        """
        参数:
            num_nodes (int): 节点数量
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层维度
            out_channels (int): 输出特征维度
            memory_dim (int): 内存维度
            time_dim (int): 时间编码维度
            edge_dim (int): 边特征维度
            num_layers (int): TGN层数
        """
        super(TGNWrapper, self).__init__()
        
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        # 内存管理器
        self.memory_wrapper = TGNMemoryWrapper(
            num_nodes=num_nodes,
            message_dim=hidden_channels,
            memory_dim=memory_dim,
            time_dim=time_dim,
            edge_dim=edge_dim
        )
        
        # 图注意力层
        self.layers = nn.ModuleList()
        
        # 第一层：将原始特征与内存结合
        self.layers.append(
            TransformerConv(
                in_channels=in_channels + memory_dim,
                out_channels=hidden_channels,
                heads=2,
                dropout=0.1,
                edge_dim=edge_dim + time_dim
            )
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_channels * 2,  # 因为使用了2个头
                    out_channels=hidden_channels,
                    heads=2,
                    dropout=0.1,
                    edge_dim=edge_dim + time_dim
                )
            )
        
        # 最后一层
        if num_layers > 1:
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_channels * 2,  # 因为使用了2个头
                    out_channels=out_channels,
                    heads=1,
                    dropout=0.1,
                    edge_dim=edge_dim + time_dim
                )
            )
        
        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * 2) for _ in range(num_layers - 1)
        ])
        if num_layers > 0:
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
    
    def forward(self, x, edge_index, edge_attr, timestamps, node_idxs=None):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            edge_attr (Tensor): 边特征 [E, edge_dim]
            timestamps (Tensor): 边的时间戳 [E]
            node_idxs (Tensor, 可选): 节点索引，默认为None表示所有节点
            
        返回:
            Tensor: 节点嵌入 [N, out_channels]
        """
        # 获取相关节点的内存
        if node_idxs is None:
            # 如果没有提供节点索引，我们需要确保内存和节点特征的大小匹配
            if x.size(0) != self.memory_wrapper.memory.size(0):
                # 获取批次中的唯一节点索引
                unique_nodes = torch.unique(edge_index.flatten())
                if len(unique_nodes) != x.size(0):
                    # 如果唯一节点数与节点特征数不匹配，我们假设需要将内存大小调整为与节点特征相同
                    # 这是一个批处理的情况，我们需要裁剪内存或者填充内存
                    if x.size(0) <= self.num_nodes:
                        # 只使用批次所需的内存
                        memory = self.memory_wrapper.memory[:x.size(0)]
                    else:
                        # 如果批次大小大于总节点数(这不应该发生)，则填充内存
                        memory = torch.cat([
                            self.memory_wrapper.memory,
                            torch.zeros(x.size(0) - self.num_nodes, self.memory_dim, device=x.device)
                        ], dim=0)
                else:
                    # 使用批次中唯一节点的内存
                    memory = self.memory_wrapper.get_memory(unique_nodes)
            else:
                memory = self.memory_wrapper.get_memory()
        else:
            memory = self.memory_wrapper.get_memory(node_idxs)
        
        # 确保内存和x的大小匹配
        if memory.size(0) != x.size(0):
            # 如果不匹配，我们需要调整内存大小
            if memory.size(0) > x.size(0):
                # 裁剪内存
                memory = memory[:x.size(0)]
            else:
                # 填充内存
                padding = torch.zeros(x.size(0) - memory.size(0), memory.size(1), device=x.device)
                memory = torch.cat([memory, padding], dim=0)
        
        # 将节点特征与内存结合
        x = torch.cat([x, memory], dim=1)
        
        # 编码时间
        time_encoding = self.memory_wrapper.time_encoder(timestamps)
        
        # 将边特征与时间编码结合
        edge_features = torch.cat([edge_attr, time_encoding], dim=1)
        
        # 通过图注意力层
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_features)
            if i < len(self.layers) - 1:
                x = F.relu(self.batch_norms[i](x))
                x = F.dropout(x, p=0.1, training=self.training)
            else:
                x = self.batch_norms[i](x)
        
        return x
    
    def update_memory(self, source_nodes, target_nodes, timestamps, edge_features):
        """
        更新内存
        
        参数:
            source_nodes (Tensor): 源节点索引 [batch_size]
            target_nodes (Tensor): 目标节点索引 [batch_size]
            timestamps (Tensor): 时间戳 [batch_size]
            edge_features (Tensor): 边特征 [batch_size, feature_dim]
        """
        self.memory_wrapper.update_memory(source_nodes, target_nodes, timestamps, edge_features)
    
    def reset_memory(self):
        """
        重置内存
        """
        self.memory_wrapper.reset_memory() 