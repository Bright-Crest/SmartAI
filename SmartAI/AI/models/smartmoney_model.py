import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
from torch_geometric.data import Batch

from .resgcn import ResGCN
from .edge_gnn import EdgeGAT, EdgeConv, MPNN
from .tgn_module import TGNWrapper
from .temporal_attention import TemporalTransformer, TemporalLSTM


class SmartMoneyEncoder(nn.Module):
    """
    SmartMoney编码器：结合ResGCN、EdgeGNN和TGN进行地址行为图建模
    """
    def __init__(
        self, 
        num_nodes: int,
        in_channels: int, 
        hidden_channels: int,
        out_channels: int,
        edge_dim: int = 4,
        dropout: float = 0.2,
        edge_gnn_type: str = "edge_gat",
        resgcn_num_layers: int = 3,
        tgn_memory_dim: int = 128,
        tgn_time_dim: int = 16,
        tgn_num_layers: int = 2
    ):
        """
        参数:
            num_nodes (int): 图中节点数量
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层特征维度
            out_channels (int): 输出特征维度
            edge_dim (int): 边特征维度
            dropout (float): Dropout概率
            edge_gnn_type (str): 边感知GNN的类型，可选值："edge_gat", "edge_conv", "mpnn"
            resgcn_num_layers (int): ResGCN层数
            tgn_memory_dim (int): TGN内存维度
            tgn_time_dim (int): 时间编码维度
            tgn_num_layers (int): TGN层数
        """
        super(SmartMoneyEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        # 1. ResGCN层 - 捕捉图的结构特征
        self.resgcn = ResGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=resgcn_num_layers,
            dropout=dropout,
            use_bn=True,
            residual=True
        )
        
        # 2. 边感知GNN层 - 处理边特征
        if edge_gnn_type == "edge_gat":
            self.edge_gnn = EdgeGAT(
                in_channels=hidden_channels,
                out_channels=hidden_channels // 2,  # 输出会是 hidden_channels 因为有2个头
                edge_dim=edge_dim,
                heads=2,
                concat=True,
                dropout=dropout
            )
        elif edge_gnn_type == "edge_conv":
            self.edge_gnn = EdgeConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim
            )
        elif edge_gnn_type == "mpnn":
            self.edge_gnn = MPNN(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim
            )
        else:
            raise ValueError(f"不支持的边感知GNN类型: {edge_gnn_type}")
        
        # 3. TGN层 - 时间动态建模
        self.tgn = TGNWrapper(
            num_nodes=num_nodes,
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            memory_dim=tgn_memory_dim,
            time_dim=tgn_time_dim,
            edge_dim=edge_dim,
            num_layers=tgn_num_layers
        )
        
        # 额外处理层
        self.norm = nn.LayerNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_idxs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            edge_attr (Tensor): 边特征 [E, edge_dim]
            timestamps (Tensor): 时间戳 [E]
            batch (Tensor, 可选): 批处理索引 [N]
            node_idxs (Tensor, 可选): 节点索引，用于获取特定节点的内存
            
        返回:
            Tensor: 节点嵌入 [N, out_channels]
        """
        # 安全检查：如果edge_index为空，确保其他边相关数据也为空，并添加自环
        if edge_index.numel() == 0 or edge_index.size(1) == 0:
            num_nodes = x.size(0)
            # 为每个节点添加自环作为默认边
            edge_index = torch.stack([
                torch.arange(num_nodes, device=x.device),
                torch.arange(num_nodes, device=x.device)
            ], dim=0)
            
            # 创建默认的边特征和时间戳
            if self.edge_gnn.in_channels > 0:
                edge_attr = torch.zeros((num_nodes, self.edge_gnn.in_channels), device=x.device)
            else:
                edge_attr = torch.zeros((num_nodes, 1), device=x.device)
                
            timestamps = torch.zeros(num_nodes, device=x.device, dtype=torch.long)
        else:
            # 确保边索引在有效范围内
            num_nodes = x.size(0)
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            
            if not torch.all(valid_mask):
                # 过滤无效的边
                edge_index = edge_index[:, valid_mask]
                if edge_attr is not None and edge_attr.size(0) > 0:
                    edge_attr = edge_attr[valid_mask]
                if timestamps is not None and timestamps.size(0) > 0:
                    timestamps = timestamps[valid_mask]
        
        # 确保edge_attr不为None
        if edge_attr is None:
            if hasattr(self.edge_gnn, 'in_channels'):
                edge_dim = self.edge_gnn.in_channels
            else:
                edge_dim = 4  # 默认边特征维度
            edge_attr = torch.zeros((edge_index.size(1), edge_dim), device=x.device)
        
        # 确保timestamps不为None
        if timestamps is None:
            timestamps = torch.zeros(edge_index.size(1), device=x.device, dtype=torch.long)
        
        # 确保维度匹配
        if edge_attr.size(0) != edge_index.size(1):
            # 如果不匹配，重新创建边特征
            if hasattr(self.edge_gnn, 'in_channels'):
                edge_dim = self.edge_gnn.in_channels
            else:
                edge_dim = 4  # 默认边特征维度
            edge_attr = torch.zeros((edge_index.size(1), edge_dim), device=x.device)
        
        if timestamps.size(0) != edge_index.size(1):
            # 如果不匹配，重新创建时间戳
            timestamps = torch.zeros(edge_index.size(1), device=x.device, dtype=torch.long)
            
        # 应用ResGCN
        x = self.resgcn(x, edge_index)
        x = self.dropout_layer(x)
        
        # 应用边感知GNN
        x = self.edge_gnn(x, edge_index, edge_attr)
        x = self.dropout_layer(x)
        
        # 应用TGN，传递节点索引
        x = self.tgn(x, edge_index, edge_attr, timestamps, node_idxs)
        
        # 规范化
        x = self.norm(x)
        
        return x
    
    def reset_memory(self):
        """
        重置TGN内存
        """
        self.tgn.reset_memory()
        
    def update_memory(self, source_nodes, target_nodes, timestamps, edge_features):
        """
        更新TGN内存
        
        参数:
            source_nodes (Tensor): 源节点索引
            target_nodes (Tensor): 目标节点索引
            timestamps (Tensor): 时间戳
            edge_features (Tensor): 边特征
        """
        # 确保edge_features的维度与TGN期望的一致
        expected_edge_dim = self.tgn.edge_dim if hasattr(self.tgn, 'edge_dim') else 4
        
        if edge_features.size(1) != expected_edge_dim:
            # 调整边特征维度
            if edge_features.size(1) > expected_edge_dim:
                # 如果边特征维度过大，截断
                edge_features = edge_features[:, :expected_edge_dim]
            else:
                # 如果边特征维度过小，填充
                padding = torch.zeros(edge_features.size(0), expected_edge_dim - edge_features.size(1), device=edge_features.device)
                edge_features = torch.cat([edge_features, padding], dim=1)
        
        self.tgn.update_memory(source_nodes, target_nodes, timestamps, edge_features)


class SmartMoneyTemporalModel(nn.Module):
    """
    SmartMoney时序模型：处理地址的行为序列
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        temporal_type: str = "transformer",
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_position_encoding: bool = True,
        max_len: int = 24
    ):
        """
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层特征维度
            output_dim (int): 输出特征维度
            temporal_type (str): 时序模型类型，可选值："transformer", "lstm"
            nhead (int): Transformer的注意力头数
            num_layers (int): 层数
            dropout (float): Dropout概率
            use_position_encoding (bool): 是否使用位置编码（仅用于Transformer）
            max_len (int): 最大序列长度（仅用于Transformer）
        """
        super(SmartMoneyTemporalModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temporal_type = temporal_type
        
        # 选择时序模型
        if temporal_type == "transformer":
            self.temporal_model = TemporalTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                use_position_encoding=use_position_encoding,
                max_len=max_len
            )
        elif temporal_type == "lstm":
            self.temporal_model = TemporalLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2,  # 因为双向LSTM会将维度翻倍
                output_dim=output_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True
            )
        else:
            raise ValueError(f"不支持的时序模型类型: {temporal_type}")
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, input_dim]
            mask (Tensor, 可选): 掩码，用于Transformer
            
        返回:
            Tensor: 序列表示 [batch_size, output_dim]
        """
        if self.temporal_type == "transformer":
            return self.temporal_model(x, mask)
        else:
            return self.temporal_model(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, input_dim]
            
        返回:
            Tensor: 注意力权重
        """
        return self.temporal_model.get_attention_weights(x)


class SmartMoneyModel(nn.Module):
    """
    完整的SmartMoney模型：结合图编码器和时序模型
    """
    def __init__(
        self,
        config: Dict,
        num_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int
    ):
        """
        参数:
            config (Dict): 模型配置
            num_nodes (int): 图中节点数量
            node_feature_dim (int): 节点特征维度
            edge_feature_dim (int): 边特征维度
        """
        super(SmartMoneyModel, self).__init__()
        
        self.config = config
        
        # 获取配置参数
        embedding_dim = config.get('embedding_dim', 128)
        hidden_dim = config.get('hidden_dim', 256)
        output_dim = config.get('output_dim', 128)
        dropout = config.get('dropout', 0.2)
        
        # ResGCN配置
        resgcn_config = config.get('resgcn', {})
        resgcn_num_layers = resgcn_config.get('num_layers', 3)
        
        # Edge GNN配置
        edge_gnn_config = config.get('edge_gnn', {})
        edge_gnn_type = edge_gnn_config.get('type', 'edge_gat')
        
        # TGN配置
        tgn_config = config.get('tgn', {})
        tgn_memory_dim = tgn_config.get('memory_dim', 128)
        tgn_time_dim = tgn_config.get('time_dim', 16)
        tgn_num_layers = tgn_config.get('num_layers', 2)
        
        # 时序模型配置
        temporal_config = config.get('temporal_attention', {})
        temporal_type = temporal_config.get('type', 'transformer')
        nhead = temporal_config.get('nhead', 8)
        temporal_num_layers = temporal_config.get('num_layers', 2)
        use_position_encoding = temporal_config.get('use_position_encoding', True)
        max_len = temporal_config.get('max_len', 24)
        
        # 1. 图编码器
        self.encoder = SmartMoneyEncoder(
            num_nodes=num_nodes,
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            edge_dim=edge_feature_dim,
            dropout=dropout,
            edge_gnn_type=edge_gnn_type,
            resgcn_num_layers=resgcn_num_layers,
            tgn_memory_dim=tgn_memory_dim,
            tgn_time_dim=tgn_time_dim,
            tgn_num_layers=tgn_num_layers
        )
        
        # 2. 时序模型
        self.temporal_model = SmartMoneyTemporalModel(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            temporal_type=temporal_type,
            nhead=nhead,
            num_layers=temporal_num_layers,
            dropout=dropout,
            use_position_encoding=use_position_encoding,
            max_len=max_len
        )
        
        # 3. 投影头 (用于对比学习)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def encode_graph(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        timestamps: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_idxs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码图结构
        
        参数:
            x (Tensor): 节点特征 [N, node_feature_dim]
            edge_index (Tensor): 边索引 [2, E]
            edge_attr (Tensor): 边特征 [E, edge_feature_dim]
            timestamps (Tensor): 时间戳 [E]
            batch (Tensor, 可选): 批处理索引 [N]
            node_idxs (Tensor, 可选): 节点索引 [N]
            
        返回:
            Tensor: 节点嵌入 [N, embedding_dim]
        """
        return self.encoder(x, edge_index, edge_attr, timestamps, batch, node_idxs)
    
    def encode_sequence(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码序列
        
        参数:
            seq (Tensor): 特征序列 [batch_size, seq_len, input_dim]
            mask (Tensor, 可选): 序列掩码
            
        返回:
            Tensor: 序列嵌入 [batch_size, output_dim]
        """
        return self.temporal_model(seq, mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        投影到对比学习空间
        
        参数:
            x (Tensor): 特征 [batch_size, output_dim]
            
        返回:
            Tensor: 投影后的特征 [batch_size, embedding_dim]
        """
        return self.projection_head(x)
    
    def forward(
        self, 
        graphs_sequence: Union[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], Batch],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，处理图序列
        
        参数:
            graphs_sequence (List[Tuple] 或 Batch): 图序列或批处理的图数据
            mask (Tensor, 可选): 序列掩码
            
        返回:
            Dict: 包含各类输出的字典
        """
        # 检查输入类型
        if isinstance(graphs_sequence, Batch):
            # 如果是PyG的Batch对象，直接进行图编码
            # 获取必要属性，并确保它们存在
            x = graphs_sequence.x
            edge_index = graphs_sequence.edge_index
            
            # 处理可选属性
            edge_attr = None
            if hasattr(graphs_sequence, 'edge_attr') and graphs_sequence.edge_attr is not None:
                edge_attr = graphs_sequence.edge_attr
            
            timestamps = None
            if hasattr(graphs_sequence, 'timestamps') and graphs_sequence.timestamps is not None:
                timestamps = graphs_sequence.timestamps
            
            # 获取批次索引
            batch = None
            if hasattr(graphs_sequence, 'batch') and graphs_sequence.batch is not None:
                batch = graphs_sequence.batch
                
            # 获取节点索引
            node_idxs = None
            if hasattr(graphs_sequence, 'node_idxs') and graphs_sequence.node_idxs is not None:
                node_idxs = graphs_sequence.node_idxs
            
            # 安全检查：确保我们有所有必要的数据，并且维度匹配
            if edge_attr is None:
                edge_dim = self.encoder.edge_gnn.in_channels if hasattr(self.encoder.edge_gnn, 'in_channels') else 4
                edge_attr = torch.zeros((edge_index.size(1), edge_dim), device=edge_index.device)
            
            if timestamps is None:
                timestamps = torch.zeros(edge_index.size(1), device=edge_index.device, dtype=torch.long)
            
            # 重置内存以确保一致性
            self.encoder.reset_memory()
            
            # 更新所有边的内存
            if edge_index.size(1) > 0:
                source_nodes, target_nodes = edge_index
                self.encoder.update_memory(source_nodes, target_nodes, timestamps, edge_attr)
            
            # 编码图，同时传递节点索引以正确获取内存
            embedding = self.encode_graph(x, edge_index, edge_attr, timestamps, batch, node_idxs)
            
            # 如果是单个图的批处理，不进行时序处理，直接投影
            projection = self.project(embedding)
            
            return {
                "sequence_embeddings": embedding.unsqueeze(1),  # 添加序列维度
                "temporal_embedding": embedding,
                "projection": projection
            }
        else:
            # 如果是图序列（时间窗口数据），进行时序处理
            # 重置内存
            self.encoder.reset_memory()
            
            # 处理每个时间点的图，构建特征序列
            sequence_embeddings = []
            for x, edge_index, edge_attr, timestamps in graphs_sequence:
                # 安全检查：确保边索引在有效范围内
                if edge_index.numel() > 0:
                    num_nodes = x.size(0)
                    valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                    
                    if not torch.all(valid_mask):
                        # 过滤无效的边
                        edge_index = edge_index[:, valid_mask]
                        if edge_attr is not None and edge_attr.size(0) > 0:
                            edge_attr = edge_attr[valid_mask]
                        if timestamps is not None and timestamps.size(0) > 0:
                            timestamps = timestamps[valid_mask]
                
                # 确保edge_attr和timestamps与edge_index长度匹配
                if edge_attr is None or edge_attr.size(0) != edge_index.size(1):
                    edge_dim = self.encoder.edge_gnn.in_channels if hasattr(self.encoder.edge_gnn, 'in_channels') else 4
                    edge_attr = torch.zeros((edge_index.size(1), edge_dim), device=edge_index.device)
                
                if timestamps is None or timestamps.size(0) != edge_index.size(1):
                    timestamps = torch.zeros(edge_index.size(1), device=edge_index.device, dtype=torch.long)
                
                # 只有当边存在时更新内存
                if edge_index.size(1) > 0:
                    source_nodes, target_nodes = edge_index
                    self.encoder.update_memory(source_nodes, target_nodes, timestamps, edge_attr)
                
                # 获取当前序列中的节点索引
                node_idxs = torch.unique(edge_index.flatten()) if edge_index.numel() > 0 else None
                
                # 编码图，同时传递节点索引以正确获取内存
                embedding = self.encode_graph(x, edge_index, edge_attr, timestamps, node_idxs=node_idxs)
                sequence_embeddings.append(embedding)
            
            # 将特征序列堆叠成张量 [batch_size, seq_len, embedding_dim]
            sequence_tensor = torch.stack(sequence_embeddings, dim=1)
            
            # 用时序模型处理特征序列
            temporal_embedding = self.encode_sequence(sequence_tensor, mask)
            
            # 投影到对比学习空间
            projection = self.project(temporal_embedding)
            
            return {
                "sequence_embeddings": sequence_tensor,
                "temporal_embedding": temporal_embedding,
                "projection": projection
            }
        
    def get_attention_weights(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """
        获取时序模型的注意力权重
        
        参数:
            sequence_tensor (Tensor): 特征序列 [batch_size, seq_len, embedding_dim]
            
        返回:
            Tensor: 注意力权重
        """
        return self.temporal_model.get_attention_weights(sequence_tensor) 