import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    """
    具有残差连接、批归一化和激活函数的GCN层
    """
    def __init__(self, in_channels, out_channels, use_bn=True, residual=True, activation="relu"):
        """
        参数:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            use_bn (bool): 是否使用批归一化
            residual (bool): 是否使用残差连接
            activation (str): 激活函数类型，可选值: "relu", "leaky_relu", "elu", "gelu"
        """
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.use_bn = use_bn
        self.residual = residual
        
        # 如果输入输出维度不一致，且需要残差连接，则添加线性映射
        if residual and in_channels != out_channels:
            self.res_conv = nn.Linear(in_channels, out_channels)
        else:
            self.res_conv = None
            
        # 批归一化层
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        
        # 设置激活函数
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            
        返回:
            Tensor: 更新后的节点特征 [N, out_channels]
        """
        # 安全检查：确保边索引不包含无效索引
        if edge_index.numel() > 0:
            num_nodes = x.size(0)
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            if not torch.all(valid_mask):
                edge_index = edge_index[:, valid_mask]
                
            # 如果过滤后没有边，添加自环以避免孤立节点
            if edge_index.size(1) == 0:
                # 为每个节点添加自环
                edge_index = torch.stack([
                    torch.arange(num_nodes, device=x.device),
                    torch.arange(num_nodes, device=x.device)
                ], dim=0)
            
            # 确保没有孤立节点（通过添加自环）
            # 计算度
            node_degrees = torch.zeros(num_nodes, device=x.device)
            node_degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=x.device))
            node_degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device))
            
            # 找出孤立节点（度为0的节点）
            isolated_nodes = torch.where(node_degrees == 0)[0]
            
            if isolated_nodes.numel() > 0:
                # 为孤立节点添加自环
                self_loops = torch.stack([isolated_nodes, isolated_nodes], dim=0)
                edge_index = torch.cat([edge_index, self_loops], dim=1)
                
        # 应用GCN卷积
        out = self.conv(x, edge_index)
        
        # 残差连接
        if self.residual:
            if self.res_conv is not None:
                x = self.res_conv(x)
            if x.size(0) == out.size(0):
                out = out + x
        
        # 批归一化
        if self.use_bn:
            out = self.bn(out)
        
        # 激活函数
        out = self.activation(out)
        
        return out


class ResGCN(nn.Module):
    """
    残差连接的多层GCN网络
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.2,
        use_bn=True,
        residual=True,
        activation="relu"
    ):
        """
        参数:
            in_channels (int): 输入特征维度
            hidden_channels (int): 隐藏层特征维度
            out_channels (int): 输出特征维度
            num_layers (int): GCN层数
            dropout (float): Dropout概率
            use_bn (bool): 是否使用批归一化
            residual (bool): 是否使用残差连接
            activation (str): 激活函数类型
        """
        super(ResGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建多层GCN
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(
            GCNLayer(in_channels, hidden_channels, use_bn, residual, activation)
        )
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(
                GCNLayer(hidden_channels, hidden_channels, use_bn, residual, activation)
            )
        
        # 输出层
        self.layers.append(
            GCNLayer(hidden_channels, out_channels, use_bn, residual, activation)
        )
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            
        返回:
            Tensor: 输出节点特征 [N, out_channels]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:  # 除了最后一层，其他层添加dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x 