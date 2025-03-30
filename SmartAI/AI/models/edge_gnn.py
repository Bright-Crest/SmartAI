import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, EdgeConv, MessagePassing
from torch_geometric.utils import softmax


class EdgeGAT(nn.Module):
    """
    边特征感知的图注意力网络 (GAT)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim=4,
        heads=2,
        concat=True,
        negative_slope=0.2,
        dropout=0.0
    ):
        """
        参数:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int): 边特征维度
            heads (int): 注意力头数
            concat (bool): 是否连接多头注意力的结果，False则取平均
            negative_slope (float): LeakyReLU的负斜率
            dropout (float): Dropout概率
        """
        super(EdgeGAT, self).__init__()
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        actual_out_channels = out_channels * heads if concat else out_channels
        self.bn = nn.BatchNorm1d(actual_out_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            edge_attr (Tensor, 可选): 边特征 [E, edge_dim]
            
        返回:
            Tensor: 更新后的节点特征 [N, out_channels*heads] 或 [N, out_channels]
        """
        x = self.gat(x, edge_index, edge_attr)
        x = self.bn(x)
        return x


class EdgeConv(nn.Module):
    """
    基于边特征的动态边卷积网络
    """
    def __init__(self, in_channels, out_channels, edge_dim=4):
        """
        参数:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int): 边特征维度
        """
        super(EdgeConv, self).__init__()
        
        # 节点特征转换
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        
        # 边特征转换
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
        
        # 消息传递网络
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + out_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            edge_attr (Tensor, 可选): 边特征 [E, edge_dim]
            
        返回:
            Tensor: 更新后的节点特征 [N, out_channels]
        """
        # 转换节点特征
        node_feat = self.node_mlp(x)
        
        # 转换边特征
        if edge_attr is not None:
            edge_feat = self.edge_mlp(edge_attr)
        else:
            edge_feat = torch.zeros((edge_index.size(1), node_feat.size(1)), 
                                   device=node_feat.device)
        
        # 计算消息
        row, col = edge_index
        message = torch.cat([x[row], x[col], edge_feat], dim=1)
        message = self.message_mlp(message)
        
        # 聚合消息（按目标节点求和）
        out = torch.zeros_like(node_feat)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, node_feat.size(1)), message)
        
        # 添加自环
        out = out + node_feat
        
        return out


class MPNN(MessagePassing):
    """
    消息传递神经网络，显式处理边特征
    """
    def __init__(self, in_channels, out_channels, edge_dim=4, aggr='add'):
        """
        参数:
            in_channels (int): 输入特征维度
            out_channels (int): 输出特征维度
            edge_dim (int): 边特征维度
            aggr (str): 聚合方法 ('add', 'mean', 'max')
        """
        super(MPNN, self).__init__(aggr=aggr)
        
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # 更新网络
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        参数:
            x (Tensor): 节点特征矩阵 [N, in_channels]
            edge_index (Tensor): 边索引 [2, E]
            edge_attr (Tensor, 可选): 边特征 [E, edge_dim]
            
        返回:
            Tensor: 更新后的节点特征 [N, out_channels]
        """
        # 变换节点特征
        node_feat = self.lin_node(x)
        
        # 执行消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 更新状态
        out = self.mlp(torch.cat([x, out], dim=1))
        out = self.bn(out)
        
        return out
    
    def message(self, x_j, edge_attr):
        """
        计算消息
        """
        if edge_attr is None:
            edge_attr = torch.zeros((x_j.size(0), 1), device=x_j.device)
            
        # 连接源节点特征和边特征
        msg = torch.cat([x_j, edge_attr], dim=1)
        msg = self.lin_msg(msg)
        
        return msg
    
    def update(self, aggr_out):
        """
        更新节点特征
        """
        return aggr_out 