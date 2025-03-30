import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块 (来自Transformer的标准位置编码)
    """
    def __init__(self, d_model, max_len=24):
        """
        参数:
            d_model (int): 模型维度
            max_len (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册位置编码为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入张量
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, d_model]
            
        返回:
            Tensor: 添加位置编码后的序列 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    """
    基于Transformer的时序注意力模块
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nhead=8,
        num_layers=2,
        dropout=0.1,
        use_position_encoding=True,
        max_len=24
    ):
        """
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            nhead (int): 注意力头数
            num_layers (int): Transformer编码器层数
            dropout (float): Dropout概率
            use_position_encoding (bool): 是否使用位置编码
            max_len (int): 最大序列长度（用于位置编码）
        """
        super(TemporalTransformer, self).__init__()
        
        self.use_position_encoding = use_position_encoding
        
        # 输入映射
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        if use_position_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True  # 确保输入格式为 [batch_size, seq_len, hidden_dim]
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 输出映射
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # LayerNorm和Dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, input_dim]
            mask (Tensor, 可选): 注意力掩码 [seq_len, seq_len]
            
        返回:
            Tensor: 整个序列的表示 [batch_size, output_dim]
        """
        # 输入映射
        x = self.input_projection(x)
        
        # 应用位置编码
        if self.use_position_encoding:
            x = self.pos_encoder(x)
        
        # 应用Transformer编码器
        x = self.norm(x)
        x = self.transformer_encoder(x, src_mask=mask)
        
        # 获取整个序列的表示（可以用不同聚合方式）
        x_mean = torch.mean(x, dim=1)  # 平均池化 [batch_size, hidden_dim]
        
        # 输出映射
        output = self.output_projection(x_mean)
        
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, input_dim]
            
        返回:
            Tensor: 注意力权重 [batch_size, nhead, seq_len, seq_len]
        """
        # 注意：这个方法需要修改Transformer的实现以访问注意力权重
        # 此处仅为占位，实际实现需根据PyTorch版本调整
        # 目前PyTorch标准Transformer不直接暴露注意力权重
        return None


class TemporalLSTM(nn.Module):
    """
    基于LSTM+注意力的时序模块
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.1,
        bidirectional=True
    ):
        """
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): LSTM隐藏层维度
            output_dim (int): 输出特征维度
            num_layers (int): LSTM层数
            dropout (float): Dropout概率
            bidirectional (bool): 是否使用双向LSTM
        """
        super(TemporalLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * self.num_directions,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出映射
        self.output_projection = nn.Linear(hidden_dim * self.num_directions, output_dim)
        
        # LayerNorm和Dropout
        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, input_dim]
            
        返回:
            Tensor: 序列表示 [batch_size, output_dim]
        """
        # 应用LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim*num_directions]
        
        # 应用自注意力
        attn_output, attn_weights = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out
        )
        
        # 残差连接和归一化
        attn_output = self.norm(lstm_out + self.dropout(attn_output))
        
        # 计算最终表示（可以用不同的聚合方式）
        output = torch.mean(attn_output, dim=1)  # [batch_size, hidden_dim*num_directions]
        
        # 输出映射
        output = self.output_projection(output)  # [batch_size, output_dim]
        
        return output
    
    def get_attention_weights(self, x):
        """
        获取注意力权重（用于可视化）
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, input_dim]
            
        返回:
            Tensor: 注意力权重 [batch_size, seq_len, seq_len]
        """
        # 首先通过LSTM
        lstm_out, _ = self.lstm(x)
        
        # 计算注意力
        _, attn_weights = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out
        )
        
        return attn_weights 