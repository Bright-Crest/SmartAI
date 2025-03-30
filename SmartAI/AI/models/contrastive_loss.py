import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NTXentLoss(nn.Module):
    """
    标准化温度缩放的交叉熵损失函数 (NT-Xent Loss)
    这是SimCLR中使用的对比损失函数
    """
    def __init__(self, temperature=0.5, use_cosine_similarity=True):
        """
        参数:
            temperature (float): 温度参数，控制分布的平滑度
            use_cosine (bool): 是否使用余弦相似度，否则使用点积
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        
    def forward(self, z_i, z_j):
        """
        计算两组嵌入之间的对比损失
        
        参数:
            z_i (Tensor): 第一组嵌入 [batch_size, dim]
            z_j (Tensor): 第二组嵌入 [batch_size, dim]
            
        返回:
            float: 对比损失值
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # 对特征进行L2归一化
        if self.use_cosine_similarity:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)
        
        # 将正样本对拼接成一个批次
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
        
        # 计算批次内所有样本对的相似度矩阵
        similarity_matrix = torch.matmul(representations, representations.t())  # [2*batch_size, 2*batch_size]
        
        # 对角线上是样本与自身的相似度，设为一个很小的值
        sim_i_i = torch.diag(similarity_matrix, 0)
        sim_j_j = torch.diag(similarity_matrix, batch_size)
        sim_i_j = torch.diag(similarity_matrix, -batch_size)
        sim_j_i = torch.diag(similarity_matrix, batch_size)
        
        # 创建掩码，过滤掉自身与自身的相似度
        mask = torch.ones_like(similarity_matrix) - torch.eye(2 * batch_size, device=device)
        
        # 应用温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        # 正样本对索引：第i个样本与第(i+batch_size)个样本构成正样本对
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device)
        
        # 标签：对于每个样本，正样本的位置即为分类目标
        labels = torch.cat(
            [torch.arange(batch_size, 2 * batch_size, device=device),
             torch.arange(0, batch_size, device=device)],
            dim=0
        )
        
        # 计算交叉熵损失
        loss = self.criterion(similarity_matrix, labels)
        
        return loss


class ClusterLoss(nn.Module):
    """
    聚类辅助损失，根据聚类分配结果优化嵌入
    """
    def __init__(self, n_clusters=10, sinkhorn_iterations=3):
        """
        参数:
            n_clusters (int): 聚类数量
            sinkhorn_iterations (int): Sinkhorn-Knopp算法迭代次数
        """
        super(ClusterLoss, self).__init__()
        self.n_clusters = n_clusters
        self.sinkhorn_iterations = sinkhorn_iterations
        self.eps = 1e-8
        
    def forward(self, features, cluster_centers):
        """
        计算特征与聚类中心之间的损失
        
        参数:
            features (Tensor): 地址特征 [batch_size, dim]
            cluster_centers (Tensor): 聚类中心 [n_clusters, dim]
            
        返回:
            float: 聚类损失值
        """
        # 特征归一化
        features = F.normalize(features, dim=1)
        cluster_centers = F.normalize(cluster_centers, dim=1)
        
        # 计算特征与聚类中心的余弦相似度
        similarity = torch.matmul(features, cluster_centers.t())  # [batch_size, n_clusters]
        
        # 应用Sinkhorn-Knopp算法获取软聚类分配
        # 这一步相当于优化传输问题，使得每个样本都被分配到某个聚类
        Q = self._sinkhorn(similarity)
        
        # 计算聚类损失 (负对数似然)
        loss = -torch.mean(torch.sum(Q * torch.log(Q + self.eps), dim=1))
        
        return loss, Q
    
    def _sinkhorn(self, similarity):
        """
        应用Sinkhorn-Knopp算法进行优化
        
        参数:
            similarity (Tensor): 相似度矩阵 [batch_size, n_clusters]
            
        返回:
            Tensor: 优化后的聚类分配 [batch_size, n_clusters]
        """
        Q = torch.exp(similarity)
        
        # 应用Sinkhorn-Knopp迭代
        for _ in range(self.sinkhorn_iterations):
            # 行归一化
            Q = Q / (torch.sum(Q, dim=1, keepdim=True) + self.eps)
            
            # 列归一化
            Q = Q / (torch.sum(Q, dim=0, keepdim=True) + self.eps)
        
        return Q


class SwAVLoss(nn.Module):
    """
    SwAV (Swapping Assignments between Views) 损失
    这种损失使用了一种聚类辅助的对比方法
    """
    def __init__(self, temperature=0.1, sinkhorn_iterations=3, epsilon=0.05):
        """
        参数:
            temperature (float): 温度参数
            sinkhorn_iterations (int): Sinkhorn-Knopp算法迭代次数
            epsilon (float): Sinkhorn-Knopp算法的正则化参数
        """
        super(SwAVLoss, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        
    def forward(self, z_i, z_j, prototype_vectors):
        """
        计算SwAV损失
        
        参数:
            z_i (Tensor): 第一组嵌入 [batch_size, dim]
            z_j (Tensor): 第二组嵌入 [batch_size, dim]
            prototype_vectors (Tensor): 原型向量 [n_prototypes, dim]
            
        返回:
            float: SwAV损失值
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # 对特征和原型向量进行L2归一化
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        prototype_vectors = F.normalize(prototype_vectors, dim=1)
        
        # 计算嵌入与原型向量的相似度
        similarity_i = torch.matmul(z_i, prototype_vectors.t()) / self.temperature
        similarity_j = torch.matmul(z_j, prototype_vectors.t()) / self.temperature
        
        # 计算软分配（通过Sinkhorn-Knopp算法）
        with torch.no_grad():
            q_i = self._sinkhorn(similarity_i)
            q_j = self._sinkhorn(similarity_j)
        
        # 计算交叉熵损失
        loss_i = -torch.mean(torch.sum(q_j * F.log_softmax(similarity_i, dim=1), dim=1))
        loss_j = -torch.mean(torch.sum(q_i * F.log_softmax(similarity_j, dim=1), dim=1))
        
        loss = (loss_i + loss_j) / 2
        
        return loss
    
    def _sinkhorn(self, scores):
        """
        应用Sinkhorn-Knopp算法获取软聚类分配
        
        参数:
            scores (Tensor): 相似度分数 [batch_size, n_prototypes]
            
        返回:
            Tensor: 软聚类分配 [batch_size, n_prototypes]
        """
        Q = torch.exp(scores / self.epsilon)
        
        # 列归一化
        Q = Q / torch.sum(Q, dim=0, keepdim=True)
        
        # 应用Sinkhorn-Knopp迭代
        for _ in range(self.sinkhorn_iterations):
            # 行归一化
            Q = Q / torch.sum(Q, dim=1, keepdim=True)
            
            # 列归一化
            Q = Q / torch.sum(Q, dim=0, keepdim=True)
        
        return Q 