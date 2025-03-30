import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    accuracy_score, classification_report,
    roc_auc_score, precision_recall_curve, average_precision_score
)


def compute_similarity_matrix(
    embeddings: np.ndarray, 
    normalize: bool = True
) -> np.ndarray:
    """
    计算嵌入向量之间的相似度矩阵
    
    参数:
        embeddings (np.ndarray): 嵌入向量，形状为 [n_samples, embedding_dim]
        normalize (bool): 是否对嵌入向量进行归一化
        
    返回:
        np.ndarray: 相似度矩阵，形状为 [n_samples, n_samples]
    """
    if normalize:
        # 对嵌入向量进行L2归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
    
    # 计算余弦相似度矩阵 (dot product of normalized vectors)
    similarity_matrix = np.matmul(embeddings, embeddings.T)
    
    return similarity_matrix


def retrieval_precision_at_k(
    similarity_matrix: np.ndarray, 
    labels: np.ndarray, 
    k: int = 10
) -> float:
    """
    计算Top-K检索精度
    
    参数:
        similarity_matrix (np.ndarray): 相似度矩阵，形状为 [n_samples, n_samples]
        labels (np.ndarray): 样本标签，形状为 [n_samples]
        k (int): 取top-k个结果
        
    返回:
        float: 平均检索精度@k
    """
    n_samples = similarity_matrix.shape[0]
    precisions = []
    
    # 对于每个查询样本
    for i in range(n_samples):
        # 获取相似度并排序（排除自身）
        similarities = similarity_matrix[i].copy()
        similarities[i] = -float('inf')  # 排除自身
        
        # 获取top-k个最相似的样本
        top_indices = np.argsort(-similarities)[:k]
        
        # 计算正确召回的数量（与查询样本同类的样本数）
        query_label = labels[i]
        relevant = (labels[top_indices] == query_label).sum()
        
        # 计算precision@k
        precision_at_k = relevant / k
        precisions.append(precision_at_k)
    
    # 计算平均精度
    return np.mean(precisions)


def mean_average_precision(
    similarity_matrix: np.ndarray, 
    labels: np.ndarray
) -> float:
    """
    计算平均精度均值(MAP)
    
    参数:
        similarity_matrix (np.ndarray): 相似度矩阵，形状为 [n_samples, n_samples]
        labels (np.ndarray): 样本标签，形状为 [n_samples]
        
    返回:
        float: 平均精度均值
    """
    n_samples = similarity_matrix.shape[0]
    aps = []
    
    # 对于每个查询样本
    for i in range(n_samples):
        # 获取相似度并排序（排除自身）
        similarities = similarity_matrix[i].copy()
        similarities[i] = -float('inf')  # 排除自身
        
        # 获取排序索引
        sorted_indices = np.argsort(-similarities)
        
        # 获取同类样本索引
        query_label = labels[i]
        relevant_indices = np.where(labels == query_label)[0]
        relevant_indices = relevant_indices[relevant_indices != i]  # 排除自身
        
        if len(relevant_indices) == 0:
            continue  # 跳过唯一类别的样本
        
        # 计算平均精度
        positions = np.where(np.isin(sorted_indices, relevant_indices))[0] + 1
        precisions = np.arange(1, len(positions) + 1) / positions
        ap = np.mean(precisions)
        aps.append(ap)
    
    # 计算平均精度均值
    return np.mean(aps)


def normalized_mutual_information(
    clusters: np.ndarray, 
    labels: np.ndarray
) -> float:
    """
    计算规范化互信息(NMI)
    
    参数:
        clusters (np.ndarray): 聚类分配，形状为 [n_samples]
        labels (np.ndarray): 真实标签，形状为 [n_samples]
        
    返回:
        float: NMI分数
    """
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels, clusters)


def adjusted_rand_index(
    clusters: np.ndarray, 
    labels: np.ndarray
) -> float:
    """
    计算调整兰德指数(ARI)
    
    参数:
        clusters (np.ndarray): 聚类分配，形状为 [n_samples]
        labels (np.ndarray): 真实标签，形状为 [n_samples]
        
    返回:
        float: ARI分数
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels, clusters)


def classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    计算分类指标
    
    参数:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        average (str): 平均方法，'micro', 'macro', 'weighted'或'samples'
        
    返回:
        Dict: 包含各种分类指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # 如果有多于两个类别，计算micro和macro平均值
    if len(np.unique(y_true)) > 2:
        metrics.update({
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        })
    
    return metrics


def binary_classification_metrics(
    y_true: np.ndarray, 
    y_score: np.ndarray
) -> Dict[str, float]:
    """
    计算二分类的评估指标
    
    参数:
        y_true (np.ndarray): 真实标签 (0或1)
        y_score (np.ndarray): 预测概率或得分
        
    返回:
        Dict: 包含各种二分类指标的字典
    """
    # 确保是二分类问题
    if len(np.unique(y_true)) != 2:
        raise ValueError("二分类指标仅适用于有两个类别的数据")
    
    # 将y_true转换为二分类标签
    y_true_binary = (y_true == 1).astype(int)
    
    # 计算ROC AUC
    auc = roc_auc_score(y_true_binary, y_score)
    
    # 计算PR AUC
    average_precision = average_precision_score(y_true_binary, y_score)
    
    # 计算各个阈值的精度和召回率
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
    
    # 在具有最高F1分数的阈值处计算指标
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_threshold_idx]
    
    # 使用最佳阈值进行预测
    y_pred_binary = (y_score >= best_threshold).astype(int)
    
    # 计算其他指标
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision_at_best = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall_at_best = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return {
        'auc': auc,
        'pr_auc': average_precision,
        'accuracy': accuracy,
        'precision': precision_at_best,
        'recall': recall_at_best,
        'f1': best_f1,
        'threshold': best_threshold
    }


def evaluate_embeddings_retrieval(
    embeddings: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    """
    通过检索任务评估嵌入向量的质量
    
    参数:
        embeddings (np.ndarray): 嵌入向量，形状为 [n_samples, embedding_dim]
        labels (np.ndarray): 样本标签，形状为 [n_samples]
        
    返回:
        Dict: 包含各种检索指标的字典
    """
    # 计算相似度矩阵
    similarity_matrix = compute_similarity_matrix(embeddings, normalize=True)
    
    # 计算检索指标
    metrics = {
        'p@1': retrieval_precision_at_k(similarity_matrix, labels, k=1),
        'p@5': retrieval_precision_at_k(similarity_matrix, labels, k=5),
        'p@10': retrieval_precision_at_k(similarity_matrix, labels, k=10),
        'map': mean_average_precision(similarity_matrix, labels)
    }
    
    return metrics


def evaluate_embeddings_clustering(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    n_clusters: Optional[int] = None
) -> Dict[str, float]:
    """
    通过聚类任务评估嵌入向量的质量
    
    参数:
        embeddings (np.ndarray): 嵌入向量，形状为 [n_samples, embedding_dim]
        labels (np.ndarray): 样本标签，形状为 [n_samples]
        n_clusters (int, 可选): 聚类数量，默认为标签中唯一类别数量
        
    返回:
        Dict: 包含各种聚类指标的字典
    """
    from sklearn.cluster import KMeans
    
    # 确定聚类数量
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # 使用K-Means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # 计算聚类指标
    metrics = {
        'nmi': normalized_mutual_information(clusters, labels),
        'ari': adjusted_rand_index(clusters, labels)
    }
    
    return metrics


def interclass_distance(
    embeddings: np.ndarray, 
    labels: np.ndarray
) -> Dict[str, float]:
    """
    计算类内和类间距离
    
    参数:
        embeddings (np.ndarray): 嵌入向量，形状为 [n_samples, embedding_dim]
        labels (np.ndarray): 样本标签，形状为 [n_samples]
        
    返回:
        Dict: 包含类内距离、类间距离和比率的字典
    """
    unique_labels = np.unique(labels)
    
    # 计算类中心
    centroids = []
    for label in unique_labels:
        mask = (labels == label)
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # 计算类内距离（样本到其所属类中心的平均距离）
    intra_distances = []
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        class_samples = embeddings[mask]
        centroid = centroids[i]
        distances = np.linalg.norm(class_samples - centroid, axis=1)
        intra_distances.append(np.mean(distances))
    intra_class_distance = np.mean(intra_distances)
    
    # 计算类间距离（类中心之间的平均距离）
    inter_class_distances = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            inter_class_distances.append(distance)
    inter_class_distance = np.mean(inter_class_distances) if inter_class_distances else 0
    
    # 计算比率（类间距离/类内距离，越大越好）
    ratio = inter_class_distance / intra_class_distance if intra_class_distance > 0 else 0
    
    return {
        'intra_class_distance': intra_class_distance,
        'inter_class_distance': inter_class_distance,
        'distance_ratio': ratio
    }


def silhouette_score(
    embeddings: np.ndarray, 
    labels: np.ndarray
) -> float:
    """
    计算轮廓系数
    
    参数:
        embeddings (np.ndarray): 嵌入向量，形状为 [n_samples, embedding_dim]
        labels (np.ndarray): 样本标签，形状为 [n_samples]
        
    返回:
        float: 轮廓系数（-1到1，越高越好）
    """
    from sklearn.metrics import silhouette_score as sk_silhouette_score
    try:
        score = sk_silhouette_score(embeddings, labels)
        return score
    except:
        return float('nan')  # 处理错误情况，例如只有一个类别 