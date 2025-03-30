import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path
import pickle
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call
from .base_model import BaseModel

class PyTorchBaseWrapper(BaseModel):
    """PyTorch模型的基础包装器"""
    
    def __init__(self, model_type, model_name, model_params):
        # 过滤掉与PyTorch模型无关的参数，只保留相关参数
        pytorch_relevant_params = {}
        for param, value in model_params.items():
            if param in ['n_classes', 'input_size', 'input_channels', 'img_size', 
                         'hidden_sizes', 'learning_rate', 'batch_size', 'epochs', 
                         'temperature']:
                pytorch_relevant_params[param] = value
        
        # 将原始参数传递给父类
        super().__init__(model_type, model_name, model_params)
        
        # 设置PyTorch特定参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.criterion = None
        self.n_classes = pytorch_relevant_params.get('n_classes', 2)
        self.input_size = pytorch_relevant_params.get('input_size', 784)  # 默认MNIST
        self.learning_rate = pytorch_relevant_params.get('learning_rate', 0.001)
        self.batch_size = pytorch_relevant_params.get('batch_size', 64)
        self.epochs = pytorch_relevant_params.get('epochs', 10)
        self.temperature = pytorch_relevant_params.get('temperature', 1.0)
    
    def _infer_input_size(self, X):
        """根据输入数据推断输入尺寸"""
        # 如果已经设置了输入尺寸且不是默认值，则不做更改
        if hasattr(self, 'input_size') and self.input_size != 784:
            return self.input_size
            
        # 确定输入尺寸
        if len(X.shape) == 2:
            # 对于扁平化数据，直接使用特征数量
            return X.shape[1]
        elif len(X.shape) == 3:
            # 对于2D数据（通常是单通道图像），计算总尺寸
            return X.shape[1] * X.shape[2]
        elif len(X.shape) == 4:
            # 对于3D数据（多通道图像），计算总尺寸
            return X.shape[1] * X.shape[2] * X.shape[3]
        else:
            # 默认返回原始输入尺寸
            return self.input_size
    
    def _get_default_optimizer(self):
        """获取默认优化器"""
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _get_default_criterion(self):
        """获取默认损失函数"""
        return F.cross_entropy
    
    def _prepare_data(self, X, y=None):
        """准备数据，转为PyTorch张量"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            if y.ndim == 2:  # 如果是软标签
                y_tensor = torch.FloatTensor(y).to(self.device)
            else:
                y_tensor = torch.LongTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor
    
    def _to_numpy(self, tensor):
        """转换张量为NumPy数组"""
        return tensor.detach().cpu().numpy()
    
    def _create_data_loader(self, X, y, batch_size=None):
        """创建数据加载器"""
        if batch_size is None:
            batch_size = self.batch_size
        
        X_tensor, y_tensor = self._prepare_data(X, y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    @log_function_call
    def save(self, filename=None):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        if filename is None:
            filename = f"{self.model_type}_{self.model_name}.pt"
        
        save_path = Path(filename)
        save_dir = save_path.parent
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 保存模型参数和优化器状态
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'model_params': self.model_params,
                'n_classes': self.n_classes
            }
            torch.save(save_dict, save_path)
            logger.info(f"PyTorch模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存PyTorch模型时出错: {str(e)}")
            raise
    
    @log_function_call
    def load(self, filename=None):
        """加载模型"""
        if filename is None:
            filename = f"{self.model_type}_{self.model_name}.pt"
        
        load_path = Path(filename)
        
        try:
            # 确保模型已经构建
            if self.model is None:
                self.build()
                
            # 加载模型参数和优化器状态
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 如果有优化器且优化器状态存在，则加载优化器状态
            if self.optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # 更新模型参数
            self.n_classes = checkpoint.get('n_classes', self.n_classes)
            
            logger.info(f"PyTorch模型已从 {load_path} 加载")
        except Exception as e:
            logger.error(f"加载PyTorch模型时出错: {str(e)}")
            raise
    
    @log_function_call
    def get_model_size(self):
        """获取模型大小"""
        if self.model is None:
            return 0
        
        try:
            # 计算参数总数
            total_params = sum(p.numel() for p in self.model.parameters())
            # 粗略估计模型大小（字节）
            model_size_kb = (total_params * 4) / 1024  # 假设每个参数是float32 (4 bytes)
            return model_size_kb
        except Exception as e:
            logger.warning(f"获取PyTorch模型大小失败: {str(e)}")
            return 0
    
    @log_function_call
    def get_model(self):
        """获取原始PyTorch模型
        
        返回原始的PyTorch模型对象(nn.Module)，而不是包装器本身
        
        Returns:
            torch.nn.Module: 原始的PyTorch模型
        """
        if self.model is None:
            logger.warning("模型尚未初始化")
            return None
        
        return self.model


# PyTorch CNN模型定义
class CNN(nn.Module):
    def __init__(self, input_channels=1, img_size=28, n_classes=10):
        super(CNN, self).__init__()
        
        # 基本参数
        self.input_channels = input_channels
        self.img_size = img_size
        self.n_classes = n_classes
        
        # 确保图像尺寸是2的整数次幂，有利于下采样
        self.is_power_of_two = (img_size & (img_size - 1) == 0)
        
        # CNN架构
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 根据图像尺寸计算卷积后的特征图大小
        # 第一次池化后的尺寸
        first_pool_size = img_size // 2
        # 第二次池化后的尺寸
        second_pool_size = first_pool_size // 2
        
        # 全连接层
        self.fc1 = nn.Linear(64 * second_pool_size * second_pool_size, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        # 重新调整输入尺寸
        if x.dim() == 2:
            # 如果输入是扁平的 (batch_size, features)
            batch_size = x.size(0)
            # 检查特征数量是否匹配预期的图像大小
            expected_features = self.input_channels * self.img_size * self.img_size
            if x.size(1) != expected_features:
                # 输入特征数量与预期不匹配，记录警告并尝试调整
                actual_features = x.size(1)
                # 假设输入通道数不变，尝试计算新的图像大小
                new_img_size = int(np.sqrt(actual_features / self.input_channels))
                if new_img_size * new_img_size * self.input_channels != actual_features:
                    # 如果不是完全平方数，则使用原始尺寸
                    x = x[:, :expected_features]  # 截断或填充
                else:
                    # 更新图像大小
                    self.img_size = new_img_size
            # 重塑为 (batch_size, channels, height, width)
            x = x.view(batch_size, self.input_channels, self.img_size, self.img_size)
        
        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        
        # 扁平化
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
        
    def get_selected_features(self):
        """获取选中的特征索引 - 此方法仅为兼容API，原生CNN不支持特征选择"""
        return np.arange(self.input_size)
        
    def get_mask_sparsity(self):
        """获取特征掩码的稀疏度 - 此方法仅为兼容API，原生CNN不支持特征选择"""
        return 1.0


# PyTorch MLP模型定义
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], n_classes=10, 
                 enable_factor_selection=False, max_factor_ratio=0.2, factor_temperature=1.0, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.enable_factor_selection = enable_factor_selection
        
        # 添加特征选择支持
        if enable_factor_selection:
            self.max_factor_ratio = max_factor_ratio
            self.factor_temperature = factor_temperature
            
            # 可学习的特征掩码，初始化为接近0的随机值
            self.feature_mask = nn.Parameter(torch.randn(input_size) * 0.01)
            
            # 低秩投影层
            max_factors = max(int(input_size * max_factor_ratio), 1)
            self.low_rank_projection = nn.Linear(input_size, max_factors)
            
            # 调整第一个隐藏层的输入维度
            first_layer_input = max_factors
        else:
            first_layer_input = input_size
        
        # 构建MLP层
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(first_layer_input, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 添加其他隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], n_classes))
        
        # 创建顺序模型
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 确保输入形状正确
        if len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
        # 如果启用了因子选择
        if self.enable_factor_selection:
            # 应用特征掩码（通过sigmoid函数实现平滑门控）
            mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
            x = x * mask
            
            # 低秩投影
            x = self.low_rank_projection(x)
            
        # 通过MLP层
        return self.layers(x)
        
    def get_mask_sparsity(self):
        """获取特征掩码的稀疏度(选择的特征比例)"""
        if not self.enable_factor_selection:
            return 1.0
            
        mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
        active_features = (mask > 0.5).sum().item()
        return active_features / self.input_size
    
    def get_selected_features(self):
        """获取选中的特征索引"""
        if not self.enable_factor_selection:
            return np.arange(self.input_size)
            
        mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
        return torch.where(mask > 0.5)[0].cpu().numpy()
        
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.enable_factor_selection:
            # 如果未启用特征选择，返回均匀重要性
            return np.ones(self.input_size)
            
        # 返回特征掩码的值作为特征重要性
        mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
        return mask.detach().cpu().numpy()


class PyTorchTeacherCNN(PyTorchBaseWrapper):
    @log_function_call
    def build(self):
        """构建CNN教师模型"""
        try:
            # 从模型参数中提取CNN参数
            input_channels = self.model_params.get('input_channels', 1)
            img_size = self.model_params.get('img_size', 28)
            
            # 创建CNN模型
            self.model = CNN(
                input_channels=input_channels,
                img_size=img_size,
                n_classes=self.n_classes
            ).to(self.device)
            
            # 设置优化器和损失函数
            self.optimizer = self._get_default_optimizer()
            self.criterion = self._get_default_criterion()
            
            logger.info(f"已创建PyTorch CNN教师模型，通道数: {input_channels}, 图像大小: {img_size}, 类别数: {self.n_classes}")
            return self
        except Exception as e:
            logger.error(f"创建PyTorch CNN教师模型失败: {str(e)}")
            raise
    
    @log_function_call
    def train(self, X, y, **kwargs):
        """训练CNN教师模型"""
        if self.model is None:
            # 尝试从输入数据推断图像尺寸
            if len(X.shape) == 2:
                # 对于扁平化数据，假设是方形图像
                img_side = int(np.sqrt(X.shape[1]))
                if img_side * img_side != X.shape[1]:
                    logger.warning(f"输入数据维度 {X.shape[1]} 不是完全平方数，无法自动确定图像尺寸，使用默认值。")
                else:
                    self.model_params['img_size'] = img_side
                    logger.info(f"从扁平化数据推断图像尺寸为 {img_side}x{img_side}")
            
            self.build()
        
        # 提取训练参数
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        # 创建数据加载器
        train_loader = self._create_data_loader(X, y, batch_size)
        
        # 训练模式
        self.model.train()
        
        try:
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    # 清零梯度
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    loss = self.criterion(outputs, labels)
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    # 统计
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    if labels.dim() == 1:  # 如果是硬标签
                        correct += (predicted == labels).sum().item()
                    
                # 打印统计信息
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100 * correct / total if total > 0 else 0
                logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            
            logger.info("PyTorch CNN教师模型训练完成")
        except Exception as e:
            logger.error(f"训练PyTorch CNN教师模型失败: {str(e)}")
            raise
    
    @log_function_call
    def predict(self, X):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 转换为张量
                X_tensor = self._prepare_data(X)
                
                # 前向传播
                outputs = self.model(X_tensor)
                
                # 获取预测结果
                _, predicted = torch.max(outputs, 1)
                
                # 转换为NumPy数组
                return self._to_numpy(predicted)
        except Exception as e:
            logger.error(f"PyTorch CNN教师模型预测失败: {str(e)}")
            raise
    
    @log_function_call
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 转换为张量
                X_tensor = self._prepare_data(X)
                
                # 前向传播
                outputs = self.model(X_tensor)
                
                # 应用softmax获取概率
                probs = nn.functional.softmax(outputs / self.temperature, dim=1)
                
                # 转换为NumPy数组
                return self._to_numpy(probs)
        except Exception as e:
            logger.error(f"PyTorch CNN教师模型概率预测失败: {str(e)}")
            raise
    
    @log_function_call
    def get_feature_importance(self):
        """获取特征重要性（CNN模型不直接支持）"""
        logger.warning("PyTorch CNN模型不直接支持特征重要性计算")
        return None


class PyTorchTeacherMLP(PyTorchBaseWrapper):
    @log_function_call
    def build(self):
        """构建PyTorch MLP教师模型"""
        try:
            # 从模型参数中提取参数
            hidden_sizes = self.model_params.get('hidden_sizes', [256, 128])
            dropout_rate = self.model_params.get('dropout_rate', 0.2)
            
            # 创建MLP模型
            self.model = MLP(
                input_size=self.input_size,
                hidden_sizes=hidden_sizes,
                n_classes=self.n_classes,
                enable_factor_selection=False,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # 设置优化器和损失函数
            self.optimizer = self._get_default_optimizer()
            self.criterion = self._get_default_criterion()
            
            logger.info(f"已创建PyTorch MLP教师模型，"
                       f"输入维度: {self.input_size}, "
                       f"隐藏层: {hidden_sizes}, "
                       f"类别数: {self.n_classes}")
            return self
        except Exception as e:
            logger.error(f"创建PyTorch MLP教师模型失败: {str(e)}")
            raise
    
    @log_function_call
    def train(self, X, y, **kwargs):
        """训练MLP教师模型"""
        if self.model is None:
            # 根据输入数据推断输入尺寸
            self.input_size = self._infer_input_size(X)
            self.build()
        
        # 提取训练参数
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        
        # 创建数据加载器
        train_loader = self._create_data_loader(X, y, batch_size)
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # 训练模式
        self.model.train()
        
        try:
            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in train_loader:
                    # 清零梯度
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    if isinstance(self.criterion, nn.CrossEntropyLoss):
                        loss = self.criterion(outputs, labels, class_weights)
                    else:
                        loss = self.criterion(outputs, labels)
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    # 统计
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    if labels.dim() == 1:  # 如果是硬标签
                        correct += (predicted == labels).sum().item()
                    
                # 打印统计信息
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100 * correct / total if total > 0 else 0
                logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            
            logger.info("PyTorch MLP教师模型训练完成")
        except Exception as e:
            logger.error(f"训练PyTorch MLP教师模型失败: {str(e)}")
            raise
    
    @log_function_call
    def predict(self, X):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 转换为张量
                X_tensor = self._prepare_data(X)
                
                # 前向传播
                outputs = self.model(X_tensor)
                
                # 获取预测结果
                _, predicted = torch.max(outputs, 1)
                
                # 转换为NumPy数组
                return self._to_numpy(predicted)
        except Exception as e:
            logger.error(f"PyTorch MLP教师模型预测失败: {str(e)}")
            raise
    
    @log_function_call
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 转换为张量
                X_tensor = self._prepare_data(X)
                
                # 前向传播
                outputs = self.model(X_tensor)
                
                # 应用softmax获取概率
                probs = nn.functional.softmax(outputs / self.temperature, dim=1)
                
                # 转换为NumPy数组
                return self._to_numpy(probs)
        except Exception as e:
            logger.error(f"PyTorch MLP教师模型概率预测失败: {str(e)}")
            raise
    
    @log_function_call
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            # 使用特征掩码的值作为特征重要性
            feature_mask = torch.sigmoid(self.model.feature_mask / self.model.factor_temperature)
            return self._to_numpy(feature_mask)
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            raise


class PyTorchStudentCNN(PyTorchTeacherCNN):
    @log_function_call
    def build(self):
        """构建PyTorch CNN学生模型，支持因子选择"""
        try:
            # 从模型参数中提取参数
            # 确保img_size和input_channels已设置
            img_size = self.model_params.get('img_size', 28)
            input_channels = self.model_params.get('input_channels', 1)
            
            # 特征选择相关参数 (CNN主要用于图像，因子选择更适合结构化数据，但我们仍提供该功能)
            enable_factor_selection = self.model_params.get('enable_factor_selection', False)
            max_factor_ratio = self.model_params.get('max_factor_ratio', 0.2)
            factor_temperature = self.model_params.get('factor_temperature', 1.0)
            
            # 计算输入大小(用于特征选择)
            input_size = input_channels * img_size * img_size
            
            # 如果启用特征选择，我们需要创建特征选择层
            if enable_factor_selection:
                # 特征掩码 - 应用于展平的输入
                self.feature_mask = nn.Parameter(torch.randn(input_size) * 0.01).to(self.device)
                self.factor_temperature = factor_temperature
                self.max_factor_ratio = max_factor_ratio
                self.input_size = input_size
                
                # 低秩投影层(可选)
                max_factors = max(int(input_size * max_factor_ratio), 1)
                self.use_low_rank = self.model_params.get('use_low_rank', False)
                if self.use_low_rank:
                    self.low_rank_projection = nn.Linear(input_size, max_factors).to(self.device)
            
            # 创建CNN模型
            self.model = CNN(
                input_channels=input_channels,
                img_size=img_size,
                n_classes=self.n_classes
            ).to(self.device)
            
            # 设置优化器和损失函数
            self.optimizer = self._get_default_optimizer()
            self.criterion = self._get_default_criterion()
            
            if enable_factor_selection:
                logger.info(f"已创建支持因子选择的PyTorch CNN学生模型，"
                          f"输入维度: {input_size}, "
                          f"最大因子比例: {max_factor_ratio}, "
                          f"图像尺寸: {img_size}x{img_size}, "
                          f"类别数: {self.n_classes}")
            else:
                logger.info(f"已创建PyTorch CNN学生模型，"
                          f"输入通道: {input_channels}, "
                          f"图像尺寸: {img_size}x{img_size}, "
                          f"类别数: {self.n_classes}")
            return self
        except Exception as e:
            logger.error(f"创建PyTorch CNN学生模型失败: {str(e)}")
            raise
            
    @log_function_call
    def train(self, X, y, teacher_probs=None, **kwargs):
        """训练CNN学生模型，支持知识蒸馏
        
        Args:
            X: 训练数据
            y: 真实标签
            teacher_probs: 教师模型的软标签预测
            **kwargs: 其他参数，支持：
                - alpha: 软标签权重 (默认0.5)
                - temperature: 温度参数 (默认1.0)
                - l1_lambda: L1正则化系数 (默认0.001) - 用于因子选择
                - factor_temperature: 特征掩码温度参数 (默认1.0)
        """
        if self.model is None:
            # 尝试从输入数据推断图像尺寸
            if len(X.shape) == 2:
                # 对于扁平化数据，假设是方形图像
                img_side = int(np.sqrt(X.shape[1]))
                if img_side * img_side != X.shape[1]:
                    logger.warning(f"输入数据维度 {X.shape[1]} 不是完全平方数，无法自动确定图像尺寸，使用默认值。")
                else:
                    self.model_params['img_size'] = img_side
                    logger.info(f"从扁平化数据推断图像尺寸为 {img_side}x{img_side}")
            
            self.build()
        
        # 提取训练参数
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        alpha = kwargs.get('alpha', 0.5)  # 软标签权重
        temperature = kwargs.get('temperature', self.temperature)  # 温度参数
        
        # 因子选择相关参数
        l1_lambda = kwargs.get('l1_lambda', 0.001)  # L1正则化系数
        
        # 检查是否使用特征选择
        enable_factor_selection = hasattr(self, 'feature_mask')
        
        # 如果有设置factor_temperature，更新模型的factor_temperature
        if enable_factor_selection and 'factor_temperature' in kwargs:
            self.factor_temperature = kwargs['factor_temperature']
        
        # 训练模式
        self.model.train()
        
        try:
            # 如果有教师模型的软标签，使用知识蒸馏
            if teacher_probs is not None:
                # 确保teacher_probs是2D数组
                if teacher_probs.ndim == 1:
                    teacher_probs = np.column_stack((1.0 - teacher_probs, teacher_probs))
                
                # 创建软标签和硬标签的数据加载器
                X_tensor = self._prepare_data(X)
                y_tensor = torch.LongTensor(y).to(self.device)
                teacher_probs_tensor = torch.FloatTensor(teacher_probs).to(self.device)
                
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, teacher_probs_tensor)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # 交叉熵损失用于真实标签
                hard_criterion = nn.CrossEntropyLoss()
                # KL散度损失用于软标签
                soft_criterion = nn.KLDivLoss(reduction='batchmean')
                
                for epoch in range(epochs):
                    running_loss = 0.0
                    hard_losses = 0.0
                    soft_losses = 0.0
                    sparsity_losses = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels, teacher_probs_batch in train_loader:
                        # 清零梯度
                        self.optimizer.zero_grad()
                        
                        # 如果启用了因子选择，应用特征掩码
                        if enable_factor_selection:
                            # 展平输入
                            batch_size = inputs.size(0)
                            flat_inputs = inputs.view(batch_size, -1)
                            
                            # 应用特征掩码
                            mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                            masked_inputs = flat_inputs * mask
                            
                            # 是否使用低秩投影
                            if hasattr(self, 'use_low_rank') and self.use_low_rank:
                                masked_inputs = self.low_rank_projection(masked_inputs)
                                # 重新调整形状为图像
                                # 这里我们简单地将低秩投影的结果重新调整为接近原始图像尺寸的形状
                                # 实际应用中可能需要更复杂的重构方法
                                new_side = int(np.sqrt(masked_inputs.size(1)))
                                if new_side * new_side == masked_inputs.size(1):
                                    inputs = masked_inputs.view(batch_size, 1, new_side, new_side)
                                else:
                                    # 如果无法完美重构，使用填充或裁剪
                                    logger.warning("低秩投影后无法完美重构为方形图像，可能影响性能")
                                    inputs = masked_inputs.view(batch_size, 1, -1, 1)
                            else:
                                # 如果不使用低秩投影，直接将掩码后的平面输入重新调整为图像形状
                                inputs = masked_inputs.view_as(inputs)
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        
                        # 计算硬标签损失
                        hard_loss = hard_criterion(outputs, labels)
                        
                        # 计算软标签损失 (KL散度)
                        log_softmax_outputs = nn.functional.log_softmax(outputs / temperature, dim=1)
                        soft_target_probs = nn.functional.softmax(teacher_probs_batch / temperature, dim=1)
                        soft_loss = soft_criterion(log_softmax_outputs, soft_target_probs) * (temperature ** 2)
                        
                        # 如果启用了因子选择，添加L1正则化损失
                        sparsity_loss = 0
                        if enable_factor_selection:
                            mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                            sparsity_loss = l1_lambda * torch.norm(mask, p=1)
                        
                        # 组合损失
                        loss = (1 - alpha) * hard_loss + alpha * soft_loss + sparsity_loss
                        
                        # 反向传播和优化
                        loss.backward()
                        self.optimizer.step()
                        
                        # 统计
                        running_loss += loss.item()
                        hard_losses += hard_loss.item()
                        soft_losses += soft_loss.item()
                        sparsity_losses += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    # 打印统计信息
                    epoch_loss = running_loss / len(train_loader)
                    epoch_hard_loss = hard_losses / len(train_loader)
                    epoch_soft_loss = soft_losses / len(train_loader)
                    epoch_sparsity_loss = sparsity_losses / len(train_loader)
                    epoch_acc = 100 * correct / total
                    
                    # 如果启用了因子选择，显示选择的特征数量
                    if enable_factor_selection:
                        selected_features = self.get_selected_features()
                        feature_sparsity = self.get_mask_sparsity()
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Hard Loss: {epoch_hard_loss:.4f}, Soft Loss: {epoch_soft_loss:.4f}, '
                                  f'Sparsity Loss: {epoch_sparsity_loss:.4f}, Acc: {epoch_acc:.2f}%, '
                                  f'Selected Features: {len(selected_features)} ({feature_sparsity*100:.1f}%)')
                    else:
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Hard Loss: {epoch_hard_loss:.4f}, Soft Loss: {epoch_soft_loss:.4f}, '
                                  f'Acc: {epoch_acc:.2f}%')
            else:
                # 如果没有教师模型的软标签，使用普通训练方法，但仍支持因子选择
                train_loader = self._create_data_loader(X, y, batch_size)
                
                for epoch in range(epochs):
                    running_loss = 0.0
                    sparsity_losses = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels in train_loader:
                        # 清零梯度
                        self.optimizer.zero_grad()
                        
                        # 如果启用了因子选择，应用特征掩码
                        if enable_factor_selection:
                            # 展平输入
                            batch_size = inputs.size(0)
                            flat_inputs = inputs.view(batch_size, -1)
                            
                            # 应用特征掩码
                            mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                            masked_inputs = flat_inputs * mask
                            
                            # 是否使用低秩投影
                            if hasattr(self, 'use_low_rank') and self.use_low_rank:
                                masked_inputs = self.low_rank_projection(masked_inputs)
                                # 重新调整形状为图像
                                new_side = int(np.sqrt(masked_inputs.size(1)))
                                if new_side * new_side == masked_inputs.size(1):
                                    inputs = masked_inputs.view(batch_size, 1, new_side, new_side)
                                else:
                                    logger.warning("低秩投影后无法完美重构为方形图像，可能影响性能")
                                    inputs = masked_inputs.view(batch_size, 1, -1, 1)
                            else:
                                # 如果不使用低秩投影，直接将掩码后的平面输入重新调整为图像形状
                                inputs = masked_inputs.view_as(inputs)
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        
                        # 计算硬标签损失
                        hard_loss = self.criterion(outputs, labels)
                        
                        # 如果启用了因子选择，添加L1正则化损失
                        sparsity_loss = 0
                        if enable_factor_selection:
                            mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                            sparsity_loss = l1_lambda * torch.norm(mask, p=1)
                        
                        # 总损失
                        loss = hard_loss + sparsity_loss
                        
                        # 反向传播和优化
                        loss.backward()
                        self.optimizer.step()
                        
                        # 统计
                        running_loss += loss.item()
                        sparsity_losses += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    # 打印统计信息
                    epoch_loss = running_loss / len(train_loader)
                    epoch_acc = 100 * correct / total
                    
                    # 如果启用了因子选择，显示选择的特征数量
                    if enable_factor_selection:
                        selected_features = self.get_selected_features()
                        feature_sparsity = self.get_mask_sparsity()
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Sparsity Loss: {sparsity_losses/len(train_loader):.4f}, '
                                  f'Acc: {epoch_acc:.2f}%, Selected Features: {len(selected_features)} '
                                  f'({feature_sparsity*100:.1f}%)')
                    else:
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Acc: {epoch_acc:.2f}%')
                
            if enable_factor_selection:
                # 训练完成后，输出选择的特征信息
                selected_features = self.get_selected_features()
                logger.info(f"模型最终选择的特征: {len(selected_features)}个 "
                          f"(总特征的{self.get_mask_sparsity()*100:.1f}%), "
                          f"特征索引: {selected_features}")
                
            logger.info("PyTorch CNN学生模型训练完成")
        except Exception as e:
            logger.error(f"训练PyTorch CNN学生模型失败: {str(e)}")
            raise
            
    @log_function_call
    def predict(self, X):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            # 检查是否使用特征选择
            enable_factor_selection = hasattr(self, 'feature_mask')
            
            with torch.no_grad():
                # 准备数据
                X_tensor = self._prepare_data(X)
                
                # 如果启用了因子选择，应用特征掩码
                if enable_factor_selection:
                    # 展平输入
                    batch_size = X_tensor.size(0)
                    flat_inputs = X_tensor.view(batch_size, -1)
                    
                    # 应用特征掩码
                    mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                    masked_inputs = flat_inputs * mask
                    
                    # 是否使用低秩投影
                    if hasattr(self, 'use_low_rank') and self.use_low_rank:
                        masked_inputs = self.low_rank_projection(masked_inputs)
                        # 重新调整形状为图像
                        new_side = int(np.sqrt(masked_inputs.size(1)))
                        if new_side * new_side == masked_inputs.size(1):
                            X_tensor = masked_inputs.view(batch_size, 1, new_side, new_side)
                        else:
                            logger.warning("低秩投影后无法完美重构为方形图像，可能影响性能")
                            X_tensor = masked_inputs.view(batch_size, 1, -1, 1)
                    else:
                        # 如果不使用低秩投影，直接将掩码后的平面输入重新调整为图像形状
                        X_tensor = masked_inputs.view_as(X_tensor)
                
                # 前向传播
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                
                return self._to_numpy(predicted)
        except Exception as e:
            logger.error(f"CNN模型预测失败: {str(e)}")
            raise
            
    @log_function_call
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            # 检查是否使用特征选择
            enable_factor_selection = hasattr(self, 'feature_mask')
            
            with torch.no_grad():
                # 准备数据
                X_tensor = self._prepare_data(X)
                
                # 如果启用了因子选择，应用特征掩码
                if enable_factor_selection:
                    # 展平输入
                    batch_size = X_tensor.size(0)
                    flat_inputs = X_tensor.view(batch_size, -1)
                    
                    # 应用特征掩码
                    mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                    masked_inputs = flat_inputs * mask
                    
                    # 是否使用低秩投影
                    if hasattr(self, 'use_low_rank') and self.use_low_rank:
                        masked_inputs = self.low_rank_projection(masked_inputs)
                        # 重新调整形状为图像
                        new_side = int(np.sqrt(masked_inputs.size(1)))
                        if new_side * new_side == masked_inputs.size(1):
                            X_tensor = masked_inputs.view(batch_size, 1, new_side, new_side)
                        else:
                            logger.warning("低秩投影后无法完美重构为方形图像，可能影响性能")
                            X_tensor = masked_inputs.view(batch_size, 1, -1, 1)
                    else:
                        # 如果不使用低秩投影，直接将掩码后的平面输入重新调整为图像形状
                        X_tensor = masked_inputs.view_as(X_tensor)
                
                # 前向传播
                outputs = self.model(X_tensor)
                probas = nn.functional.softmax(outputs / self.temperature, dim=1)
                
                return self._to_numpy(probas)
        except Exception as e:
            logger.error(f"CNN模型概率预测失败: {str(e)}")
            raise
            
    @log_function_call
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            if hasattr(self, 'feature_mask'):
                # 使用特征掩码的值作为特征重要性
                feature_mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
                return self._to_numpy(feature_mask)
            else:
                return super().get_feature_importance()
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            raise
            
    @log_function_call
    def get_selected_features(self):
        """获取选中的特征索引"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        if hasattr(self, 'feature_mask'):
            mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
            return torch.where(mask > 0.5)[0].cpu().numpy()
        else:
            # 如果没有启用特征选择，返回所有特征
            if hasattr(self, 'input_size'):
                return np.arange(self.input_size)
            else:
                # 推断输入大小
                img_size = self.model_params.get('img_size', 28)
                input_channels = self.model_params.get('input_channels', 1)
                input_size = input_channels * img_size * img_size
                return np.arange(input_size)
                
    @log_function_call
    def get_mask_sparsity(self):
        """获取特征掩码的稀疏度(选择的特征比例)"""
        if not hasattr(self, 'feature_mask'):
            return 1.0
            
        mask = torch.sigmoid(self.feature_mask / self.factor_temperature)
        active_features = (mask > 0.5).sum().item()
        input_size = self.feature_mask.size(0)
        return active_features / input_size


class PyTorchStudentMLP(PyTorchTeacherMLP):
    @log_function_call
    def build(self):
        """构建PyTorch MLP学生模型，支持因子选择"""
        try:
            # 从模型参数中提取参数
            hidden_sizes = self.model_params.get('hidden_sizes', [128, 64])
            dropout_rate = self.model_params.get('dropout_rate', 0.2)
            
            # 特征选择相关参数
            enable_factor_selection = self.model_params.get('enable_factor_selection', True)
            max_factor_ratio = self.model_params.get('max_factor_ratio', 0.2)
            factor_temperature = self.model_params.get('factor_temperature', 1.0)
            
            # 创建MLP模型
            self.model = MLP(
                input_size=self.input_size,
                hidden_sizes=hidden_sizes,
                n_classes=self.n_classes,
                enable_factor_selection=enable_factor_selection,
                max_factor_ratio=max_factor_ratio,
                factor_temperature=factor_temperature,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # 设置优化器和损失函数
            self.optimizer = self._get_default_optimizer()
            self.criterion = self._get_default_criterion()
            
            if enable_factor_selection:
                logger.info(f"已创建支持因子选择的PyTorch MLP学生模型，"
                          f"输入维度: {self.input_size}, "
                          f"最大因子比例: {max_factor_ratio}, "
                          f"隐藏层: {hidden_sizes}, "
                          f"类别数: {self.n_classes}")
            else:
                logger.info(f"已创建PyTorch MLP学生模型(未启用因子选择)，"
                          f"输入维度: {self.input_size}, "
                          f"隐藏层: {hidden_sizes}, "
                          f"类别数: {self.n_classes}")
            return self
        except Exception as e:
            logger.error(f"创建PyTorch MLP学生模型失败: {str(e)}")
            raise
            
    @log_function_call
    def train(self, X, y, teacher_probs=None, **kwargs):
        """训练MLP学生模型，支持知识蒸馏
        
        Args:
            X: 训练数据
            y: 真实标签
            teacher_probs: 教师模型的软标签预测
            **kwargs: 其他参数，支持：
                - alpha: 软标签权重 (默认0.5)
                - temperature: 温度参数 (默认1.0)
                - l1_lambda: L1正则化系数 (默认0.001) - 用于因子选择
                - factor_temperature: 特征掩码温度参数 (默认1.0)
        """
        if self.model is None:
            # 根据输入数据推断输入尺寸
            self.input_size = self._infer_input_size(X)
            self.build()
        
        # 提取训练参数
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        alpha = kwargs.get('alpha', 0.5)  # 软标签权重
        temperature = kwargs.get('temperature', self.temperature)  # 温度参数
        
        # 因子选择相关参数
        l1_lambda = kwargs.get('l1_lambda', 0.001)  # L1正则化系数
        
        # 如果有设置factor_temperature，更新模型的factor_temperature
        if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
            if 'factor_temperature' in kwargs:
                self.model.factor_temperature = kwargs['factor_temperature']
        
        # 训练模式
        self.model.train()
        
        try:
            # 如果有教师模型的软标签，使用知识蒸馏
            if teacher_probs is not None:
                # 确保teacher_probs是2D数组
                if teacher_probs.ndim == 1:
                    teacher_probs = np.column_stack((1.0 - teacher_probs, teacher_probs))
                
                # 创建软标签和硬标签的数据加载器
                X_tensor = self._prepare_data(X)
                y_tensor = torch.LongTensor(y).to(self.device)
                teacher_probs_tensor = torch.FloatTensor(teacher_probs).to(self.device)
                
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, teacher_probs_tensor)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # 交叉熵损失用于真实标签
                hard_criterion = nn.CrossEntropyLoss()
                # KL散度损失用于软标签
                soft_criterion = nn.KLDivLoss(reduction='batchmean')
                
                for epoch in range(epochs):
                    running_loss = 0.0
                    hard_losses = 0.0
                    soft_losses = 0.0
                    sparsity_losses = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels, teacher_probs_batch in train_loader:
                        # 清零梯度
                        self.optimizer.zero_grad()
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        
                        # 计算硬标签损失
                        hard_loss = hard_criterion(outputs, labels)
                        
                        # 计算软标签损失 (KL散度)
                        log_softmax_outputs = nn.functional.log_softmax(outputs / temperature, dim=1)
                        soft_target_probs = nn.functional.softmax(teacher_probs_batch / temperature, dim=1)
                        soft_loss = soft_criterion(log_softmax_outputs, soft_target_probs) * (temperature ** 2)
                        
                        # 如果启用了因子选择，添加L1正则化损失
                        sparsity_loss = 0
                        if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
                            feature_mask = torch.sigmoid(self.model.feature_mask / self.model.factor_temperature)
                            sparsity_loss = l1_lambda * torch.norm(feature_mask, p=1)
                        
                        # 组合损失
                        loss = (1 - alpha) * hard_loss + alpha * soft_loss + sparsity_loss
                        
                        # 反向传播和优化
                        loss.backward()
                        self.optimizer.step()
                        
                        # 统计
                        running_loss += loss.item()
                        hard_losses += hard_loss.item()
                        soft_losses += soft_loss.item()
                        sparsity_losses += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    # 打印统计信息
                    epoch_loss = running_loss / len(train_loader)
                    epoch_hard_loss = hard_losses / len(train_loader)
                    epoch_soft_loss = soft_losses / len(train_loader)
                    epoch_sparsity_loss = sparsity_losses / len(train_loader)
                    epoch_acc = 100 * correct / total
                    
                    # 如果启用了因子选择，显示选择的特征数量
                    if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
                        selected_features = self.model.get_selected_features()
                        feature_sparsity = self.model.get_mask_sparsity()
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Hard Loss: {epoch_hard_loss:.4f}, Soft Loss: {epoch_soft_loss:.4f}, '
                                  f'Sparsity Loss: {epoch_sparsity_loss:.4f}, Acc: {epoch_acc:.2f}%, '
                                  f'Selected Features: {len(selected_features)} ({feature_sparsity*100:.1f}%)')
                    else:
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Hard Loss: {epoch_hard_loss:.4f}, Soft Loss: {epoch_soft_loss:.4f}, '
                                  f'Acc: {epoch_acc:.2f}%')
            else:
                # 如果没有教师模型的软标签，使用普通训练方法
                train_loader = self._create_data_loader(X, y, batch_size)
                
                for epoch in range(epochs):
                    running_loss = 0.0
                    sparsity_losses = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels in train_loader:
                        # 清零梯度
                        self.optimizer.zero_grad()
                        
                        # 前向传播
                        outputs = self.model(inputs)
                        
                        # 计算硬标签损失
                        hard_loss = self.criterion(outputs, labels)
                        
                        # 如果启用了因子选择，添加L1正则化损失
                        sparsity_loss = 0
                        if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
                            feature_mask = torch.sigmoid(self.model.feature_mask / self.model.factor_temperature)
                            sparsity_loss = l1_lambda * torch.norm(feature_mask, p=1)
                        
                        # 总损失
                        loss = hard_loss + sparsity_loss
                        
                        # 反向传播和优化
                        loss.backward()
                        self.optimizer.step()
                        
                        # 统计
                        running_loss += loss.item()
                        sparsity_losses += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    # 打印统计信息
                    epoch_loss = running_loss / len(train_loader)
                    epoch_acc = 100 * correct / total
                    
                    # 如果启用了因子选择，显示选择的特征数量
                    if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
                        selected_features = self.model.get_selected_features()
                        feature_sparsity = self.model.get_mask_sparsity()
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Sparsity Loss: {sparsity_losses/len(train_loader):.4f}, '
                                  f'Acc: {epoch_acc:.2f}%, Selected Features: {len(selected_features)} '
                                  f'({feature_sparsity*100:.1f}%)')
                    else:
                        logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                                  f'Acc: {epoch_acc:.2f}%')
                
            if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
                # 训练完成后，输出选择的特征信息
                selected_features = self.model.get_selected_features()
                logger.info(f"模型最终选择的特征: {len(selected_features)}个 "
                          f"(总特征的{self.model.get_mask_sparsity()*100:.1f}%), "
                          f"特征索引: {selected_features}")
                
            logger.info("PyTorch MLP学生模型训练完成")
        except Exception as e:
            logger.error(f"训练PyTorch MLP学生模型失败: {str(e)}")
            raise
            
    @log_function_call
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
                # 使用特征掩码的值作为特征重要性
                feature_mask = torch.sigmoid(self.model.feature_mask / self.model.factor_temperature)
                return self._to_numpy(feature_mask)
            else:
                return super().get_feature_importance()
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            raise
            
    @log_function_call
    def get_selected_features(self):
        """获取选中的特征索引"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        if hasattr(self.model, 'enable_factor_selection') and self.model.enable_factor_selection:
            return self.model.get_selected_features()
        else:
            # 如果没有启用特征选择，返回所有特征
            return np.arange(self.input_size)


class PyTorchFactorStudentMLP(PyTorchStudentMLP):
    """基于因子选择的学生MLP模型
    
    这是为了向后兼容而保留的类，现在所有学生模型都支持因子选择
    """
    
    @log_function_call
    def build(self):
        """构建支持因子选择的MLP学生模型"""
        try:
            # 强制启用因子选择
            self.model_params['enable_factor_selection'] = True
            
            # 调用父类的build方法
            return super().build()
        except Exception as e:
            logger.error(f"创建支持因子选择的PyTorch MLP学生模型失败: {str(e)}")
            raise 


# 因子化MLP模型，支持多重损失优化
class FactorizedMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], n_classes=10, 
                 max_factors=None, factor_temperature=1.0, dropout_rate=0.2):
        super(FactorizedMLP, self).__init__()
        
        self.input_size = input_size
        self.n_classes = n_classes
        self.factor_temperature = factor_temperature
        
        # 确定最大因子数量，默认为输入维度的40%
        if max_factors is None:
            max_factors = max(int(input_size * 0.4), 1)
        self.max_factors = max_factors
        
        # 构建特征提取MLP层 (input_size -> hidden_sizes)
        feature_layers = [] 
        
        # 输入层到第一个隐藏层
        feature_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        feature_layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.Dropout(dropout_rate))
        
        # 添加其他隐藏层
        for i in range(len(hidden_sizes) - 1):
            feature_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            feature_layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout_rate))
        
        # 创建特征提取网络
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 因子映射层（hidden_sizes[-1] -> max_factors）- 从最后一个隐藏层映射到因子空间
        self.factor_weights = nn.Parameter(torch.randn(hidden_sizes[-1], max_factors) * 0.01)
        self.factor_gate = nn.Parameter(torch.zeros(max_factors))  # 因子门控参数
        
        # 分类器网络 (max_factors -> n_classes)
        self.classifier = nn.Linear(max_factors, n_classes)
        
        # 用于保存类中心的字典（用于计算类间分离损失）
        self.class_centers = {}
        
        # 对于每个可能的类别预初始化一个零中心
        # 这样可以避免在开始训练时出现键错误
        # 注意：在训练过程中这些中心会被更新为实际的类别中心
        self.register_buffer('initialized', torch.tensor(False))
        
        logger.info(f"创建了因子化MLP模型：输入维度{input_size}，"
                  f"隐藏层维度{hidden_sizes}，最大因子数{max_factors}，"
                  f"输出类别数{n_classes}")
    
    def forward(self, x):
        # 确保输入形状正确
        if len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        # 通过特征提取网络
        features = self.feature_extractor(x)
        
        # 计算因子表示
        factors = torch.matmul(features, self.factor_weights)
        
        # 门控机制：控制哪些因子被激活
        gate_values = torch.sigmoid(self.factor_gate / self.factor_temperature)
        self.selected_factors = factors * gate_values
        
        # 分类任务输出
        logits = self.classifier(self.selected_factors)
        
        # 如果想获取因子表示，可以通过返回元组实现
        # 注意：通过这种方式，可以在调用model(x)时同时获取logits和factors
        # 但是某些需要单一返回值的PyTorch函数可能会有问题
        return logits, self.selected_factors
    
    def compute_losses(self, logits, factors, labels, teacher_probs=None, temperature=1.0, alpha=0.5, l1_lambda=0.001, triplet_margin=1.0, class_weights=None, current_epoch=0, warmup_epochs=10):
        """
        计算多重损失：任务损失、因子稀疏性损失和类间分离损失
        
        Args:
            logits: 模型输出的logits
            factors: 因子表示
            labels: 真实标签
            teacher_probs: 教师模型的软标签预测
            temperature: 知识蒸馏的温度参数
            alpha: 软标签权重
            l1_lambda: L1正则化系数基准值
            triplet_margin: 三元组损失的边界值
            class_weights: 类别权重，用于处理不平衡数据
            current_epoch: 当前训练轮次，用于动态调整正则化强度
            warmup_epochs: 预热轮次，在此之前逐步增加正则化强度
            
        Returns:
            task_loss: 任务损失（交叉熵损失或KL散度）
            sparse_loss: 因子稀疏性损失（L1正则化）
            sep_loss: 类间分离损失（三元组损失）
        """
        # 计算任务损失（标签蒸馏监督）
        if teacher_probs is not None:
            # 使用KL散度计算与教师模型预测的差异
            log_softmax_outputs = F.log_softmax(logits / temperature, dim=1)
            soft_target_probs = F.softmax(teacher_probs / temperature, dim=1)
            
            # 针对不平衡数据的优化：使用类别权重调整KL散度损失
            if class_weights is not None and labels is not None:
                # 获取每个样本的类别权重
                sample_weights = class_weights[labels]  # shape: (batch_size,)
                # 标准化到平均为1（防止梯度爆炸）
                sample_weights = sample_weights / sample_weights.mean()
                
                # 计算逐样本KL散度（per-sample KL）
                per_sample_kl = F.kl_div(
                    log_softmax_outputs, 
                    soft_target_probs, 
                    reduction='none'
                ).sum(dim=1)  # shape: (batch_size,)
                
                # 使用样本权重加权平均
                task_loss = (per_sample_kl * sample_weights).mean() * (temperature ** 2)
            else:
                # 如果没有类别权重，使用标准KL散度
                task_loss = F.kl_div(log_softmax_outputs, soft_target_probs, reduction='batchmean') * (temperature ** 2)
            
            # 如果有硬标签，结合交叉熵损失
            if labels is not None:
                hard_loss = F.cross_entropy(logits, labels, weight=class_weights)
                task_loss = (1 - alpha) * hard_loss + alpha * task_loss
        else:
            # 如果没有教师模型预测，使用交叉熵损失
            task_loss = F.cross_entropy(logits, labels, weight=class_weights)
        
        # 动态调整稀疏正则权重 - 前几轮训练降低 l1_lambda，避免因子过早被压缩
        dynamic_l1_lambda = l1_lambda * min(1.0, current_epoch / warmup_epochs)
        
        # 计算因子稀疏性损失
        gate_values = torch.sigmoid(self.factor_gate / self.factor_temperature)
        sparse_loss = dynamic_l1_lambda * torch.norm(gate_values, p=1)
        
        # 计算类间分离损失（三元组损失）- 如果类别数量足够则计算
        unique_labels = torch.unique(labels)
        if len(unique_labels) >= 2:
            # 分离计算三元组损失，避免计算图共享导致的多次反向传播问题
            with torch.no_grad():
                # 先用分离的计算图计算三元组损失
                sep_loss_detached = self.triplet_loss(factors.detach(), labels, margin=triplet_margin, class_weights=class_weights)
                
            if sep_loss_detached > 0:
                # 如果分离后的损失值大于0，才在原始计算图上计算（减少不必要的计算）
                sep_loss = self.triplet_loss(factors, labels, margin=triplet_margin, class_weights=class_weights)
            else:
                # 否则使用零张量
                sep_loss = torch.tensor(0.0, device=factors.device)
        else:
            # 如果类别不足，设为0
            sep_loss = torch.tensor(0.0, device=factors.device)
        
        return task_loss, sparse_loss, sep_loss
    
    def triplet_loss(self, factors, labels, margin=1.0, class_weights=None):
        """
        计算三元组损失，用于优化类间分离
        
        Args:
            factors: 因子表示
            labels: 真实标签
            margin: 三元组损失的边界值
            class_weights: 类别权重，用于处理不平衡数据
            
        Returns:
            triplet_loss: 三元组损失
        """
        try:
            batch_size = factors.size(0)
            if batch_size <= 1:
                return torch.tensor(0.0, device=factors.device)
            
            # 获取唯一的类别
            unique_labels = torch.unique(labels)
            if len(unique_labels) <= 1:
                return torch.tensor(0.0, device=factors.device)
            
            triplet_loss = torch.tensor(0.0, device=factors.device)
            valid_triplets = 0
            
            try:
                # 将标签转换为CPU上的Python整数
                unique_labels_cpu = [label.item() for label in unique_labels]
            except:
                # 如果标签转换失败，返回0损失
                logger.warning("三元组损失计算中标签转换失败，跳过分离损失计算")
                return torch.tensor(0.0, device=factors.device)
            
            # 更新类中心 - 使用EMA并限制中心大小
            for label in unique_labels_cpu:
                try:
                    # 创建与原始标签类型匹配的张量以便查找
                    label_tensor = torch.tensor(label, device=labels.device, dtype=labels.dtype)
                    label_factors = factors[labels == label_tensor]
                    if len(label_factors) == 0:
                        continue
                        
                    # 计算当前批次的类中心
                    current_center = label_factors.mean(dim=0)
                    
                    if label in self.class_centers:
                        # 根据类别样本数量动态调整更新率
                        # 少数类使用较小的更新率以保持稳定性
                        update_rate = 0.1
                        if class_weights is not None:
                            # 如果类别权重大，说明是少数类，降低更新率
                            class_weight = class_weights[label_tensor].item()
                            if class_weight > 1.0:  # 少数类
                                update_rate = max(0.01, 0.1 / class_weight)  # 减缓更新速度但不小于0.01
                        
                        # 使用指数滑动平均(EMA)更新类中心
                        self.class_centers[label] = (1 - update_rate) * self.class_centers[label] + update_rate * current_center
                        
                        # 归一化类中心向量以防止大小差异
                        center_norm = torch.norm(self.class_centers[label])
                        if center_norm > 0:
                            self.class_centers[label] = self.class_centers[label] / center_norm
                    else:
                        # 初始化类中心
                        self.class_centers[label] = current_center
                        
                        # 归一化类中心向量
                        center_norm = torch.norm(self.class_centers[label])
                        if center_norm > 0:
                            self.class_centers[label] = self.class_centers[label] / center_norm
                except Exception as e:
                    logger.warning(f"更新类中心时出错: {str(e)}")
                    continue
            
            # 类均衡采样和自适应margin的三元组损失计算
            label_indices = {}
            # 将样本索引按类别分组
            for i in range(batch_size):
                try:
                    label = labels[i].item()
                    if label not in label_indices:
                        label_indices[label] = []
                    label_indices[label].append(i)
                except:
                    continue
            
            # 确保至少有两个不同的类别
            if len(label_indices) < 2:
                return torch.tensor(0.0, device=factors.device)
            
            # 类均衡采样 - 从每个类中均匀抽取样本
            processed_samples = 0
            max_samples_per_class = 5  # 每个类别最多使用的样本数
            
            for anchor_label in label_indices:
                # 对于当前类，选择一些样本作为anchor
                anchor_indices = label_indices[anchor_label]
                n_anchors = min(len(anchor_indices), max_samples_per_class)
                
                # 随机选择一部分样本
                selected_anchor_indices = anchor_indices
                if len(anchor_indices) > max_samples_per_class:
                    selected_anchor_indices = random.sample(anchor_indices, n_anchors)
                
                for anchor_idx in selected_anchor_indices:
                    anchor = factors[anchor_idx]
                    
                    # 获取自适应margin - 少数类使用更大的margin
                    adaptive_margin = margin
                    if class_weights is not None:
                        try:
                            class_weight = class_weights[labels[anchor_idx]].item()
                            # 增大少数类的margin
                            adaptive_margin = margin * min(2.0, max(1.0, class_weight))
                        except:
                            pass
                    
                    # 确保anchor_label在self.class_centers中
                    if anchor_label not in self.class_centers:
                        continue
                        
                    # 正样本距离：与相同类别中心的距离
                    pos_dist = F.pairwise_distance(anchor.unsqueeze(0), self.class_centers[anchor_label].unsqueeze(0)).squeeze()
                    
                    # 负样本距离：与不同类别中心的最小距离，但注意类均衡选择负样本
                    neg_dists = []
                    
                    # 类均衡负样本选择
                    other_labels = [l for l in unique_labels_cpu if l != anchor_label and l in self.class_centers]
                    if not other_labels:
                        continue
                    
                    # 从每个类别中选择负样本
                    for neg_label in other_labels:
                        neg_dist = F.pairwise_distance(anchor.unsqueeze(0), self.class_centers[neg_label].unsqueeze(0)).squeeze()
                        neg_dists.append(neg_dist)
                    
                    if neg_dists:  # 确保有负样本
                        neg_dist = torch.min(torch.stack(neg_dists))
                        # 使用自适应margin计算损失
                        loss = F.relu(pos_dist - neg_dist + adaptive_margin)
                        # 累加标量损失
                        triplet_loss = triplet_loss + loss
                        valid_triplets += 1
                    
                    processed_samples += 1
                    # 限制处理的样本总数以提高效率
                    if processed_samples >= 50:  # 每批次最多处理50个样本
                        break
                
                # 如果已经处理了足够多的样本，跳出循环
                if processed_samples >= 50:
                    break
            
            if valid_triplets > 0:
                triplet_loss = triplet_loss / valid_triplets
                
            return triplet_loss
            
        except Exception as e:
            logger.error(f"三元组损失计算失败: {str(e)}")
            return torch.tensor(0.0, device=factors.device)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        # 计算因子重要性并导出特征重要性
        try:
            with torch.no_grad():
                # 获取因子门控值
                gate_values = torch.sigmoid(self.factor_gate / self.factor_temperature).detach().cpu().numpy()
                
                # 提取第一层权重 (input_size -> hidden_sizes[0])
                first_layer_weights = None
                for module in self.feature_extractor:
                    if isinstance(module, nn.Linear):
                        first_layer_weights = module.weight.detach().cpu().numpy()
                        break
                
                if first_layer_weights is None:
                    # 如果无法获取第一层权重，返回均匀重要性
                    return np.ones(self.input_size)
                
                # 获取最后一层到因子层的权重
                factor_weights = self.factor_weights.detach().cpu().numpy()
                
                # 计算每个特征的重要性
                feature_importance = np.zeros(self.input_size)
                
                # 对每个因子计算重要性
                for i, gate_value in enumerate(gate_values):
                    if gate_value > 0.01:  # 只考虑有意义的因子
                        # 从输入到该因子的全路径贡献
                        factor_contrib = np.abs(factor_weights[:, i])
                        # 传播到输入特征
                        for j, contrib in enumerate(factor_contrib):
                            feature_importance += contrib * np.abs(first_layer_weights[j, :]) * gate_value
                
                # 归一化
                if feature_importance.max() > 0:
                    feature_importance = feature_importance / feature_importance.max()
                
                return feature_importance
        except Exception as e:
            # 如果出错，返回均匀重要性
            logger.error(f"计算特征重要性时出错: {str(e)}")
            return np.ones(self.input_size)
    
    def get_factor_importance(self):
        gate_values = torch.sigmoid(self.factor_gate / self.factor_temperature)
        return gate_values.detach().cpu().numpy()
            
    def get_mask_sparsity(self):
        """获取因子掩码的稀疏度(选择的因子比例)"""
        gate_values = torch.sigmoid(self.factor_gate / self.factor_temperature)
        active_factors = (gate_values > 0.5).sum().item()
        return active_factors / self.max_factors
    
    def get_selected_factors(self):
        """获取选中的因子索引"""
        gate_values = torch.sigmoid(self.factor_gate / self.factor_temperature)
        return torch.where(gate_values > 0.5)[0].cpu().numpy()
    
    def get_selected_features(self):
        """获取选中的特征索引（通过重要性计算）"""
        # 获取特征重要性
        feature_importance = self.get_feature_importance()
        
        # 选择重要性高于中位数的特征
        median_importance = np.median(feature_importance)
        selected_features = np.where(feature_importance > median_importance)[0]
        
        # 如果没有选中任何特征，则选择前20%的特征
        if len(selected_features) == 0:
            top_k = max(int(self.input_size * 0.2), 1)
            selected_features = np.argsort(feature_importance)[-top_k:]
            
        return selected_features
        
    def compute_factor_feature_shap(self, X_background, X_explain=None, threshold=0.05):
        """
        使用SHAP计算每个因子与原始特征之间的关联关系
        
        Args:
            X_background: 用于建立解释器的背景数据
            X_explain: 需要解释的数据点（可选，默认使用背景数据）
            threshold: 特征影响的阈值，只保留影响大于此阈值的特征
            
        Returns:
            factor_feature_mapping: 每个因子依赖的特征映射字典
            {factor_idx: [(feature_idx, shap_value), ...], ...}
        """
        try:
            import shap
            import numpy as np
            
            # 确保模型处于评估模式
            self.eval()
            
            # 如果没有指定要解释的数据，使用背景数据
            if X_explain is None:
                X_explain = X_background
                
            # 将数据转为NumPy数组
            if isinstance(X_background, torch.Tensor):
                X_background_np = X_background.detach().cpu().numpy()
            else:
                X_background_np = np.array(X_background)
                
            if isinstance(X_explain, torch.Tensor):
                X_explain_np = X_explain.detach().cpu().numpy()
            else:
                X_explain_np = np.array(X_explain)
            
            # 获取选中的因子索引
            selected_factors = self.get_selected_factors()
            if len(selected_factors) == 0:
                logger.warning("没有激活的因子，无法计算SHAP值")
                return {}
                
            # 创建结果字典
            factor_feature_mapping = {}
            
            # 对每个选中的因子计算SHAP值
            for factor_idx in selected_factors:
                # 创建一个函数，它返回模型对特定因子的输出
                def factor_predictor(x):
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x).to(next(self.parameters()).device)
                        features = self.feature_extractor(x_tensor)
                        # 计算每个输入对该因子的贡献
                        factor_values = torch.matmul(features, self.factor_weights[:, factor_idx])
                        # 应用门控值
                        gate_value = torch.sigmoid(self.factor_gate[factor_idx] / self.factor_temperature)
                        factor_output = factor_values * gate_value
                        return factor_output.cpu().numpy()
                
                # 创建SHAP解释器
                explainer = shap.KernelExplainer(factor_predictor, X_background_np)
                
                # 计算SHAP值
                shap_values = explainer.shap_values(X_explain_np)
                
                # 计算每个特征的平均绝对SHAP值
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                
                # 如果均值太小，则均值都为0，则重新标准化
                if np.max(mean_abs_shap) > 0:
                    # 标准化SHAP值
                    normalized_shap = mean_abs_shap / np.max(mean_abs_shap)
                else:
                    normalized_shap = mean_abs_shap
                
                # 获取重要特征索引（SHAP值大于阈值的特征）
                important_features = [(i, normalized_shap[i]) for i in range(len(normalized_shap)) 
                                      if normalized_shap[i] > threshold]
                
                # 按照SHAP值排序
                important_features.sort(key=lambda x: x[1], reverse=True)
                
                # 将结果添加到映射字典
                factor_feature_mapping[int(factor_idx)] = important_features
                
            return factor_feature_mapping
            
        except ImportError:
            logger.error("计算SHAP值需要安装shap库")
            return {}
        except Exception as e:
            logger.error(f"计算因子-特征SHAP关联失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}


class PyTorchFactorizedStudentMLP(PyTorchBaseWrapper):
    """
    基于因子化和多重损失的学生MLP模型包装器
    
    支持：
    1. 标签蒸馏（任务损失）
    2. 因子稀疏性（稀疏性损失）
    3. 类间分离（分离损失）
    """
    
    @log_function_call
    def build(self):
        """构建支持多重损失的因子化MLP学生模型"""
        try:
            # 从模型参数中提取参数
            hidden_sizes = self.model_params.get('hidden_sizes', [128, 64])
            dropout_rate = self.model_params.get('dropout_rate', 0.2)
            
            # 因子化相关参数
            max_factors = self.model_params.get('max_factors', None)
            factor_temperature = self.model_params.get('factor_temperature', 1.0)
            
            # 创建因子化MLP模型
            self.model = FactorizedMLP(
                input_size=self.input_size,
                hidden_sizes=hidden_sizes,
                n_classes=self.n_classes,
                max_factors=max_factors,
                factor_temperature=factor_temperature,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # 设置优化器和损失函数
            self.optimizer = self._get_default_optimizer()
            self.criterion = self._get_default_criterion()
            
            if max_factors:
                factor_ratio = max_factors / self.input_size
            else:
                factor_ratio = 0.2
                
            logger.info(f"已创建支持多重损失的因子化MLP学生模型，"
                      f"输入维度: {self.input_size}, "
                      f"最大因子数: {max_factors if max_factors else int(self.input_size * 0.2)}, "
                      f"因子比例: {factor_ratio:.2f}, "
                      f"隐藏层: {hidden_sizes}, "
                      f"类别数: {self.n_classes}")
            return self
        except Exception as e:
            logger.error(f"创建多重损失的因子化MLP学生模型失败: {str(e)}")
            raise
            
    @log_function_call
    def train(self, X, y, teacher_probs=None, **kwargs):
        """
        训练支持多重损失的因子化MLP学生模型
        
        Args:
            X: 训练数据
            y: 真实标签
            teacher_probs: 教师模型的软标签预测
            **kwargs: 其他参数，支持：
                - epochs: 训练轮数 (默认10)
                - batch_size: 批量大小 (默认32)
                - alpha: 软标签权重 (默认0.5)
                - temperature: 温度参数 (默认1.0)
                - l1_lambda: L1正则化系数 (默认0.001)
                - triplet_margin: 三元组损失的边界值 (默认1.0)
                - sep_lambda: 分离损失权重 (默认0.1)
                - factor_temperature: 因子温度参数 (默认1.0)
        """
        if self.model is None:
            # 根据输入数据推断输入尺寸
            self.input_size = self._infer_input_size(X)
            self.build()
        
        # 提取训练参数
        epochs = kwargs.get('epochs', self.epochs)
        batch_size = kwargs.get('batch_size', self.batch_size)
        alpha = kwargs.get('alpha', 0.5)  # 软标签权重
        temperature = kwargs.get('temperature', self.temperature)  # 温度参数
        
        # 损失相关参数
        l1_lambda = kwargs.get('l1_lambda', 0.001)  # L1正则化系数
        triplet_margin = kwargs.get('triplet_margin', 1.0)  # 三元组损失的边界值
        sep_lambda = kwargs.get('sep_lambda', 0.1)  # 分离损失权重
        
        # 如果有设置factor_temperature，更新模型的factor_temperature
        if 'factor_temperature' in kwargs:
            self.model.factor_temperature = kwargs['factor_temperature']
        
        # 在训练开始前重置类中心字典
        self.model.class_centers = {}
        
        # 训练模式
        self.model.train()
        
        try:
            # 如果有教师模型的软标签，使用知识蒸馏
            if teacher_probs is not None:
                # 确保teacher_probs是2D数组
                if teacher_probs.ndim == 1:
                    teacher_probs = np.column_stack((1.0 - teacher_probs, teacher_probs))
                
                # 创建软标签和硬标签的数据加载器
                X_tensor = self._prepare_data(X)
                y_tensor = torch.LongTensor(y).to(self.device)
                teacher_probs_tensor = torch.FloatTensor(teacher_probs).to(self.device)
                
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, teacher_probs_tensor)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

                class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                
                for epoch in range(epochs):
                    running_loss = 0.0
                    task_losses = 0.0
                    sparse_losses = 0.0
                    sep_losses = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels, teacher_probs_batch in train_loader:
                        # 清零梯度
                        self.optimizer.zero_grad()
                        
                        # 前向传播 - 获取logits和因子表示
                        logits, factors = self.model(inputs)
                        
                        # 计算多重损失
                        task_loss, sparse_loss, sep_loss = self.model.compute_losses(
                            logits=logits,
                            factors=factors,
                            labels=labels,
                            teacher_probs=teacher_probs_batch,
                            temperature=temperature,
                            alpha=alpha,
                            l1_lambda=l1_lambda,
                            triplet_margin=triplet_margin,
                            class_weights=class_weights,
                            current_epoch=epoch,
                            warmup_epochs=epochs * 0.5
                        )
                        
                        # 组合损失 - 直接计算总损失而不是单独使用每个损失
                        total_loss = 0.8 * task_loss + sparse_loss + sep_lambda * sep_loss
                        
                        # 反向传播和优化
                        total_loss.backward()
                        self.optimizer.step()
                        
                        # 统计
                        running_loss += total_loss.item()
                        task_losses += task_loss.item()
                        sparse_losses += sparse_loss.item()
                        sep_losses += sep_loss.item()
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    # 打印统计信息
                    epoch_loss = running_loss / len(train_loader)
                    epoch_task_loss = task_losses / len(train_loader)
                    epoch_sparse_loss = sparse_losses / len(train_loader)
                    epoch_sep_loss = sep_losses / len(train_loader)
                    epoch_acc = 100 * correct / total
                    
                    # 计算因子稀疏度
                    factor_sparsity = self.model.get_mask_sparsity()
                    selected_factors = self.model.get_selected_factors()
                    
                    logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                              f'Task Loss: {epoch_task_loss:.4f}, Sparse Loss: {epoch_sparse_loss:.4f}, '
                              f'Sep Loss: {epoch_sep_loss:.4f}, Acc: {epoch_acc:.2f}%, '
                              f'Selected Factors: {len(selected_factors)} ({factor_sparsity*100:.1f}%)')
            else:
                # 如果没有教师模型的软标签，使用普通训练方法
                train_loader = self._create_data_loader(X, y, batch_size)
                
                for epoch in range(epochs):
                    running_loss = 0.0
                    task_losses = 0.0
                    sparse_losses = 0.0
                    sep_losses = 0.0
                    correct = 0
                    total = 0
                    
                    for inputs, labels in train_loader:
                        # 清零梯度
                        self.optimizer.zero_grad()
                        
                        # 前向传播 - 获取logits和因子表示
                        logits, factors = self.model(inputs)
                        
                        # 计算多重损失
                        task_loss, sparse_loss, sep_loss = self.model.compute_losses(
                            logits=logits,
                            factors=factors,
                            labels=labels,
                            teacher_probs=None,
                            l1_lambda=l1_lambda,
                            triplet_margin=triplet_margin
                        )
                        
                        # 组合损失 - 直接计算总损失而不是单独使用每个损失
                        total_loss = task_loss + sparse_loss + sep_lambda * sep_loss
                        
                        # 反向传播和优化
                        total_loss.backward()
                        self.optimizer.step()
                        
                        # 统计
                        running_loss += total_loss.item()
                        task_losses += task_loss.item()
                        sparse_losses += sparse_loss.item()
                        sep_losses += sep_loss.item()
                        _, predicted = torch.max(logits.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    # 打印统计信息
                    epoch_loss = running_loss / len(train_loader)
                    epoch_task_loss = task_losses / len(train_loader)
                    epoch_sparse_loss = sparse_losses / len(train_loader)
                    epoch_sep_loss = sep_losses / len(train_loader)
                    epoch_acc = 100 * correct / total
                    
                    # 计算因子稀疏度
                    factor_sparsity = self.model.get_mask_sparsity()
                    selected_factors = self.model.get_selected_factors()
                    
                    logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, '
                              f'Task Loss: {epoch_task_loss:.4f}, Sparse Loss: {epoch_sparse_loss:.4f}, '
                              f'Sep Loss: {epoch_sep_loss:.4f}, Acc: {epoch_acc:.2f}%, '
                              f'Selected Factors: {len(selected_factors)} ({factor_sparsity*100:.1f}%)')
                
            # 训练完成后，输出选择的因子信息
            selected_factors = self.model.get_selected_factors()
            selected_features = self.model.get_selected_features()
            logger.info(f"模型最终选择的因子: {len(selected_factors)}个 "
                      f"(总因子的{self.model.get_mask_sparsity()*100:.1f}%)")
            logger.info(f"这些因子对应的重要特征: {len(selected_features)}个")
                
            logger.info("PyTorch多重损失因子化MLP学生模型训练完成")
        except Exception as e:
            logger.error(f"训练多重损失因子化MLP学生模型失败: {str(e)}")
            raise
    
    @log_function_call
    def predict(self, X):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 转换为张量
                X_tensor = self._prepare_data(X)
                
                # 前向传播 - 只需要logits
                logits, _ = self.model(X_tensor)
                
                # 获取预测结果
                _, predicted = torch.max(logits.data, 1)
                
                # 转换为NumPy数组
                return self._to_numpy(predicted)
        except Exception as e:
            logger.error(f"因子化MLP学生模型预测失败: {str(e)}")
            raise
    
    @log_function_call
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 评估模式
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 转换为张量
                X_tensor = self._prepare_data(X)
                
                # 前向传播 - 只需要logits
                logits, _ = self.model(X_tensor)
                
                # 应用softmax获取概率
                probs = nn.functional.softmax(logits / self.temperature, dim=1)
                
                # 转换为NumPy数组
                return self._to_numpy(probs)
        except Exception as e:
            logger.error(f"因子化MLP学生模型概率预测失败: {str(e)}")
            raise
    
    @log_function_call
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            return self.model.get_feature_importance()
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            raise
    
    @log_function_call
    def get_selected_features(self):
        """获取选中的特征索引"""
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            return self.model.get_selected_features()
        except Exception as e:
            logger.error(f"获取选中特征失败: {str(e)}")
            raise
    
    def get_selected_factors(self):
        """获取选中的因子"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        try:
            return self.model.get_selected_factors()
        except Exception as e:
            logger.error(f"获取选中的因子失败: {str(e)}")
            raise
    
    def get_mask_sparsity(self):
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        try:
            return self.model.get_mask_sparsity()
        except Exception as e:
            logger.error(f"{str(e)}")
            raise
            
    @log_function_call
    def compute_shap_values(self, X_background, X_explain=None):
        """
        计算SHAP值以解释模型预测
        
        Args:
            X_background: 用于建立解释器的背景数据
            X_explain: 需要解释的数据点（可选，默认使用背景数据）
            
        Returns:
            shap_values: SHAP值
            feature_importance: 特征重要性
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            import shap
            
            # 评估模式
            self.model.eval()
            
            # 如果没有指定要解释的数据，使用背景数据
            if X_explain is None:
                X_explain = X_background
            
            # 将数据转为张量
            X_background_tensor = self._prepare_data(X_background)
            
            # 创建一个包装函数作为解释目标
            def model_predict(x):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x).to(self.device)
                    logits, _ = self.model(x_tensor)
                    probs = F.softmax(logits, dim=1)
                    return self._to_numpy(probs)
            
            # 使用KernelExplainer（模型无关的解释器）
            explainer = shap.KernelExplainer(model_predict, self._to_numpy(X_background_tensor))
            shap_values = explainer.shap_values(X_explain)
            
            # 计算特征重要性
            if isinstance(shap_values, list):  # 多类别情况
                feature_importance = np.abs(np.array(shap_values)).mean(axis=1).mean(axis=0)
            else:  # 二分类情况
                feature_importance = np.abs(shap_values).mean(axis=0)
            
            return shap_values, feature_importance
        except ImportError:
            logger.error("计算SHAP值需要安装shap库")
            return None, self.model.get_feature_importance()
        except Exception as e:
            logger.error(f"计算SHAP值失败: {str(e)}")
            return None, self.model.get_feature_importance()
            
    @log_function_call
    def compute_factor_feature_dependencies(self, X_background, X_explain=None, threshold=0.05):
        """
        计算每个因子与原始特征之间的依赖关系（使用SHAP值）
        
        Args:
            X_background: 用于建立解释器的背景数据
            X_explain: 需要解释的数据点（可选，默认使用背景数据）
            threshold: 特征影响的阈值，只保留影响大于此阈值的特征
            
        Returns:
            factor_feature_mapping: 每个因子依赖的特征映射字典
            {factor_idx: [(feature_idx, shap_value), ...], ...}
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            # 准备数据
            X_background_tensor = self._prepare_data(X_background)
            
            if X_explain is not None:
                X_explain_tensor = self._prepare_data(X_explain)
            else:
                X_explain_tensor = None
            
            # 调用底层模型的compute_factor_feature_shap方法
            factor_feature_mapping = self.model.compute_factor_feature_shap(
                X_background_tensor, 
                X_explain_tensor, 
                threshold=threshold
            )
            
            # 展示每个因子的关联特征
            for factor_idx, features in factor_feature_mapping.items():
                if features:
                    logger.info(f"因子 {factor_idx} 依赖的前5个特征: " + 
                               ", ".join([f"特征{idx}({value:.3f})" for idx, value in features[:5]]))
                else:
                    logger.info(f"因子 {factor_idx} 没有发现显著依赖的特征")
            
            return factor_feature_mapping
            
        except Exception as e:
            logger.error(f"计算因子-特征依赖关系失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}