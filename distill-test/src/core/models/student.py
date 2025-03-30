import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call
from .base_model import BaseModel
from .model_registry import ModelRegistry

class StudentModel(BaseModel):
    @log_function_call
    def build(self):
        """构建学生模型"""
        try:
            # 使用模型注册表获取模型类
            model_class = ModelRegistry.get_model_class("student", self.model_name)
            
            # 对于SVM等特殊模型，确保启用概率输出
            if self.model_name == "svm":
                self.model_params["probability"] = True
            
            # 对于逻辑回归模型，检查并设置多类别支持
            if self.model_name == "logistic":
                # 确保设置多类别参数，以便能够处理软标签
                if 'multi_class' not in self.model_params:
                    # 默认使用'ovr'（一对多），但如果想要支持软标签训练，应该使用'multinomial'
                    self.model_params['multi_class'] = 'ovr'
                    
                # 如果需要multinomial支持，确保设置合适的求解器
                if self.model_params.get('multi_class') == 'multinomial' and 'solver' not in self.model_params:
                    self.model_params['solver'] = 'lbfgs'  # 支持multinomial的求解器
            
            # 实例化模型
            # 对PyTorch模型进行特殊处理
            if self.model_name.startswith("pytorch_"):
                # 如果是PyTorch模型，使用model_type和model_name参数初始化
                self.model = model_class(
                    model_type=self.model_type,
                    model_name=self.model_name,
                    model_params=self.model_params
                )
            else:
                # 非PyTorch模型，直接使用参数字典初始化
                self.model = model_class(**self.model_params)
            
            logger.info(f"已创建学生模型 {self.model_name}")
            return self
        except Exception as e:
            logger.error(f"创建学生模型失败: {str(e)}")
            raise
    
    @log_function_call
    def train(self, X, y, teacher_probs=None, **kwargs):
        """训练学生模型
        
        Args:
            X: 训练数据
            y: 真实标签
            teacher_probs: 教师模型的软标签预测
            **kwargs: 其他参数
        """
        if self.model is None:
            self.build()
        
        # 移除模型fit方法不支持的参数
        for param in ['random_state']:
            kwargs.pop(param, None)
            
        try:
            # 针对PyTorch模型的特殊处理
            if self.model_name.startswith("pytorch_"):
                # PyTorch模型直接使用自己的训练方法，它已经实现了知识蒸馏
                self.model.train(X, y, teacher_probs=teacher_probs, **kwargs)
            # 如果有教师模型的软标签，使用软标签和真实标签的组合进行训练
            elif teacher_probs is not None and hasattr(self.model, 'fit') and self._supports_soft_targets():
                alpha = kwargs.pop('alpha', 0.5)  # 软标签权重
                temperature = kwargs.pop('temperature', 1.0)  # 温度参数
                
                # 确保teacher_probs是2D数组
                if teacher_probs.ndim == 1:
                    # 如果是一维数组（二分类问题的第二类概率），转换为二维
                    teacher_probs = np.column_stack((1.0 - teacher_probs, teacher_probs))
                
                # 软标签处理
                soft_targets = np.exp(np.log(np.clip(teacher_probs, 1e-10, 1.0)) / temperature)
                soft_targets = soft_targets / soft_targets.sum(axis=1, keepdims=True)
                
                # 获取类别数量
                n_classes = soft_targets.shape[1]
                
                # 对于不同类型的模型使用不同的方法
                if self._supports_sample_weight():
                    # 将软目标转换为样本权重
                    sample_weights = np.max(soft_targets, axis=1)
                    self.model.fit(X, y, sample_weight=sample_weights, **kwargs)
                else:
                    # 使用软目标和真实标签的组合进行训练
                    # 将y转换为one-hot编码
                    y_one_hot = np.zeros((len(y), n_classes))
                    for i, label in enumerate(y):
                        if 0 <= label < n_classes:  # 确保标签在有效范围内
                            y_one_hot[i, int(label)] = 1
                        
                    # 组合软标签和真实标签
                    combined_targets = alpha * soft_targets + (1 - alpha) * y_one_hot
                    self._fit_with_soft_targets(X, combined_targets, **kwargs)
            else:
                # 如果不支持软标签或没有提供软标签，使用普通训练
                self.model.fit(X, y, **kwargs)
                
            logger.info("学生模型训练完成")
        except Exception as e:
            logger.error(f"学生模型训练失败: {str(e)}")
            raise
    
    @log_function_call
    def _supports_soft_targets(self):
        """检查模型是否支持软标签训练"""
        # PyTorch模型、逻辑回归和MLP支持软标签训练
        return self.model_name in ["logistic", "mlp"] or self.model_name.startswith("pytorch_")
    
    @log_function_call
    def _supports_sample_weight(self):
        """检查模型是否支持样本权重"""
        try:
            return 'sample_weight' in self.model.fit.__code__.co_varnames
        except AttributeError:
            return False
    
    @log_function_call
    def _fit_with_soft_targets(self, X, soft_targets, **kwargs):
        """使用软标签进行训练
        
        Args:
            X: 训练数据
            soft_targets: 软标签
            **kwargs: 其他参数
        """
        # 移除模型fit方法不支持的参数
        for param in ['random_state']:
            kwargs.pop(param, None)
            
        # 针对不同模型类型进行适配
        if self.model_name == "logistic":
            # 检查逻辑回归模型是否配置为支持多标签
            is_multinomial = (
                hasattr(self.model, 'multi_class') and 
                self.model.multi_class == 'multinomial' and
                hasattr(self.model, 'solver') and 
                self.model.solver in ['lbfgs', 'newton-cg', 'sag', 'saga']
            )
            
            if is_multinomial:
                try:
                    # 对于配置了multinomial的逻辑回归，使用y_predict_proba而不是fit
                    # 设置适当的参数以实现多输出支持
                    kwargs['multi_output'] = True
                    self.model.fit(X, soft_targets, **kwargs)
                except (ValueError, TypeError) as e:
                    logger.warning(f"多标签训练失败，回退到硬标签训练: {str(e)}")
                    y_hard = np.argmax(soft_targets, axis=1)
                    self.model.fit(X, y_hard, **kwargs)
            else:
                # 如果不是multinomial，使用硬标签
                logger.info("逻辑回归模型未配置为multinomial，使用硬标签训练")
                y_hard = np.argmax(soft_targets, axis=1)
                self.model.fit(X, y_hard, **kwargs)
        elif self.model_name == "mlp":
            # MLP能够处理多标签
            try:
                # 保存原来的label_binarizer_属性
                original_label_binarizer = None
                if hasattr(self.model, '_label_binarizer'):
                    original_label_binarizer = self.model._label_binarizer
                
                # 临时禁用标签二值化器
                self.model._label_binarizer = None
                
                # 尝试直接用软标签训练
                self.model.fit(X, soft_targets, **kwargs)
                
                # 恢复原来的设置
                if original_label_binarizer is not None:
                    self.model._label_binarizer = original_label_binarizer
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"MLP多标签训练失败，尝试另一种方法: {str(e)}")
                try:
                    # 备选方案：使用warm_start方式分批次训练
                    if not hasattr(self.model, 'warm_start') or not self.model.warm_start:
                        self.model.warm_start = True
                        was_warm_start_enabled = False
                    else:
                        was_warm_start_enabled = True
                    
                    # 首先用硬标签训练一次
                    y_hard = np.argmax(soft_targets, axis=1)
                    self.model.fit(X, y_hard, **kwargs)
                    
                    # 然后使用软标签进行微调
                    # 为每个类别创建一个数据子集
                    n_classes = soft_targets.shape[1]
                    for class_idx in range(n_classes):
                        # 选择概率高于阈值的样本
                        threshold = 0.1  # 可调整的阈值
                        mask = soft_targets[:, class_idx] > threshold
                        if np.sum(mask) > 0:  # 只有当有足够的样本时才训练
                            X_subset = X[mask]
                            y_subset = np.full(X_subset.shape[0], class_idx)
                            # 使用样本权重表示软标签的置信度
                            sample_weight = soft_targets[mask, class_idx]
                            self.model.fit(X_subset, y_subset, sample_weight=sample_weight, **kwargs)
                    
                    # 恢复原始的warm_start设置
                    if not was_warm_start_enabled:
                        self.model.warm_start = False
                        
                except Exception as e2:
                    logger.warning(f"MLP软标签训练的备选方案也失败: {str(e2)}")
                    # 最后的回退：使用硬标签
                    y_hard = np.argmax(soft_targets, axis=1)
                    self.model.fit(X, y_hard, **kwargs)
        else:
            # 默认方法：转换为类别
            y_hard = np.argmax(soft_targets, axis=1)
            self.model.fit(X, y_hard, **kwargs)
    
    @log_function_call
    def predict(self, X):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    @log_function_call
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        try:
            probs = self.model.predict_proba(X)
            
            # 确保返回的是二维概率数组
            if probs.ndim == 1:
                # 如果是一维数组（通常是二分类的正类概率），转换为二维
                probs = np.column_stack((1.0 - probs, probs))
                
            return probs
        except (AttributeError, NotImplementedError):
            # 如果模型不支持概率预测，则使用独热编码
            logger.warning("模型不支持概率预测，将使用独热编码")
            y_pred = self.predict(X)
            n_classes = len(np.unique(y_pred))
            return np.eye(n_classes)[y_pred]
    
    @log_function_call
    def get_model_size(self):
        """获取模型大小"""
        if self.model is None:
            return 0
        
        try:
            # 根据不同模型类型获取大小
            if self.model_name == "logistic":
                return np.prod(self.model.coef_.shape)
            elif self.model_name == "mlp":
                return sum(np.prod(layer.shape) for layer in self.model.coefs_)
            elif self.model_name == "svm":
                return len(self.model.support_vectors_)
            elif self.model_name == "decision_tree":
                return self.model.tree_.node_count
            else:
                # 尝试使用通用方法估计大小
                import sys
                return sys.getsizeof(self.model) / 1024  # 大小（KB）
        except Exception as e:
            logger.warning(f"获取模型大小失败: {str(e)}")
            return 0