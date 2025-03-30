import numpy as np
import sys
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score, precision_recall_curve
)
from .logger import logger, log_function_call
import math
import traceback

class ModelEvaluator:
    @staticmethod
    @log_function_call
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """计算模型评估指标"""
        # 确保y_true是一维数组，用于标准指标计算
        if isinstance(y_true, np.ndarray) and y_true.ndim > 1:
            logger.info(f"将二维标签数组转换为一维数组进行评估，原形状: {y_true.shape}")
            # 对于二分类问题，取第1列作为正类的标签
            if y_true.shape[1] == 2:
                y_true_1d = y_true[:, 1]
            else:
                # 对于多分类问题，取argmax
                y_true_1d = np.argmax(y_true, axis=1)
        else:
            y_true_1d = y_true
            
        metrics = {
            'accuracy': accuracy_score(y_true_1d, y_pred),
            'precision': precision_score(y_true_1d, y_pred, average='weighted'),
            'recall': recall_score(y_true_1d, y_pred, average='weighted'),
            'f1': f1_score(y_true_1d, y_pred, average='weighted')
        }
        
        if y_prob is not None:
            try:
                # 二分类问题
                if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1 and y_prob.shape[1] == 2:
                    # 只使用第2类（正类）的概率
                    metrics['auc_roc'] = roc_auc_score(y_true_1d, y_prob[:, 1])
                    # 计算AUC-PR (精确率-召回率曲线下面积)
                    metrics['auc_pr'] = average_precision_score(y_true_1d, y_prob[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true_1d, y_prob)
                    # 计算AUC-PR (精确率-召回率曲线下面积)
                    metrics['auc_pr'] = average_precision_score(y_true_1d, y_prob)
            except Exception as e:
                logger.warning(f"无法计算AUC-ROC或AUC-PR: {str(e)}")
        
        return metrics
    
    @staticmethod
    @log_function_call
    def calculate_prediction_consistency(teacher_pred, student_pred):
        """计算教师模型和学生模型预测一致性"""
        # 确保输入是一维数组
        teacher_pred_1d = np.argmax(teacher_pred, axis=1) if teacher_pred.ndim > 1 else teacher_pred
        student_pred_1d = np.argmax(student_pred, axis=1) if student_pred.ndim > 1 else student_pred
        
        consistency = np.mean(teacher_pred_1d == student_pred_1d)
        return {
            'prediction_consistency': consistency,
            'disagreement_rate': 1 - consistency
        }
    
    @staticmethod
    @log_function_call
    def calculate_model_complexity(model):
        """计算模型复杂度，包括参数数量、计算复杂度（FLOPs）和内存使用
        
        Args:
            model: 模型对象
            
        Returns:
            dict: 包含复杂度指标的字典
        """
        try:
            # 日志记录模型类型，帮助调试
            logger.debug(f"计算模型复杂度: {type(model).__name__}")
            
            # 首先尝试从包装器中获取原始模型（如果有get_model方法）
            if hasattr(model, 'get_model') and callable(getattr(model, 'get_model')):
                # 记录这是一个包装器模型
                logger.debug(f"模型是包装器类型 {type(model).__name__}，尝试获取原始模型")
                original_model = model.get_model()
                if original_model is not None:
                    logger.debug(f"从包装器获取到原始模型: {type(original_model).__name__}")
                    model = original_model
                else:
                    logger.debug(f"从包装器获取原始模型失败，使用包装器模型继续")
            
            # XGBoost模型
            if str(type(model)).find('xgboost') != -1:
                logger.debug("检测到XGBoost模型")
                booster = model.get_booster() if hasattr(model, 'get_booster') else model
                
                # 参数数量
                json_model = booster.save_config()
                model_dump = booster.get_dump()
                num_trees = len(model_dump)
                
                # 估算参数数量：每个树节点4个参数（特征id、分裂值、左子树、右子树）
                if num_trees > 0:
                    total_nodes = sum([model_dump[i].count(':') for i in range(num_trees)])
                    params = total_nodes * 4
                else:
                    params = 0
                
                # FLOPs估算：决策过程中的比较操作
                # 每个样本遍历的节点数 ≈ 树的深度 * 树的数量
                # 假设平均深度为log2(节点数)
                if total_nodes > 0 and num_trees > 0:
                    avg_nodes_per_tree = total_nodes / num_trees
                    avg_depth = math.log2(avg_nodes_per_tree + 1) if avg_nodes_per_tree > 0 else 0
                    flops = avg_depth * num_trees  # 每个样本的FLOPs
                else:
                    flops = 0
                
                # 内存使用
                try:
                    # 获取模型二进制表示
                    raw_data = booster.save_raw()
                    # bytearray对象使用len()获取大小，而不是nbytes属性
                    memory_usage = len(raw_data) / 1024  # KB
                except Exception as e:
                    logger.warning(f"计算XGBoost模型内存大小时出错: {str(e)}")
                    # 使用备选方法估算大小
                    memory_usage = params * 4 / 1024  # 假设每个参数4字节
                
                logger.debug(f"XGBoost模型复杂度: 参数={params}, FLOPs={flops}, 内存={memory_usage}KB")
                return {
                    '总参数量': params,
                    '每样本FLOPS': flops,
                    '内存大小估计(KB)': memory_usage,
                    '树数量': num_trees,
                    '总节点数': total_nodes
                }
                
            # Scikit-learn模型
            elif hasattr(model, 'get_params'):
                logger.debug("检测到Scikit-learn模型")
                try:
                    # 基于模型类型的粗略估计
                    if hasattr(model, 'n_support_'):  # SVM
                        params = sum(model.n_support_) * model.n_features_in_
                        flops = params  # 简化估计
                    elif hasattr(model, 'coef_'):  # 线性模型
                        params = model.coef_.size + (model.intercept_.size if hasattr(model, 'intercept_') else 0)
                        flops = params  # 简化估计
                    else:
                        params = 1000  # 默认估计
                        flops = 1000  # 默认估计
                    
                    # 内存使用
                    memory_usage = 0
                    for attr in ['coef_', 'intercept_', 'support_vectors_']:
                        if hasattr(model, attr):
                            obj = getattr(model, attr)
                            if hasattr(obj, 'nbytes'):
                                memory_usage += obj.nbytes
                    
                    memory_usage = memory_usage / 1024  # KB
                    
                    logger.debug(f"Scikit-learn模型复杂度: 参数={params}, FLOPs={flops}, 内存={memory_usage}KB")
                    return {
                        '总参数量': params,
                        '每样本FLOPS': flops,
                        '内存大小估计(KB)': memory_usage
                    }
                except Exception as e:
                    logger.warning(f"计算Scikit-learn模型复杂度时出错: {str(e)}")
                    return {'总参数量': 0, '每样本FLOPS': 0, '内存大小估计(KB)': 0}
            
            # PyTorch模型 - 详细计算
            elif (hasattr(model, 'parameters') and callable(getattr(model, 'parameters')) and 
                  str(type(model)).find('torch') != -1):
                logger.debug(f"检测到PyTorch模型: {type(model).__name__}")
                
                # 参数数量计算
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                logger.debug(f"PyTorch模型参数: 总数={total_params}, 可训练={trainable_params}")
                
                # 内存使用计算
                memory_usage = 0
                for param in model.parameters():
                    memory_usage += param.nelement() * param.element_size()
                
                # 加上缓冲区大小
                for buffer in model.buffers():
                    memory_usage += buffer.nelement() * buffer.element_size()
                
                memory_usage = memory_usage / 1024  # KB
                logger.debug(f"PyTorch模型内存使用: {memory_usage}KB")
                
                # FLOPS计算
                flops = 0
                
                # 对MLP模型的特殊处理
                if hasattr(model, 'hidden_layers'):  # 如果是MLP
                    logger.debug("计算MLP模型的FLOPs")
                    try:
                        input_size = model.input_size if hasattr(model, 'input_size') else None
                        output_size = model.output_size if hasattr(model, 'output_size') else None
                        
                        if not input_size and hasattr(model, 'fc_layers') and model.fc_layers:
                            # 尝试从第一层获取输入大小
                            first_layer = model.fc_layers[0]
                            if hasattr(first_layer, 'in_features'):
                                input_size = first_layer.in_features
                        
                        if not output_size and hasattr(model, 'fc_layers') and model.fc_layers:
                            # 尝试从最后一层获取输出大小
                            last_layer = model.fc_layers[-1]
                            if hasattr(last_layer, 'out_features'):
                                output_size = last_layer.out_features
                        
                        logger.debug(f"MLP维度: 输入={input_size}, 输出={output_size}")
                        
                        if hasattr(model, 'hidden_sizes') and input_size and output_size:
                            # 计算每层的FLOPs
                            prev_size = input_size
                            for h_size in model.hidden_sizes:
                                # 每层: 乘加操作 = 输入维度*输出维度
                                layer_flops = prev_size * h_size
                                flops += layer_flops
                                logger.debug(f"层大小 {prev_size} -> {h_size}: {layer_flops} FLOPs")
                                prev_size = h_size
                            
                            # 输出层
                            output_flops = prev_size * output_size
                            flops += output_flops
                            logger.debug(f"输出层 {prev_size} -> {output_size}: {output_flops} FLOPs")
                    except Exception as e:
                        logger.warning(f"计算MLP FLOPs时出错: {str(e)}")
                
                # 遍历模型的所有模块，为常见层计算FLOPs
                if flops == 0:
                    logger.debug("通过遍历模型层计算FLOPs")
                    for name, module in model.named_modules():
                        # 线性层(全连接)
                        if isinstance(module, torch.nn.Linear):
                            in_features = module.in_features
                            out_features = module.out_features
                            # FLOPs = 乘法和加法操作 = 输入*输出
                            layer_flops = in_features * out_features
                            flops += layer_flops
                            logger.debug(f"线性层 {name}: {in_features} -> {out_features} = {layer_flops} FLOPs")
                        
                        # 卷积层
                        elif isinstance(module, torch.nn.Conv2d):
                            in_channels = module.in_channels
                            out_channels = module.out_channels
                            kernel_size = module.kernel_size
                            if hasattr(module, 'output_size') and module.output_size:
                                # 如果知道输出大小，精确计算FLOPs
                                out_h, out_w = module.output_size
                                layer_flops = out_h * out_w * in_channels * out_channels * kernel_size[0] * kernel_size[1]
                            else:
                                # 粗略估计
                                estimated_output_size = 32  # 假设值
                                layer_flops = estimated_output_size**2 * in_channels * out_channels * kernel_size[0] * kernel_size[1]
                            flops += layer_flops
                            logger.debug(f"卷积层 {name}: in={in_channels}, out={out_channels}, k={kernel_size} = {layer_flops} FLOPs")
                    
                    # 如果没有计算到任何FLOPs，使用简单的估计方法
                    if flops == 0:
                        logger.debug("无法通过层计算FLOPs，使用参数总数作为估计")
                        flops = total_params * 2  # 简单估计：每个参数执行约2次操作
                
                # 收集隐藏层大小信息
                hidden_sizes_str = None
                if hasattr(model, 'hidden_sizes'):
                    hidden_sizes_str = str(model.hidden_sizes)
                
                # 收集输入大小信息
                input_size_val = None
                if hasattr(model, 'input_size'):
                    input_size_val = model.input_size
                
                # 检查是否有特征选择信息
                selected_features_count = None
                feature_usage_rate = None
                if hasattr(model, 'get_selected_features') and callable(getattr(model, 'get_selected_features')):
                    try:
                        selected_features = model.get_selected_features()
                        selected_features_count = len(selected_features)
                    except Exception as e:
                        logger.warning(f"获取选择的特征数量失败: {str(e)}")
                
                if hasattr(model, 'get_mask_sparsity') and callable(getattr(model, 'get_mask_sparsity')):
                    try:
                        feature_usage_rate = model.get_mask_sparsity()
                    except Exception as e:
                        logger.warning(f"获取特征使用率失败: {str(e)}")
                
                # 构建结果字典
                result = {
                    '总参数量': total_params,
                    '可学习参数数量': trainable_params,
                    '每样本FLOPS': flops,
                    '内存大小估计(KB)': memory_usage
                }
                
                # 添加可选信息
                if hidden_sizes_str:
                    result['隐藏层大小'] = hidden_sizes_str
                
                if input_size_val:
                    result['输入维度'] = input_size_val
                
                if selected_features_count is not None:
                    result['选择的特征数量'] = selected_features_count
                
                if feature_usage_rate is not None:
                    result['特征使用率'] = feature_usage_rate
                
                # 如果内存大于1024KB，也提供MB的值
                if memory_usage > 1024:
                    result['内存大小估计(MB)'] = memory_usage / 1024
                
                logger.debug(f"PyTorch模型最终复杂度: {result}")
                return result
            
            # 未知模型类型
            else:
                logger.warning(f"未知模型类型: {type(model).__name__}")
                return {
                    '总参数量': 1000, 
                    '每样本FLOPS': 1000, 
                    '内存大小估计(KB)': 1000
                }
        
        except Exception as e:
            logger.error(f"计算模型复杂度时发生错误: {str(e)}")
            traceback.print_exc()
            return {
                '总参数量': 0, 
                '每样本FLOPS': 0, 
                '内存大小估计(KB)': 0
            }
    
    @staticmethod
    @log_function_call
    def format_metrics(metrics_dict):
        """格式化指标输出"""
        if not isinstance(metrics_dict, dict):
            # 如果不是字典类型，尝试将它当作其他格式处理
            if isinstance(metrics_dict, (list, tuple)) and len(metrics_dict) == 2:
                # 可能是(预测一致性, 不一致率)的元组
                return f"一致性: {metrics_dict[0]:.4f}\n不一致率: {metrics_dict[1]:.4f}"
            else:
                # 其他情况，直接返回字符串表示
                return str(metrics_dict)
        
        # 如果是字典类型，正常格式化
        return "\n".join([f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in metrics_dict.items()])