import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call
from utils.config import config, Config
from utils.metrics import ModelEvaluator
from visualization.plotters import VisualizationManager
from core.models.pytorch_wrappers import PyTorchBaseWrapper

class ReportGenerator:
    def __init__(self, conf: Optional[Config] = None):
        """初始化报告生成器"""
        if conf is None:
            conf = config

        self.config = conf
        self.reports_dir = conf.results_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = conf.results_dir / "plots"
        self.evaluator = ModelEvaluator()
        self.visualization = VisualizationManager()
    
    @log_function_call
    def generate_experiment_report(self, teacher_model, student_model, X_test, y_test, feature_names=None):
        """生成实验报告
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            X_test: 测试特征
            y_test: 测试标签
            feature_names: 特征名称列表（可选），用于展示更有意义的特征描述
        """
        try:
            # 获取时间
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 模型预测
            teacher_pred = teacher_model.predict(X_test)
            teacher_prob = teacher_model.predict_proba(X_test)
            student_pred = student_model.predict(X_test)
            student_prob = student_model.predict_proba(X_test)
            
            # 计算指标
            teacher_metrics = self.evaluator.calculate_metrics(y_test, teacher_pred, teacher_prob)
            student_metrics = self.evaluator.calculate_metrics(y_test, student_pred, student_prob)
            consistency_metrics = self.evaluator.calculate_prediction_consistency(
                teacher_pred, student_pred
            )
            
            # 获取原始模型对象
            teacher_raw_model = teacher_model.get_model()
            student_raw_model = student_model.get_model()
            
            # 计算模型复杂度
            teacher_complexity = self.evaluator.calculate_model_complexity(teacher_raw_model)
            student_complexity = self.evaluator.calculate_model_complexity(student_raw_model)
            
            # 生成特征重要性可视化（如果可用）
            if hasattr(student_raw_model, 'get_feature_importance') and callable(student_raw_model.get_feature_importance):
                try:
                    # 获取特征重要性
                    feature_importance = student_raw_model.get_feature_importance()
                    
                    # 获取选择的特征索引
                    if hasattr(student_raw_model, 'get_selected_features') and callable(student_raw_model.get_selected_features):
                        selected_features = student_raw_model.get_selected_features()
                    else:
                        selected_features = None
                    
                    # 绘制特征重要性图
                    self.visualization.plot_feature_importance(
                        feature_importance, 
                        feature_names=feature_names, 
                        selected_features=selected_features
                    )
                    
                    # 如果数据维度不太高，也可以绘制特征分布图
                    if X_test.shape[1] <= 50:
                        self.visualization.plot_feature_distribution(X_test, feature_names)
                        
                    # 如果数据维度不太高，可以绘制相关性矩阵
                    if X_test.shape[1] <= 30:
                        self.visualization.plot_correlation_matrix(X_test, feature_names)
                    
                    # 使用PCA进行降维可视化
                    self.visualization.plot_dim_reduction(X_test, y_test, method='pca', feature_names=feature_names)
                    
                    # 绘制精确率-召回率曲线
                    self.visualization.plot_pr_curve(y_test, teacher_prob[:, 1] if teacher_prob.ndim > 1 and teacher_prob.shape[1] > 1 else teacher_prob, 
                                                 student_prob[:, 1] if student_prob.ndim > 1 and student_prob.shape[1] > 1 else student_prob)
                    
                    # 添加：教师模型和学生模型的特征重要性分布对比
                    if hasattr(teacher_raw_model, 'get_feature_importance') and callable(teacher_raw_model.get_feature_importance) and \
                       hasattr(student_raw_model, 'get_feature_importance') and callable(student_raw_model.get_feature_importance):
                        try:
                            # 获取两个模型的特征重要性
                            teacher_importance = teacher_raw_model.get_feature_importance()
                            student_importance = student_raw_model.get_feature_importance()
                            
                            # 如果学生模型有特征选择，则需要调整特征重要性数组
                            if hasattr(student_raw_model, 'get_selected_features') and callable(student_raw_model.get_selected_features):
                                selected_features = student_raw_model.get_selected_features()
                                # 创建与教师模型相同长度的数组，未选中特征的重要性设为0
                                if len(teacher_importance) != len(student_importance):
                                    full_student_importance = np.zeros_like(teacher_importance)
                                    for i, idx in enumerate(selected_features):
                                        if idx < len(full_student_importance) and i < len(student_importance):
                                            full_student_importance[idx] = student_importance[i]
                                    student_importance = full_student_importance
                            
                            # 绘制特征重要性分布对比图
                            self.visualization.plot_feature_importance_comparison(
                                teacher_importance, 
                                student_importance,
                                feature_names=feature_names
                            )
                            
                            logger.info("已生成教师模型和学生模型的特征重要性分布对比图")
                        except Exception as e:
                            logger.warning(f"生成特征重要性分布对比图失败: {str(e)}")
                    
                    logger.info("已生成数据可视化图表")
                except Exception as e:
                    logger.warning(f"生成特征重要性可视化失败: {str(e)}")
            
            # 生成Markdown报告
            report = []
            
            # 标题
            report.append("# 知识蒸馏实验报告")
            report.append(f"**生成时间**: {now}\n")
            
            # 模型概览
            report.append("## 模型概览")
            report.append(f"* **教师模型**: {teacher_model.model_name}")
            report.append(f"* **学生模型**: {student_model.model_name}\n")
            
            # 添加学生模型的特征选择信息
            self._add_feature_selection_info(report, student_raw_model, student_raw_model, feature_names)
            
            # 添加因子化模型信息，并传递X_test用于SHAP分析
            self._add_factorized_model_info(report, student_raw_model, student_raw_model, feature_names, X_test)

            # 性能指标
            report.append("## 性能指标")
            
            # 教师模型指标
            report.append("### 教师模型")
            for metric, value in teacher_metrics.items():
                report.append(f"* **{metric}**: {value:.4f}")
            report.append("")
            
            # 学生模型指标
            report.append("### 学生模型")
            for metric, value in student_metrics.items():
                report.append(f"* **{metric}**: {value:.4f}")
            report.append("")
            
            # 预测一致性
            report.append("### 预测一致性")
            for metric, value in consistency_metrics.items():
                report.append(f"* **{metric}**: {value:.4f}")
            report.append("")
            
            # 模型复杂度对比
            report.append("## 模型复杂度对比")
            
            # 创建复杂度对比表格
            self._add_model_complexity_comparison(report, teacher_complexity, student_complexity)
            
            # 可视化结果
            report.append("## 可视化结果")
            
            # 如果有可视化图像，添加到报告中
            vis_files = list(self.plots_dir.glob("*.png"))
            for img_path in sorted(vis_files):
                img_name = img_path.name
                img_title = img_name.replace("_", " ").replace(".png", "").title()
                rel_path = os.path.relpath(img_path, self.reports_dir)
                report.append(f"### {img_title}")
                report.append(f"![{img_title}]({rel_path})")
                report.append("")
            
            # 结论
            report.append("## 结论")
            
            # 比较准确率
            acc_diff = student_metrics.get('accuracy', 0) - teacher_metrics.get('accuracy', 0)
            if acc_diff > 0:
                conclusion = f"学生模型的准确率比教师模型高出 {acc_diff:.4f}，说明蒸馏效果非常好。"
            elif acc_diff > -0.05:
                conclusion = f"学生模型的准确率与教师模型相近，仅差 {-acc_diff:.4f}，同时模型复杂度显著降低，蒸馏效果良好。"
            else:
                conclusion = f"学生模型的准确率比教师模型低 {-acc_diff:.4f}，但模型复杂度显著降低，在资源受限场景中可以考虑使用。"
            
            # 比较AUC-PR指标
            if 'auc_pr' in teacher_metrics and 'auc_pr' in student_metrics:
                auc_pr_diff = student_metrics.get('auc_pr', 0) - teacher_metrics.get('auc_pr', 0)
                if auc_pr_diff > 0:
                    conclusion += f"\n\n学生模型的AUC-PR（精确率-召回率曲线下面积）比教师模型高出 {auc_pr_diff:.4f}，在处理不平衡数据集时表现优异，显示了蒸馏的高效性。"
                elif auc_pr_diff > -0.05:
                    conclusion += f"\n\n学生模型的AUC-PR（精确率-召回率曲线下面积）与教师模型相近，仅差 {-auc_pr_diff:.4f}，在处理不平衡类别问题时保持了教师模型的性能。"
                else:
                    conclusion += f"\n\n学生模型的AUC-PR（精确率-召回率曲线下面积）比教师模型低 {-auc_pr_diff:.4f}，在处理不平衡类别问题上略有性能损失，但考虑到模型复杂度的显著降低，这是可接受的折中。"
            
            # 添加特征选择总结
            if 'selected_features' in locals() and len(locals()['selected_features']) > 0:
                feature_reduction = 1.0 - len(locals()['selected_features']) / X_test.shape[1]
                conclusion += f"\n\n学生模型通过特征选择，将特征数量从 {X_test.shape[1]} 减少到 {len(locals()['selected_features'])}，"
                conclusion += f"特征减少率为 {feature_reduction:.2%}，显著降低了模型复杂度，提高了模型部署效率。"
            
            # 添加模型大小对比总结
            try:
                # 获取压缩率数据
                teacher_params = teacher_complexity.get('总参数量', 0)
                student_params = student_complexity.get('总参数量', 0)
                param_compression = 1.0 - (student_params / teacher_params) if teacher_params > 0 and student_params > 0 else 0
                
                teacher_flops = teacher_complexity.get('每样本FLOPS', 0)
                student_flops = student_complexity.get('每样本FLOPS', 0)
                flops_compression = 1.0 - (student_flops / teacher_flops) if teacher_flops > 0 and student_flops > 0 else 0
                
                # 首先尝试使用MB单位的内存估计
                teacher_size_mb = teacher_complexity.get('估计内存占用(MB)', teacher_complexity.get('内存大小估计(MB)', 0))
                student_size_mb = student_complexity.get('估计内存占用(MB)', student_complexity.get('内存大小估计(MB)', 0))
                
                # 如果MB单位估计为0，尝试使用KB单位
                if teacher_size_mb == 0 or student_size_mb == 0:
                    teacher_size_kb = teacher_complexity.get('内存大小估计(KB)', 0)
                    student_size_kb = student_complexity.get('内存大小估计(KB)', 0)
                    
                    # 计算内存压缩率
                    mem_compression = 1.0 - (student_size_kb / teacher_size_kb) if teacher_size_kb > 0 and student_size_kb > 0 else 0
                    
                    if teacher_size_kb > 0 and student_size_kb > 0:
                        size_reduction = 1.0 - student_size_kb / teacher_size_kb
                        
                        # 构建详细的结论
                        compression_conclusion = f"\n\n通过知识蒸馏，成功将模型复杂度显著降低："
                        
                        # 参数压缩
                        if param_compression > 0:
                            if param_compression >= 0.9:
                                compression_conclusion += f"\n- **参数量**: 从 {teacher_params:,} 减少到 {student_params:,}，减少了 **{param_compression:.2%}**，**成功达到压缩目标**"
                            else:
                                compression_conclusion += f"\n- **参数量**: 从 {teacher_params:,} 减少到 {student_params:,}，减少了 {param_compression:.2%}"
                        
                        # 计算复杂度压缩
                        if flops_compression > 0:
                            if flops_compression >= 0.9:
                                compression_conclusion += f"\n- **计算复杂度**: 从 {teacher_flops:,} FLOPS 减少到 {student_flops:,} FLOPS，减少了 **{flops_compression:.2%}**，**成功达到压缩目标**"
                            else:
                                compression_conclusion += f"\n- **计算复杂度**: 从 {teacher_flops:,} FLOPS 减少到 {student_flops:,} FLOPS，减少了 {flops_compression:.2%}"
                        
                        # 内存压缩
                        compression_conclusion += f"\n- **内存占用**: 从 {teacher_size_kb:.2f}KB 减少到 {student_size_kb:.2f}KB，减少了 {size_reduction:.2%}"
                        
                        # 添加总体评价
                        if param_compression >= 0.9 or flops_compression >= 0.9:
                            compression_conclusion += f"\n\n**总体评价**: 蒸馏效果优秀，学生模型在保持预测能力的同时，大幅降低了复杂度，适合在资源受限环境中部署。"
                        else:
                            compression_conclusion += f"\n\n**总体评价**: 蒸馏取得了一定效果，学生模型能够以较低的复杂度实现相近的预测效果。"
                        
                        conclusion += compression_conclusion
                else:
                    # 使用MB单位
                    if teacher_size_mb > 0 and student_size_mb > 0:
                        size_reduction = 1.0 - student_size_mb / teacher_size_mb
                        
                        # 构建详细的结论
                        compression_conclusion = f"\n\n通过知识蒸馏，成功将模型复杂度显著降低："
                        
                        # 参数压缩
                        if param_compression > 0:
                            if param_compression >= 0.9:
                                compression_conclusion += f"\n- **参数量**: 从 {teacher_params:,} 减少到 {student_params:,}，减少了 **{param_compression:.2%}**，**成功达到压缩目标**"
                            else:
                                compression_conclusion += f"\n- **参数量**: 从 {teacher_params:,} 减少到 {student_params:,}，减少了 {param_compression:.2%}"
                        
                        # 计算复杂度压缩
                        if flops_compression > 0:
                            if flops_compression >= 0.9:
                                compression_conclusion += f"\n- **计算复杂度**: 从 {teacher_flops:,} FLOPS 减少到 {student_flops:,} FLOPS，减少了 **{flops_compression:.2%}**，**成功达到压缩目标**"
                            else:
                                compression_conclusion += f"\n- **计算复杂度**: 从 {teacher_flops:,} FLOPS 减少到 {student_flops:,} FLOPS，减少了 {flops_compression:.2%}"
                        
                        # 内存压缩
                        compression_conclusion += f"\n- **内存占用**: 从 {teacher_size_mb:.2f}MB 减少到 {student_size_mb:.2f}MB，减少了 {size_reduction:.2%}"
                        
                        # 添加总体评价
                        if param_compression >= 0.9 or flops_compression >= 0.9:
                            compression_conclusion += f"\n\n**总体评价**: 蒸馏效果优秀，学生模型在保持预测能力的同时，大幅降低了复杂度，适合在资源受限环境中部署。"
                        else:
                            compression_conclusion += f"\n\n**总体评价**: 蒸馏取得了一定效果，学生模型能够以较低的复杂度实现相近的预测效果。"
                        
                        conclusion += compression_conclusion
            except Exception as e:
                logger.warning(f"计算模型大小对比时出错: {str(e)}")
            
            report.append(conclusion)
            report.append("")
            
            # 加入特征重要性图表（如果有）
            if 'selected_features_info' in locals() and len(locals()['selected_features_info']) > 0:
                # 导出特征重要性数据为JSON
                selected_features_json_path = self.plots_dir / "selected_features.json"
                try:
                    with open(selected_features_json_path, 'w', encoding='utf-8') as f:
                        json.dump(locals()['selected_features_info'], f, ensure_ascii=False, indent=2)
                    logger.info(f"特征重要性数据已保存至 {selected_features_json_path}")
                except Exception as e:
                    logger.warning(f"保存特征重要性数据时出错: {str(e)}")
                
            # 保存报告
            report_path = self.reports_dir / "experiment_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
                
            logger.info(f"实验报告已生成并保存至 {report_path}")
            
            return report_path
        
        except Exception as e:
            logger.error(f"生成实验报告失败: {str(e)}")
            raise
    
    def _add_feature_selection_info(self, report, student_model, student_raw_model, feature_names=None):
        """添加特征选择信息到报告中"""
        try:
            # 判断学生模型是否支持因子选择
            has_feature_selection = False
            
            # 检查是否为PyTorch模型，并且支持特征选择
            if hasattr(student_raw_model, 'get_selected_features'):
                has_feature_selection = True
                report.append("### 特征选择信息")
                
                # 获取选择的特征
                selected_features = student_raw_model.get_selected_features()
                
                # 特征使用率
                if hasattr(student_raw_model, 'get_mask_sparsity') and not hasattr(student_raw_model, 'get_selected_features'):
                    sparsity = student_raw_model.get_mask_sparsity()
                    report.append(f"* **特征使用率**: {sparsity:.2%}")
                    
                # 选择的特征数量
                report.append(f"* **选择的特征数量**: {len(selected_features)} / {student_raw_model.input_size if hasattr(student_raw_model, 'input_size') else len(selected_features)}")
                
                # 获取特征重要性（如果可用）
                if hasattr(student_model, 'get_feature_importance') and callable(student_model.get_feature_importance):
                    try:
                        feature_importance = student_model.get_feature_importance()
                        
                        # 将特征索引与重要性分数配对
                        selected_features_info = []
                        for idx in selected_features:
                            if idx < len(feature_importance):
                                importance = feature_importance[idx]
                                feature_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"特征_{idx}"
                                selected_features_info.append({
                                    "index": int(idx),
                                    "name": feature_name,
                                    "importance": float(importance)
                                })
                        
                        # 按重要性降序排序
                        selected_features_info.sort(key=lambda x: x["importance"], reverse=True)
                        
                        # 展示顶部10个最重要的特征（或者全部，如果少于10个）
                        report.append("\n**选择的关键特征 (重要性降序排列)**:")
                        report.append("| 特征名称 | 重要性分数 |")
                        report.append("| --- | --- |")
                        
                        top_n = min(10, len(selected_features_info))
                        for feature in selected_features_info[:top_n]:
                            report.append(f"| {feature['name']} | {feature['importance']:.4f} |")
                        
                        # 如果有更多特征，添加省略号
                        if len(selected_features_info) > top_n:
                            report.append(f"| ... 等 {len(selected_features_info) - top_n} 个特征 | ... |")
                        
                        report.append("")
                        
                        # 保存本地变量以便后续使用
                        locals()['selected_features_info'] = selected_features_info
                        locals()['selected_features'] = selected_features
                        
                    except Exception as e:
                        logger.warning(f"获取特征重要性失败: {str(e)}")
                        
                report.append("")
            
            # 如果不支持特征选择，添加普通信息
            if not has_feature_selection:
                report.append("### 模型信息")
                report.append("* 学生模型**未启用**特征选择功能")
                
                # 尝试获取模型的输入大小
                if hasattr(student_raw_model, 'input_size'):
                    report.append(f"* **输入特征维度**: {student_raw_model.input_size}")
                
                # 尝试获取隐藏层大小（对于神经网络）
                if hasattr(student_raw_model, 'hidden_sizes'):
                    report.append(f"* **隐藏层大小**: {student_raw_model.hidden_sizes}")
                elif hasattr(student_raw_model, 'hidden_layer_sizes'):
                    report.append(f"* **隐藏层大小**: {student_raw_model.hidden_layer_sizes}")
                    
                report.append("")
                
        except Exception as e:
            logger.warning(f"添加特征选择信息到报告时出错: {str(e)}")
            report.append("* 无法获取特征选择信息\n")

    def _add_factorized_model_info(self, report, student_model, student_raw_model, feature_names=None, X_test=None):
        """添加多重损失因子化模型的详细信息到报告"""
        try:
            if isinstance(student_model.get_model(), PyTorchBaseWrapper):
                student_model = student_model.get_model()
            if hasattr(student_raw_model, 'get_model'):
                student_raw_model = student_raw_model.get_model()

            # 获取选中的因子
            if hasattr(student_raw_model, 'get_selected_factors'):
                selected_factors = student_raw_model.get_selected_factors()
                max_factors = student_raw_model.max_factors if hasattr(student_raw_model, 'max_factors') else (student_raw_model.factor_weights.shape[-1] if hasattr(student_raw_model, 'factor_weights') else 1)
                
                report.append("\n### 因子分析")
                report.append(f"* **总因子数量**: {max_factors}")
                report.append(f"* **选中的因子数量**: {len(selected_factors)}")
                report.append(f"* **因子使用率**: {len(selected_factors)/max_factors:.2%}")
                
                # 获取因子权重
                if hasattr(student_raw_model, 'factor_gate') and hasattr(student_raw_model, 'factor_weights'):
                    import torch
                    import numpy as np
                    
                    # 获取门控值和因子权重
                    gates = student_raw_model.get_factor_importance()
                    factor_weights = student_raw_model.factor_weights.weight.detach().cpu().numpy() if isinstance(student_raw_model.factor_weights, torch.nn.Linear) else (student_raw_model.factor_weights.detach().cpu().numpy() if isinstance(student_raw_model.factor_weights, torch.Tensor) else student_raw_model.factor_weights)
                    
                    # 如果模型使用特征提取器，需要获取特征提取器的最后一层输出维度
                    last_hidden_size = student_raw_model.hidden_sizes[-1] if hasattr(student_raw_model, 'hidden_sizes') and student_raw_model.hidden_sizes else factor_weights.shape[0]
                    
                    # 展示选中因子的门控值
                    report.append("\n**选中的因子及其门控值**:")
                    report.append("| 因子 ID | 门控值 |")
                    report.append("| --- | --- |")
                    
                    # 门控值按降序排列
                    factors_with_gates = [(i, gates[i]) for i in range(len(gates))]
                    factors_with_gates.sort(key=lambda x: x[1], reverse=True)
                    
                    for factor_id, gate_value in factors_with_gates[:min(10, len(factors_with_gates))]:
                        report.append(f"| 因子_{factor_id} | {gate_value:.4f} |")
                    
                    if len(factors_with_gates) > 10:
                        report.append(f"| ... 等 {len(factors_with_gates) - 10} 个因子 | ... |")
                    
                    report.append("")
                    
                    # 分析因子与原始特征的关联
                    report.append("### 因子与特征提取器输出的关联")
                    report.append("每个因子通过特征提取器的输出进行加权，下面展示每个选中因子与隐藏特征的关联强度。")
                    
                    # 因子与最后隐藏层的关联
                    report.append("\n**因子与特征提取器输出的关联**:")
                    report.append("| 因子 ID | 最相关的隐藏特征 | 关联强度 |")
                    report.append("| --- | --- | --- |")
                    
                    # 对于每个选中的因子，找出最强关联的隐藏特征
                    for factor_id in selected_factors[:min(5, len(selected_factors))]:
                        # 获取该因子与隐藏层的权重
                        factor_to_hidden_weights = np.abs(factor_weights[factor_id]) if factor_weights.shape[0] == max_factors else np.abs(factor_weights[:, factor_id])
                        
                        # 找出最大权重及其索引
                        max_idx = np.argmax(factor_to_hidden_weights)
                        max_weight = factor_to_hidden_weights[max_idx]
                        
                        report.append(f"| 因子_{factor_id} | 隐藏特征_{max_idx} | {max_weight:.4f} |")
                    
                    if len(selected_factors) > 5:
                        report.append(f"| ... 等 {len(selected_factors) - 5} 个因子 | ... | ... |")
                    
                    report.append("")
                    
                    # 尝试分析因子与原始特征的关联 - 通过模型结构反向传播和SHAP计算
                    try:
                        if self.config.use_shap:
                            # 计算特征重要性
                            feature_importance = student_model.get_feature_importance()
                            
                            # 使用SHAP值计算因子与原始特征的关联关系
                            if hasattr(student_model, 'compute_factor_feature_dependencies') and callable(student_model.compute_factor_feature_dependencies):
                                report.append("\n### 因子与原始特征的SHAP关联分析")
                                report.append("使用SHAP计算每个因子与原始特征的依赖关系，揭示因子如何依赖于原始特征。")
                                
                                # 获取数据集用于SHAP分析
                                X_background = None
                                if X_test is not None:
                                    X_background = X_test[:min(100, len(X_test))]  # 最多使用100个样本
                                
                                if X_background is not None:
                                    try:
                                        # 计算因子与特征的依赖关系
                                        factor_feature_mapping = student_model.compute_factor_feature_dependencies(
                                            X_background=X_background,
                                            threshold=0.05  # 只保留影响大于5%的特征
                                        )
                                        
                                        # 展示每个因子的关联特征
                                        if factor_feature_mapping:
                                            for factor_idx, features in factor_feature_mapping.items():
                                                if features:
                                                    report.append(f"\n**因子_{factor_idx} 依赖的原始特征（SHAP值）**:")
                                                    report.append("| 特征名称 | SHAP值（归一化） |")
                                                    report.append("| --- | --- |")
                                                    
                                                    # 展示每个因子的前10个最重要特征
                                                    for feature_idx, importance in features[:min(10, len(features))]:
                                                        feature_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f"特征_{feature_idx}"
                                                        report.append(f"| {feature_name} | {importance:.4f} |")
                                                    
                                                    if len(features) > 10:
                                                        report.append(f"| ... 等 {len(features) - 10} 个特征 | ... |")
                                                    
                                                    report.append("")
                                                else:
                                                    report.append(f"\n**因子_{factor_idx}**: 没有发现显著依赖的特征")
                                        else:
                                            report.append("\n*SHAP分析未发现任何因子与特征之间的显著关联*\n")
                                    except Exception as e:
                                        logger.warning(f"计算因子与特征SHAP关联失败: {str(e)}")
                                        import traceback
                                        logger.warning(f"错误详情: {traceback.format_exc()}")
                                        report.append("\n*计算因子与特征的SHAP关联关系失败*\n")
                                else:
                                    report.append("\n*无法获取背景数据进行SHAP分析*\n")
                            
                            # 为每个因子创建与原始特征的关联表（基于特征重要性的简化方法）
                            report.append("\n### 因子与原始特征的关联（基于特征重要性）")
                            report.append("下表展示了每个选中因子最相关的原始特征:")
                            
                            for factor_id in selected_factors[:min(5, len(selected_factors))]:
                                report.append(f"\n**因子_{factor_id} 最相关的原始特征**:")
                                report.append("| 特征名称 | 关联强度 |")
                                report.append("| --- | --- |")
                                
                                # 对于多层模型，我们需要近似计算因子与原始特征的关联
                                # 这里我们使用特征重要性作为简化的关联度量
                                # 实际上更准确的方法是计算完整的梯度路径，但这超出了报告范围
                                
                                # 假设特征重要性已经考虑了因子的贡献
                                # 我们选择前5个最重要的特征与因子关联
                                sorted_features = np.argsort(feature_importance)[::-1]
                                top_features = sorted_features[:5]
                                
                                for feature_idx in top_features:
                                    feature_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f"特征_{feature_idx}"
                                    importance = feature_importance[feature_idx]
                                    report.append(f"| {feature_name} | {importance:.4f} |")
                                
                                report.append("")
                            
                            if len(selected_factors) > 5:
                                report.append(f"\n*注：仅展示前5个因子的关联信息，共有 {len(selected_factors)} 个选中的因子。*\n")
                    
                    except Exception as e:
                        logger.warning(f"计算因子与原始特征关联失败: {str(e)}")
                        report.append("\n*无法获取因子与原始特征的详细关联*\n")
            
        except Exception as e:
            logger.warning(f"添加因子模型信息失败: {str(e)}")
            import traceback
            logger.warning(f"错误堆栈: \n{traceback.format_exc()}")
            report.append("\n*无法获取因子模型详细信息*\n")
    
    def _add_model_complexity_comparison(self, report, teacher_complexity, student_complexity):
        """添加模型复杂度比较到报告中"""
        try:
            # 将复杂度指标分类为不同组
            categories = {
                "计算复杂度": ["每样本FLOPS", "推理复杂度(路径数)", "推理时间复杂度"],
                "参数量": ["总参数量", "可学习参数数量", "总节点数", "叶子节点数", "非叶子节点数"],
                "内存占用": ["内存大小估计(KB)", "内存大小估计(MB)", "权重大小(KB)", "估计内存占用(MB)"],
                "网络结构": ["层数", "隐藏层大小", "树数量", "最大树深", "超参数数量"],
                "特征使用": ["特征数量", "选择的特征数量", "特征使用率", "稀疏度", "非零系数数量"]
            }
            
            # 计算总体压缩率
            # 1. 参数量压缩率
            teacher_params = teacher_complexity.get('总参数量', 0)
            student_params = student_complexity.get('总参数量', 0)
            
            param_compression = 0
            if teacher_params > 0 and student_params > 0:
                param_compression = 1.0 - (student_params / teacher_params)
            
            # 2. 计算复杂度压缩率
            teacher_flops = teacher_complexity.get('每样本FLOPS', 0)
            student_flops = student_complexity.get('每样本FLOPS', 0)
            
            flops_compression = 0
            if teacher_flops > 0 and student_flops > 0:
                flops_compression = 1.0 - (student_flops / teacher_flops)
            
            # 3. 内存占用压缩率
            teacher_mem_kb = teacher_complexity.get('内存大小估计(KB)', 0)
            student_mem_kb = student_complexity.get('内存大小估计(KB)', 0)
            
            memory_compression = 0
            if teacher_mem_kb > 0 and student_mem_kb > 0:
                memory_compression = 1.0 - (student_mem_kb / teacher_mem_kb)
            
            # 添加压缩率总结信息
            report.append("### 模型压缩率总结")
            report.append("| 压缩指标 | 压缩率 | 说明 |")
            report.append("| --- | --- | --- |")
            
            # 参数量压缩率
            report.append(f"| **参数量压缩率** | {param_compression:.2%} | 从 {teacher_params:,} 参数减少到 {student_params:,} 参数 |")
            
            # 计算复杂度压缩率
            report.append(f"| **计算复杂度压缩率** | {flops_compression:.2%} | 从 {teacher_flops:,} FLOPS减少到 {student_flops:,} FLOPS |")
            
            # 内存占用压缩率
            report.append(f"| **内存占用压缩率** | {memory_compression:.2%} | 从 {teacher_mem_kb:.2f}KB 减少到 {student_mem_kb:.2f}KB |")
            
            # 如果压缩率超过目标，添加特别标记
            if param_compression >= 0.9:
                report.append("\n> **🎉 成功达成参数量削减90%以上的目标！**")
            
            if flops_compression >= 0.9:
                report.append("\n> **🚀 成功达成计算复杂度削减90%以上的目标！**")
            
            # 如果有特征选择，添加特征选择的信息
            if '选择的特征数量' in student_complexity and '特征使用率' in student_complexity:
                feature_count = student_complexity.get('选择的特征数量', 0)
                total_features = teacher_complexity.get('特征数量', 0)
                if total_features == 0:
                    # 尝试从其它地方推断特征总数
                    if '系数数量' in teacher_complexity:
                        total_features = teacher_complexity['系数数量']
                
                if total_features > 0:
                    feature_reduction = 1.0 - (feature_count / total_features)
                    report.append(f"\n> **🔍 学生模型通过特征选择，从 {total_features} 个特征中选择了 {feature_count} 个关键特征，特征减少率为 {feature_reduction:.2%}**")
            
            report.append("\n")
            
            # 合并所有指标并去重
            all_metrics = set(teacher_complexity.keys()) | set(student_complexity.keys())
            
            # 创建复杂度对比表格
            for category, metrics in categories.items():
                # 过滤出当前类别中存在的指标
                category_metrics = [m for m in metrics if m in all_metrics]
                
                if category_metrics:
                    report.append(f"\n### {category}")
                    report.append("| 指标 | 教师模型 | 学生模型 | 变化率 |")
                    report.append("| --- | --- | --- | --- |")
                    
                    for metric in category_metrics:
                        teacher_value = teacher_complexity.get(metric, "N/A")
                        student_value = student_complexity.get(metric, "N/A")
                        
                        # 格式化值
                        t_value_str = f"{teacher_value:,}" if isinstance(teacher_value, int) else (f"{teacher_value:.4f}" if isinstance(teacher_value, float) else str(teacher_value))
                        s_value_str = f"{student_value:,}" if isinstance(student_value, int) else (f"{student_value:.4f}" if isinstance(student_value, float) else str(student_value))
                        
                        # 计算比例变化（如果两个值都是数字）
                        change_ratio = "N/A"
                        if isinstance(teacher_value, (int, float)) and isinstance(student_value, (int, float)) and teacher_value != 0:
                            ratio = (student_value - teacher_value) / teacher_value
                            if ratio < 0:  # 减少了
                                change_str = "减少"
                                change_ratio = f"{change_str} {abs(ratio):.2%}"
                                # 如果减少超过90%，特别标记
                                if abs(ratio) >= 0.9:
                                    change_ratio = f"**{change_ratio}** 🎯"
                            else:  # 增加了
                                change_str = "增加"
                                change_ratio = f"{change_str} {abs(ratio):.2%}"
                        
                        report.append(f"| {metric} | {t_value_str} | {s_value_str} | {change_ratio} |")
                    
                    report.append("")
                    
            # 如果以上类别没有覆盖所有指标，添加其他指标
            other_metrics = all_metrics - set().union(*categories.values())
            if other_metrics:
                report.append("\n### 其他指标")
                report.append("| 指标 | 教师模型 | 学生模型 | 变化率 |")
                report.append("| --- | --- | --- | --- |")
                
                for metric in other_metrics:
                    teacher_value = teacher_complexity.get(metric, "N/A")
                    student_value = student_complexity.get(metric, "N/A")
                    
                    # 格式化值
                    t_value_str = f"{teacher_value:,}" if isinstance(teacher_value, int) else (f"{teacher_value:.4f}" if isinstance(teacher_value, float) else str(teacher_value))
                    s_value_str = f"{student_value:,}" if isinstance(student_value, int) else (f"{student_value:.4f}" if isinstance(student_value, float) else str(student_value))
                    
                    # 计算比例变化（如果两个值都是数字）
                    change_ratio = "N/A"
                    if isinstance(teacher_value, (int, float)) and isinstance(student_value, (int, float)) and teacher_value != 0:
                        ratio = (student_value - teacher_value) / teacher_value
                        change_str = "增加" if ratio > 0 else "减少"
                        change_ratio = f"{change_str} {abs(ratio):.2%}"
                    
                    report.append(f"| {metric} | {t_value_str} | {s_value_str} | {change_ratio} |")
                
                report.append("")
                
        except Exception as e:
            logger.warning(f"添加模型复杂度比较到报告时出错: {str(e)}")
            # 退回到简单的表格
            report.append("| 指标 | 教师模型 | 学生模型 |")
            report.append("| --- | --- | --- |")
            
            for key in set(teacher_complexity.keys()) | set(student_complexity.keys()):
                teacher_value = teacher_complexity.get(key, "N/A")
                student_value = student_complexity.get(key, "N/A")
                
                if teacher_value != "N/A":
                    teacher_value = f"{teacher_value:.4f}" if isinstance(teacher_value, float) else str(teacher_value)
                
                if student_value != "N/A":
                    student_value = f"{student_value:.4f}" if isinstance(student_value, float) else str(student_value)
                
                report.append(f"| {key} | {teacher_value} | {student_value} |")
            
            report.append("")