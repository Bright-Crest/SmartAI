# 知识蒸馏实验报告
**生成时间**: 2025-03-30 21:36:12

## 模型概览
* **教师模型**: pytorch_mlp
* **学生模型**: pytorch_factorized_mlp

### 特征选择信息
* **选择的特征数量**: 6 / 13

**选择的关键特征 (重要性降序排列)**:
| 特征名称 | 重要性分数 |
| --- | --- |
| 特征_8 | 1.0000 |
| 特征_9 | 0.8533 |
| 特征_2 | 0.7761 |
| 特征_6 | 0.7691 |
| 特征_4 | 0.7647 |
| 特征_5 | 0.6728 |


## 性能指标
### 教师模型
* **accuracy**: 0.9820
* **precision**: 0.9820
* **recall**: 0.9820
* **f1**: 0.9820
* **auc_roc**: 0.9968
* **auc_pr**: 0.9629

### 学生模型
* **accuracy**: 0.9440
* **precision**: 0.9561
* **recall**: 0.9440
* **f1**: 0.9467
* **auc_roc**: 0.9944
* **auc_pr**: 0.9476

### 预测一致性
* **prediction_consistency**: 0.9470
* **disagreement_rate**: 0.0530

## 模型复杂度对比
### 模型压缩率总结
| 压缩指标 | 压缩率 | 说明 |
| --- | --- | --- |
| **参数量压缩率** | 71.29% | 从 37,506 参数减少到 10,769 参数 |
| **计算复杂度压缩率** | 72.86% | 从 36,352 FLOPS减少到 9,866 FLOPS |
| **内存占用压缩率** | 70.85% | 从 149.52KB 减少到 43.58KB |



### 计算复杂度
| 指标 | 教师模型 | 学生模型 | 变化率 |
| --- | --- | --- | --- |
| 每样本FLOPS | 36,352 | 9,866 | 减少 72.86% |


### 参数量
| 指标 | 教师模型 | 学生模型 | 变化率 |
| --- | --- | --- | --- |
| 总参数量 | 37,506 | 10,769 | 减少 71.29% |
| 可学习参数数量 | 37,506 | 10,769 | 减少 71.29% |


### 内存占用
| 指标 | 教师模型 | 学生模型 | 变化率 |
| --- | --- | --- | --- |
| 内存大小估计(KB) | 149.5234 | 43.5830 | 减少 70.85% |


### 特征使用
| 指标 | 教师模型 | 学生模型 | 变化率 |
| --- | --- | --- | --- |
| 选择的特征数量 | 13 | 6 | 减少 53.85% |
| 特征使用率 | 1.0000 | 0.0000 | **减少 100.00%** 🎯 |


### 其他指标
| 指标 | 教师模型 | 学生模型 | 变化率 |
| --- | --- | --- | --- |
| 输入维度 | 13 | 13 | 减少 0.00% |

## 可视化结果
### Correlation Matrix
![Correlation Matrix](..\plots\correlation_matrix.png)

### Decision Boundary Pytorchfactorizedstudentmlp F0 1
![Decision Boundary Pytorchfactorizedstudentmlp F0 1](..\plots\decision_boundary_PyTorchFactorizedStudentMLP_f0_1.png)

### Decision Boundary Pytorchteachermlp F0 1
![Decision Boundary Pytorchteachermlp F0 1](..\plots\decision_boundary_PyTorchTeacherMLP_f0_1.png)

### Factor Feature Heatmap
![Factor Feature Heatmap](..\plots\factor_feature_heatmap.png)

### Factor Gates
![Factor Gates](..\plots\factor_gates.png)

### Feature Distribution
![Feature Distribution](..\plots\feature_distribution.png)

### Feature Importance
![Feature Importance](..\plots\feature_importance.png)

### Feature Importance Distribution
![Feature Importance Distribution](..\plots\feature_importance_distribution.png)

### Pca Scatter
![Pca Scatter](..\plots\pca_scatter.png)

### Pca Visualization
![Pca Visualization](..\plots\pca_visualization.png)

### Precision Recall Curve
![Precision Recall Curve](..\plots\precision_recall_curve.png)

### Prediction Comparison
![Prediction Comparison](..\plots\prediction_comparison.png)

### Tsne Visualization
![Tsne Visualization](..\plots\tsne_visualization.png)

## 结论
学生模型的准确率与教师模型相近，仅差 0.0380，同时模型复杂度显著降低，蒸馏效果良好。

学生模型的AUC-PR（精确率-召回率曲线下面积）与教师模型相近，仅差 0.0153，在处理不平衡类别问题时保持了教师模型的性能。

学生模型通过特征选择，将特征数量从 13 减少到 6，特征减少率为 53.85%，显著降低了模型复杂度，提高了模型部署效率。

通过知识蒸馏，成功将模型复杂度显著降低：
- **参数量**: 从 37,506 减少到 10,769，减少了 71.29%
- **计算复杂度**: 从 36,352 FLOPS 减少到 9,866 FLOPS，减少了 72.86%
- **内存占用**: 从 149.52KB 减少到 43.58KB，减少了 70.85%

**总体评价**: 蒸馏取得了一定效果，学生模型能够以较低的复杂度实现相近的预测效果。
