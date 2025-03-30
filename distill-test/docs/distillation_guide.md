# 自主因子选择功能

本文档介绍了在蒸馏模型中添加的自主因子选择功能，该功能允许学生模型自动选择最关键的特征因子，从而减少模型复杂度并提高泛化能力。

## 功能概述

- **自主选择因子**：学生模型能够自动学习每个特征的重要性，并选择最关键的特征子集
- **降低模型复杂度**：通过只使用最重要的特征，模型可以变得更轻量级
- **提高泛化能力**：去除不重要的特征有助于减少过拟合，提高模型泛化能力
- **可解释性提升**：自动选择的关键特征可以帮助理解模型决策过程

## 支持的模型

目前以下学生模型支持自主因子选择：

- `pytorch_mlp`：PyTorch实现的多层感知机模型
- `pytorch_cnn`：PyTorch实现的卷积神经网络模型
- `pytorch_factor_mlp`：特化的因子选择MLP模型

## 使用方法

在创建学生模型时，可以通过以下参数控制因子选择功能：

```python
student_params = {
    'input_size': X_train.shape[1],  # 输入特征维度
    'hidden_sizes': [64, 32],        # 隐藏层大小
    'n_classes': 2,                  # 分类类别数
    
    # 因子选择相关参数
    'enable_factor_selection': True,       # 是否启用因子选择
    'max_factor_ratio': 0.1,               # 最大因子比例（0.1表示最多使用10%的特征）
    'factor_temperature': 2.0,             # 温度参数（控制选择的软硬程度）
}

student = StudentModel(
    model_type="student", 
    model_name="pytorch_mlp",
    model_params=student_params
)
```

在训练过程中，可以通过以下参数进一步调整因子选择行为：

```python
student.train(
    X_train, y_train, 
    teacher_probs=teacher_probs,
    alpha=0.5,                    # 软标签权重
    temperature=2.0,              # 知识蒸馏温度
    l1_lambda=0.01,               # L1正则化系数（控制特征稀疏程度）
    factor_temperature=1.5        # 可以动态调整特征掩码温度
)
```

## 命令行使用

使用`run_pipeline.py`脚本时可以通过以下参数控制因子选择：

```bash
python src/run_pipeline.py --data-path data/your_dataset.csv \
                           --teacher xgboost \
                           --student pytorch_mlp \
                           --enable-factor-selection \
                           --max-factor-ratio 0.2 \
                           --factor-temperature 1.5 \
                           --l1-lambda 0.005
```

## 查看选择的特征

训练完成后，可以使用以下方法查看模型选择的特征：

```python
# 获取选择的特征索引
selected_features = student.get_selected_features()
print(f"模型选择了 {len(selected_features)} 个特征: {selected_features}")

# 获取特征重要性分数
feature_importance = student.get_feature_importance()
```

## 参数说明

- **enable_factor_selection**：是否启用特征自动选择功能，默认为True
- **max_factor_ratio**：最大因子比例，控制最多可以选择多少比例的特征，范围[0, 1]，默认0.2
- **factor_temperature**：控制特征选择的软硬程度，较大的值使选择更加模糊，较小的值使选择更加明确，默认1.0
- **l1_lambda**：L1正则化系数，控制特征稀疏程度，较大的值会导致选择更少的特征，默认0.001

## 工作原理

该功能使用以下技术实现自主因子选择：

1. **可学习的特征掩码**：为每个输入特征分配一个可学习的权重（掩码）
2. **Sigmoid激活**：通过Sigmoid函数将掩码值转换为0-1之间的值
3. **温度参数**：控制Sigmoid函数的陡峭程度，影响特征选择的软硬程度
4. **L1正则化**：通过对掩码值添加L1正则化，鼓励模型选择稀疏的特征子集
5. **特征门控**：训练数据通过掩码进行过滤，使模型只关注重要特征

## 实现细节

对于PyTorch模型，特征选择通过以下方式实现：

```python
class FeatureSelector(nn.Module):
    def __init__(self, input_size, max_factor_ratio=0.2, temperature=1.0):
        super().__init__()
        self.input_size = input_size
        self.max_factor_ratio = max_factor_ratio
        self.temperature = temperature
        self.feature_masks = nn.Parameter(torch.zeros(input_size))
    
    def forward(self, x):
        # 应用Sigmoid和温度参数
        masks = torch.sigmoid(self.feature_masks / self.temperature)
        # 应用掩码
        return x * masks
    
    def get_mask_values(self):
        with torch.no_grad():
            return torch.sigmoid(self.feature_masks / self.temperature).cpu().numpy()
```

## 最佳实践

- 从较大的`max_factor_ratio`开始（如0.5），然后根据需要减小该值
- 调整`l1_lambda`以控制特征选择的稀疏程度
- 分析选择的特征与领域知识的一致性，评估模型的可解释性
- 观察选择特征数量与模型性能之间的权衡关系

## 可视化与报告功能

### 特征重要性可视化

```python
from visualization.plotters import VisualizationManager

# 创建可视化管理器
vis_manager = VisualizationManager()

# 获取特征重要性和选定的特征
feature_importance = student_model.get_feature_importance()
selected_features = student_model.get_selected_features()

# 绘制特征重要性图
vis_manager.plot_feature_importance(
    feature_importance, 
    feature_names=feature_names,
    selected_features=selected_features
)
```

### 实验报告生成

使用`ReportGenerator`可以生成包含以下内容的详细Markdown实验报告：

```python
from visualization.report_generator import ReportGenerator

report_gen = ReportGenerator()
report_path = report_gen.generate_experiment_report(
    teacher_model, student_model, X_test, y_test, feature_names=feature_names
)
print(f"实验报告已生成: {report_path}")
```

报告内容包括：

- 模型概览信息
- 特征选择详情，包括选择的特征数量和特征使用率
- 选择的关键特征及其重要性分数
- 性能指标对比（教师模型 vs 学生模型）
- 预测一致性评估
- 模型复杂度详细对比（内存占用、参数数量等）
- 特征可视化结果

## 应用场景

自主因子选择功能在以下场景特别有用：

1. **高维特征数据**：当原始特征维度较高时，可以自动筛选最有用的特征
2. **嘈杂数据**：包含许多不相关特征的数据集可受益于自动特征筛选
3. **资源受限环境**：在内存或计算资源有限的环境中运行模型
4. **可解释性需求**：需要理解模型决策主要基于哪些特征的场景