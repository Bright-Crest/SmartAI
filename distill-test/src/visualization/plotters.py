import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import matplotlib
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import json
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.matplotlib_config import setup_chinese_fonts
from utils.logger import logger, log_function_call
from utils.config import config

# 设置中文字体支持
try:
    # 获取用户系统中的中文字体
    import matplotlib.font_manager as font_manager
    # 优先尝试微软雅黑，如果没有则使用系统中其他含有中文的字体
    chinese_fonts = [f for f in font_manager.findSystemFonts() if os.path.basename(f).startswith(('msyh', 'simhei', 'simsun', 'simkai', 'kaiti', 'stkaiti', 'fangson'))]
    if chinese_fonts:
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=chinese_fonts[0]).get_name()
    else:
        # 如果没有找到中文字体，则使用系统默认字体，可能会显示乱码
        logger.warning("未找到支持中文的字体，图表中的中文可能会显示为乱码")
except Exception as e:
    logger.warning(f"设置中文字体支持时出错: {str(e)}")
    
# 添加中文支持的字体路径(备选方案)
font_paths = [
    # Windows字体路径
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/msyh.ttf",    # 微软雅黑
    "C:/Windows/Fonts/simsun.ttc",  # 宋体
    # Linux字体路径
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # macOS字体路径
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Songti.ttc"
]

# 查找有效的字体文件
chinese_font = None
for font_path in font_paths:
    if os.path.exists(font_path):
        chinese_font = FontProperties(fname=font_path)
        break

if chinese_font is None:
    logger.warning("未找到中文字体文件，可能导致图表中文显示问题")

# 设置默认颜色风格
current_palette = sns.color_palette("Set2", 10)
sns.set_palette(current_palette)
# 设置全局样式
sns.set(style="whitegrid")

class VisualizationManager:
    def __init__(self):
        """初始化可视化管理器"""
        self.plots_dir = config.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 确保每次创建实例时都应用中文字体设置
        setup_chinese_fonts()
        
    @log_function_call
    def plot_feature_distributions(self, X, feature_names=None, n_cols=3, figsize=(16, 12)):
        """绘制特征分布图
        
        Args:
            X: 特征矩阵
            feature_names: 特征名列表
            n_cols: 每行子图数量
            figsize: 图像大小
        """
        try:
            # 确保在绘图前设置中文字体
            setup_chinese_fonts()
            
            if feature_names is None:
                feature_names = [f"特征_{i+1}" for i in range(X.shape[1])]
            
            n_features = X.shape[1]
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()
            
            for i, (name, ax) in enumerate(zip(feature_names, axes)):
                if i < n_features:
                    sns.histplot(X[:, i], ax=ax, kde=True)
                    ax.set_title(name)
                    ax.set_xlabel('')
            
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            save_path = self.plots_dir / "feature_distributions.png"
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"特征分布图已保存至 {save_path}")
            
        except Exception as e:
            logger.error(f"绘制特征分布图失败: {str(e)}")
            plt.close('all')
    
    @log_function_call
    def plot_correlation_matrix(self, X, feature_names=None, figsize=(12, 10)):
        """绘制相关性矩阵
        
        Args:
            X: 特征矩阵
            feature_names: 特征名列表
            figsize: 图像大小
        """
        try:
            # 确保在绘图前设置中文字体
            setup_chinese_fonts()
            
            if feature_names is None:
                feature_names = [f"特征_{i+1}" for i in range(X.shape[1])]
            
            df = pd.DataFrame(X, columns=feature_names)
            corr = df.corr()
            
            fig, ax = plt.subplots(figsize=figsize)
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(
                corr, 
                mask=mask, 
                cmap=cmap, 
                vmax=1, 
                vmin=-1, 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5},
                ax=ax
            )
            
            plt.title('特征相关性矩阵')
            save_path = self.plots_dir / "correlation_matrix.png"
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"相关性矩阵图已保存至 {save_path}")
            
        except Exception as e:
            logger.error(f"绘制相关性矩阵失败: {str(e)}")
            plt.close('all')
            
    @log_function_call
    def plot_dimension_reduction(self, X, y, method='pca', figsize=(10, 8)):
        """使用降维方法可视化数据
        
        Args:
            X: 特征矩阵
            y: 标签
            method: 降维方法 ('pca' 或 'tsne')
            figsize: 图像大小
        """
        try:
            # 确保在绘图前设置中文字体
            setup_chinese_fonts()
            
            if method.lower() == 'pca':
                reducer = PCA(n_components=2)
                title = 'PCA降维可视化'
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
                title = 't-SNE降维可视化'
            else:
                raise ValueError(f"不支持的降维方法: {method}")
            
            X_reduced = reducer.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=figsize)
            scatter = ax.scatter(
                X_reduced[:, 0], 
                X_reduced[:, 1], 
                c=y, 
                cmap='viridis', 
                alpha=0.7,
                s=50,
                edgecolors='w'
            )
            
            legend = ax.legend(*scatter.legend_elements(), title="类别")
            ax.add_artist(legend)
            
            ax.set_title(title)
            ax.set_xlabel('维度 1')
            ax.set_ylabel('维度 2')
            
            save_path = self.plots_dir / f"{method.lower()}_visualization.png"
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"{title}已保存至 {save_path}")
            
        except Exception as e:
            logger.error(f"绘制降维图失败: {str(e)}")
            plt.close('all')
    
    @log_function_call
    def plot_decision_boundary(self, X, y, model, feature_idx=[0, 1], figsize=(10, 8), mesh_stepsize=0.02):
        """绘制决策边界
        
        Args:
            X: 特征矩阵
            y: 标签数组
            model: 模型对象（需要有predict方法）
            feature_idx: 用于可视化的两个特征的索引
            figsize: 图表大小
            mesh_stepsize: 网格步长，较小的值会使边界更加平滑但计算更慢
        """
        try:
            # 检查是否为二维数据
            if X.shape[1] < 2:
                logger.warning("特征维度少于2，无法绘制决策边界")
                return None
                
            # 确保feature_idx是有效的整数列表或数组
            if isinstance(feature_idx, list) or isinstance(feature_idx, tuple) or isinstance(feature_idx, np.ndarray):
                # 确保索引是整数
                feature_idx = [int(idx) for idx in feature_idx[:2]]  # 只取前两个
                # 检查索引是否超出范围
                if max(feature_idx) >= X.shape[1]:
                    logger.warning(f"特征索引{feature_idx}超出范围，将使用前两个特征")
                    feature_idx = [0, 1]
                # 提取两个特征
                X_subset = X[:, feature_idx]
            else:
                # 如果是单个整数或其他类型，默认使用前两个特征
                logger.warning(f"无效的feature_idx类型: {type(feature_idx)}，将使用前两个特征")
                feature_idx = [0, 1]
                X_subset = X[:, :2]
            
            # 设置图表
            plt.figure(figsize=figsize)
            
            # 定义边界
            x_min, x_max = X_subset[:, 0].min() - 0.5, X_subset[:, 0].max() + 0.5
            y_min, y_max = X_subset[:, 1].min() - 0.5, X_subset[:, 1].max() + 0.5
            
            # 创建网格点
            xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                               np.arange(y_min, y_max, mesh_stepsize))
            
            # 准备预测的输入 - 使用X的平均值填充所有特征
            Z_input = np.zeros((xx.ravel().shape[0], X.shape[1]))
            
            # 先用整个X的特征平均值填充所有Z_input
            X_mean = np.mean(X, axis=0)
            for i in range(X.shape[1]):
                Z_input[:, i] = X_mean[i]
            
            # 然后用网格点的值替换要可视化的两个特征
            Z_input[:, feature_idx[0]] = xx.ravel()
            Z_input[:, feature_idx[1]] = yy.ravel()
            
            logger.info(f"Z_input形状: {Z_input.shape}, 原始X形状: {X.shape}")
            
            # 预测
            try:
                # 尝试使用predict_proba方法（对于分类问题）
                probs = model.predict_proba(Z_input)
                if probs.shape[1] == 2:  # 二分类
                    Z = probs[:, 1]
                else:  # 多分类
                    Z = np.argmax(probs, axis=1)
                logger.info(f"使用predict_proba方法生成决策边界")
            except Exception as e:
                logger.warning(f"使用predict_proba失败: {str(e)}, 尝试使用predict")
                try:
                    # 降级到predict方法
                    Z = model.predict(Z_input)
                    logger.info(f"使用predict方法生成决策边界")
                except Exception as e2:
                    logger.error(f"使用predict也失败: {str(e2)}")
                    raise ValueError(f"模型无法预测数据: {str(e2)}")
                
            # 将结果重塑为网格
            Z = Z.reshape(xx.shape)
            
            # 绘制决策边界
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
            plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')
            
            # 绘制训练点
            scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y, 
                               edgecolors='k', s=50, cmap=plt.cm.RdBu_r)
            
            # 添加图例
            legend1 = plt.legend(*scatter.legend_elements(),
                               title="类别", loc="best")
            plt.gca().add_artist(legend1)
            
            # 设置标题和标签
            title_text = f"模型决策边界 - 特征{feature_idx[0]}和特征{feature_idx[1]}"
            plt.title(title_text, fontproperties=chinese_font)
            plt.xlabel(f"特征 {feature_idx[0]}", fontproperties=chinese_font)
            plt.ylabel(f"特征 {feature_idx[1]}", fontproperties=chinese_font)
            plt.grid(True)
            plt.tight_layout()
            
            # 保存图表
            model_name = model.__class__.__name__
            save_path = self.plots_dir / f"decision_boundary_{model_name}_f{feature_idx[0]}_{feature_idx[1]}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"决策边界图已保存至 {save_path}")
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"绘制决策边界失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            plt.close()
            raise
    
    @log_function_call
    def plot_prediction_comparison(self, y_true, teacher_pred, student_pred, figsize=(12, 6)):
        """绘制预测对比图
        
        Args:
            y_true: 真实标签
            teacher_pred: 教师模型预测
            student_pred: 学生模型预测
            figsize: 图像大小
        """
        try:
            # 确保在绘图前设置中文字体
            setup_chinese_fonts()
            
            n_samples = len(y_true)
            indices = np.random.choice(range(n_samples), min(100, n_samples), replace=False)
            indices = np.sort(indices)
            
            df = pd.DataFrame({
                '样本索引': indices,
                '真实标签': y_true[indices],
                '教师预测': teacher_pred[indices],
                '学生预测': student_pred[indices]
            })
            
            # 计算预测正确/错误
            df['教师正确'] = df['教师预测'] == df['真实标签']
            df['学生正确'] = df['学生预测'] == df['真实标签']
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 绘制真实标签
            ax.scatter(df['样本索引'], df['真实标签'], s=80, marker='o', color='black', label='真实标签')
            
            # 绘制教师预测
            for i, row in df.iterrows():
                ax.scatter(row['样本索引'], row['教师预测'], s=50, 
                           marker='^', color='green' if row['教师正确'] else 'red',
                           alpha=0.7)
            
            # 绘制学生预测
            for i, row in df.iterrows():
                ax.scatter(row['样本索引'], row['学生预测'], s=50, 
                           marker='s', color='blue' if row['学生正确'] else 'orange',
                           alpha=0.7)
            
            # 绘制图例
            handles, labels = [], []
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10))
            labels.append('真实标签')
            
            handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10))
            labels.append('教师正确预测')
            
            handles.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10))
            labels.append('教师错误预测')
            
            handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10))
            labels.append('学生正确预测')
            
            handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10))
            labels.append('学生错误预测')
            
            ax.legend(handles, labels, loc='upper right')
            
            ax.set_xlabel('样本索引')
            ax.set_ylabel('类别')
            ax.set_title('模型预测对比')
            
            plt.tight_layout()
            save_path = self.plots_dir / "prediction_comparison.png"
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            logger.info(f"预测对比图已保存至 {save_path}")
            
        except Exception as e:
            logger.error(f"绘制预测对比图失败: {str(e)}")
            plt.close('all')

    @log_function_call
    def plot_feature_distribution(self, X, feature_names=None, figsize=(12, 8)):
        """绘制特征分布图
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            figsize: 图表大小
        """
        try:
            if isinstance(X, pd.DataFrame):
                # 如果是pandas DataFrame，直接使用它的列名
                data = X
            else:
                # 转换为pandas DataFrame
                if feature_names is None:
                    feature_names = [f"特征_{i}" for i in range(X.shape[1])]
                
                # 如果特征太多，只选择前30个
                if X.shape[1] > 30:
                    logger.info(f"特征数量太多({X.shape[1]})，只展示前30个特征的分布")
                    X = X[:, :30]
                    feature_names = feature_names[:30]
                
                data = pd.DataFrame(X, columns=feature_names)
            
            # 设置图表
            plt.figure(figsize=figsize)
            
            # 使用Seaborn绘制箱线图
            ax = sns.boxplot(data=data)
            
            # 设置标题和标签
            plt.title("特征分布", fontproperties=chinese_font)
            plt.xlabel("特征", fontproperties=chinese_font)
            plt.ylabel("值", fontproperties=chinese_font)
            
            # 设置x轴标签旋转以适应更多文本
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            save_path = self.plots_dir / "feature_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征分布图已保存至 {save_path}")
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"绘制特征分布图失败: {str(e)}")
            raise
            
    @log_function_call
    def plot_correlation_matrix(self, X, feature_names=None, figsize=(12, 10)):
        """绘制相关性矩阵热图
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            figsize: 图表大小
        """
        try:
            # 转换为pandas DataFrame
            if isinstance(X, pd.DataFrame):
                # 如果是pandas DataFrame，直接使用它
                data = X
            else:
                # 否则转换为DataFrame
                if feature_names is None:
                    feature_names = [f"特征_{i}" for i in range(X.shape[1])]
                
                # 如果特征太多，只选择前30个
                if X.shape[1] > 30:
                    logger.info(f"特征数量太多({X.shape[1]})，只展示前30个特征的相关性")
                    X = X[:, :30]
                    feature_names = feature_names[:30]
                    
                data = pd.DataFrame(X, columns=feature_names)
            
            # 计算相关性矩阵
            corr = data.corr()
            
            # 设置图表
            plt.figure(figsize=figsize)
            
            # 使用Seaborn绘制热图
            mask = np.triu(np.ones_like(corr, dtype=bool))  # 只显示下三角
            sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
                        vmin=-1, vmax=1, square=True, linewidths=0.5)
            
            # 设置标题
            plt.title("特征相关性矩阵", fontproperties=chinese_font)
            
            # 设置坐标轴标签
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 保存图表
            save_path = self.plots_dir / "correlation_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性矩阵热图已保存至 {save_path}")
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"绘制相关性矩阵热图失败: {str(e)}")
            raise
            
    @log_function_call
    def plot_dim_reduction(self, X, y=None, method='pca', feature_names=None, figsize=(12, 10)):
        """使用降维方法可视化高维数据
        
        Args:
            X: 特征矩阵
            y: 标签数组（可选）
            method: 降维方法，'pca'或'tsne'
            feature_names: 特征名称列表（用于PCA载荷图）
            figsize: 图表大小
        """
        try:
            # 数据标准化
            X_scaled = StandardScaler().fit_transform(X)
            
            if method.lower() == 'pca':
                # 使用PCA降维
                pca = PCA(n_components=2)
                X_reduced = pca.fit_transform(X_scaled)
                
                # 设置图表
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                
                # 绘制散点图
                if y is not None:
                    scatter = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                                         alpha=0.7, cmap='tab10', edgecolors='k', s=50)
                    # 添加图例
                    legend1 = ax1.legend(*scatter.legend_elements(),
                                        title="类别", loc="best")
                    ax1.add_artist(legend1)
                else:
                    ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, edgecolors='k', s=50)
                
                # 设置标题和标签
                ax1.set_title("PCA降维结果", fontproperties=chinese_font)
                ax1.set_xlabel(f"主成分1 ({pca.explained_variance_ratio_[0]:.2%})", fontproperties=chinese_font)
                ax1.set_ylabel(f"主成分2 ({pca.explained_variance_ratio_[1]:.2%})", fontproperties=chinese_font)
                ax1.grid(True)
                
                # 绘制PCA载荷图（特征贡献图）
                if feature_names is None:
                    feature_names = [f"特征_{i}" for i in range(X.shape[1])]
                
                # 如果特征太多，只显示贡献最大的前15个
                if len(feature_names) > 15:
                    # 计算每个特征的重要性
                    importance = np.sum(np.abs(pca.components_[:2, :]), axis=0)
                    top_indices = np.argsort(importance)[-15:]
                    
                    # 筛选出重要的特征
                    filtered_loadings = pca.components_[:2, top_indices]
                    filtered_names = [feature_names[i] for i in top_indices]
                else:
                    filtered_loadings = pca.components_[:2, :]
                    filtered_names = feature_names
                
                # 绘制载荷图
                for i, (name, loading) in enumerate(zip(filtered_names, filtered_loadings.T)):
                    ax2.arrow(0, 0, loading[0], loading[1], head_width=0.05, head_length=0.05, 
                             fc='blue', ec='blue', alpha=0.5)
                    ax2.text(loading[0] * 1.1, loading[1] * 1.1, name, fontproperties=chinese_font)
                
                # 设置载荷图的坐标轴和标题
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.set_xlabel("主成分1载荷", fontproperties=chinese_font)
                ax2.set_ylabel("主成分2载荷", fontproperties=chinese_font)
                ax2.set_title("PCA特征贡献", fontproperties=chinese_font)
                ax2.grid(True)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                
                # 调整布局
                plt.tight_layout()
                
                # 保存图表
                save_path = self.plots_dir / "pca_visualization.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"PCA可视化图已保存至 {save_path}")
                plt.close()
                
            elif method.lower() == 'tsne':
                # 使用t-SNE降维
                tsne = TSNE(n_components=2, perplexity=min(30, X.shape[0]//2), random_state=42)
                X_reduced = tsne.fit_transform(X_scaled)
                
                # 设置图表
                plt.figure(figsize=(10, 8))
                
                # 绘制散点图
                if y is not None:
                    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
                                         alpha=0.7, cmap='tab10', edgecolors='k', s=50)
                    # 添加图例
                    legend1 = plt.legend(*scatter.legend_elements(),
                                        title="类别", loc="best")
                    plt.gca().add_artist(legend1)
                else:
                    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, edgecolors='k', s=50)
                
                # 设置标题和标签
                plt.title("t-SNE降维结果", fontproperties=chinese_font)
                plt.xlabel("t-SNE维度1", fontproperties=chinese_font)
                plt.ylabel("t-SNE维度2", fontproperties=chinese_font)
                plt.grid(True)
                plt.tight_layout()
                
                # 保存图表
                save_path = self.plots_dir / "tsne_visualization.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"t-SNE可视化图已保存至 {save_path}")
                plt.close()
                
            return save_path
        except Exception as e:
            logger.error(f"绘制降维可视化图失败: {str(e)}")
            raise
            
    @log_function_call
    def plot_feature_importance(self, feature_importance, feature_names=None, selected_features=None, figsize=(12, 8)):
        """绘制特征重要性图
        
        Args:
            feature_importance: 特征重要性数组或字典
            feature_names: 特征名称列表（可选）
            selected_features: 选定的特征索引（可选），用于突出显示
            figsize: 图表大小
        
        Returns:
            save_path: 保存的图像路径
        """
        try:
            # 处理输入数据
            if isinstance(feature_importance, dict):
                features = list(feature_importance.keys())
                importances = list(feature_importance.values())
            else:
                importances = feature_importance
                if feature_names is None:
                    features = [f"特征_{i}" for i in range(len(importances))]
                else:
                    features = feature_names[:len(importances)]
            
            # 转换为numpy数组以便操作
            importances = np.array(importances)
            
            # 如果特征太多，只显示前30个最重要的
            if len(importances) > 30:
                logger.info(f"特征数量太多({len(importances)})，只显示前30个最重要的特征")
                idx = np.argsort(importances)[-30:]
                importances = importances[idx]
                features = [features[i] for i in idx]
                
                # 调整selected_features以匹配新的索引
                if selected_features is not None:
                    selected_features = [i for i, f_idx in enumerate(idx) if f_idx in selected_features]
            
            # 创建排序索引
            sorted_idx = np.argsort(importances)
            
            # 设置图表
            plt.figure(figsize=figsize)
            
            # 创建颜色映射，选定的特征使用不同颜色
            colors = ['#1f77b4'] * len(importances)  # 默认蓝色
            if selected_features is not None:
                for i in range(len(importances)):
                    if i in selected_features:
                        colors[i] = '#ff7f0e'  # 选定的特征用橙色
            
            # 绘制水平条形图
            bars = plt.barh(np.arange(len(sorted_idx)), importances[sorted_idx], color=[colors[i] for i in sorted_idx])
            
            # 设置y轴标签
            plt.yticks(np.arange(len(sorted_idx)), [features[i] for i in sorted_idx])
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{importances[sorted_idx[i]]:.3f}', 
                         va='center', fontsize=10)
            
            # 添加图例
            if selected_features is not None:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#1f77b4', label='未选择特征'),
                    Patch(facecolor='#ff7f0e', label='选择的特征')
                ]
                plt.legend(handles=legend_elements, loc='lower right', prop=chinese_font)
            
            # 设置标题和标签
            plt.title("特征重要性", fontproperties=chinese_font)
            plt.xlabel("重要性", fontproperties=chinese_font)
            plt.ylabel("特征", fontproperties=chinese_font)
            plt.tight_layout()
            
            # 保存图表
            save_path = self.plots_dir / "feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存至 {save_path}")
            plt.close()
            
            # 如果提供了选定特征，还可以绘制特征分布图
            if selected_features is not None:
                # 提取选定特征和未选定特征的索引
                selected = np.zeros(len(importances), dtype=bool)
                for idx in selected_features:
                    if idx < len(selected):
                        selected[idx] = True
                
                # 分别计算选定特征和未选定特征的重要性分布
                selected_importance = importances[selected]
                not_selected_importance = importances[~selected]
                
                # 创建直方图
                plt.figure(figsize=(10, 6))
                
                # 绘制两个直方图
                if len(selected_importance) > 0:
                    plt.hist(selected_importance, bins=10, alpha=0.7, label='选定特征', color='#ff7f0e')
                if len(not_selected_importance) > 0:
                    plt.hist(not_selected_importance, bins=10, alpha=0.7, label='未选定特征', color='#1f77b4')
                
                # 添加图例和标题
                plt.legend(prop=chinese_font)
                plt.title("特征重要性分布对比", fontproperties=chinese_font)
                plt.xlabel("重要性", fontproperties=chinese_font)
                plt.ylabel("特征数量", fontproperties=chinese_font)
                plt.tight_layout()
                
                # 保存图表
                dist_save_path = self.plots_dir / "feature_importance_distribution.png"
                plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"特征重要性分布图已保存至 {dist_save_path}")
                plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"绘制特征重要性图失败: {str(e)}")
            raise

    @log_function_call
    def plot_feature_importance_comparison(self, teacher_importance, student_importance, 
                                          feature_names=None, figsize=(12, 8), bins=10):
        """绘制教师模型和学生模型的特征重要性分布对比图
        
        Args:
            teacher_importance: 教师模型的特征重要性数组
            student_importance: 学生模型的特征重要性数组
            feature_names: 特征名称列表（可选）
            figsize: 图表大小
            bins: 直方图的分箱数量
            
        Returns:
            save_path: 保存的图像路径
        """
        try:
            # 转换为numpy数组以便操作
            teacher_importance = np.array(teacher_importance)
            student_importance = np.array(student_importance)
            
            # 创建图形
            plt.figure(figsize=figsize)
            
            # 计算直方图的范围，使两个分布使用相同的范围
            min_val = min(teacher_importance.min(), student_importance.min())
            max_val = max(teacher_importance.max(), student_importance.max())
            bin_range = (min_val, max_val)
            
            # 绘制两个直方图
            plt.hist(teacher_importance, bins=bins, alpha=0.7, label='教师模型', color='#2166ac', range=bin_range, density=True)
            plt.hist(student_importance, bins=bins, alpha=0.7, label='学生模型', color='#b2182b', range=bin_range, density=True)
            
            # 添加图例和标题
            plt.legend(prop=chinese_font)
            plt.title("教师模型与学生模型特征重要性分布对比", fontproperties=chinese_font)
            plt.xlabel("特征重要性", fontproperties=chinese_font)
            plt.ylabel("密度", fontproperties=chinese_font)
            plt.grid(True, alpha=0.3)
            
            # 添加分布统计信息文本框
            teacher_stats = f"教师模型特征: {len(teacher_importance)}\n" \
                           f"平均重要性: {teacher_importance.mean():.3f}\n" \
                           f"最大重要性: {teacher_importance.max():.3f}"
            
            student_stats = f"学生模型特征: {len(student_importance)}\n" \
                           f"平均重要性: {student_importance.mean():.3f}\n" \
                           f"最大重要性: {student_importance.max():.3f}"
            
            # 放置文本框在右上角
            plt.text(0.98, 0.98, teacher_stats, transform=plt.gca().transAxes, 
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='#d1e5f0', alpha=0.8))
            
            plt.text(0.98, 0.78, student_stats, transform=plt.gca().transAxes, 
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='#fddbc7', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图表
            save_path = self.plots_dir / "feature_importance_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性分布对比图已保存至 {save_path}")
            plt.close()
            
            return save_path
        except Exception as e:
            logger.error(f"绘制特征重要性分布对比图失败: {str(e)}")
            plt.close()
            raise

    @log_function_call
    def plot_factor_analysis(self, model, feature_names=None, figsize=(14, 10)):
        """绘制因子分析图，用于多重损失因子化模型
        
        Args:
            model: 因子化模型实例
            feature_names: 特征名称列表，可选
            figsize: 图表大小
        
        Returns:
            保存的图表路径列表
        """
        try:
            import torch
            import numpy as np
            import seaborn as sns
            from matplotlib.colors import LinearSegmentedColormap
            
            # 获取模型原始对象
            if hasattr(model, 'get_model'):
                model_raw = model.get_model()
                if hasattr(model_raw, 'get_model'):
                    model_raw = model_raw.get_model()
            else:
                model_raw = model
            
            # 检查是否为因子化模型
            if not hasattr(model_raw, 'get_selected_factors') or not hasattr(model_raw, 'factor_weights'):
                logger.warning(f"提供的模型{model_raw}不是因子化模型，无法生成因子分析图")
                return []
            
            # 准备数据
            selected_factors = model_raw.get_selected_factors()
            gates = model_raw.factor_gate.detach().cpu().numpy() if isinstance(model_raw.factor_gate, torch.Tensor) else model_raw.factor_gate
            
            # 收集图表路径
            fig_paths = []
            
            # 1. 绘制因子热力图
            plt.figure(figsize=figsize)
            
            # 创建自定义的颜色映射，从蓝色(低值)到红色(高值)
            colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", 
                    "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
            custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
            
            if hasattr(model_raw, 'factor_weights'):
                # 获取因子权重
                factor_weights = model_raw.factor_weights.weight.detach().cpu().numpy() if isinstance(model_raw.factor_weights, torch.nn.Linear) else model_raw.factor_weights
                
                # 如果是多层模型，权重可能连接到隐藏层
                if hasattr(model_raw, 'feature_extractor') and hasattr(model_raw, 'hidden_sizes'):
                    # 获取最后一个隐藏层的大小
                    last_hidden_size = model_raw.hidden_sizes[-1]
                    
                    # 调整权重数据以适应热力图
                    if factor_weights.shape[0] == model_raw.max_factors:
                        # 权重形状为 [max_factors, last_hidden_size]
                        weights_for_heatmap = factor_weights[:, :last_hidden_size]
                        plt.title('因子与隐藏特征的关联热力图')
                        y_labels = [f"因子_{i}" for i in range(factor_weights.shape[0])]
                        x_labels = [f"隐藏特征_{i}" for i in range(last_hidden_size)]
                    else:
                        # 权重形状可能是 [last_hidden_size, max_factors]
                        weights_for_heatmap = factor_weights[:last_hidden_size, :].T
                        plt.title('因子与隐藏特征的关联热力图')
                        y_labels = [f"因子_{i}" for i in range(weights_for_heatmap.shape[0])]
                        x_labels = [f"隐藏特征_{i}" for i in range(weights_for_heatmap.shape[1])]
                else:
                    # 直接使用权重，假设是原始特征与因子的关联
                    if factor_weights.shape[0] == model_raw.max_factors:
                        weights_for_heatmap = factor_weights
                        plt.title('因子与特征的关联热力图')
                        y_labels = [f"因子_{i}" for i in range(factor_weights.shape[0])]
                        x_labels = [feature_names[i] if feature_names and i < len(feature_names) else f"特征_{i}" 
                                  for i in range(factor_weights.shape[1])]
                    else:
                        weights_for_heatmap = factor_weights.T
                        plt.title('因子与特征的关联热力图')
                        y_labels = [f"因子_{i}" for i in range(weights_for_heatmap.shape[0])]
                        x_labels = [feature_names[i] if feature_names and i < len(feature_names) else f"特征_{i}" 
                                  for i in range(weights_for_heatmap.shape[1])]
                
                # 只绘制选中的因子
                if selected_factors is not None and len(selected_factors) > 0:
                    weights_for_heatmap = weights_for_heatmap[selected_factors]
                    y_labels = [f"因子_{i}" for i in selected_factors]
                
                # 绘制热力图
                if isinstance(weights_for_heatmap, torch.Tensor):
                    weights_for_heatmap = weights_for_heatmap.detach().cpu().numpy()
                ax = sns.heatmap(weights_for_heatmap, cmap=custom_cmap, center=0, 
                               xticklabels=x_labels if len(x_labels) < 30 else [], 
                               yticklabels=y_labels)
                
                # 设置标签
                plt.xlabel('特征')
                plt.ylabel('因子')
                
                # 如果特征太多，则省略x刻度标签
                if len(x_labels) >= 30:
                    plt.xticks([])
                    plt.xlabel(f'特征 (共{len(x_labels)}个)')
                
                # 调整图表显示
                plt.tight_layout()
                
                # 保存图表
                fig_path = self.plots_dir / "factor_feature_heatmap.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"因子特征热力图已保存为: {fig_path}")
                fig_paths.append(fig_path)
            
            # 2. 绘制因子门控值条形图
            plt.figure(figsize=(10, 6))
            
            # 排序的门控值
            sorted_indices = np.argsort(gates)[::-1]
            sorted_gates = gates[sorted_indices]
            
            # 创建条形图
            bars = plt.bar(range(len(sorted_gates)), sorted_gates, 
                         color=['r' if i in selected_factors else 'b' for i in sorted_indices])
            
            # 设置图表属性
            plt.xlabel('因子索引 (按门控值排序)')
            plt.ylabel('门控值')
            plt.title('因子门控值分布')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            
            # 添加选择阈值线
            if hasattr(model_raw, 'gate_threshold'):
                threshold = model_raw.gate_threshold
                plt.axhline(y=threshold, color='g', linestyle='--', label=f'选择阈值 ({threshold:.2f})')
                plt.legend()
            
            # 美化图表
            plt.tight_layout()
            
            # 保存图表
            fig_path = self.plots_dir / "factor_gates.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"因子门控值图已保存为: {fig_path}")
            fig_paths.append(fig_path)
            
            # 3. 尝试绘制因子与原始特征的关联分析
            if hasattr(model, 'get_feature_importance'):
                # 获取特征重要性
                feature_importance = model.get_feature_importance()
                
                # 创建一个新的图表，展示每个选中因子的特征贡献
                plt.figure(figsize=(15, 10))
                
                # 对于多层模型，我们需要近似计算因子与原始特征的关联
                # 这里使用特征重要性作为简化，实际上应该使用完整的梯度分析
                
                # 显示前N个最重要的特征
                top_n = min(10, len(feature_importance))
                top_feature_indices = np.argsort(feature_importance)[::-1][:top_n]
                
                # 获取特征名称
                if feature_names is not None:
                    top_feature_names = [feature_names[i] if i < len(feature_names) else f"特征_{i}" for i in top_feature_indices]
                else:
                    top_feature_names = [f"特征_{i}" for i in top_feature_indices]
                
                # 绘制特征重要性与因子关联的桑基图或关系图
                # 由于桑基图实现复杂，这里简化为特征重要性柱状图，并标注关联的因子
                
                plt.barh(range(len(top_feature_indices)), feature_importance[top_feature_indices], color='skyblue')
                plt.yticks(range(len(top_feature_indices)), top_feature_names)
                plt.xlabel('特征重要性')
                plt.title('关键特征与因子关联分析')
                plt.grid(axis='x', linestyle='--', alpha=0.6)
                
                # 添加因子关联标注
                for i, idx in enumerate(top_feature_indices):
                    importance = feature_importance[idx]
                    plt.text(importance + max(feature_importance) * 0.02, i, 
                           f"关联因子: {', '.join([f'F{f}' for f in selected_factors[:min(3, len(selected_factors))]])}", 
                           va='center')
                
                plt.tight_layout()
                
                # 保存图表
                fig_path = self.plots_dir / "factor_feature_association.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"因子特征关联分析图已保存为: {fig_path}")
                fig_paths.append(fig_path)
            
            return fig_paths
            
        except Exception as e:
            logger.error(f"绘制因子分析图失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    @log_function_call
    def plot_from_json(self, json_path):
        """从JSON文件中读取特征重要性数据并绘制图表
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            save_path: 保存的图像路径
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                logger.warning("JSON文件为空或格式不正确")
                return None
                
            # 提取特征名称和重要性
            feature_names = [item['name'] for item in data]
            importance_scores = [item['importance'] for item in data]
            
            # 绘制特征重要性图
            save_path = self.plot_feature_importance(importance_scores, feature_names)
            logger.info(f"从JSON文件绘制的特征重要性图已保存至 {save_path}")
            
            return save_path
        except Exception as e:
            logger.error(f"从JSON文件绘制特征重要性图失败: {str(e)}")
            raise

    @log_function_call
    def plot_pr_curve(self, y_true, y_prob_teacher, y_prob_student=None, figsize=(10, 8)):
        """绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_prob_teacher: 教师模型预测概率
            y_prob_student: 学生模型预测概率(可选)
            figsize: 图像大小
        """
        try:
            # 确保在绘图前设置中文字体
            setup_chinese_fonts()
            
            plt.figure(figsize=figsize)
            
            # 计算教师模型的PR曲线
            precision_teacher, recall_teacher, _ = precision_recall_curve(y_true, y_prob_teacher)
            ap_teacher = average_precision_score(y_true, y_prob_teacher)
            
            # 绘制教师模型的PR曲线
            plt.plot(
                recall_teacher, precision_teacher, 
                color='blue', linestyle='-', linewidth=2,
                label=f'教师模型 (AP = {ap_teacher:.4f})'
            )
            
            # 如果提供了学生模型的预测，也绘制它的PR曲线
            if y_prob_student is not None:
                precision_student, recall_student, _ = precision_recall_curve(y_true, y_prob_student)
                ap_student = average_precision_score(y_true, y_prob_student)
                
                plt.plot(
                    recall_student, precision_student, 
                    color='red', linestyle='-', linewidth=2,
                    label=f'学生模型 (AP = {ap_student:.4f})'
                )
            
            # 添加无技巧分类器基准线
            no_skill = len(y_true[y_true == 1]) / len(y_true)
            plt.plot(
                [0, 1], [no_skill, no_skill], 
                color='grey', linestyle='--', linewidth=1.5,
                label=f'无技巧分类器 ({no_skill:.4f})'
            )
            
            # 设置图表元素
            plt.xlabel('召回率', fontproperties=chinese_font)
            plt.ylabel('精确率', fontproperties=chinese_font)
            plt.title('精确率-召回率曲线', fontproperties=chinese_font)
            plt.legend(loc='best', prop=chinese_font)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            save_path = self.plots_dir / "precision_recall_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"精确率-召回率曲线已保存至 {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"绘制精确率-召回率曲线失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            plt.close('all')
            return None