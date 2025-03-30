import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call, log_step
from utils.metrics import ModelEvaluator
from utils.config import config
from utils.metrics import ModelEvaluator as MetricsCalculator

# 导入中文字体设置函数（如果还不存在）
try:
    from utils.matplotlib_config import setup_chinese_fonts
except ImportError:
    # 如果无法导入，重新定义该函数
    from matplotlib import font_manager
    
    def setup_chinese_fonts():
        """设置matplotlib中文字体支持"""
        # 尝试找到系统中的中文字体
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'KaiTi', 'NSimSun', 'FangSong',
            'Arial Unicode MS', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Source Han Sans CN', 
            'Source Han Sans SC', 'Noto Sans CJK SC', 'Noto Sans SC', 'Noto Sans CJK TC', 
            'Droid Sans Fallback', 'Microsoft JhengHei'
        ]
        found_font = None
        
        # 优先尝试加载内置字体（解决Windows上的问题）
        try:
            # 查找系统中的字体文件
            import platform
            system = platform.system()
            
            if system == 'Windows':
                # Windows系统常见中文字体路径
                potential_font_paths = [
                    r'C:\Windows\Fonts\simhei.ttf',
                    r'C:\Windows\Fonts\msyh.ttc',
                    r'C:\Windows\Fonts\simsun.ttc'
                ]
                
                for font_path in potential_font_paths:
                    try:
                        if os.path.exists(font_path):
                            font_prop = font_manager.FontProperties(fname=font_path)
                            plt.rcParams['font.family'] = font_prop.get_name()
                            logger.info(f"成功加载Windows中文字体: {font_path}")
                            return True
                    except:
                        continue
            
            elif system == 'Linux':
                # Linux系统常见中文字体路径
                potential_font_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                    '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'
                ]
                
                for font_path in potential_font_paths:
                    try:
                        if os.path.exists(font_path):
                            font_prop = font_manager.FontProperties(fname=font_path)
                            plt.rcParams['font.family'] = font_prop.get_name()
                            logger.info(f"成功加载Linux中文字体: {font_path}")
                            return True
                    except:
                        continue
            
            elif system == 'Darwin':  # MacOS
                # MacOS系统常见中文字体路径
                potential_font_paths = [
                    '/System/Library/Fonts/PingFang.ttc',
                    '/System/Library/Fonts/STHeiti Light.ttc',
                    '/System/Library/Fonts/STHeiti Medium.ttc'
                ]
                
                for font_path in potential_font_paths:
                    try:
                        if os.path.exists(font_path):
                            font_prop = font_manager.FontProperties(fname=font_path)
                            plt.rcParams['font.family'] = font_prop.get_name()
                            logger.info(f"成功加载MacOS中文字体: {font_path}")
                            return True
                    except:
                        continue
        except Exception as e:
            logger.warning(f"直接加载字体文件失败: {str(e)}")
        
        # 回退方法：按名称查找字体
        for font in chinese_fonts:
            try:
                font_path = font_manager.findfont(font_manager.FontProperties(family=font))
                if font_path and font_path.strip():
                    found_font = font
                    plt.rcParams['font.family'] = found_font
                    logger.info(f"使用中文字体: {found_font}")
                    break
            except:
                continue
        
        if not found_font:
            # 最终回退方法：使用sans-serif字体族
            try:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Source Han Sans CN', 'Source Han Sans SC', 'Noto Sans SC', 'Microsoft YaHei'] + plt.rcParams.get('font.sans-serif', [])
                logger.info("使用sans-serif字体族")
            except:
                logger.warning("无法设置中文字体，图形中的中文可能无法正确显示")
        
        # 修复负号显示
        plt.rcParams['axes.unicode_minus'] = False
        
        # 确保字体缓存更新
        try:
            font_manager._rebuild()
            logger.info("已重新构建字体缓存")
        except:
            pass
        
        return True

# 初始化中文字体设置
setup_chinese_fonts()

class ModelEvaluator:
    def __init__(self):
        """初始化模型评估器"""
        self.results_dir = config.results_dir / "evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        
        # 确保每次创建实例时都应用中文字体设置
        setup_chinese_fonts()
    
    @log_step("评估模型性能")
    def evaluate_model(self, model, X, y, model_name="model"):
        """评估单个模型性能
        
        Args:
            model: 模型对象
            X: 特征数据
            y: 标签数据
            model_name: 模型名称
        
        Returns:
            包含评估指标的字典
        """
        try:
            # 预测
            y_pred = model.predict(X)
            y_prob = None
            
            try:
                y_prob = model.predict_proba(X)
            except (AttributeError, NotImplementedError):
                logger.warning(f"{model_name} 不支持概率预测")
            
            # 使用工具类计算指标，直接传递原始标签
            metrics = MetricsCalculator.calculate_metrics(y, y_pred, y_prob)
            
            # 保存指标
            self.metrics[model_name] = metrics
            
            # 输出评估结果
            logger.info(f"{model_name} 评估结果:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"评估 {model_name} 时出错: {str(e)}")
            raise
    
    @log_step("比较模型性能")
    def compare_models(self, teacher_model, student_model, X, y):
        """比较教师模型和学生模型性能
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            X: 特征数据
            y: 标签数据
        
        Returns:
            包含比较结果的字典
        """
        # 确保y是一维数组，如果是二维的话
        y_1d = np.argmax(y, axis=1) if y.ndim > 1 else y
        
        # 评估两个模型
        teacher_metrics = self.evaluate_model(teacher_model, X, y_1d, "teacher")
        student_metrics = self.evaluate_model(student_model, X, y_1d, "student")
        
        # 预测一致性评估
        teacher_pred = teacher_model.predict(X)
        student_pred = student_model.predict(X)
        
        # 确保预测结果是一维数组
        teacher_pred_1d = np.argmax(teacher_pred, axis=1) if teacher_pred.ndim > 1 else teacher_pred
        student_pred_1d = np.argmax(student_pred, axis=1) if student_pred.ndim > 1 else student_pred
        
        # 计算预测一致性
        consistency = np.mean(teacher_pred_1d == student_pred_1d)
        logger.info(f"预测一致性: {consistency:.4f}")
        
        # 创建比较表
        comparison = {}
        for metric in teacher_metrics:
            if metric in student_metrics:
                diff = student_metrics[metric] - teacher_metrics[metric]
                comparison[metric] = {
                    'teacher': teacher_metrics[metric],
                    'student': student_metrics[metric],
                    'diff': diff,
                    'relative_diff': diff / teacher_metrics[metric] if teacher_metrics[metric] != 0 else 0
                }
        
        # 保存比较结果
        self._save_comparison_results(comparison, consistency)
        
        return {
            'teacher': teacher_metrics,
            'student': student_metrics,
            'consistency': consistency,
            'comparison': comparison
        }
    
    @log_function_call
    def _save_comparison_results(self, comparison, consistency):
        """保存比较结果
        
        Args:
            comparison: 比较结果字典
            consistency: 预测一致性
        """
        # 创建DataFrame
        data = []
        for metric, values in comparison.items():
            data.append({
                'Metric': metric,
                'Teacher': values['teacher'],
                'Student': values['student'],
                'Difference': values['diff'],
                'Relative Difference (%)': values['relative_diff'] * 100
            })
        
        df = pd.DataFrame(data)
        
        # 保存为CSV
        csv_path = self.results_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"模型比较结果已保存至 {csv_path}")
        
        # 创建可视化
        self._create_comparison_visualization(df, consistency)
    
    @log_function_call
    def _create_comparison_visualization(self, comparison_df, consistency):
        """创建比较可视化
        
        Args:
            comparison_df: 比较结果DataFrame
            consistency: 预测一致性
        """
        try:
            # 引入所需模块
            from matplotlib.font_manager import FontProperties
            from utils.matplotlib_config import get_chinese_font
            
            # 应用中文字体设置 - 确保在每次绘图前都重新设置
            from utils.matplotlib_config import setup_chinese_fonts
            setup_chinese_fonts()
            
            # 获取中文字体属性对象
            chinese_font = get_chinese_font()
            
            # 设置绘图风格
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制柱状图
            metrics = comparison_df['Metric'].tolist()
            teacher_values = comparison_df['Teacher'].tolist()
            student_values = comparison_df['Student'].tolist()
            
            x = np.arange(len(metrics))
            width = 0.35
            
            teacher_bar = ax.bar(x - width/2, teacher_values, width, label='教师模型')
            student_bar = ax.bar(x + width/2, student_values, width, label='学生模型')
            
            # 添加标签和标题 - 使用更通用的字体设置方式
            ax.set_xlabel('指标', fontproperties=chinese_font)
            ax.set_ylabel('得分', fontproperties=chinese_font)
            ax.set_title(f'模型性能比较 (预测一致性: {consistency:.4f})', 
                        fontproperties=chinese_font)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            
            # 图例中也设置字体
            legend = ax.legend(prop=chinese_font)
            
            plt.tight_layout()
            
            # 保存图像
            plot_path = self.results_dir / "model_comparison.png"
            
            # 使用额外的dpi参数以及更好的字体抗锯齿设置
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1, 
                       facecolor='white', edgecolor='none', transparent=False)
            plt.close(fig)
            
            logger.info(f"模型比较可视化已保存至 {plot_path}")
            
        except Exception as e:
            logger.error(f"创建比较可视化时出错: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            plt.close('all')
    
    @log_step("分析模型复杂度")
    def analyze_model_complexity(self, teacher_model, student_model):
        """分析模型复杂度
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
        
        Returns:
            包含复杂度分析的字典
        """
        try:
            # 从底层模型获取复杂度指标
            
            teacher_complexity = MetricsCalculator.calculate_model_complexity(teacher_model.get_model())
            student_complexity = MetricsCalculator.calculate_model_complexity(student_model.get_model())
            
            # 计算模型大小比率
            teacher_size = teacher_model.get_model_size()
            student_size = student_model.get_model_size()
            
            size_ratio = student_size / teacher_size if teacher_size > 0 else 0
            
            logger.info(f"模型大小比较:")
            logger.info(f"  教师模型: {teacher_size}")
            logger.info(f"  学生模型: {student_size}")
            logger.info(f"  大小比率: {size_ratio:.4f}")
            
            return {
                'teacher': teacher_complexity,
                'student': student_complexity,
                'size_ratio': size_ratio
            }
            
        except Exception as e:
            logger.error(f"分析模型复杂度时出错: {str(e)}")
            raise 