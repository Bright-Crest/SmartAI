import argparse
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置路径
root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(root_dir))

# 导入中文字体配置（确保在导入其他模块前设置）
from utils.matplotlib_config import setup_chinese_fonts

# 重新应用中文字体设置
setup_chinese_fonts()

from core.data.data_loader import DataLoader
from core.distillation.distiller import Distiller
from core.models.model_registry import ModelRegistry
from core.models.pytorch_wrappers import PyTorchBaseWrapper
from utils.config import config
from utils.logger import logger, log_step
from visualization.plotters import VisualizationManager
from visualization.report_generator import ReportGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='知识蒸馏训练流程')
    
    # 模型选项
    parser.add_argument('--teacher', type=str, default='xgboost',
                      choices=list(ModelRegistry.get_available_models('teacher').keys()),
                      help='教师模型类型')
    parser.add_argument('--student', type=str, default='logistic',
                      choices=list(ModelRegistry.get_available_models('student').keys()),
                      help='学生模型类型')
    parser.add_argument('--data-path', type=str, required=True,
                      help='训练数据路径')
    parser.add_argument('--epochs', type=int, default=None,
                      help='训练轮数，如果为None则使用配置默认值')
    parser.add_argument('--alpha', type=float, default=None,
                      help='软标签权重，如果为None则使用配置默认值')
    parser.add_argument('--temperature', type=float, default=None,
                      help='蒸馏温度参数，如果为None则使用配置默认值')
    
    # 特征选择选项
    parser.add_argument('--enable-factor-selection', action='store_true',
                      help='启用特征因子选择')
    parser.add_argument('--disable-factor-selection', action='store_true',
                      help='禁用特征因子选择')
    parser.add_argument('--max-factor-ratio', type=float, default=None,
                      help='最大特征因子比例，范围[0,1]')
    parser.add_argument('--factor-temperature', type=float, default=None,
                      help='特征选择温度参数，控制选择的软硬程度')
    parser.add_argument('--l1-lambda', type=float, default=None,
                      help='L1正则化系数，控制特征稀疏度')
    
    # 多重损失模型参数
    parser.add_argument('--max-factors', type=int, default=None,
                      help='最大因子数量，用于因子化模型')
    parser.add_argument('--triplet-margin', type=float, default=None,
                      help='三元组损失边界值，用于多重损失模型')
    parser.add_argument('--sep-lambda', type=float, default=None,
                      help='分离损失权重，用于多重损失模型')
    
    # 控制选项
    parser.add_argument('--no-visualization', action='store_true',
                      help='禁用可视化')
    parser.add_argument('--seed', type=int,
                      help='随机种子')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='输出目录，如果为None则使用时间戳创建目录')
    
    # 运行模式选项
    parser.add_argument('--online-mode', action='store_true',
                      help='启用在线模式')
    parser.add_argument('--load-teacher', type=str, default=None,
                      help='加载教师模型路径')
    parser.add_argument('--load-student', type=str, default=None,
                      help='加载学生模型路径')
    
    # 数据特征选项
    parser.add_argument('--feature-names', type=str, default=None,
                      help='特征名称文件路径，CSV或JSON格式，用于增强报告可读性')
    
    parser.add_argument('--use-shap', action='store_true',
                      help='使用SHAP解释模型')

    return parser.parse_args()

def setup_environment(args):
    """设置环境，包括随机种子、输出目录等
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    if args.seed is None:
        args.seed = np.random.randint(1000000)
        logger.info(f"未指定随机种子，使用随机种子: {args.seed}")
    np.random.seed(args.seed)
    
    # 确保中文字体设置生效
    setup_chinese_fonts()
    
    # 设置输出目录
    if args.output_dir:
        results_dir = Path(args.output_dir)
        # 更新配置
        config.update_results_dir(results_dir)
        # 创建必要的目录
        config.make_dirs()
    
    # 更新配置参数
    config.update(**{
        "model_config.teacher.name": args.teacher,
        "model_config.student.name": args.student,
        "use_shap": args.use_shap
    })
    
    # 更新训练参数
    if args.epochs is not None:
        config.train_config["epochs"] = args.epochs
    if args.alpha is not None:
        config.train_config["alpha"] = args.alpha
    if args.temperature is not None:
        config.train_config["temperature"] = args.temperature
    
    # 更新特征选择参数
    if args.enable_factor_selection:
        config.train_config["feature_selection"]["enable"] = True
    elif args.disable_factor_selection:
        config.train_config["feature_selection"]["enable"] = False
    
    if args.max_factor_ratio is not None:
        config.train_config["feature_selection"]["max_factor_ratio"] = args.max_factor_ratio
    
    if args.factor_temperature is not None:
        config.train_config["feature_selection"]["factor_temperature"] = args.factor_temperature
    
    if args.l1_lambda is not None:
        config.train_config["feature_selection"]["l1_lambda"] = args.l1_lambda
    
    # 更新多重损失模型参数
    if args.max_factors is not None:
        config.train_config["multi_loss"] = config.train_config.get("multi_loss", {})
        config.train_config["multi_loss"]["max_factors"] = args.max_factors
    
    if args.triplet_margin is not None:
        config.train_config["multi_loss"] = config.train_config.get("multi_loss", {})
        config.train_config["multi_loss"]["triplet_margin"] = args.triplet_margin
    
    if args.sep_lambda is not None:
        config.train_config["multi_loss"] = config.train_config.get("multi_loss", {})
        config.train_config["multi_loss"]["sep_lambda"] = args.sep_lambda
    
    # 记录配置信息
    logger.info(f"配置信息:")
    logger.info(f"  结果目录: {config.results_dir}")
    logger.info(f"  教师模型: {config.get_model_name('teacher')}")
    logger.info(f"  学生模型: {config.get_model_name('student')}")
    logger.info(f"  训练参数: epochs={config.train_config['epochs']}, "
                f"alpha={config.train_config['alpha']}, "
                f"temperature={config.train_config['temperature']}")
    
    # 记录特征选择配置
    feature_selection_config = config.get_feature_selection_config()
    logger.info(f"  特征选择: 启用={feature_selection_config['enable']}, "
                f"max_factor_ratio={feature_selection_config['max_factor_ratio']}, "
                f"factor_temperature={feature_selection_config['factor_temperature']}, "
                f"l1_lambda={feature_selection_config['l1_lambda']}")
    

def get_model_specific_params(model_type, **kwargs):
    """根据模型类型筛选适合的参数
    
    Args:
        model_type: 模型类型 (xgboost, lightgbm, logistic, mlp等)
        **kwargs: 所有可能的参数
        
    Returns:
        筛选后的参数字典
    """
    # 不同模型支持的参数
    model_params = {
        # 树模型参数
        'xgboost': ['n_estimators', 'max_depth', 'learning_rate', 'objective', 'eval_set', 'early_stopping_rounds'],
        'lightgbm': ['n_estimators', 'max_depth', 'learning_rate', 'objective', 'eval_set', 'early_stopping_rounds'],
        'random_forest': ['n_estimators', 'max_depth', 'criterion', 'max_features'],
        'catboost': ['iterations', 'depth', 'learning_rate', 'eval_set'],
        
        # 线性模型参数
        'logistic': ['C', 'penalty', 'solver', 'max_iter', 'alpha', 'temperature'],
        'svm': ['C', 'kernel', 'gamma', 'probability', 'alpha', 'temperature'],
        'mlp': ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'learning_rate', 'max_iter', 'temperature'],
        'decision_tree': ['max_depth', 'criterion', 'min_samples_split'],
        
        # PyTorch模型参数
        'pytorch_mlp': ['input_size', 'hidden_sizes', 'n_classes', 'dropout_rate', 'learning_rate', 
                        'enable_factor_selection', 'max_factor_ratio', 'factor_temperature'],
        'pytorch_cnn': ['input_channels', 'img_size', 'n_classes', 'learning_rate', 
                        'enable_factor_selection', 'max_factor_ratio', 'factor_temperature'],
        'pytorch_factor_mlp': ['input_size', 'hidden_sizes', 'n_classes', 'dropout_rate', 'learning_rate', 
                              'max_factor_ratio', 'factor_temperature'],
        'pytorch_factorized_mlp': ['input_size', 'hidden_sizes', 'n_classes', 'dropout_rate', 'learning_rate',
                                  'max_factors', 'factor_temperature'],
    }
    
    # 只保留当前模型支持的参数
    supported_params = model_params.get(model_type, [])
    
    # 特殊情况：所有模型都支持的通用参数 
    # 注意：random_state应该在模型初始化时使用，而不是在fit方法中
    # 但这里我们还是添加verbose，因为大多数模型的fit方法支持它
    supported_params.append('verbose')
    
    # 蒸馏特有参数
    if model_type in ['logistic', 'mlp', 'svm']:
        supported_params.extend(['alpha', 'temperature', 'teacher_probs'])
    
    # PyTorch模型蒸馏参数
    if model_type in ['pytorch_mlp', 'pytorch_cnn', 'pytorch_factor_mlp']:
        supported_params.extend(['alpha', 'temperature', 'teacher_probs', 'l1_lambda'])
    
    # 多重损失模型参数
    if model_type in ['pytorch_factorized_mlp']:
        supported_params.extend(['alpha', 'temperature', 'teacher_probs', 'l1_lambda', 
                               'triplet_margin', 'sep_lambda', 'max_factors'])
    
    # 筛选参数
    filtered_params = {k: v for k, v in kwargs.items() if k in supported_params}
    
    return filtered_params

def load_feature_names(file_path):
    """加载特征名称文件
    
    Args:
        file_path: 特征名称文件路径，支持CSV或JSON格式
    
    Returns:
        特征名称列表
    """
    if file_path is None:
        return None
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"特征名称文件不存在: {file_path}")
        return None
    
    try:
        # 根据文件扩展名选择加载方式
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
            # 假设特征名在第一列或名为'feature_name'的列
            if 'feature_name' in df.columns:
                feature_names = df['feature_name'].tolist()
            else:
                feature_names = df.iloc[:, 0].tolist()
            
        elif file_path.suffix.lower() == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 支持多种JSON格式
            if isinstance(data, list):
                # 如果是列表，直接使用
                feature_names = data
            elif isinstance(data, dict):
                # 如果是字典，尝试获取特征名称
                if 'feature_names' in data:
                    feature_names = data['feature_names']
                elif 'features' in data:
                    feature_names = data['features']
                else:
                    # 直接使用键作为特征名
                    feature_names = list(data.keys())
            else:
                logger.warning(f"无法解析JSON文件: {file_path}")
                return None
        else:
            logger.warning(f"不支持的文件格式: {file_path}")
            return None
        
        logger.info(f"已加载 {len(feature_names)} 个特征名称")
        return feature_names
    
    except Exception as e:
        logger.error(f"加载特征名称文件失败: {str(e)}")
        return None

@log_step("运行完整蒸馏流程")
def run_pipeline(args):
    """运行完整的蒸馏流程"""
    start_time = time.time()
    
    try:
        # 加载数据
        logger.info(f"从 {args.data_path} 加载数据")
        data_loader = DataLoader()
        X, y = data_loader.load_data(args.data_path)
        data_loader.save_data_summary(X, y)
        
        # 加载特征名称
        feature_names = load_feature_names(args.feature_names)
        
        # 数据分割和预处理
        logger.info("数据分割和预处理")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.preprocess_data(X, y)
        
        # 创建蒸馏器
        distiller = Distiller()
        
        # 在线模式：加载预训练模型
        if args.online_mode:
            logger.info("在线模式：加载预训练模型")
            distiller.initialize_models()
            distiller.load_models(args.load_teacher, args.load_student)
        else:
            # 准备训练参数
            train_params = {
                'random_state': args.seed
            }
            
            # 针对神经网络模型，添加epochs参数
            if config.get_model_name('student') in ['mlp']:
                train_params['max_iter'] = config.train_config['epochs']
            
            # 训练教师模型
            logger.info("训练教师模型")
            teacher_params = get_model_specific_params(
                config.get_model_name('teacher'), 
                eval_set=[(X_val, y_val)],
                **train_params
            )
            
            distiller.train_teacher(
                X_train, y_train,
                X_val=X_val,
                y_val=y_val,
                **teacher_params
            )
            
            # 准备蒸馏参数
            distill_params = {
                'alpha': config.train_config['alpha'],
                'temperature': config.train_config['temperature'],
                'random_state': args.seed
            }
            
            # 获取特征选择配置
            feature_selection_config = config.get_feature_selection_config()

            # 获取多重损失配置（如果存在）
            multi_loss_config = {}
            if hasattr(config, 'get_multi_loss_config'):
                multi_loss_config = config.get_multi_loss_config()
            
            # 如果是PyTorch模型，添加特征选择参数
            if config.get_model_name('student') in ['pytorch_mlp', 'pytorch_cnn', 'pytorch_factor_mlp']:
                distill_params.update({
                    'enable_factor_selection': feature_selection_config['enable'],
                    'max_factor_ratio': feature_selection_config['max_factor_ratio'],
                    'factor_temperature': feature_selection_config['factor_temperature'],
                    'l1_lambda': feature_selection_config['l1_lambda']
                })
            
            # 如果是多重损失因子化模型，添加相关参数
            if config.get_model_name('student') in ['pytorch_factorized_mlp']:
                distill_params.update({
                    'max_factors': multi_loss_config.get('max_factors'),
                    'factor_temperature': feature_selection_config['factor_temperature'],
                    'l1_lambda': feature_selection_config['l1_lambda'],
                    'triplet_margin': multi_loss_config.get('triplet_margin', 1.0),
                    'sep_lambda': multi_loss_config.get('sep_lambda', 0.1)
                })
                        
            # 为了支持动态调整输入维度，添加input_size参数
            if config.get_model_name('student') in ['pytorch_mlp', 'pytorch_factor_mlp', 'pytorch_factorized_mlp']:
                distill_params['input_size'] = X_train.shape[1]
            
            # 针对CNN模型，推断图像尺寸
            if config.get_model_name('student') in ['pytorch_cnn']:
                # 尝试推断图像尺寸（假设是方形图像）
                if len(X_train.shape) == 2:  # 扁平化数据
                    img_side = int(np.sqrt(X_train.shape[1]))
                    if img_side * img_side == X_train.shape[1]:
                        distill_params['img_size'] = img_side
                        distill_params['input_channels'] = 1
                    else:
                        logger.warning(f"无法自动推断图像尺寸，请手动设置img_size和input_channels参数")
                elif len(X_train.shape) == 4:  # 已经是图像格式 [N, C, H, W]
                    distill_params['img_size'] = X_train.shape[2]
                    distill_params['input_channels'] = X_train.shape[1]
            
            # 针对神经网络模型，添加epochs参数
            if config.get_model_name('student') in ['pytorch_mlp', 'pytorch_cnn', 'pytorch_factor_mlp', 'mlp']:
                distill_params['epochs'] = config.train_config['epochs']
            
            # 为学生模型筛选参数
            student_params = get_model_specific_params(
                config.get_model_name('student'), 
                **distill_params
            )
            
            # 执行知识蒸馏
            logger.info("执行知识蒸馏")
            distiller.distill(
                X_train, y_train,
                X_val=X_val,
                y_val=y_val,
                **student_params
            )
            
            # 保存模型
            logger.info("保存模型")
            distiller.save_models()
        
        # 在测试集上进行最终评估
        logger.info("在测试集上进行最终评估")
        evaluation_results = distiller._evaluate_and_compare(X_test, y_test)
        
        # 可视化和报告生成
        if not args.no_visualization:
            visualize_and_report(X_train, y_train, X_test, y_test, distiller, feature_names)
        
        # 记录总耗时
        total_time = time.time() - start_time
        logger.info(f"蒸馏流程完成，总耗时: {total_time:.2f} 秒")
        
        return distiller, evaluation_results
        
    except Exception as e:
        logger.error(f"蒸馏流程失败: {str(e)}")
        import traceback
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        raise

@log_step("可视化和报告生成")
def visualize_and_report(X_train, y_train, X_test, y_test, distiller, feature_names=None):
    """执行数据可视化和生成报告"""
    try:
        # 创建可视化管理器
        viz_manager = VisualizationManager()
        
        # 数据可视化
        logger.info("生成数据可视化")
        try:
            # 使用新的方法名
            viz_manager.plot_feature_distribution(X_train, feature_names)
            viz_manager.plot_correlation_matrix(X_train, feature_names)
        except Exception as e:
            logger.warning(f"特征分布可视化失败: {str(e)}")
            # 尝试使用旧的方法名（向后兼容）
            try:
                viz_manager.plot_feature_distributions(X_train)
                viz_manager.plot_correlation_matrix(X_train)
            except Exception as e2:
                logger.warning(f"特征分布可视化完全失败: {str(e2)}")
        
        # 降维可视化
        try:
            logger.info("生成降维可视化")
            # 使用新的方法名
            viz_manager.plot_dim_reduction(X_train, y_train, method='pca', feature_names=feature_names)
            # 如果样本数足够多，也可以使用t-SNE
            if X_train.shape[0] >= 50:  # t-SNE需要一定数量的样本
                viz_manager.plot_dim_reduction(X_train, y_train, method='tsne')
        except Exception as e:
            logger.warning(f"降维可视化失败: {str(e)}")
            # 尝试使用旧的方法名（向后兼容）
            try:
                viz_manager.plot_dimension_reduction(X_train, y_train, method='pca')
                if X_train.shape[0] >= 50:
                    viz_manager.plot_dimension_reduction(X_train, y_train, method='tsne')
            except Exception as e2:
                logger.warning(f"降维可视化完全失败: {str(e2)}")
        
        # 特征重要性可视化（如果学生模型支持）
        if hasattr(distiller.student, 'get_feature_importance') and callable(distiller.student.get_feature_importance):
            try:
                logger.info("生成特征重要性可视化")
                feature_importance = distiller.student.get_feature_importance()
                
                # 如果学生模型支持获取选择的特征
                selected_features = None
                if hasattr(distiller.student, 'get_selected_features') and callable(distiller.student.get_selected_features):
                    selected_features = distiller.student.get_selected_features()
                
                viz_manager.plot_feature_importance(
                    feature_importance, 
                    feature_names=feature_names,
                    selected_features=selected_features
                )
            except Exception as e:
                logger.warning(f"特征重要性可视化失败: {str(e)}")
        
        # 如果是多重损失因子化模型，添加因子分析可视化
        if hasattr(distiller.student, 'model_name') and distiller.student.model_name == 'pytorch_factorized_mlp':
            try:
                student = distiller.student.get_model() if isinstance(distiller.student.get_model(), PyTorchBaseWrapper) else distiller.student
                logger.info(f"对{distiller.student}生成因子分析可视化")
                viz_manager.plot_factor_analysis(
                    student,
                    feature_names=feature_names
                )
            except Exception as e:
                logger.warning(f"因子分析可视化失败: {str(e)}")
                import traceback
                logger.warning(f"错误详情: {traceback.format_exc()}")
        
        # 决策边界可视化
        if X_train.shape[1] >= 2:  # 至少需要2个特征
            try:
                logger.info("生成决策边界可视化")
                
                # 获取教师和学生模型对象
                teacher_model = distiller.teacher.get_model()
                student_model = distiller.student.get_model()
                
                # 记录模型类型
                logger.info(f"教师模型类型: {type(teacher_model)}")
                logger.info(f"学生模型类型: {type(student_model)}")
                
                # 尝试多个特征组合
                # 1. 首先尝试前两个特征
                try:
                    logger.info("尝试使用前两个特征绘制决策边界")
                    # 对于教师模型
                    viz_manager.plot_decision_boundary(
                        X_test, y_test, 
                        teacher_model,
                        feature_idx=[0, 1],
                        figsize=(10, 8)
                    )
                    
                    # 对于学生模型
                    viz_manager.plot_decision_boundary(
                        X_test, y_test, 
                        student_model,
                        feature_idx=[0, 1],
                        figsize=(10, 8)
                    )
                except Exception as e:
                    logger.warning(f"使用前两个特征绘制决策边界失败: {str(e)}")
                
                # 2. 如果学生模型支持特征选择，尝试使用最重要的两个特征
                if hasattr(distiller.student, 'get_feature_importance') and callable(distiller.student.get_feature_importance):
                    try:
                        feature_importance = distiller.student.get_feature_importance()
                        # 找出最重要的两个特征
                        top_indices = np.argsort(feature_importance)[-2:]
                        
                        logger.info(f"尝试使用最重要的两个特征绘制决策边界: {top_indices}")
                        # 对于教师模型
                        viz_manager.plot_decision_boundary(
                            X_test, y_test, 
                            teacher_model,
                            feature_idx=top_indices,
                            figsize=(10, 8)
                        )
                        
                        # 对于学生模型
                        viz_manager.plot_decision_boundary(
                            X_test, y_test, 
                            student_model,
                            feature_idx=top_indices,
                            figsize=(10, 8)
                        )
                    except Exception as e:
                        logger.warning(f"使用最重要特征绘制决策边界失败: {str(e)}")
                
                # 3. 如果数据维度较高，尝试使用PCA降维后的前两个主成分
                if X_test.shape[1] > 5:
                    try:
                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        from matplotlib.font_manager import FontProperties
                        
                        logger.info("尝试使用PCA降维后的特征绘制决策边界")
                        # 标准化数据
                        X_scaled = StandardScaler().fit_transform(X_test)
                        # PCA降维
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # 记录解释方差比例
                        variance_ratio = pca.explained_variance_ratio_
                        logger.info(f"PCA解释方差比例: {variance_ratio}")
                        
                        # 获取中文字体
                        from utils.matplotlib_config import get_chinese_font
                        chinese_font = get_chinese_font()
                        
                        # 为教师模型创建一个临时数据集
                        plt.figure(figsize=(10, 8))
                        # 绘制散点图
                        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, 
                                            edgecolors='k', s=50, cmap=plt.cm.RdBu_r)
                        
                        # 添加图例
                        legend1 = plt.legend(*scatter.legend_elements(),
                                           title="类别", loc="best")
                        plt.gca().add_artist(legend1)
                        
                        # 设置标题和标签
                        plt.title(f"PCA降维数据 - 教师模型: {distiller.teacher.model_name}, 学生模型: {distiller.student.model_name}", 
                                fontproperties=chinese_font)
                        plt.xlabel(f"主成分1 ({variance_ratio[0]:.2%})", fontproperties=chinese_font)
                        plt.ylabel(f"主成分2 ({variance_ratio[1]:.2%})", fontproperties=chinese_font)
                        plt.grid(True)
                        plt.tight_layout()
                        
                        # 保存图表
                        save_path = viz_manager.plots_dir / "pca_scatter.png"
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        logger.info(f"PCA降维散点图已保存至 {save_path}")
                        plt.close()
                        
                        # 注: 我们不能在PCA空间直接绘制决策边界，因为模型是在原始特征空间训练的
                        # 这只是提供一个数据在降维空间的可视化
                    except Exception as e:
                        logger.warning(f"使用PCA降维可视化失败: {str(e)}")
                        import traceback
                        logger.warning(f"错误堆栈: {traceback.format_exc()}")
                
            except Exception as e:
                logger.warning(f"决策边界可视化失败: {str(e)}")
                # 记录更多诊断信息
                logger.warning(f"X_test 形状: {X_test.shape}")
                logger.warning(f"y_test 形状: {y_test.shape}")
        
        # 预测对比
        try:
            teacher_pred = distiller.teacher.predict(X_test)
            student_pred = distiller.student.predict(X_test)
            viz_manager.plot_prediction_comparison(
                y_test, teacher_pred, student_pred
            )
        except Exception as e:
            logger.warning(f"预测对比可视化失败: {str(e)}")
        
        student = distiller.student.get_model() if isinstance(distiller.student.get_model(), PyTorchBaseWrapper) else distiller.student
        
        # 生成报告
        logger.info("生成实验报告")
        report_generator = ReportGenerator(config)
        report_path = report_generator.generate_experiment_report(
            distiller.teacher,
            student,
            X_test,
            y_test,
            feature_names=feature_names
        )
        logger.info(f"实验报告已生成: {report_path}")
        
    except Exception as e:
        logger.error(f"可视化和报告生成失败: {str(e)}")
        raise

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置环境
    setup_environment(args)
    
    # 运行流程
    run_pipeline(args)

if __name__ == "__main__":
    main()