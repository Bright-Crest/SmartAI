import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call

class ModelRegistry:
    """模型注册表，用于管理和加载不同类型的模型
    
    这个类实现了工厂模式，根据配置动态加载不同的模型实现。
    它允许在不修改核心代码的情况下添加新的模型类型。
    
    学生模型支持的特殊参数：
    - enable_factor_selection: 是否启用特征因子自动选择 (默认: PyTorch模型为True, 其他为False)
    - max_factor_ratio: 最大因子比例，控制最多使用多少比例的特征 (默认: 0.2)
    - factor_temperature: 特征掩码温度参数，控制特征选择的软硬程度 (默认: 1.0)
    - l1_lambda: L1正则化系数，控制特征稀疏程度 (默认: 0.001)
    """
    
    # 教师模型映射表
    TEACHER_MODELS = {
        "xgboost": "xgboost.XGBClassifier",
        "random_forest": "sklearn.ensemble.RandomForestClassifier",
        "lightgbm": "lightgbm.LGBMClassifier",
        "catboost": "catboost.CatBoostClassifier",
        "pytorch_cnn": "src.core.models.pytorch_wrappers.PyTorchTeacherCNN",
        "pytorch_mlp": "src.core.models.pytorch_wrappers.PyTorchTeacherMLP"
    }
    
    # 学生模型映射表
    STUDENT_MODELS = {
        "logistic": "sklearn.linear_model.LogisticRegression",
        "svm": "sklearn.svm.SVC",
        "mlp": "sklearn.neural_network.MLPClassifier",
        "decision_tree": "sklearn.tree.DecisionTreeClassifier",
        "pytorch_cnn": "src.core.models.pytorch_wrappers.PyTorchStudentCNN",
        "pytorch_mlp": "src.core.models.pytorch_wrappers.PyTorchStudentMLP",
        "pytorch_factor_mlp": "src.core.models.pytorch_wrappers.PyTorchFactorStudentMLP",  # 为向后兼容而保留
        "pytorch_factorized_mlp": "src.core.models.pytorch_wrappers.PyTorchFactorizedStudentMLP"  # 新的多重损失因子化学生模型
    }
    
    @classmethod
    @log_function_call
    def get_model_class(cls, model_type, model_name):
        """根据模型类型和名称获取模型类
        
        Args:
            model_type: 模型类型 ('teacher' 或 'student')
            model_name: 模型名称
        
        Returns:
            模型类
        """
        # 选择模型映射表
        if model_type == "teacher":
            models_map = cls.TEACHER_MODELS
        elif model_type == "student":
            models_map = cls.STUDENT_MODELS
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 检查模型名称是否存在
        if model_name not in models_map:
            raise ValueError(f"未注册的{model_type}模型: {model_name}")
        
        # 获取模型类路径
        model_path = models_map[model_name]
        module_path, class_name = model_path.rsplit(".", 1)
        
        try:
            # 动态导入模块
            module = importlib.import_module(module_path)
            # 获取模型类
            model_class = getattr(module, class_name)
            return model_class
        except (ImportError, AttributeError) as e:
            logger.error(f"加载模型 {model_name} 失败: {str(e)}")
            raise
    
    @classmethod
    @log_function_call
    def register_teacher_model(cls, name, model_path):
        """注册新的教师模型
        
        Args:
            name: 模型名称
            model_path: 模型类路径 (例如 'module.submodule.ClassName')
        """
        cls.TEACHER_MODELS[name] = model_path
        logger.info(f"已注册教师模型: {name} -> {model_path}")
    
    @classmethod
    @log_function_call
    def register_student_model(cls, name, model_path):
        """注册新的学生模型
        
        Args:
            name: 模型名称
            model_path: 模型类路径 (例如 'module.submodule.ClassName')
        """
        cls.STUDENT_MODELS[name] = model_path
        logger.info(f"已注册学生模型: {name} -> {model_path}")
    
    @classmethod
    @log_function_call
    def get_available_models(cls, model_type=None):
        """获取可用的模型列表
        
        Args:
            model_type: 模型类型 ('teacher', 'student' 或 None 表示所有)
        
        Returns:
            可用模型字典
        """
        if model_type == "teacher":
            return cls.TEACHER_MODELS
        elif model_type == "student":
            return cls.STUDENT_MODELS
        else:
            return {
                "teacher": cls.TEACHER_MODELS,
                "student": cls.STUDENT_MODELS
            } 