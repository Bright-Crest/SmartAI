import os
from datetime import datetime
from pathlib import Path

class Config:
    def __init__(self):
        # 项目根目录
        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 时间戳
        self.timestamp = datetime.now().strftime("%d_%H%M%S")

        self.results_dir = self.root_dir / "results" / self.timestamp
        self.update_results_dir(self.results_dir)
        self.make_dirs()

        self.log_level = "INFO"
        
        # 模型配置
        self.model_config = {
            "teacher": {
                "name": "xgboost",  # 可选: xgboost, lightgbm, etc.
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                }
            },
            "student": {
                "name": "logistic",  # 可选: logistic, mlp, etc.
                "params": {
                }
            }
        }
        
        # 训练配置
        self.train_config = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "temperature": 2.0,
            "alpha": 1.0,  # 软标签权重
            
            # 特征选择相关配置
            "feature_selection": {
                "enable": True,         # 是否启用特征选择
                "max_factor_ratio": 0.4,  # 最大特征比例
                "factor_temperature": 1.0, # 特征选择温度参数
                "l1_lambda": 0.2      # L1正则化系数
            },
            
            # 多重损失模型相关配置
            "multi_loss": {
                "max_factors": None,     # 最大因子数量，None表示自动计算（输入维度的20%）
                "factor_temperature": 1.0, # 因子选择温度参数
                "l1_lambda": 0.2,      # L1正则化系数
                "triplet_margin": 1.0,   # 三元组损失边界值
                "sep_lambda": 0.2        # 分离损失权重
            }
        }

        self.use_shap = False
    
    def update_results_dir(self, results_dir):
        """更新结果目录"""
        self.results_dir = results_dir
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.log_file = self.logs_dir / f"log_{self.timestamp}.log"

    def make_dirs(self):
        """创建必要的目录"""
        for dir_path in [self.logs_dir, self.results_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif "." in key:
                # 处理嵌套配置，如 "model_config.teacher.params.n_estimators"
                parts = key.split(".")
                current = self
                for part in parts[:-1]:
                    current = getattr(current, part) if hasattr(current, part) else current[part]
                current[parts[-1]] = value

    def get_model_params(self, model_type):
        """获取指定类型模型的参数"""
        return self.model_config[model_type]["params"]

    def get_model_name(self, model_type):
        """获取指定类型模型的名称"""
        return self.model_config[model_type]["name"]
    
    def get_feature_selection_config(self):
        """获取特征选择配置"""
        return self.train_config["feature_selection"]
    
    def get_multi_loss_config(self):
        """获取多重损失配置"""
        return self.train_config["multi_loss"]

# 全局配置实例
config = Config()