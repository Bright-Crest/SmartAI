import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call, log_step
from utils.config import config
from .data_generator import DataGenerator
from .fetch_blockchain_data import BlockchainDataFetcher

class DataLoader:
    def __init__(self):
        """初始化数据加载器"""
        self.data_dir = config.results_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_generator = DataGenerator()
        self.blockchain_fetcher = None  # 延迟初始化
    
    @log_step("加载数据")
    def load_data(self, file_path):
        """加载数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            X: 特征矩阵
            y: 标签向量
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {file_path}")
                
                # 尝试生成数据
                logger.info("尝试生成合成数据...")
                generated_path = self.generate_data()
                
                if generated_path:
                    logger.info(f"使用生成的数据: {generated_path}")
                    file_path = generated_path
                else:
                    raise FileNotFoundError(f"无法找到或生成数据: {file_path}")
            
            # 根据文件扩展名决定加载方式
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                data = pd.read_csv(file_path)
                
                # 确定哪一列是标签
                if 'is_smart_money' in data.columns:
                    y_col = 'is_smart_money'
                else:
                    # 假设最后一列是标签
                    y_col = data.columns[-1]
                
                # 移除非特征列
                non_feature_cols = ['eth_address', y_col]
                feature_cols = [col for col in data.columns if col not in non_feature_cols]
                
                X = data[feature_cols].values
                y = data[y_col].values
                
            elif ext == '.npz':
                data = np.load(file_path)
                X, y = data['X'], data['y']
                
            elif ext == '.npy':
                data = np.load(file_path, allow_pickle=True)
                if isinstance(data, dict):
                    X, y = data['X'], data['y']
                else:
                    # 假设最后一列是标签
                    X = data[:, :-1]
                    y = data[:, -1]
            else:
                raise ValueError(f"不支持的文件类型: {ext}")
            
            logger.info(f"数据加载成功，特征形状: {X.shape}, 标签形状: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    @log_step("数据分割和预处理")
    def preprocess_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """数据分割和预处理
        
        Args:
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            val_size: 验证集比例（相对于训练集）
            random_state: 随机种子
            
        Returns:
            训练集、验证集和测试集元组
        """
        try:
            # 数据分割
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=val_size, 
                random_state=random_state,
                stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None
            )
            
            # 特征标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            # 保存标准化器
            self._save_scaler(scaler)
            
            logger.info(f"数据预处理完成")
            logger.info(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    @log_function_call
    def _save_scaler(self, scaler):
        """保存标准化器
        
        Args:
            scaler: 标准化器实例
        """
        import joblib
        scaler_path = self.data_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"标准化器已保存至 {scaler_path}")
    
    @log_function_call
    def load_scaler(self):
        """加载标准化器
        
        Returns:
            标准化器实例
        """
        import joblib
        try:
            scaler_path = self.data_dir / "scaler.joblib"
            scaler = joblib.load(scaler_path)
            logger.info(f"标准化器已从 {scaler_path} 加载")
            return scaler
        except Exception as e:
            logger.error(f"加载标准化器失败: {str(e)}")
            raise
    
    @log_function_call
    def save_data_summary(self, X, y):
        """保存数据摘要信息
        
        Args:
            X: 特征矩阵
            y: 标签向量
        """
        try:
            # 基本统计信息
            n_samples, n_features = X.shape
            n_classes = len(np.unique(y))
            class_distribution = {int(cls): int(np.sum(y == cls)) for cls in np.unique(y)}
            
            # 创建摘要
            summary = {
                "n_samples": n_samples,
                "n_features": n_features,
                "n_classes": n_classes,
                "class_distribution": class_distribution,
                "feature_means": X.mean(axis=0).tolist(),
                "feature_stds": X.std(axis=0).tolist()
            }
            
            # 保存为JSON
            import json
            summary_path = self.data_dir / "data_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"数据摘要已保存至 {summary_path}")
            
        except Exception as e:
            logger.error(f"保存数据摘要失败: {str(e)}")
            raise
    
    @log_step("生成合成数据")
    def generate_data(self, num_samples=1000, smart_money_ratio=0.5, random_state=42):
        """生成合成数据
        
        Args:
            num_samples: 样本数量
            smart_money_ratio: 聪明钱比例
            random_state: 随机种子
            
        Returns:
            生成的数据文件路径
        """
        try:
            logger.info(f"生成合成数据 (样本数量={num_samples}, 聪明钱比例={smart_money_ratio})")
            
            # 使用数据生成器生成数据
            _, processed_data_path = self.data_generator.generate_and_save_data(
                num_samples=num_samples,
                smart_money_ratio=smart_money_ratio,
                random_state=random_state
            )
            
            return processed_data_path
            
        except Exception as e:
            logger.error(f"生成合成数据失败: {str(e)}")
            return None
    
    @log_step("获取区块链数据")
    def fetch_blockchain_data(self, api_key=None, addresses_file=None, num_addresses=100, use_demo_data=True):
        """获取区块链数据
        
        Args:
            api_key: Etherscan API密钥
            addresses_file: 地址文件路径
            num_addresses: 随机地址数量
            use_demo_data: 是否使用演示数据
            
        Returns:
            获取的数据文件路径
        """
        try:
            logger.info(f"获取区块链数据 (地址数量={num_addresses}, 使用演示数据={use_demo_data})")
            
            # 延迟初始化区块链数据获取器
            if self.blockchain_fetcher is None:
                self.blockchain_fetcher = BlockchainDataFetcher(api_key)
            
            # 获取数据
            processed_data_path = self.blockchain_fetcher.fetch_and_save_data(
                addresses_file=addresses_file,
                count=num_addresses,
                use_demo_data=use_demo_data
            )
            
            return processed_data_path
            
        except Exception as e:
            logger.error(f"获取区块链数据失败: {str(e)}")
            return None
    
    @log_step("准备数据")
    def prepare_data(self, data_path=None, use_real_data=False, api_key=None, 
                    addresses_file=None, num_samples=1000, smart_money_ratio=0.5):
        """准备数据，自动选择数据来源
        
        Args:
            data_path: 数据文件路径（可选）
            use_real_data: 是否使用真实区块链数据
            api_key: Etherscan API密钥（使用真实数据时需要）
            addresses_file: 地址文件路径（使用真实数据时可选）
            num_samples: 合成数据样本数量
            smart_money_ratio: 聪明钱比例（合成数据）
            
        Returns:
            数据文件路径
        """
        # 如果指定了数据路径，首先尝试加载
        if data_path and os.path.exists(data_path):
            logger.info(f"使用指定的数据文件: {data_path}")
            return data_path
        
        # 根据选项决定数据来源
        if use_real_data:
            logger.info("获取真实区块链数据")
            return self.fetch_blockchain_data(
                api_key=api_key,
                addresses_file=addresses_file,
                num_addresses=num_samples,
                use_demo_data=(api_key is None)
            )
        else:
            logger.info("生成合成区块链数据")
            return self.generate_data(
                num_samples=num_samples,
                smart_money_ratio=smart_money_ratio
            )