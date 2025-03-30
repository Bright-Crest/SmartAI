import logging
import sys
from pathlib import Path
from .config import config

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """初始化日志配置"""
        self.logger = logging.getLogger('distillation')
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # 清除现有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """获取logger实例"""
        return self.logger

# 创建全局logger实例
logger = Logger().get_logger()

def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def log_step(step_name):
    """步骤日志装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {step_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {step_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {step_name}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator