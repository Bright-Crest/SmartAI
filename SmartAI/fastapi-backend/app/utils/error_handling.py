"""
区块链API错误处理模块
提供统一的错误类和处理功能
"""

import logging
from typing import Dict, Any, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainError(Exception):
    """区块链API基础异常类"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class BlockchainConnectionError(BlockchainError):
    """区块链连接异常"""
    pass

class InvalidAddressError(BlockchainError):
    """无效的以太坊地址异常"""
    pass

class InvalidTransactionError(BlockchainError):
    """无效的交易哈希异常"""
    pass

class ContractInteractionError(BlockchainError):
    """合约交互异常"""
    pass

def log_error(error: Exception, logger: logging.Logger = logger):
    """
    记录错误信息
    
    参数:
        error: 异常对象
        logger: 日志记录器
    """
    if isinstance(error, BlockchainError):
        logger.error(f"{error.__class__.__name__}: {error.message}")
        if hasattr(error, 'details') and error.details:
            logger.error(f"错误详情: {error.details}")
    else:
        logger.error(f"未预期的错误: {str(error)}") 