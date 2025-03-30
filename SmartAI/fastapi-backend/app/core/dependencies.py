"""
核心依赖模块，包含各种辅助函数和共享依赖项
"""

import time
from typing import Generator, Optional, Dict, Any, Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from functools import wraps
import hashlib
import logging
import inspect

from app.core.config import settings
from app.models.user import User

logger = logging.getLogger(__name__)

# 模拟缓存，生产环境应使用Redis等
mock_cache = {}

# 内存缓存实现 (简化版)
_cache = {}

# 模拟数据库会话
def get_db() -> Generator:
    """
    获取数据库会话
    这是一个模拟实现，实际项目中应该连接到真实数据库
    """
    db = None
    try:
        # 在实际项目中，这里将创建数据库会话
        # db = SessionLocal()
        db = {"mock": "database_session"}
        yield db
    finally:
        # 在实际项目中，这里将关闭数据库会话
        # db.close()
        pass

# 缓存装饰器
def cache(ttl: int = 3600):
    """
    缓存装饰器，用于缓存函数结果
    支持同步和异步函数
    
    Args:
        ttl: 缓存生存时间（秒）
    """
    def decorator(func):
        # 检查函数是否是异步函数
        is_async = inspect.iscoroutinefunction(func)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}:{v}" for k, v in kwargs.items()])
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # 检查缓存
            now = time.time()
            if cache_key in _cache and now < _cache[cache_key]["expire_at"]:
                logger.debug(f"缓存命中: {func.__name__}")
                return _cache[cache_key]["data"]
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 更新缓存
            _cache[cache_key] = {
                "data": result,
                "expire_at": now + ttl
            }
            
            # 清理过期缓存
            expired_keys = [k for k, v in _cache.items() if now > v["expire_at"]]
            for k in expired_keys:
                del _cache[k]
                
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}:{v}" for k, v in kwargs.items()])
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # 检查缓存
            now = time.time()
            if cache_key in _cache and now < _cache[cache_key]["expire_at"]:
                logger.debug(f"缓存命中: {func.__name__}")
                return _cache[cache_key]["data"]
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 更新缓存
            _cache[cache_key] = {
                "data": result,
                "expire_at": now + ttl
            }
            
            # 清理过期缓存
            expired_keys = [k for k, v in _cache.items() if now > v["expire_at"]]
            for k in expired_keys:
                del _cache[k]
                
            return result
            
        return async_wrapper if is_async else sync_wrapper
    return decorator

# 基本认证依赖
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

# 模拟用户认证
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    获取当前用户
    这是一个模拟实现，实际项目中应该验证JWT令牌并从数据库获取用户
    """
    # 在实际项目中，这里将解码JWT令牌并验证用户
    if token == "invalid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 模拟用户对象
    return User(
        id=1,
        email="user@example.com",
        username="testuser",
        is_active=True,
        is_admin=False
    )

# 管理员权限依赖
def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """获取当前管理员用户"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足",
        )
    return current_user

# 风险级别计算
def calculate_risk_score(risk_factors: Dict[str, bool]) -> int:
    """
    计算风险评分，基于提供的风险因素
    
    Args:
        risk_factors: 包含各种风险因素的字典，键为风险名称，值为布尔值
        
    Returns:
        int: 风险评分 (0-100)
    """
    # 风险因素权重
    risk_weights = {
        "high_value": 10,               # 大额资产/交易
        "new_address": 5,               # 新地址
        "exchange_interaction": -5,      # 与主流交易所交互
        "mixer_interaction": 40,         # 与混币器交互
        "unusual_pattern": 30,           # 不寻常交易模式
        "sanctioned_address": 60,        # 与受制裁地址交互
        "darkmarket_interaction": 50,    # 与暗网市场交互
        "high_frequency": 15             # 高频交易
    }
    
    # 计算总评分
    score = 0
    for factor, present in risk_factors.items():
        if present and factor in risk_weights:
            score += risk_weights[factor]
    
    # 确保评分在0-100范围内
    return max(0, min(100, score))

# 风险级别确定
def determine_risk_level(risk_score: int) -> str:
    """
    根据风险评分确定风险等级
    
    Args:
        risk_score: 风险评分 (0-100)
        
    Returns:
        str: 风险等级 ("none", "low", "medium", "high", "critical")
    """
    if risk_score < 10:
        return "none"
    elif risk_score < 40:
        return "low"
    elif risk_score < 70:
        return "medium"
    elif risk_score < 90:
        return "high"
    else:
        return "critical"

# JWT令牌验证依赖（占位，实际实现应更加复杂）
async def verify_token(token: str) -> Dict[str, Any]:
    """
    验证JWT令牌
    
    Args:
        token: JWT令牌
    
    Returns:
        Dict: 用户信息
    """
    if token == "invalid":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据"
        )
    
    # 模拟用户信息
    return {
        "id": "user123",
        "username": "test_user",
        "email": "user@example.com",
        "is_active": True
    } 