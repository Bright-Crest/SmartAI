import os
from typing import Any, Dict, List, Optional, Union
from pydantic import validator
# 兼容pydantic 1.10.x版本
try:
    # 尝试从pydantic直接导入（适用于1.10.x版本）
    from pydantic import BaseSettings
except ImportError:
    # 更高版本需要从pydantic_settings导入
    from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用程序设置配置"""
    
    # API 配置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "区块链情报分析平台"
    
    # CORS 配置
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3006",
        "https://chaininteligence.ai",
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # 区块链 API 配置
    ETHERSCAN_API_KEY: Optional[str] = None
    INFURA_API_KEY: Optional[str] = None
    ALCHEMY_API_KEY: Optional[str] = None
    
    # 区块链节点配置
    ETHEREUM_RPC_URL: Optional[str] = None
    INFURA_URL: Optional[str] = None
    
    @validator("INFURA_URL", pre=True)
    def assemble_infura_url(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if v:
            return v
        if values.get("INFURA_API_KEY"):
            return f"https://mainnet.infura.io/v3/{values.get('INFURA_API_KEY')}"
        return None
    
    # 数据库配置
    DATABASE_URL: Optional[str] = None
    
    # 缓存配置
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 缓存过期时间（秒）
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    
    # 风险分析配置
    RISK_THRESHOLD_HIGH: int = 75  # 高风险阈值
    RISK_THRESHOLD_MEDIUM: int = 40  # 中风险阈值
    
    # 其他配置
    MAX_NETWORK_NODES: int = 200  # 网络分析最大节点数
    MAX_DEPTH: int = 3  # 网络分析最大深度
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings() 