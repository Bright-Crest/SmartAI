"""
风险分析API服务
提供区块链地址风险分析服务
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 设置项目根目录导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) if current_dir not in sys.path else None

# 导入风险分析路由
from app.risk_analysis.routes import router as risk_router

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="区块链风险分析API",
    description="提供以太坊地址风险评分和用户画像分类服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加风险分析路由
app.include_router(risk_router)

# API根路径
@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "status": "online",
        "service": "区块链风险分析API",
        "version": "1.0.0"
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "区块链风险分析API"
    }

# 预加载模型
from app.risk_analysis.routes import load_models
@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    logger.info("服务启动，预加载模型...")
    try:
        load_models()
        logger.info("模型加载完成")
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")

# 启动应用
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    logger.info(f"启动风险分析API服务，端口: {port}")
    uvicorn.run(
        "risk_analysis_api:app", 
        host="127.0.0.1", 
        port=port, 
        reload=True
    ) 