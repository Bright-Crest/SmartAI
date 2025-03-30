import logging
from fastapi import FastAPI, Request, status, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
import time
from typing import Callable, List
import uvicorn

from app.core.config import settings
from app.routers import (
    user,
    auth,
    address,
    transaction,
    network,
    dashboard,
    kol,
    users,
    utils,
    ai_model
)
from app.api_bridge import router as bridge_router

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
origins = settings.BACKEND_CORS_ORIGINS
if not origins:
    origins = [
        "http://localhost:3000",  # 前端默认端口
        "http://localhost:3006",  # 可能的备用前端端口
        "https://chaininteligence.ai",  # 生产环境域名
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局依赖
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

# 请求拦截中间件
@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    """记录所有请求的日志"""
    request_id = str(time.time())
    logger.info(f"开始请求 [{request_id}]: {request.method} {request.url.path}")
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"完成请求 [{request_id}]: {request.method} {request.url.path} - "
            f"状态: {response.status_code}, 处理时间: {process_time:.4f}s"
        )
        
        # 添加自定义响应头
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"请求异常 [{request_id}]: {request.method} {request.url.path} - "
            f"错误: {str(e)}, 处理时间: {process_time:.4f}s"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "服务器内部错误"}
        )

# 注册API前缀路由
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(user.router, prefix=settings.API_V1_STR)
app.include_router(address.router, prefix=settings.API_V1_STR)
app.include_router(transaction.router, prefix=settings.API_V1_STR)
app.include_router(network.router, prefix=settings.API_V1_STR)
app.include_router(dashboard.router, prefix=settings.API_V1_STR)
app.include_router(kol.router, prefix=settings.API_V1_STR)
app.include_router(ai_model.router, prefix=settings.API_V1_STR)

# 注册API桥接路由 - 用于Ethers.js前端连接
app.include_router(bridge_router, prefix=settings.API_V1_STR)

# 注册区块链API路由
from app.api.routes.blockchain import router as blockchain_router
app.include_router(
    blockchain_router,
    prefix=f"{settings.API_V1_STR}/blockchain",
    tags=["区块链"]
)

# 注册路由
app.include_router(
    auth.router,
    prefix=settings.API_V1_STR,
    tags=["认证"]
)
app.include_router(
    users.router,
    prefix=settings.API_V1_STR,
    tags=["用户"]
)
app.include_router(
    utils.router,
    prefix=settings.API_V1_STR,
    tags=["工具"]
)

@app.get("/")
async def root():
    """
    API根端点 - 返回API状态信息
    """
    return {
        "status": "online",
        "api_version": "v1",
        "message": "欢迎使用区块链情报分析平台API"
    }

@app.get("/health")
async def health_check():
    """
    健康检查端点 - 用于监控和容器健康检查
    """
    return {
        "status": "healthy",
        "api_version": "v1"
    }

@app.get(f"{settings.API_V1_STR}/info")
async def api_info():
    """
    返回API基本信息
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "区块链情报分析平台API",
        "endpoints": [
            {"path": "/docs", "description": "API文档"},
            {"path": "/redoc", "description": "ReDoc API文档"},
            {"path": f"{settings.API_V1_STR}/auth", "description": "认证接口"},
            {"path": f"{settings.API_V1_STR}/users", "description": "用户接口"},
            {"path": f"{settings.API_V1_STR}/address", "description": "地址分析接口"},
            {"path": f"{settings.API_V1_STR}/transactions", "description": "交易分析接口"},
            {"path": f"{settings.API_V1_STR}/network", "description": "网络分析接口"},
            {"path": f"{settings.API_V1_STR}/dashboard", "description": "仪表盘接口"},
            {"path": f"{settings.API_V1_STR}/kol", "description": "意见领袖分析接口"},
            {"path": f"{settings.API_V1_STR}/bridge", "description": "以太坊API桥接接口"},
            {"path": f"{settings.API_V1_STR}/blockchain", "description": "区块链分析接口"},
            {"path": f"{settings.API_V1_STR}/ai_model", "description": "智能模型接口"}
        ]
    }

if __name__ == "__main__":
    """当直接运行文件时启动应用"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    ) 