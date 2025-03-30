"""
API桥接模块 - 为前端Ethers.js提供与Web3.py后端的交互接口
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import json

from app.blockchain.eth_client import get_eth_client

# 创建日志记录器
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/bridge",
    tags=["bridge"],
    responses={404: {"description": "Not found"}},
)

# 请求模型
class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: List[Any] = []
    id: int

# 响应模型
class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: int

@router.post("/eth", response_model=JsonRpcResponse)
async def ethereum_bridge(request: JsonRpcRequest):
    """
    以太坊JSON-RPC桥接端点，允许前端通过标准以太坊JSON-RPC接口与后端通信
    
    该接口模拟以太坊节点的JSON-RPC接口，但实际由后端的Web3.py处理，
    使前端Ethers.js可以无缝连接到该接口。
    """
    try:
        # 获取Web3客户端
        eth_client = get_eth_client()

        # 支持的JSON-RPC方法
        supported_methods = {
            # 网络信息
            "net_version": lambda params: eth_client.w3.net.version,
            "eth_chainId": lambda params: hex(int(eth_client.w3.eth.chain_id)),
            "eth_blockNumber": lambda params: hex(eth_client.w3.eth.block_number),
            
            # 账户信息
            "eth_getBalance": lambda params: hex(eth_client.w3.eth.get_balance(params[0], params[1] if len(params) > 1 else "latest")),
            "eth_getTransactionCount": lambda params: hex(eth_client.w3.eth.get_transaction_count(params[0], params[1] if len(params) > 1 else "latest")),
            
            # 交易信息
            "eth_getTransactionByHash": lambda params: eth_client.w3.eth.get_transaction(params[0]),
            "eth_getTransactionReceipt": lambda params: eth_client.w3.eth.get_transaction_receipt(params[0]),
            
            # Gas
            "eth_gasPrice": lambda params: hex(eth_client.w3.eth.gas_price),
            "eth_estimateGas": lambda params: hex(eth_client.w3.eth.estimate_gas(params[0])),
            
            # 区块信息
            "eth_getBlockByNumber": lambda params: eth_client.w3.eth.get_block(params[0], full_transactions=params[1] if len(params) > 1 else False),
            "eth_getBlockByHash": lambda params: eth_client.w3.eth.get_block(params[0], full_transactions=params[1] if len(params) > 1 else False),
            
            # 合约交互
            "eth_call": lambda params: eth_client.w3.eth.call(params[0], params[1] if len(params) > 1 else "latest").hex(),
            
            # 代币信息
            "eth_getTokenBalance": lambda params: hex(int(eth_client.get_token_balance(params[0], params[1]) * (10 ** 18)))
        }
        
        # 日志请求信息
        logger.info(f"收到RPC请求: method={request.method}, params={json.dumps(request.params)}")
        
        if request.method not in supported_methods:
            logger.warning(f"不支持的方法: {request.method}")
            return JsonRpcResponse(
                jsonrpc="2.0",
                error={"code": -32601, "message": f"不支持的方法: {request.method}"},
                id=request.id
            )
            
        # 执行对应的方法
        result = supported_methods[request.method](request.params)
        
        # 转换结果为可JSON序列化格式
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        
        # 确保hex字符串格式正确 (添加0x前缀)
        if isinstance(result, str) and result.startswith("0x"):
            pass
        elif isinstance(result, str) and all(c in "0123456789abcdefABCDEF" for c in result):
            result = "0x" + result
            
        # 日志响应结果
        logger.info(f"RPC响应: {result}")
        
        return JsonRpcResponse(
            jsonrpc="2.0",
            result=result,
            id=request.id
        )
        
    except Exception as e:
        logger.error(f"处理RPC请求出错: {str(e)}")
        return JsonRpcResponse(
            jsonrpc="2.0",
            error={"code": -32603, "message": f"内部错误: {str(e)}"},
            id=request.id
        )

@router.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        eth_client = get_eth_client()
        connected = eth_client.w3.is_connected()
        return {
            "status": "ok" if connected else "error",
            "blockchain_connected": connected,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "error",
            "blockchain_connected": False,
            "error": str(e),
            "version": "1.0.0"
        } 