"""
极简区块链API服务
提供以太坊区块链数据模拟访问接口
"""

import logging
import uvicorn
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 模拟数据
MOCK_BALANCES = {
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e": 1000000000000000000,  # 1 ETH
    "0x1111111111111111111111111111111111111111": 2000000000000000000,  # 2 ETH
    "0x2222222222222222222222222222222222222222": 5000000000000000000   # 5 ETH
}

MOCK_RECEIPTS = {
    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef": {
        "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "blockNumber": 12345678,
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
        "status": 1  # 交易成功
    }
}

MOCK_TOKEN_BALANCES = {
    "0x742d35Cc6634C0532925a3b844Bc454e4438f44e": [
        {
            "contract_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "symbol": "USDT",
            "name": "Tether USD",
            "decimals": 6,
            "balance": "1000000000"  # 1000 USDT
        },
        {
            "contract_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "symbol": "USDC",
            "name": "USD Coin",
            "decimals": 6,
            "balance": "2000000000"  # 2000 USDC
        }
    ]
}

MOCK_CONTRACT_RESULTS = {
    "totalSupply": 1000000000000000000000000,
    "balanceOf": 100000000000000000000,
    "allowance": 1000000000000000000,
    "symbol": "TKN",
    "name": "Token",
    "decimals": 18
}

# API版本
API_VERSION = "v1"

# 创建FastAPI应用
app = FastAPI(
    title="极简区块链API",
    description="提供以太坊区块链数据模拟访问接口",
    version=API_VERSION
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义数据模型
class AddressBalanceResponse(BaseModel):
    """地址余额响应模型"""
    address: str
    balance_wei: str
    balance_eth: float
    block_number: int

class TransactionReceiptResponse(BaseModel):
    """交易收据响应模型"""
    hash: str
    block_number: int
    from_address: str = Field(..., alias="from")
    to_address: Optional[str] = Field(None, alias="to")
    status: int

class ContractCallRequest(BaseModel):
    """合约调用请求模型"""
    address: str
    abi: Union[str, List[Dict[str, Any]]]
    function_name: str
    args: Optional[List[Any]] = []
    kwargs: Optional[Dict[str, Any]] = {}

# 帮助函数
def validate_address(address: str) -> str:
    """验证以太坊地址格式"""
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail=f"无效的以太坊地址格式: {address}")
    return address

def validate_tx_hash(tx_hash: str) -> str:
    """验证交易哈希格式"""
    if not tx_hash.startswith("0x") or len(tx_hash) != 66:
        raise HTTPException(status_code=400, detail=f"无效的交易哈希格式: {tx_hash}")
    return tx_hash

# API路由
@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "status": "online",
        "version": API_VERSION
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": API_VERSION
    }

@app.get("/address/{address}", response_model=AddressBalanceResponse)
async def query_address(address: str):
    """
    查询以太坊地址的ETH余额
    
    参数:
        address: 以太坊地址
    """
    try:
        # 验证地址
        checksum_address = validate_address(address)
        
        # 获取模拟余额
        balance_wei = MOCK_BALANCES.get(checksum_address, 1000000000000000000)  # 默认1 ETH
        balance_eth = float(balance_wei) / 10**18
        
        return {
            "address": checksum_address,
            "balance_wei": str(balance_wei),
            "balance_eth": balance_eth,
            "block_number": 12345678
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询地址余额时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/transaction/{tx_hash}", response_model=TransactionReceiptResponse)
async def query_transaction(tx_hash: str):
    """
    查询交易收据信息
    
    参数:
        tx_hash: 交易哈希
    """
    try:
        # 验证交易哈希
        tx_hash = validate_tx_hash(tx_hash)
        
        # 获取模拟收据
        receipt = MOCK_RECEIPTS.get(tx_hash, {
            "hash": tx_hash,
            "blockNumber": 12345678,
            "from": "0x1111111111111111111111111111111111111111",
            "to": "0x2222222222222222222222222222222222222222",
            "status": 1  # 交易成功
        })
            
        return {
            "hash": receipt["hash"],
            "block_number": receipt["blockNumber"],
            "from": receipt["from"],
            "to": receipt["to"],
            "status": receipt["status"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询交易收据时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/address/{address}/tokens")
async def query_token_balances(address: str):
    """
    查询地址持有的ERC20代币余额
    
    参数:
        address: 以太坊地址
    """
    try:
        # 验证地址
        checksum_address = validate_address(address)
        
        # 获取模拟代币余额
        tokens = MOCK_TOKEN_BALANCES.get(checksum_address, [
            {
                "contract_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "symbol": "USDT",
                "name": "Tether USD",
                "decimals": 6,
                "balance": "1000000000"  # 1000 USDT
            }
        ])
        
        # 过滤余额为0的代币
        non_zero_tokens = [
            token for token in tokens
            if token.get("balance", "0") != "0"
        ]
        
        return non_zero_tokens
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询代币余额时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.post("/contract/call")
async def call_contract_endpoint(contract_request: ContractCallRequest):
    """
    调用合约的只读函数
    
    参数:
        contract_request: 合约调用请求
    """
    try:
        # 验证地址
        contract_address = validate_address(contract_request.address)
        
        # 获取模拟结果
        function_name = contract_request.function_name
        result = MOCK_CONTRACT_RESULTS.get(function_name, 123456789)  # 默认返回数字
        
        # 处理结果，确保可JSON序列化
        if isinstance(result, (bytes, bytearray)):
            result = result.hex()
        
        return {
            "contract_address": contract_address,
            "function": function_name,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"调用合约函数时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# 启动应用
if __name__ == "__main__":
    # 使用 uvicorn 启动 API 服务
    uvicorn.run(
        "blockchain_mini_api:app", 
        host="127.0.0.1", 
        port=8001, 
        reload=True
    ) 