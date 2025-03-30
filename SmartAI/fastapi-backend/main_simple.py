from fastapi import FastAPI
import uvicorn
from app.blockchain.ethereum import get_balance, get_transaction_receipt
from app.blockchain.contract import load_contract, call_contract_function
from typing import Optional

app = FastAPI(title="区块链情报分析平台")

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

@app.get("/blockchain/address/{address}")
async def query_address(address: str):
    """
    获取以太坊地址余额
    
    Args:
        address: 以太坊地址
        
    Returns:
        Dict: 地址的余额信息
    """
    try:
        result = get_balance(address)
        return result
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"获取余额时出错: {str(e)}"}

@app.get("/blockchain/transaction/{tx_hash}")
async def query_transaction(tx_hash: str):
    """
    获取交易收据
    
    Args:
        tx_hash: 交易哈希
        
    Returns:
        Dict: 交易收据信息
    """
    try:
        result = get_transaction_receipt(tx_hash)
        if not result:
            return {"error": "交易未找到"}
            
        # 处理特殊字段名(from是Python关键字)
        if 'from' in result:
            result['from_address'] = result.pop('from')
            
        return result
    except Exception as e:
        return {"error": f"获取交易收据时出错: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 