"""
区块链API路由
提供与区块链交互的各种API端点
"""

from fastapi import APIRouter, HTTPException, Path, Query, Depends
from typing import Dict, Any, Optional
from pydantic import BaseModel, validator

from app.blockchain.ethereum import get_balance, get_transaction_receipt
from app.core.config import settings

router = APIRouter()

class AddressResponse(BaseModel):
    """以太坊地址余额响应模型"""
    address: str
    balance_wei: str
    balance_eth: float
    block_number: int
    
    @validator('address')
    def validate_address(cls, v):
        if not v or not v.startswith('0x') or len(v) != 42:
            raise ValueError('无效的以太坊地址格式')
        return v

@router.get("/address/{address}", response_model=AddressResponse)
async def query_address(
    address: str = Path(..., description="以太坊地址，以0x开头的42个字符"),
):
    """
    获取以太坊地址余额
    
    Args:
        address: 以太坊地址
        
    Returns:
        AddressResponse: 地址的余额信息
    """
    try:
        result = get_balance(address)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TransactionResponse(BaseModel):
    """交易收据响应模型"""
    blockHash: Optional[str] = None
    blockNumber: Optional[int] = None
    contractAddress: Optional[str] = None
    cumulativeGasUsed: Optional[int] = None
    effectiveGasPrice: Optional[int] = None
    from_address: Optional[str] = None
    gasUsed: Optional[int] = None
    logs: Optional[list] = None
    logsBloom: Optional[str] = None
    status: Optional[int] = None
    to: Optional[str] = None
    transactionHash: str
    transactionIndex: Optional[int] = None
    type: Optional[int] = None
    
    class Config:
        fields = {
            'from_address': 'from'
        }

@router.get("/transaction/{tx_hash}", response_model=Optional[TransactionResponse])
async def query_transaction(
    tx_hash: str = Path(..., description="交易哈希，以0x开头的66个字符"),
):
    """
    获取交易收据
    
    Args:
        tx_hash: 交易哈希
        
    Returns:
        TransactionResponse: 交易收据详情，如果交易不存在返回None
    """
    try:
        result = get_transaction_receipt(tx_hash)
        if not result:
            return None
        
        # 处理特殊字段名(from是Python关键字)
        if 'from' in result:
            result['from_address'] = result.pop('from')
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 