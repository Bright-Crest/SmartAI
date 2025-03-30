import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from web3 import Web3, AsyncWeb3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_typing import BlockIdentifier, ChecksumAddress
from hexbytes import HexBytes

from app.core.config import settings

logger = logging.getLogger(__name__)

class EthClient:
    """以太坊网络客户端"""
    
    def __init__(self, provider_url: Optional[str] = None):
        """
        初始化以太坊客户端
        
        Args:
            provider_url: 以太坊节点URL，如果为None则使用配置文件中的URL
        """
        self.provider_url = provider_url or settings.INFURA_URL or settings.ETHEREUM_RPC_URL
        if not self.provider_url:
            raise ValueError("未配置以太坊节点URL，请在配置文件中设置INFURA_URL或ETHEREUM_RPC_URL")
        
        # 同步Web3实例
        self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
        
        # 异步Web3实例
        self.async_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.provider_url))
        
        # 支持PoA链
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        logger.info(f"以太坊客户端初始化，连接到: {self.provider_url}")
        logger.info(f"连接状态: {'已连接' if self.w3.is_connected() else '未连接'}")
    
    def get_balance(self, address: str) -> float:
        """
        获取地址余额（以ETH为单位）
        
        Args:
            address: 要查询的地址
        
        Returns:
            float: 余额（以ETH为单位）
        """
        address = Web3.to_checksum_address(address)
        balance_wei = self.w3.eth.get_balance(address)
        balance_eth = Web3.from_wei(balance_wei, 'ether')
        return float(balance_eth)
    
    async def get_balance_async(self, address: str) -> float:
        """
        异步获取地址余额（以ETH为单位）
        
        Args:
            address: 要查询的地址
        
        Returns:
            float: 余额（以ETH为单位）
        """
        address = Web3.to_checksum_address(address)
        balance_wei = await self.async_w3.eth.get_balance(address)
        balance_eth = Web3.from_wei(balance_wei, 'ether')
        return float(balance_eth)
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        获取交易详情
        
        Args:
            tx_hash: 交易哈希
            
        Returns:
            Dict: 交易详情
        """
        tx_hash_bytes = HexBytes(tx_hash)
        tx = self.w3.eth.get_transaction(tx_hash_bytes)
        receipt = self.w3.eth.get_transaction_receipt(tx_hash_bytes)
        
        # 转换为可序列化的字典
        tx_dict = dict(tx)
        receipt_dict = dict(receipt)
        
        # 处理HexBytes类型
        for key, value in tx_dict.items():
            if isinstance(value, HexBytes):
                tx_dict[key] = value.hex()
            elif isinstance(value, bytes):
                tx_dict[key] = "0x" + value.hex()
        
        for key, value in receipt_dict.items():
            if isinstance(value, HexBytes):
                receipt_dict[key] = value.hex()
            elif isinstance(value, bytes):
                receipt_dict[key] = "0x" + value.hex()
        
        # 合并交易和收据信息
        result = {**tx_dict, "receipt": receipt_dict}
        return result
    
    async def get_transaction_async(self, tx_hash: str) -> Dict[str, Any]:
        """
        异步获取交易详情
        
        Args:
            tx_hash: 交易哈希
            
        Returns:
            Dict: 交易详情
        """
        tx_hash_bytes = HexBytes(tx_hash)
        tx, receipt = await asyncio.gather(
            self.async_w3.eth.get_transaction(tx_hash_bytes),
            self.async_w3.eth.get_transaction_receipt(tx_hash_bytes)
        )
        
        # 转换为可序列化的字典
        tx_dict = dict(tx)
        receipt_dict = dict(receipt)
        
        # 处理HexBytes类型
        for key, value in tx_dict.items():
            if isinstance(value, HexBytes):
                tx_dict[key] = value.hex()
            elif isinstance(value, bytes):
                tx_dict[key] = "0x" + value.hex()
        
        for key, value in receipt_dict.items():
            if isinstance(value, HexBytes):
                receipt_dict[key] = value.hex()
            elif isinstance(value, bytes):
                receipt_dict[key] = "0x" + value.hex()
        
        # 合并交易和收据信息
        result = {**tx_dict, "receipt": receipt_dict}
        return result
    
    def get_transactions_by_address(self, address: str, start_block: int = 0, end_block: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取地址的所有交易（注意：这个方法效率很低，应该使用第三方API如Etherscan）
        
        Args:
            address: 要查询的地址
            start_block: 起始区块
            end_block: 结束区块，默认为最新区块
            
        Returns:
            List[Dict]: 交易列表
        """
        address = Web3.to_checksum_address(address)
        end_block = end_block or self.w3.eth.block_number
        
        transactions = []
        for block_num in range(start_block, end_block + 1):
            block = self.w3.eth.get_block(block_num, full_transactions=True)
            block_txs = block['transactions']
            
            for tx in block_txs:
                tx_dict = dict(tx)
                if tx_dict.get('from') == address or tx_dict.get('to') == address:
                    # 处理HexBytes类型
                    for key, value in tx_dict.items():
                        if isinstance(value, HexBytes):
                            tx_dict[key] = value.hex()
                        elif isinstance(value, bytes):
                            tx_dict[key] = "0x" + value.hex()
                    
                    transactions.append(tx_dict)
        
        return transactions
    
    def get_token_balance(self, token_address: str, wallet_address: str) -> float:
        """
        获取ERC20代币余额
        
        Args:
            token_address: 代币合约地址
            wallet_address: 钱包地址
            
        Returns:
            float: 代币余额
        """
        # ERC20 ABI
        abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]
        
        # 创建合约实例
        token_address = Web3.to_checksum_address(token_address)
        wallet_address = Web3.to_checksum_address(wallet_address)
        token_contract = self.w3.eth.contract(address=token_address, abi=abi)
        
        # 获取代币精度
        decimals = token_contract.functions.decimals().call()
        
        # 获取余额
        balance_wei = token_contract.functions.balanceOf(wallet_address).call()
        balance = balance_wei / (10 ** decimals)
        
        return balance
    
    async def get_token_balance_async(self, token_address: str, wallet_address: str) -> float:
        """
        异步获取ERC20代币余额
        
        Args:
            token_address: 代币合约地址
            wallet_address: 钱包地址
            
        Returns:
            float: 代币余额
        """
        # ERC20 ABI
        abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            },
            {
                "constant": True,
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "type": "function"
            }
        ]
        
        # 创建合约实例
        token_address = Web3.to_checksum_address(token_address)
        wallet_address = Web3.to_checksum_address(wallet_address)
        token_contract = self.async_w3.eth.contract(address=token_address, abi=abi)
        
        # 获取代币精度和余额
        decimals, balance_wei = await asyncio.gather(
            token_contract.functions.decimals().call(),
            token_contract.functions.balanceOf(wallet_address).call()
        )
        
        balance = balance_wei / (10 ** decimals)
        return balance


# 创建单例实例
eth_client = None

def get_eth_client() -> EthClient:
    """
    获取以太坊客户端单例实例
    
    Returns:
        EthClient: 以太坊客户端实例
    """
    global eth_client
    if eth_client is None:
        eth_client = EthClient()
    return eth_client 