"""
区块链数据存储和缓存模块
提供数据持久化和缓存功能
"""

import os
import time
import logging
import json
import sqlite3
import functools
from typing import Any, Dict, List, Optional, Callable, Tuple, Union

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认缓存过期时间（秒）
DEFAULT_CACHE_EXPIRY = 3600  # 1小时

class BlockchainDataStore:
    """区块链数据存储类，提供数据缓存和持久化功能"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据存储
        
        参数:
            db_path: SQLite数据库文件路径，如果为None，则使用内存数据库
        """
        # 获取数据库路径
        self.db_path = db_path or os.getenv("DB_PATH", ":memory:")
        logger.info(f"初始化数据存储，数据库路径: {self.db_path}")
        
        # 创建内存缓存
        self.cache = {}
        
        # 初始化数据库
        self._initialize_db()
    
    def _initialize_db(self):
        """初始化数据库表"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # 创建缓存表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            timestamp INTEGER
        )
        ''')
        
        # 创建地址余额表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS address_balances (
            address TEXT PRIMARY KEY,
            balance_wei TEXT,
            balance_eth REAL,
            block_number INTEGER,
            timestamp INTEGER
        )
        ''')
        
        # 创建交易收据表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transaction_receipts (
            hash TEXT PRIMARY KEY,
            block_number INTEGER,
            from_address TEXT,
            to_address TEXT,
            status INTEGER,
            timestamp INTEGER
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("数据库表初始化完成")
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用列名访问
        return conn
    
    def store_cache(self, key: str, value: Any, expiry: int = DEFAULT_CACHE_EXPIRY):
        """
        存储缓存数据
        
        参数:
            key: 缓存键
            value: 缓存值
            expiry: 过期时间（秒）
        """
        # 存储到内存缓存
        self.cache[key] = {
            "value": value,
            "timestamp": int(time.time())
        }
        
        # 存储到数据库（如果不是内存数据库）
        if self.db_path != ":memory:":
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 序列化值
                serialized_value = json.dumps(value)
                
                # 存储数据
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (key, value, timestamp) VALUES (?, ?, ?)",
                    (key, serialized_value, int(time.time()))
                )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"存储缓存到数据库失败: {str(e)}")
    
    def get_cache(self, key: str, max_age: int = DEFAULT_CACHE_EXPIRY) -> Optional[Any]:
        """
        获取缓存数据
        
        参数:
            key: 缓存键
            max_age: 最大缓存年龄（秒）
            
        返回:
            缓存值，如果不存在或已过期则返回None
        """
        # 检查内存缓存
        if key in self.cache:
            cache_entry = self.cache[key]
            current_time = int(time.time())
            
            # 检查是否过期
            if current_time - cache_entry["timestamp"] <= max_age:
                return cache_entry["value"]
            else:
                # 从内存缓存中删除过期数据
                del self.cache[key]
        
        # 如果内存缓存中不存在或已过期，检查数据库
        if self.db_path != ":memory:":
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 查询缓存
                cursor.execute(
                    "SELECT value, timestamp FROM cache WHERE key = ?",
                    (key,)
                )
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    value = json.loads(row["value"])
                    timestamp = row["timestamp"]
                    current_time = int(time.time())
                    
                    # 检查是否过期
                    if current_time - timestamp <= max_age:
                        # 更新内存缓存
                        self.cache[key] = {
                            "value": value,
                            "timestamp": timestamp
                        }
                        return value
            except Exception as e:
                logger.error(f"从数据库获取缓存失败: {str(e)}")
        
        return None
    
    def clear_cache(self):
        """清除所有缓存"""
        # 清除内存缓存
        self.cache.clear()
        
        # 清除数据库缓存
        if self.db_path != ":memory:":
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 删除缓存数据
                cursor.execute("DELETE FROM cache")
                
                conn.commit()
                conn.close()
                
                logger.info("数据库缓存已清除")
            except Exception as e:
                logger.error(f"清除数据库缓存失败: {str(e)}")
    
    def clear_expired_cache(self, max_age: int = DEFAULT_CACHE_EXPIRY):
        """
        清除过期缓存
        
        参数:
            max_age: 最大缓存年龄（秒）
        """
        # 清除内存中的过期缓存
        current_time = int(time.time())
        expired_keys = []
        
        for key, cache_entry in self.cache.items():
            if current_time - cache_entry["timestamp"] > max_age:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        # 清除数据库中的过期缓存
        if self.db_path != ":memory:":
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # 删除过期缓存
                cursor.execute(
                    "DELETE FROM cache WHERE timestamp < ?",
                    (current_time - max_age,)
                )
                
                deleted_count = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                logger.info(f"已清除 {deleted_count} 条过期数据库缓存")
            except Exception as e:
                logger.error(f"清除过期数据库缓存失败: {str(e)}")
    
    def get_address_balance(self, address: str, max_age: int = DEFAULT_CACHE_EXPIRY) -> Optional[Dict[str, Any]]:
        """获取地址余额"""
        cache_key = f"balance_{address}"
        return self.get_cache(cache_key, max_age)
    
    def store_address_balance(self, address: str, balance_data: Dict[str, Any]):
        """存储地址余额"""
        cache_key = f"balance_{address}"
        self.store_cache(cache_key, balance_data)
        
        # 同时存储到余额表
        if self.db_path != ":memory:":
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO address_balances 
                    (address, balance_wei, balance_eth, block_number, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        address,
                        str(balance_data.get("balance_wei", "0")),
                        float(balance_data.get("balance_eth", 0)),
                        int(balance_data.get("block_number", 0)),
                        int(time.time())
                    )
                )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"存储地址余额到数据库失败: {str(e)}")
    
    def get_transaction_receipt(self, tx_hash: str, max_age: int = DEFAULT_CACHE_EXPIRY) -> Optional[Dict[str, Any]]:
        """获取交易收据"""
        cache_key = f"transaction_{tx_hash}"
        return self.get_cache(cache_key, max_age)
    
    def store_transaction_receipt(self, tx_hash: str, receipt_data: Dict[str, Any]):
        """存储交易收据"""
        cache_key = f"transaction_{tx_hash}"
        self.store_cache(cache_key, receipt_data)
        
        # 同时存储到交易表
        if self.db_path != ":memory:":
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO transaction_receipts 
                    (hash, block_number, from_address, to_address, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx_hash,
                        int(receipt_data.get("block_number", 0)),
                        receipt_data.get("from", ""),
                        receipt_data.get("to", ""),
                        int(receipt_data.get("status", 0)),
                        int(time.time())
                    )
                )
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"存储交易收据到数据库失败: {str(e)}")

# 单例数据存储实例
_data_store = None

def get_data_store() -> BlockchainDataStore:
    """获取数据存储实例（单例模式）"""
    global _data_store
    if _data_store is None:
        db_path = os.getenv("DB_PATH")
        _data_store = BlockchainDataStore(db_path)
    return _data_store

def cached(key_prefix: str, ttl: int = DEFAULT_CACHE_EXPIRY):
    """
    缓存装饰器，用于缓存函数返回值
    
    参数:
        key_prefix: 缓存键前缀
        ttl: 缓存过期时间（秒）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            args_str = "_".join(str(arg) for arg in args)
            kwargs_str = "_".join(f"{key}_{value}" for key, value in sorted(kwargs.items()))
            cache_key = f"{key_prefix}_{func.__name__}_{args_str}_{kwargs_str}"
            
            # 获取数据存储实例
            data_store = get_data_store()
            
            # 尝试从缓存获取
            cached_result = data_store.get_cache(cache_key, ttl)
            if cached_result is not None:
                logger.debug(f"从缓存获取结果: {cache_key}")
                return cached_result
            
            # 执行原函数
            result = func(*args, **kwargs)
            
            # 存储结果到缓存
            data_store.store_cache(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator 