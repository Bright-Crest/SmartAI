# 区块链 API 数据持久化与缓存功能

## 功能概述

区块链 API 现已集成数据持久化和缓存功能，通过 SQLite 数据库存储查询结果并实现多级缓存机制，显著提升 API 性能、降低外部依赖并减少网络请求。

## 主要特性

1. **多级缓存机制**

   - 内存缓存：提供最快的数据访问速度
   - SQLite 持久化存储：在应用重启后保留数据
   - 可配置的缓存过期时间

2. **支持的缓存数据类型**

   - 地址余额数据
   - 交易收据数据
   - 代币余额数据
   - 交易历史数据
   - 合约调用结果
   - 合约事件数据

3. **智能缓存管理**
   - 自动过期机制
   - 缓存统计和监控
   - 手动清理接口

## 配置选项

可以通过环境变量或`.env`文件配置缓存行为：

```
# 是否启用缓存
CACHE_ENABLED=true

# 默认缓存过期时间（秒）
CACHE_EXPIRY=3600

# 地址余额缓存过期时间
ADDRESS_CACHE_EXPIRY=3600

# 交易收据缓存过期时间
TX_CACHE_EXPIRY=86400

# 代币余额缓存过期时间
TOKEN_CACHE_EXPIRY=3600

# 合约调用缓存过期时间
CONTRACT_CACHE_EXPIRY=3600

# 合约事件缓存过期时间
EVENTS_CACHE_EXPIRY=86400

# 数据库文件路径
DB_PATH=/path/to/blockchain_data.db
```

## API 接口

### 缓存管理接口

#### 获取缓存统计信息

```
GET /cache/stats
```

响应示例：

```json
{
  "status": "正常",
  "stats": {
    "total_items": 42,
    "types": {
      "balance": 10,
      "tx": 15,
      "tokens": 5,
      "txs": 5,
      "contract": 5,
      "events": 2,
      "other": 0
    }
  }
}
```

#### 清除缓存

```
POST /cache/clear
```

请求体：

```json
{
  "expired_only": true
}
```

响应示例：

```json
{
  "status": "成功",
  "message": "已清除过期缓存"
}
```

## 缓存工作流程

1. **读取数据流程**：

   - API 接收请求 → 检查内存缓存 → 检查数据库 → 从区块链获取数据 → 更新缓存

2. **写入数据流程**：

   - 获取区块链数据 → 同时更新内存缓存和数据库

3. **缓存失效**：
   - 基于时间的自动过期
   - 通过 API 手动清理

## 性能优势

1. **减少外部 API 调用**：减轻对 Etherscan/Alchemy/Infura 等服务的依赖
2. **降低延迟**：常用数据可以直接从内存或本地数据库读取
3. **提高稳定性**：即使外部服务暂时不可用，缓存数据仍可访问
4. **降低成本**：减少对付费 API 的调用次数

## 使用示例

### 带缓存的地址余额查询

首次查询时，数据将从区块链获取并缓存：

```
GET /address/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

后续相同查询将直接从缓存返回，显著提高响应速度。

### 合约调用结果缓存

对于相同参数的合约调用，API 会缓存结果以提高性能：

```
POST /contract/call
{
  "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
  "abi": [...],
  "function_name": "balanceOf",
  "args": ["0x742d35Cc6634C0532925a3b844Bc454e4438f44e"]
}
```

## 开发者信息

缓存装饰器可以用于自定义函数：

```python
from app.data import cached

@cached(ttl=3600, key_prefix="my_function")
def my_function(param1, param2):
    # 函数逻辑
    return result
```

数据存储实例可以在自定义代码中使用：

```python
from app.data import get_data_store

data_store = get_data_store()
data_store.store_custom_data(...)
```
