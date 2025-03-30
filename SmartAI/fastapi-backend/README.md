# 极简区块链 API

这是一个提供以太坊区块链数据模拟访问的 API 服务，专为开发测试环境设计。该 API 不需要连接真实的以太坊节点，完全使用模拟数据进行响应。

## 特点

- 独立运行，无需外部依赖
- 模拟以太坊区块链数据，无需实际区块链连接
- 支持地址余额查询、交易信息查询、代币余额查询和合约调用
- 使用 FastAPI 构建，性能出色
- 包含完善的错误处理和日志记录

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/blockchain-api.git

# 进入项目目录
cd blockchain-api

# 安装依赖
pip install fastapi uvicorn
```

## 运行

```bash
# 直接运行
python blockchain_mini_api.py

# 或者使用uvicorn
uvicorn blockchain_mini_api:app --reload --port 8001
```

服务将在 http://localhost:8001 启动。

## API 端点

### 基础信息

- `GET /` - API 基本信息
- `GET /health` - 健康检查

### 区块链数据

- `GET /address/{address}` - 获取 ETH 余额
- `GET /transaction/{tx_hash}` - 获取交易收据
- `GET /address/{address}/tokens` - 获取 ERC20 代币余额
- `POST /contract/call` - 调用智能合约

## 使用示例

### 查询地址余额

```bash
curl http://localhost:8001/address/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

响应：

```json
{
  "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
  "balance_wei": "1000000000000000000",
  "balance_eth": 1.0,
  "block_number": 12345678
}
```

### 查询交易信息

```bash
curl http://localhost:8001/transaction/0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

响应：

```json
{
  "hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  "block_number": 12345678,
  "from": "0x1111111111111111111111111111111111111111",
  "to": "0x2222222222222222222222222222222222222222",
  "status": 1
}
```

### 查询代币余额

```bash
curl http://localhost:8001/address/0x742d35Cc6634C0532925a3b844Bc454e4438f44e/tokens
```

响应：

```json
[
  {
    "contract_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "symbol": "USDT",
    "name": "Tether USD",
    "decimals": 6,
    "balance": "1000000000"
  },
  {
    "contract_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "symbol": "USDC",
    "name": "USD Coin",
    "decimals": 6,
    "balance": "2000000000"
  }
]
```

### 调用合约

```bash
curl -X POST http://localhost:8001/contract/call \
  -H "Content-Type: application/json" \
  -d '{
    "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "abi": "[]",
    "function_name": "totalSupply",
    "args": []
  }'
```

响应：

```json
{
  "contract_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
  "function": "totalSupply",
  "result": 1000000000000000000000000
}
```

## 文档

API 文档可以通过访问 http://localhost:8001/docs 获得，这是由 FastAPI 自动生成的 Swagger UI 文档。

## 自定义

您可以通过修改`blockchain_mini_api.py`文件中的模拟数据变量来自定义返回结果：

- `MOCK_BALANCES` - 地址余额
- `MOCK_RECEIPTS` - 交易收据
- `MOCK_TOKEN_BALANCES` - 代币余额
- `MOCK_CONTRACT_RESULTS` - 合约调用结果

## 许可证

MIT
