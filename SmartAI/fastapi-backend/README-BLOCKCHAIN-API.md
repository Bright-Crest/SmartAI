# 区块链智能 API

## 项目简介

区块链智能 API 是一个提供以太坊区块链数据查询和智能合约交互功能的 RESTful API 服务。该 API 支持查询地址余额、交易历史、代币持有情况，以及调用智能合约函数等功能。

## 主要特性

- **区块链数据查询**：查询地址余额、交易收据、代币余额和交易历史
- **智能合约交互**：调用合约函数、获取合约事件日志
- **多数据源支持**：集成 Web3、Etherscan API 和 Alchemy API
- **自动降级机制**：当一个数据源不可用时自动尝试其他数据源
- **模拟模式**：在无法连接区块链时提供合理的模拟数据用于开发和测试
- **统一错误处理**：友好的错误信息和日志记录
- **API 文档**：自动生成的 API 文档（访问`/docs`）

## 技术栈

- FastAPI：高性能的异步 API 框架
- Web3.py：以太坊交互库
- Pydantic：数据验证
- Uvicorn：ASGI 服务器

## 快速开始

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/chainintellai.git
cd chainintellai/fastapi-backend
```

2. 安装依赖：

```bash
# 创建Python 3.11虚拟环境
python3.11 -m venv venv311
source venv311/bin/activate

# 安装依赖
pip install -r requirements-py311.txt
```

3. 配置环境变量：

创建`.env`文件并添加以下配置：

```
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
INFURA_API_KEY=YOUR_INFURA_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
ALCHEMY_API_KEY=YOUR_ALCHEMY_KEY
DEFAULT_MOCK_MODE=auto
```

### 运行

```bash
# 启动API服务
python blockchain_api.py
```

或者使用 Uvicorn 直接运行：

```bash
uvicorn blockchain_api:app --reload --port 8001
```

服务将在`http://localhost:8001`上运行，API 文档可在`http://localhost:8001/docs`查看。

## API 端点

### 基础端点

- `GET /` - 返回 API 状态信息
- `GET /health` - 健康检查端点

### 区块链数据端点

- `GET /api/v1/blockchain/address/{address}` - 获取地址 ETH 余额
- `GET /api/v1/blockchain/transaction/{tx_hash}` - 获取交易详情
- `GET /api/v1/blockchain/address/{address}/tokens` - 获取地址持有的代币
- `GET /api/v1/blockchain/address/{address}/transactions` - 获取地址交易历史

### 智能合约端点

- `POST /api/v1/blockchain/contract/call` - 调用智能合约函数
- `POST /api/v1/blockchain/contract/events` - 获取合约事件日志

## 使用示例

### 查询地址余额

```bash
curl -X GET "http://localhost:8001/api/v1/blockchain/address/0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
```

响应：

```json
{
  "success": true,
  "message": "操作成功",
  "data": {
    "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
    "balance_wei": "1000000000000000000",
    "balance_eth": 1.0,
    "block_number": 12345678
  }
}
```

### 调用合约函数

```bash
curl -X POST "http://localhost:8001/api/v1/blockchain/contract/call" \
  -H "Content-Type: application/json" \
  -d '{
    "contract_address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "function_name": "balanceOf",
    "args": ["0x742d35Cc6634C0532925a3b844Bc454e4438f44e"]
  }'
```

响应：

```json
{
  "success": true,
  "message": "操作成功",
  "data": {
    "contract_address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "function_name": "balanceOf",
    "result": "1000000000000000000"
  }
}
```

## 模拟模式

在没有区块链连接或 API 密钥时，服务会自动进入模拟模式，返回合理的模拟数据。这对于开发和测试非常有用。

设置`DEFAULT_MOCK_MODE=true`在`.env`文件中强制启用模拟模式，或者设置为`auto`让服务根据连接状态自动决定是否使用模拟模式。

## 错误处理

API 使用统一的错误响应格式：

```json
{
  "success": false,
  "error": {
    "code": 1001,
    "message": "无法连接到区块链节点",
    "details": {
      "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    }
  }
}
```

常见错误代码：

- 1001：区块链连接错误
- 1002：Etherscan API 错误
- 1003：Alchemy API 错误
- 1004：合约交互错误
- 1005：无效地址错误
- 1006：无效交易哈希错误
- 9999：未知错误

## 注意事项

- 对于生产环境，建议设置适当的 API 密钥限制和缓存机制
- 当使用 Infura、Etherscan 或 Alchemy API 时，请注意其使用限制和条款
- 合约交互可能受到区块链网络拥堵的影响
- 返回数据的格式可能会根据使用的数据源略有不同

## 贡献

欢迎贡献代码、报告问题或提出功能建议。请提交 PR 或 Issue 到项目仓库。

## 许可

[MIT 许可证](LICENSE)
