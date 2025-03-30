# 区块链情报分析平台 API

## 项目概述

区块链情报分析平台 API 提供了一系列用于查询和分析区块链数据的接口。这些接口包括查询地址余额、交易收据、合约调用等功能，支持与 Etherscan 和 Alchemy API 的集成。

## 主要功能

- 查询以太坊地址余额
- 查询交易收据和详情
- 智能合约函数调用
- 查询地址代币余额
- 查询地址交易历史
- 与 Etherscan/Alchemy API 集成，提供更丰富的数据查询能力

## 安装和设置

### 依赖项

- Python 3.11+
- FastAPI
- Web3.py
- 其他依赖详见 `requirements-py311.txt`

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/chainintellai.git
cd chainintellai
```

2. 创建并激活虚拟环境

```bash
# 使用Python 3.11创建虚拟环境
python3.11 -m venv venv311
source venv311/bin/activate  # Linux/Mac
venv311\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r fastapi-backend/requirements-py311.txt
```

4. 配置环境变量

创建`.env`文件，参考`.env.example`填写必要的配置：

```
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_INFURA_KEY
INFURA_API_KEY=YOUR_INFURA_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
ALCHEMY_API_KEY=YOUR_ALCHEMY_KEY
```

## 启动 API 服务

```bash
cd fastapi-backend
python blockchain_api.py
```

服务默认会在 `http://localhost:8001` 上启动。

## API 端点说明

### 基础端点

- **GET /** - API 根端点，返回 API 状态信息
- **GET /health** - 健康检查端点

### 区块链数据查询端点

- **GET /api/v1/blockchain/address/{address}** - 查询以太坊地址余额

  - 参数: `address` (路径参数) - 以太坊地址
  - 返回: 地址余额信息（ETH 余额、Wei 余额、区块号）

- **GET /api/v1/blockchain/transaction/{tx_hash}** - 查询交易收据

  - 参数: `tx_hash` (路径参数) - 交易哈希
  - 返回: 交易收据信息（区块号、发送方、接收方、状态等）

- **POST /api/v1/blockchain/contract/call** - 调用合约函数

  - 请求体:
    ```json
    {
      "contract_address": "0x...",  // 合约地址
      "function_name": "balanceOf",  // 函数名称
      "abi": [...],  // 合约ABI（可选，如不提供会尝试从Etherscan获取）
      "args": ["0x..."],  // 函数参数
      "kwargs": {}  // 函数关键字参数
    }
    ```
  - 返回: 函数调用结果

- **GET /api/v1/blockchain/address/{address}/tokens** - 查询地址代币余额

  - 参数: `address` (路径参数) - 以太坊地址
  - 返回: 地址持有的所有代币余额列表

- **GET /api/v1/blockchain/address/{address}/transactions** - 查询地址交易历史
  - 参数:
    - `address` (路径参数) - 以太坊地址
    - `start_block` (查询参数, 默认: 0) - 开始区块
    - `end_block` (查询参数, 默认: 99999999) - 结束区块
    - `page` (查询参数, 默认: 1) - 页码
    - `offset` (查询参数, 默认: 10) - 每页记录数
  - 返回: 地址的交易记录列表

## Etherscan/Alchemy API 集成

本 API 支持与 Etherscan 和 Alchemy API 集成，以提供更丰富的数据查询能力。当配置了相应的 API 密钥后，系统会优先使用这些外部 API 查询数据，从而获取更准确和更全面的区块链数据。

### Etherscan API 功能

- 查询地址 ETH 余额
- 查询交易收据
- 查询地址交易历史
- 获取合约 ABI

### Alchemy API 功能

- 查询地址 ETH 余额
- 查询交易收据
- 查询地址代币余额

### 配置方法

1. 在`.env`文件中设置相应的 API 密钥：

```
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
ALCHEMY_API_KEY=YOUR_ALCHEMY_KEY
```

2. 重启 API 服务以应用新的配置

## 模拟模式

当无法连接到以太坊节点或者未配置外部 API 密钥时，系统会自动切换到模拟模式，返回模拟数据。这对于开发和测试非常有用。

## 测试

使用提供的测试脚本测试 API 功能：

```bash
python test_blockchain_api.py
```

可选参数:

- `--url` - API 基础 URL（默认: http://localhost:8001）
- `--address` - 测试用的以太坊地址
- `--tx` - 测试用的交易哈希
- `--skip-contract` - 跳过合约调用测试

## 错误处理

API 使用 HTTP 状态码表示请求处理状态：

- 200: 请求成功
- 400: 请求参数错误
- 404: 资源未找到
- 500: 服务器内部错误

错误响应格式:

```json
{
  "detail": "错误信息"
}
```

## 开发说明

### 模块结构

- `blockchain_api.py` - API 主入口和路由定义
- `app/blockchain/ethereum.py` - 以太坊交互模块，包含 Etherscan 和 Alchemy API 集成
- `app/blockchain/contract.py` - 智能合约交互模块

### 扩展 API

要添加新的 API 端点，请在`blockchain_api.py`中定义新的路由和处理函数。

### 添加新的数据源

要支持新的数据源，可以在`app/blockchain/ethereum.py`中添加新的客户端类和集成代码。

## 局限性

- 当前版本主要支持以太坊主网和兼容 EVM 的区块链
- 某些高级查询可能需要外部 API 支持
- 区块链重组可能导致历史数据不准确

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。

## 许可证

[MIT License](LICENSE)
