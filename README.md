# SmartAI - 区块链风险智能分析平台

![版本](https://img.shields.io/badge/版本-1.0.0-blue.svg)
![测试](https://img.shields.io/badge/测试-通过-green.svg)
![许可证](https://img.shields.io/badge/许可证-MIT-yellow.svg)

SmartAI 是一个区块链风险智能分析平台，专注于区块链钱包分析、投资组合评估和风险监控。本平台帮助用户洞察钱包资产分布、交易历史及风险状况，为投资决策提供数据支持。

## 🚀 功能亮点

- **钱包分析**：全面分析区块链钱包资产构成与价值
- **投资组合评估**：提供详细的加密资产投资组合分析
- **交易历史**：追踪并可视化历史交易记录
- **风险监控**：评估不同加密资产和交易的风险度

## 📋 项目架构

项目采用现代化全栈架构：

- **前端**：React + Next.js + Chakra UI
- **后端**：Python + FastAPI + Web3.py
- **数据分析**：Pandas + NumPy
- **区块链交互**：Web3.py

## AI

### 黑盒因子蒸馏可行性验证

参见[黑盒因子蒸馏可行性验证](distill-test/README.md)

### HiFi-GNN: A Multi-dimensional Heterogeneous Graph Neural Network for Smart Money Identification

[多维异构聪明钱图神经网络识别系统](SmartAI/AI/README.md)

Stephenzhu:
[图片]


## 🔧 快速开始

### 前提条件

- Node.js 16+
- Python 3.8+
- Yarn
- 区块链 API 密钥（如 Infura, Etherscan 等）

### 本地开发环境设置

1. 克隆仓库

   ```bash
   git clone https://github.com/Bright-Crest/SmartAI.git
   cd SmartAI
   ```

2. 安装前端依赖

   ```bash
   cd frontend
   yarn install
   ```

3. 安装后端依赖

   ```bash
   cd fastapi-backend
   python -m pip install -r requirements.txt
   ```

4. 设置环境变量

   ```bash
   # 前端环境变量
   cd frontend
   cp .env.example .env.local

   # 后端环境变量
   cd fastapi-backend
   cp .env.example .env
   # 编辑.env文件，填入必要的API密钥和配置
   ```

5. 启动开发服务器

   ```bash
   # 启动前端
   cd frontend
   yarn dev

   # 启动后端 (新终端)
   cd fastapi-backend
   python minimal_app.py  # 使用简化版API
   # 或
   python blockchain_mini_api.py  # 使用带有区块链功能的API
   ```

### 访问应用

- 前端界面: http://localhost:3000
- API 文档: http://localhost:8003/docs

## 📱 主要功能

### 钱包分析

输入区块链钱包地址，可获取：

- 总资产价值
- 资产构成与分布
- 代币持仓详情
- 历史交易记录

### API 接口

主要 API 端点：

- `GET /wallet/{address}` - 获取钱包分析数据
- `GET /health` - 健康检查

## 📚 项目文档

- [安装指南](SmartAI/docs/SETUP_GUIDE.md) - 详细的环境配置指南
- [项目架构](SmartAI/docs/ARCHITECTURE.md) - 系统架构与组件交互
- [API 文档](SmartAI/docs/api.md) - API 端点与使用说明
- [配置指南](SmartAI/docs/configuration.md) - 环境配置与设置
- [CI/CD 文档](SmartAI/docs/CI_CD.md) - 持续集成与部署流程

## 🤝 贡献

欢迎通过 Issues 和 Pull Requests 参与项目改进！

## 📜 许可证

本项目采用 MIT 许可证 - 详情见[LICENSE](SmartAI/LICENSE)文件

---

© 2025 SmartAI. 保留所有权利。
