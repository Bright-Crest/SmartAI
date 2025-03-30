# SmartAI 开发环境设置指南

本文档为开发人员提供了详细的环境设置指南，包括依赖安装、配置项说明和常见开发问题解决方案。

## 目录

- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [环境配置详解](#环境配置详解)
- [本地开发工作流](#本地开发工作流)
- [代码规范](#代码规范)
- [问题排查](#问题排查)

## 环境要求

### 必要软件

| 软件           | 最低版本 | 推荐版本 | 说明           |
| -------------- | -------- | -------- | -------------- |
| Node.js        | 16.0.0   | 18.16.0  | 运行时环境     |
| Yarn           | 1.22.0   | 3.6.0    | 包管理器       |
| Git            | 2.30.0   | 2.40.0   | 版本控制       |
| MongoDB        | 4.4.0    | 6.0.0    | 数据库         |
| Redis          | 6.0.0    | 7.0.0    | 缓存服务       |
| Docker         | 20.10.0  | 24.0.0   | 容器环境(可选) |
| Docker Compose | 2.0.0    | 2.19.0   | 容器编排(可选) |

### IDE 推荐设置

推荐使用 Visual Studio Code 并安装以下插件：

- ESLint: 代码质量检查
- Prettier: 代码格式化
- Jest: 测试运行和调试
- Docker: Docker 容器管理
- MongoDB: 数据库管理
- GitLens: Git 历史查看

### VS Code 设置

在项目根目录创建 `.vscode/settings.json` 文件，推荐的配置如下：

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ],
  "jest.autoRun": "off"
}
```

## 安装步骤

### 快速安装

我们提供了一键安装脚本，可以自动完成大部分安装步骤：

```bash
# 添加执行权限
chmod +x setup.sh

# 运行脚本
./setup.sh
```

Windows 用户可以直接运行 `setup.bat` 文件。

### 手动安装

如果您更喜欢手动安装或需要更多控制，请按照以下步骤操作：

1. **克隆仓库**

```bash
git clone https://github.com/yourusername/SmartAI.git
cd SmartAI
```

2. **安装根目录依赖**

```bash
yarn install
```

3. **安装前端依赖**

```bash
cd frontend
yarn install
```

4. **安装后端依赖**

```bash
cd ../backend
yarn install
```

5. **创建环境配置文件**

```bash
# 根目录
cp .env.example .env

# 前端目录
cd frontend
cp .env.example .env

# 后端目录
cd ../backend
cp .env.example .env
```

6. **启动本地数据库** (如果未使用 Docker)

```bash
# 启动MongoDB
mongod --dbpath=/path/to/data --port 27017

# 启动Redis
redis-server
```

7. **启动开发服务器**

```bash
# 终端1 - 启动前端服务
cd frontend
yarn dev

# 终端2 - 启动后端服务
cd backend
yarn dev
```

## 环境配置详解

项目使用 `.env` 文件管理环境变量配置。以下是主要配置项的说明：

### 全局配置 (根目录 .env)

```
# 环境模式
NODE_ENV=development  # development, test, production

# 服务端口
PORT=3000

# 日志配置
LOG_LEVEL=debug  # debug, info, warn, error
```

### 前端配置 (frontend/.env)

```
# API 服务地址
NEXT_PUBLIC_API_URL=http://localhost:3000

# 主题配置
NEXT_PUBLIC_DEFAULT_THEME=light  # light, dark

# 特性标记
NEXT_PUBLIC_ENABLE_ANALYTICS=false
```

### 后端配置 (backend/.env)

```
# 数据库配置
DATABASE_URL=mongodb://localhost:27017/smartai
REDIS_HOST=localhost
REDIS_PORT=6379

# 区块链配置
ETHERSCAN_API_KEY=your_api_key
WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/your_infura_key

# 安全配置
JWT_SECRET=your_jwt_secret
JWT_EXPIRES_IN=7d
```

## 本地开发工作流

### 代码结构

项目的代码结构组织如下：

```
SmartAI/
├── frontend/               # 前端应用
│   ├── components/         # React组件
│   │   ├── pages/              # Next.js页面
│   │   ├── public/             # 静态资源
│   │   ├── styles/             # 样式文件
│   │   ├── utils/              # 前端工具函数
│   │   └── api/                # API客户端
│   ├── backend/                # 后端服务
│   │   ├── src/                # 源代码
│   │   │   ├── controllers/    # 控制器
│   │   │   ├── routes/         # 路由定义
│   │   │   ├── services/       # 服务逻辑
│   │   │   ├── middleware/     # 中间件
│   │   │   ├── database/       # 数据库交互
│   │   │   └── utils/          # 后端工具函数
│   │   └── tests/              # 测试文件
│   ├── shared/                 # 前后端共享代码
│   └── docs/                   # 项目文档
```

### 开发流程

1. **功能分支**

创建功能分支进行开发：

```bash
git checkout -b feature/your-feature-name
```

2. **运行测试**

定期运行测试，确保代码质量：

```bash
# 前端测试
cd frontend
yarn test

# 后端测试
cd backend
yarn test
```

3. **提交代码**

提交前确保通过代码检查：

```bash
# 运行代码检查
yarn lint

# 提交代码
git add .
git commit -m "feat: 添加新功能"
```

4. **创建 PR**

将功能分支推送到远程仓库并创建 PR：

```bash
git push origin feature/your-feature-name
```

## 代码规范

项目使用 ESLint 和 Prettier 进行代码风格检查和格式化。关键规则包括：

- 使用 2 空格缩进
- 使用单引号 (`'`)
- 句末不使用分号
- 箭头函数总是使用括号和花括号
- React 组件使用函数组件

详细规则请查看 `.eslintrc.js` 和 `.prettierrc` 文件。

## 问题排查

### 常见问题

#### 1. `yarn install` 失败

可能的解决方案：

```bash
# 清除 yarn 缓存
yarn cache clean

# 使用 --network-timeout 参数
yarn install --network-timeout 100000
```

#### 2. MongoDB 连接失败

检查 MongoDB 服务是否正在运行：

```bash
# 检查 MongoDB 进程
ps aux | grep mongod

# 手动启动 MongoDB
mongod --dbpath=/path/to/data
```

#### 3. 前端热重载不工作

检查以下几点：

- Next.js 版本是否更新
- Node.js 版本是否兼容
- `next.config.js` 中是否正确配置了热重载

#### 4. 区块链 API 调用失败

确保已正确设置 API 密钥：

```bash
# 检查 .env 文件中的 API 密钥
grep -r "ETHERSCAN_API_KEY" .env*

# 获取新的 API 密钥
# 访问 https://etherscan.io/apis
```

#### 5. 端口冲突

检查端口占用情况并终止相关进程：

```bash
# 查找占用端口的进程
lsof -i :3000

# 终止进程
kill -9 <PID>
```

### 调试工具

- **前端调试**：使用 React DevTools 和 Chrome DevTools
- **后端调试**：使用 VS Code 调试器或 `node --inspect`
- **API 调试**：使用 Postman 或 Insomnia

## 获取帮助

如果您遇到未在本文档中列出的问题，请：

1. 查看 [项目 Wiki](https://github.com/yourusername/SmartAI/wiki)
2. 在 GitHub Issues 中搜索类似问题
3. 向项目维护者寻求帮助

## 相关文档

- [项目架构](ARCHITECTURE.md)
- [API 文档](api.md)
- [部署指南](../DEPLOYMENT.md)
- [CI/CD 流程](CI_CD.md)
