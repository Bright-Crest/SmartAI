# SmartAI 快速启动指南

本文档提供了快速搭建和运行 SmartAI 项目的步骤，适合开发者和非技术用户使用。

## 一键安装设置（推荐）

为了简化安装流程，我们提供了一键安装脚本，可以自动检查环境、安装依赖、配置设置并启动服务。

### Linux/macOS 用户

1. 打开终端
2. 切换到项目根目录
3. 运行以下命令：

```bash
# 添加执行权限
chmod +x setup.sh

# 运行脚本
./setup.sh
```

### Windows 用户

1. 打开命令提示符或 PowerShell
2. 切换到项目根目录
3. 直接双击`setup.bat`文件，或在命令行中运行：

```
setup.bat
```

### 脚本功能说明

安装脚本会自动执行以下操作：

- ✅ 检查系统依赖（Node.js、Yarn、Git 等）
- ✅ 创建必要的环境配置文件
- ✅ 安装项目依赖
- ✅ 提供启动选项
- ✅ 创建服务启动和停止脚本

## 手动安装（高级用户）

如果您希望手动控制安装过程，可以按照以下步骤操作：

### 前提条件

确保您的系统已安装以下软件：

- Node.js（v16 或更高版本）
- Yarn
- Git
- 可选：Docker 和 Docker Compose

### 步骤 1：克隆仓库（如果尚未克隆）

```bash
git clone https://github.com/yourusername/SmartAI.git
cd SmartAI
```

### 步骤 2：设置环境变量

```bash
cp .env.example .env
# 使用文本编辑器编辑.env文件，填入必要的API密钥和配置
```

### 步骤 3：安装依赖

```bash
# 安装根目录依赖
yarn install

# 安装前端依赖
cd frontend
yarn install
cd ..

# 安装后端依赖
cd backend
yarn install
cd ..
```

### 步骤 4：启动服务

```bash
# 方法1：分别启动前端和后端（推荐开发环境）
# 终端1
cd frontend
yarn dev

# 终端2
cd backend
yarn dev

# 方法2：使用Docker启动所有服务（需要安装Docker）
docker-compose up -d
```

## 验证安装

安装完成后，您可以通过以下 URL 访问应用程序：

- 前端界面：http://localhost:3000
- API 端点：http://localhost:3000/api

## 常见问题排查

如果遇到问题，请尝试以下解决方案：

### 1. 安装依赖失败

```bash
# 清除yarn缓存
yarn cache clean

# 重新安装
yarn install
```

### 2. 服务启动失败

检查端口占用情况：

```bash
# Linux/macOS
lsof -i :3000

# Windows
netstat -ano | findstr :3000
```

### 3. 无法连接到数据库

确保 MongoDB 和 Redis 服务正在运行：

```bash
# 检查MongoDB状态
mongod --version

# 检查Redis状态
redis-cli ping
```

## 更多资源

- [项目架构](docs/ARCHITECTURE.md)
- [API 文档](docs/api.md)
- [配置指南](docs/configuration.md)
- [CI/CD 文档](docs/CI_CD.md)

如有其他问题，请在 GitHub Issues 页面提问或联系项目维护者。
