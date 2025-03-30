# CI/CD 流程与配置指南

本文档详细介绍 SmartAI 项目的持续集成和持续部署(CI/CD)流程，包括环境配置、工作流程和最佳实践。

## CI/CD 概览

SmartAI 项目使用 GitHub Actions 作为 CI/CD 平台，实现了以下自动化流程：

1. **代码质量检查**：自动化代码审查
2. **自动化测试**：单元测试、集成测试和端到端测试
3. **构建过程**：自动化构建前端和后端应用
4. **部署流程**：自动部署到不同环境

## 工作流配置

所有 CI/CD 工作流配置位于 `.github/workflows/` 目录下。主要工作流包括：

### 1. 主工作流 (main.yml)

当代码推送到主分支或创建针对主分支的 PR 时触发，执行完整的测试、构建和部署流程。

```yaml
name: Main Workflow
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

### 2. 前端测试工作流 (frontend-tests.yml)

专门针对前端代码的测试工作流，当前端代码更改时触发。

```yaml
name: Frontend Tests
on:
  push:
    paths:
      - "frontend/**"
  pull_request:
    paths:
      - "frontend/**"
```

### 3. 后端测试工作流 (backend-tests.yml)

专门针对后端代码的测试工作流，当后端代码更改时触发。

```yaml
name: Backend Tests
on:
  push:
    paths:
      - "backend/**"
  pull_request:
    paths:
      - "backend/**"
```

### 4. 部署工作流 (deploy.yml)

用于部署代码到不同环境的工作流。仅在通过所有测试后触发。

## 环境配置

### GitHub Secrets 配置

CI/CD 流程需要以下 GitHub Secrets:

- `DOCKER_USERNAME`: Docker Hub 用户名
- `DOCKER_PASSWORD`: Docker Hub 密码
- `DEPLOY_SSH_KEY`: 部署服务器 SSH 私钥
- `MONGODB_URI`: MongoDB 连接 URI
- `REDIS_HOST`: Redis 主机
- `REDIS_PORT`: Redis 端口
- `API_SECRET_KEY`: API 密钥
- `ETHERSCAN_API_KEY`: Etherscan API 密钥

### 环境变量

根据不同的部署环境，使用如下环境变量：

- `NODE_ENV`: 环境名称 (development, testing, production)
- `LOG_LEVEL`: 日志级别
- `PORT`: 应用端口
- `DATABASE_URL`: 数据库连接 URL
- `CACHE_ENABLED`: 缓存是否启用
- `WEB3_PROVIDER_URL`: Web3 提供者 URL

## 部署策略

### 环境定义

系统包含以下部署环境：

1. **开发环境(Development)**

   - 用途：开发和测试
   - 触发条件：手动触发或推送到 develop 分支
   - URL: https://dev.smartai.example.com

2. **测试环境(Staging)**

   - 用途：QA 测试和功能验收
   - 触发条件：成功合并到 main 分支
   - URL: https://staging.smartai.example.com

3. **生产环境(Production)**
   - 用途：最终用户使用
   - 触发条件：手动批准生产发布
   - URL: https://smartai.example.com

### 部署流程

1. **构建阶段**：

   - 安装依赖
   - 运行测试
   - 构建前端和后端应用
   - 创建 Docker 镜像

2. **部署阶段**：

   - 推送 Docker 镜像到仓库
   - 执行部署脚本
   - 更新服务
   - 运行数据库迁移(如需要)

3. **验证阶段**：
   - 执行冒烟测试
   - 验证端点可访问
   - 检查日志是否有错误

### 回滚流程

如果部署失败，系统支持以下回滚策略：

1. 自动回滚到上一个稳定版本
2. 保留历史 Docker 镜像以便快速回滚
3. 提供手动回滚命令

## 本地开发流程

### 预提交检查

使用 husky 和 lint-staged 在提交前自动运行代码质量检查：

```bash
# 安装husky和lint-staged
yarn add --dev husky lint-staged

# 配置预提交钩子
npx husky add .husky/pre-commit "npx lint-staged"
```

配置 lint-staged 在 package.json 中：

```json
{
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{css,scss}": ["prettier --write"]
  }
}
```

### 本地测试工作流

本地开发环境推荐的工作流：

1. 创建功能分支：`git checkout -b feature/new-feature`
2. 开发功能并提交代码：`git commit -m "Add new feature"`
3. 本地运行测试：`yarn test`
4. 推送代码：`git push origin feature/new-feature`
5. 创建 Pull Request

## Docker 部署

### Docker 镜像构建

前端和后端分别有独立的 Dockerfile：

1. 前端 Dockerfile (`frontend/Dockerfile`):

```dockerfile
FROM node:16-alpine as builder
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile
COPY . .
RUN yarn build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

2. 后端 Dockerfile (`backend/Dockerfile`):

```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile --production
COPY . .
EXPOSE 3000
CMD ["node", "src/index.js"]
```

### Docker Compose 配置

用于本地开发和测试的 docker-compose.yml：

```yaml
version: "3"
services:
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - PORT=3000
      - MONGODB_URI=mongodb://mongo:27017/smartai
      - REDIS_HOST=redis
    depends_on:
      - mongo
      - redis

  mongo:
    image: mongo:4.4
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  mongo-data:
```

## 持续监控

### 性能监控

使用 Prometheus 和 Grafana 监控应用性能：

1. API 响应时间
2. 错误率
3. 资源使用情况

### 部署通知

部署状态通过以下渠道通知：

1. GitHub Actions 状态更新
2. Slack 通知
3. 电子邮件通知(仅限重要事件)

## 最佳实践

1. **频繁集成**：鼓励开发人员频繁提交小型更改
2. **自动化测试**：保持高测试覆盖率
3. **特性分支**：为每个功能创建独立分支
4. **环境一致性**：开发、测试和生产环境保持配置一致
5. **安全性检查**：自动化扫描安全漏洞
6. **版本控制**：遵循语义化版本控制
7. **文档更新**：代码更改应包含相应的文档更新

## 故障排除

常见 CI/CD 问题及解决方案：

1. **测试失败**：检查测试日志，修复失败的测试
2. **构建错误**：验证依赖关系，检查构建配置
3. **部署失败**：检查凭据和网络连接，查看服务器日志
4. **环境配置**：确保所有必要的环境变量已正确设置

## 参考资源

- [GitHub Actions 文档](https://docs.github.com/cn/actions)
- [Docker 文档](https://docs.docker.com/)
- [Jest 测试框架](https://jestjs.io/docs/zh-Hans/getting-started)
- [Cypress 端到端测试](https://docs.cypress.io/)
