# SmartAI API 接口统一集成文档

## 架构概述

SmartAI 平台采用微服务架构，使用不同的技术栈实现各种服务功能。本文档详细说明了 API 接口的统一策略和集成方法。

### 技术栈

1. **Express.js (Node.js)** - 提供通用 API 和数据处理服务

   - 端口: 3000
   - 路径前缀: `/api/*`
   - 主要职责: 业务逻辑、数据处理、基本风险分析
   - 特点: 高性能、事件驱动、适合处理大量并发请求

2. **FastAPI (Python)** - 提供高级风险分析功能

   - 端口: 8002
   - 路径前缀: `/risk/*`
   - 主要职责: 机器学习风险分析、用户画像聚类
   - 特点: 支持异步处理、自动文档生成、内置数据验证

3. **API 网关 (Nginx)** - 统一路由和响应格式
   - 端口: 80
   - 主要职责: 请求路由、负载均衡、统一入口
   - 特点: 高性能、可配置性强、支持反向代理

## API 响应格式统一

为了确保前端获得一致的 API 响应，我们统一了两个后端的响应格式：

### 标准响应结构

```typescript
// Express.js标准响应格式
interface ApiResponse<T = any> {
  code: number; // 状态码
  message: string; // 消息
  data?: T; // 数据
  requestId?: string; // 请求ID
  timestamp: number; // 时间戳
}

// FastAPI标准响应格式 (直接返回数据)
interface FastAPIResponse {
  // 直接返回业务数据
  [key: string]: any;
}
```

### 风险评分响应统一

为了统一 FastAPI 和 Express.js 的风险评分响应，我们采用以下字段映射：

| FastAPI 字段     | Express.js 字段 | 说明                  |
| ---------------- | --------------- | --------------------- |
| risk_score       | smartScore      | 标准化风险分数(0-100) |
| risk_level       | riskLevel       | 风险等级              |
| risk_description | riskDescription | 风险等级描述          |
| risk_factors     | (新增)          | 风险因素列表          |
| attention_points | (新增)          | 需要关注的点          |

### 前端适配器

前端通过适配器处理不同后端的响应格式差异：

```javascript
function adaptRiskResponse(response) {
  // 提取Express.js的data部分
  if (response.data && response.code !== undefined) {
    response = response.data;
  }

  // 统一字段名称 (保留原始字段以兼容旧代码)
  return {
    ...response,
    risk_score: response.smartScore || response.risk_score || 0,
    risk_level: response.riskLevel || response.risk_level || "unknown",
    risk_description:
      response.riskDescription || response.risk_description || "",
  };
}
```

## API 路由规则

API 网关根据以下规则路由请求到不同的后端服务：

```
/api/risk/(score|analyze|high-risk|kol) → FastAPI (Python)
/api/risk/metrics                        → Express.js (Node.js)
/api/*                                   → Express.js (Node.js)
```

## 具体 API 接口说明

### 1. 风险评分 API (FastAPI)

获取地址的风险评分和分析结果。

**请求**:

```
GET /api/risk/score/{address}
```

**参数**:

- `address`: 区块链地址 (必填)

**响应**:

```json
{
  "address": "0x1234567890abcdef1234567890abcdef12345678",
  "risk_score": 85.7,
  "risk_level": "high",
  "risk_description": "该地址存在高风险，与多个已知风险地址有交互",
  "risk_explanation": "该地址风险评分为85.70，风险等级为high。该地址存在高风险，与多个已知风险地址有交互",
  "risk_factors": [
    {
      "name": "异常资金流动模式",
      "description": "资金流动模式异常，影响风险评分。",
      "impact": 0.85,
      "category": "资金流动"
    }
  ],
  "attention_points": ["该地址存在明显风险，与该地址交互前请充分了解其背景。"],
  "dimensions": {
    "flow": 0.85,
    "behavior": 0.72,
    "association": 0.65,
    "historical": 0.45,
    "technical": 0.51
  },
  "features": {
    "transaction_count": 156,
    "event_count": 42,
    "age_days": 125,
    "tag_count": 7,
    "dimensions": {
      "flow": 0.85,
      "behavior": 0.72,
      "association": 0.65,
      "historical": 0.45,
      "technical": 0.51
    }
  }
}
```

### 2. 五维指标 API (Express.js)

获取地址风险的详细五维指标数据。

**请求**:

```
GET /api/risk/metrics/{address}
```

**参数**:

- `address`: 区块链地址 (必填)

**响应**:

```json
{
  "code": 200,
  "message": "五维指标数据获取成功",
  "data": [
    {
      "name": "资金流动",
      "key": "flow",
      "description": "资金流动模式分析",
      "score": 85,
      "weight": 0.35,
      "indicators": [
        {
          "name": "资金流量异常度",
          "value": 85,
          "description": "资金流入流出模式的异常程度",
          "isAnomaly": true
        },
        {
          "name": "大额交易比例",
          "value": 92,
          "description": "大额交易在总交易中的比例",
          "isAnomaly": true
        },
        {
          "name": "混币器关联度",
          "value": 34,
          "description": "与混币服务的关联程度",
          "isAnomaly": false
        }
      ]
    }
    // ... 其他维度
  ],
  "requestId": "f8e7d6c5b4a3210987654321fedcba09",
  "timestamp": 1679000000000
}
```

### 3. 批量风险评分 API (Express.js)

同时计算多个地址的风险评分。

**请求**:

```
POST /api/risk/batch-score
```

**请求体**:

```json
{
  "addresses": [
    "0x1234567890abcdef1234567890abcdef12345678",
    "0xabcdef1234567890abcdef1234567890abcdef12"
  ]
}
```

**响应**:

```json
{
  "code": 200,
  "message": "批量风险评分计算完成",
  "data": {
    "total": 2,
    "successful": 2,
    "failed": 0,
    "results": [
      {
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "risk_score": 85.7,
        "risk_level": "high",
        "risk_description": "该地址存在高风险，与多个已知风险地址有交互",
        "success": true
        // ... 其他字段
      },
      {
        "address": "0xabcdef1234567890abcdef1234567890abcdef12",
        "risk_score": 32.5,
        "risk_level": "medium",
        "risk_description": "该地址存在一定风险，请谨慎交互",
        "success": true
        // ... 其他字段
      }
    ]
  },
  "requestId": "a1b2c3d4e5f6789012345678fedcba98",
  "timestamp": 1679000000000
}
```

## 错误处理

两个后端服务使用统一的错误处理机制：

### Express.js 错误响应

```json
{
  "code": 400,
  "message": "无效的以太坊地址",
  "timestamp": 1679000000000,
  "requestId": "12345678abcdef0987654321"
}
```

### FastAPI 错误响应

```json
{
  "detail": "无效的以太坊地址"
}
```

### 前端错误处理

前端统一处理两种错误格式：

```javascript
try {
  const data = await getAddressRiskScore(address);
  // 处理成功响应
} catch (error) {
  // 提取错误消息
  const errorMessage = error.data?.detail || error.message || "未知错误";
  // 显示错误
  showErrorNotification(errorMessage);
}
```

## 部署与配置

### Nginx 配置

Nginx 作为 API 网关，统一路由请求：

```nginx
# FastAPI风险分析API (Python后端)
location /api/risk/ {
    # 判断请求类型，分发到不同后端
    if ($request_uri ~* "^/api/risk/(score|analyze|high-risk|kol)") {
        # 基础风险分析请求转发到FastAPI
        proxy_pass http://localhost:8002/risk/;
        rewrite ^/api/risk/(.*) /risk/$1 break;
    }

    if ($request_uri ~* "^/api/risk/metrics") {
        # 五维指标数据请求转发到Express
        proxy_pass http://localhost:3000/api/risk/;
    }

    # 默认转发到FastAPI
    proxy_pass http://localhost:8002/risk/;
    rewrite ^/api/risk/(.*) /risk/$1 break;
}

# Express.js其他API (Node.js后端)
location /api/ {
    proxy_pass http://localhost:3000/api/;
}
```

### 环境变量配置

前端通过环境变量配置 API 端点：

```
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:3000
NEXT_PUBLIC_RISK_API_URL=http://localhost:8002
```

## 最佳实践

1. **API 版本控制**:

   - 在 URL 中包含版本号 (如 `/api/v1/risk/score`)
   - 在请求头中指定版本 (`Accept: application/vnd.smartai.v1+json`)

2. **请求跟踪**:

   - 使用统一的 requestId 跟踪请求
   - 在日志中记录 requestId 便于故障排查

3. **性能优化**:

   - 使用缓存减少重复计算
   - 批量 API 减少请求次数
   - 启用 GZIP 压缩减少传输数据量

4. **扩展建议**:
   - 考虑引入 GraphQL 统一查询接口
   - 使用消息队列处理长时间运行的任务
   - 实现熔断机制处理服务故障
