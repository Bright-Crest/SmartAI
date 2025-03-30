# ChainIntelAI 前端项目

ChainIntelAI 是一个区块链智能分析平台，用于实时监控区块链交易，智能识别风险。

## 项目结构

```
frontend/
├── app/                    # Next.js 应用目录
│   ├── components/         # 可复用组件
│   ├── contexts/           # React 上下文
│   ├── hooks/              # 自定义 React hooks
│   ├── i18n/               # 国际化配置和翻译文件
│   ├── utils/              # 工具函数
│   ├── addresses/          # 地址画像页面
│   ├── alerts/             # 风险告警页面
│   ├── dashboard/          # 仪表盘页面
│   ├── transactions/       # 交易分析页面
│   ├── layout.js           # 布局组件
│   ├── page.js             # 首页组件
│   └── globals.css         # 全局样式
├── __tests__/              # 测试目录
│   └── components/         # 组件测试
├── cypress/                # E2E 测试
│   ├── e2e/                # E2E 测试文件
│   └── support/            # 测试支持文件
├── public/                 # 静态资源
├── .eslintrc.js            # ESLint 配置
├── .prettierrc             # Prettier 配置
├── jest.config.js          # Jest 配置
├── jest.setup.js           # Jest 设置
├── cypress.config.js       # Cypress 配置
├── next.config.js          # Next.js 配置
└── package.json            # 项目依赖
```

## 组件结构

我们将页面中的重复组件提取为可复用组件，包括：

- `PageHeader`: 页面标题组件
- `SearchBar`: 搜索栏组件
- `RiskBadge`: 风险等级标签组件
- `StatusBadge`: 状态标签组件
- `AddressFormatter`: 地址格式化组件
- `DateFormatter`: 日期格式化组件
- `StatCard`: 统计卡片组件
- `AlertsTable`: 风险告警表格组件
- `TransactionsTable`: 交易表格组件

## 工具函数

我们将重复的工具函数提取为可复用函数，包括：

- `formatters.js`: 格式化工具函数
- `api.js`: API 请求工具函数

## 自定义 Hooks

我们创建了一些自定义 hooks，用于处理常见的状态管理：

- `useSearch`: 搜索 hook，用于处理搜索功能
- `useFetch`: 数据获取 hook，用于处理数据获取
- `useLocalStorage`: 本地存储 hook，用于处理本地存储
- `useApi`: API 请求 hook，用于处理 API 请求

## 全局状态管理

我们使用 React Context API 实现了全局状态管理：

- `AppContext`: 应用全局状态
- `LocalStorageContext`: 本地存储状态
- `ThemeContext`: 主题状态

## 国际化支持

项目使用 i18next 和 react-i18next 实现国际化支持。翻译文件位于 `app/i18n/locales/` 目录下。

目前支持的语言:

- 中文 (zh-CN)
- 英文 (en-US)

## 测试

我们添加了测试配置和示例测试：

- Jest 单元测试和组件测试
- Cypress E2E 测试

## 代码规范

我们添加了代码规范配置：

- ESLint 配置
- Prettier 配置

## 优化内容

1. **组件复用**：提取了多个可复用组件，减少了代码重复
2. **状态管理**：使用 Context API 和自定义 hooks 管理状态，使代码更清晰
3. **工具函数**：提取了工具函数，减少了代码重复
4. **错误处理**：添加了错误处理和加载状态
5. **代码规范**：添加了 ESLint 和 Prettier 配置，统一代码风格
6. **测试**：添加了测试配置和示例测试
7. **国际化**：添加了多语言支持
8. **主题**：添加了暗黑模式支持

## 性能优化组件

项目中包含了多个性能优化的React组件示例，展示了不同的优化技术：

### 1. MemoizedComponent

使用 `React.memo` 和 `useCallback` 优化的组件示例，避免不必要的重新渲染。

```tsx
// 使用React.memo优化的子组件
const MemoizedChild = React.memo(({ name, count, onClick }: ChildProps) => {
  // 组件内容
});

// 使用useCallback优化事件处理函数
const handleClick = useCallback(() => {
  // 处理逻辑
}, []);
```

### 2. OptimizedForm

优化的表单组件，使用多种性能优化技术：

- 使用 `React.memo` 避免表单项不必要的重新渲染
- 使用 `useCallback` 优化事件处理函数
- 使用 `useMemo` 缓存验证逻辑
- 使用防抖函数优化表单验证

### 3. VirtualizedList

虚拟列表组件，只渲染可视区域内的列表项，适用于展示大量数据：

- 只渲染可视区域内的列表项，而不是全部项目
- 使用绝对定位放置列表项，避免重排
- 使用节流函数优化滚动事件处理
- 使用overscan参数预渲染额外的项目，使滚动更平滑

### 4. 性能优化工具函数

`utils/performance.ts` 文件包含了多个性能优化工具函数：

- `debounce`: 防抖函数，延迟执行函数，适用于输入搜索、窗口调整大小等
- `throttle`: 节流函数，限制函数在一定时间内只能执行一次，适用于滚动事件处理
- `memoize`: 缓存函数结果，避免重复计算
- `batch`: 批量处理函数，将多次操作合并为一次执行
- `rafThrottle`: 使用requestAnimationFrame进行节流
- `idleLoad`: 延迟加载函数，在浏览器空闲时执行

## 性能优化

### D3.js 网络图性能优化

为了解决大规模数据集下的性能问题，我们进行了以下优化：

#### 1. WebGL 渲染 (PixiJS)

使用 PixiJS 实现 WebGL 渲染，相比 SVG 渲染有以下优势：

- 硬件加速：利用 GPU 进行图形渲染
- 更高效的内存管理
- 支持高达数千个节点的流畅渲染
- 大幅减少 DOM 节点数量

实现方式：

```typescript
// 使用 PixiJS 创建渲染器
const app = new PIXI.Application({
  width,
  height,
  backgroundColor: 0xffffff,
  antialias: true,
  autoDensity: true,
  resolution: window.devicePixelRatio || 1,
});
```

#### 2. 视口裁剪 (Viewport Culling)

只渲染当前视口可见的节点和链接，大幅减少绘制开销：

- 节点仅在进入视口时被渲染
- 减少了 GPU 处理的对象数量
- 支持平滑滚动和缩放大型网络

实现方式：

```typescript
// 检查节点是否在视口内
const isNodeInViewport = (node: NetworkNode) => {
  // 计算视口边界
  const vpLeft = -offsetX / scale - margin;
  const vpRight = (width - offsetX) / scale + margin;
  const vpTop = -offsetY / scale - margin;
  const vpBottom = (height - offsetY) / scale + margin;

  // 检查节点是否在视口内
  return node.x >= vpLeft && node.x <= vpRight && node.y >= vpTop && node.y <= vpBottom;
};
```

#### 3. 节点聚合 (Node Clustering)

动态聚合节点以减少渲染对象数量：

- 基于空间距离自动聚合节点
- 用户可调整聚合距离
- 支持点击展开聚合节点查看详情
- 适用于大规模网络分析

实现方式：

```typescript
// 聚合节点算法
const clusterNodes = (nodes, links, distance) => {
  // 根据距离聚合节点
  // 处理链接关系
  // 返回聚合后的节点和链接
};
```

#### 性能对比

| 优化方法   | 常规 SVG (D3) | 优化版 (WebGL) |
| ---------- | ------------- | -------------- |
| 1,000 节点 | 5-10 FPS      | 50-60 FPS      |
| 视口裁剪   | 不支持        | 支持           |
| 节点聚合   | 不支持        | 支持           |
| CPU 使用率 | 高            | 低             |
| 内存占用   | 高            | 低             |

### 使用方法

项目中提供了两个网络图组件：

1. `BlockchainNetwork`: 基础版，使用 D3.js + SVG 渲染，适用于小型网络
2. `BlockchainNetworkOptimized`: 优化版，使用 PixiJS + WebGL 渲染，支持大规模数据集

根据数据规模选择合适的渲染模式：

```tsx
{
  isLargeDataset ? (
    <BlockchainNetworkOptimized nodes={nodes} links={links} />
  ) : (
    <BlockchainNetwork nodes={nodes} links={links} />
  );
}
```

## 依赖管理

项目使用 Yarn 作为包管理工具。添加新依赖:

```bash
yarn add [package-name]
```

添加开发依赖:

```bash
yarn add [package-name] --dev
```

## 运行项目

```bash
# 安装依赖
yarn install

# 开发模式运行
yarn dev

# 构建项目
yarn build

# 运行构建后的项目
yarn start

# 运行代码检查
yarn lint

# 格式化代码
yarn format

# 运行测试
yarn test

# 监视模式运行测试
yarn test:watch

# 打开 Cypress 测试工具
yarn cypress

# 无头模式运行 Cypress 测试
yarn cypress:headless
```

## 后续优化方向

1. **性能优化**：添加组件懒加载和代码分割
2. **可访问性**：提高组件的可访问性
3. **测试覆盖率**：提高测试覆盖率
4. **微前端**：考虑使用微前端架构
5. **PWA**：添加 PWA 支持

# 前端测试改进

## 测试文件概述

我们已经为多个前端组件创建和改进了测试，包括：

1. **核心组件测试**:

   - `TransactionList.test.jsx` - 测试交易列表组件的渲染、筛选、搜索和错误处理
   - `VirtualizedList.test.tsx` - 测试虚拟滚动列表的渲染和性能优化
   - `GlobalSearch.test.tsx` - 测试全局搜索功能，包括交互和路由
   - `RiskBadge.test.tsx` - 测试风险徽章组件的不同风险级别显示
   - `ErrorBoundary.test.tsx` - 测试错误边界组件的错误捕获和UI显示
   - `OptimizedForm.test.tsx` - 测试优化表单组件的验证和提交
   - `MemoizedComponent.test.tsx` - 测试记忆化组件的缓存和重新渲染逻辑
   - `LazyImage.test.tsx` - 测试延迟加载图片组件的加载状态和错误处理
   - `StatisticsCards.test.tsx` - 测试统计卡片的数据展示和动画
   - `Skeleton.test.tsx` - 测试骨架屏组件的加载状态展示

2. **模拟服务**:
   - `mockService.js` - 提供统一的模拟数据和API响应，供测试使用

## 测试覆盖的功能

我们的测试文件覆盖了以下方面：

### 1. 基本渲染

- 验证组件在不同状态下的正确渲染
- 验证组件内容和结构

### 2. 交互行为

- 用户输入和表单提交
- 点击、悬停和键盘交互
- 搜索和筛选功能

### 3. 状态管理

- 加载状态展示
- 错误状态处理
- 边缘情况（如空数据）

### 4. 性能优化

- 记忆化组件测试
- 懒加载和虚拟化列表测试
- 防抖和节流功能测试

### 5. 无障碍支持

- ARIA属性验证
- 键盘导航支持
- 屏幕阅读器友好元素

## 测试工具和技术

- **Jest** - 测试框架
- **React Testing Library** - 组件测试库
- **Mock Service Worker** - API模拟
- **userEvent** - 用户交互模拟

## 后续改进计划

1. **提高测试覆盖率**:

   - 为更多组件添加测试，特别是复杂的图表组件
   - 为路由和页面组件添加测试

2. **端到端测试**:

   - 使用Cypress或Playwright添加端到端测试
   - 测试完整的用户流程和场景

3. **性能测试**:

   - 使用React性能工具测试组件渲染性能
   - 使用Lighthouse自动化测试页面性能指标

4. **可访问性测试**:

   - 使用专业工具（如axe-core）自动化测试无障碍合规性
   - 进行手动的屏幕阅读器测试

5. **测试环境改进**:
   - 修复TypeScript兼容性问题
   - 优化测试运行速度
   - 添加测试覆盖率报告

## 测试运行指南

通过以下命令运行测试：

```bash
# 运行所有测试
yarn test

# 运行单个测试文件
yarn test TransactionList

# 运行带有覆盖率报告的测试
yarn test --coverage
```

## 测试最佳实践

1. **组件测试原则**:

   - 专注于测试行为而不是实现细节
   - 使用用户视角编写测试
   - 避免测试内部状态，而是测试可观察到的输出

2. **测试组织**:

   - 清晰的测试描述和分组
   - 一致的测试模式和风格
   - 测试模块化，避免重复代码

3. **模拟数据和服务**:
   - 使用一致的模拟数据
   - 隔离外部依赖
   - 全局模拟配置
