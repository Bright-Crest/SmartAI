# ChainIntelAI 前端测试框架

本文档描述了 ChainIntelAI 前端应用程序的测试框架和策略。

## 测试类型

我们的测试框架包含以下几种测试类型：

1. **单元测试**：使用 Jest 测试框架测试独立组件和函数
2. **API 测试**：测试 API 路由和处理程序
3. **组件测试**：使用 Cypress 组件测试功能测试 React 组件
4. **端到端测试**：使用 Cypress 测试整个应用程序流程

## 测试目录结构

```
frontend/
├── tests/                  # Jest 测试目录
│   ├── components/         # 组件单元测试
│   ├── pages/              # 页面单元测试
│   ├── hooks/              # 自定义 Hook 测试
│   ├── utils/              # 工具函数测试
│   └── api/                # API 路由测试
├── cypress/                # Cypress 测试目录
│   ├── e2e/                # 端到端测试
│   ├── component/          # 组件测试
│   ├── fixtures/           # 测试数据
│   └── support/            # 支持文件和命令
└── jest.config.js          # Jest 配置
```

## 运行测试

### 单元测试

```bash
# 运行所有单元测试
yarn test

# 运行特定测试文件
yarn test tests/components/RiskAlerts.test.js

# 运行带有覆盖率报告的测试
yarn test --coverage
```

### API 测试

```bash
# 运行所有 API 测试
yarn test tests/api

# 运行特定 API 测试
yarn test tests/api/addresses.test.js

# 运行带有覆盖率报告的 API 测试
yarn test --coverage tests/api
```

### 端到端测试

```bash
# 打开 Cypress 测试运行器
yarn cypress

# 运行所有端到端测试（无头模式）
yarn cypress:headless

# 运行特定端到端测试
yarn cypress run --spec "cypress/e2e/api.cy.js"

# 运行端到端测试（启动开发服务器）
yarn e2e
```

### 组件测试

```bash
# 打开 Cypress 组件测试运行器
yarn cypress:component

# 运行所有组件测试（无头模式）
yarn cypress:component:headless
```

## 测试覆盖率

我们的目标是保持以下测试覆盖率：

- 单元测试：90% 以上的代码覆盖率
- API 测试：85% 以上的代码覆盖率
- 端到端测试：覆盖所有关键用户流程

覆盖率报告可以通过运行 `yarn test --coverage` 生成，报告将保存在 `coverage/` 目录中。

## 测试策略

### 单元测试策略

- 测试每个组件的渲染和基本功能
- 测试自定义 Hook 的行为
- 测试工具函数的输入和输出
- 使用模拟（mock）隔离测试单元

### API 测试策略

- 测试每个 API 路由的正常响应
- 测试错误处理和边缘情况
- 测试参数验证
- 测试数据库连接失败和外部 API 超时等异常情况

### 端到端测试策略

- 测试关键用户流程（如搜索地址、查看交易详情等）
- 测试错误处理和用户反馈
- 测试响应式设计在不同设备上的表现
- 使用模拟数据测试 API 响应

## CI/CD 集成

我们使用 GitHub Actions 自动运行测试。工作流配置文件位于 `.github/workflows/frontend-tests.yml`。

每次推送到 `main` 或 `develop` 分支，或创建针对这些分支的拉取请求时，都会触发测试工作流。

工作流包括以下任务：

1. 运行单元测试和生成覆盖率报告
2. 运行 API 测试和生成覆盖率报告
3. 运行端到端测试
4. 运行代码质量检查（ESLint 和类型检查）

## 最佳实践

1. **测试驱动开发**：先编写测试，再实现功能
2. **测试可维护性**：保持测试简单、可读和可维护
3. **测试独立性**：每个测试应该独立运行，不依赖其他测试的状态
4. **测试命名**：使用描述性的测试名称，清晰表达测试的目的
5. **测试覆盖率**：定期检查测试覆盖率，确保关键代码路径被测试
6. **测试速度**：保持测试运行速度快，特别是单元测试
7. **测试稳定性**：避免不稳定的测试，如依赖外部服务或时间的测试

## 边缘情况测试

我们特别关注以下边缘情况的测试：

1. **数据库连接失败**：测试当数据库连接失败时应用程序的行为
2. **外部 API 超时**：测试当外部 API 请求超时时的错误处理
3. **无效输入格式**：测试当用户提供无效输入时的验证和错误消息
4. **并发请求**：测试应用程序处理多个并发请求的能力
5. **极端数据值**：测试处理极大或极小数据值的能力
6. **空数据集**：测试当没有数据返回时的行为
7. **权限错误**：测试当用户没有足够权限时的行为

## 模拟策略

在测试中，我们使用以下模拟策略：

1. **API 响应模拟**：使用 Cypress 拦截器模拟 API 响应
2. **服务模拟**：使用 Jest 模拟服务和依赖
3. **用户交互模拟**：使用 Cypress 模拟用户交互
4. **时间模拟**：使用 Jest 的 `jest.useFakeTimers()` 模拟时间相关功能

# 前端组件测试指南

本文档提供了关于如何在项目中创建和运行组件测试的指南。

## 目录

- [测试环境](#测试环境)
- [测试工具](#测试工具)
- [测试脚本](#测试脚本)
- [创建测试](#创建测试)
- [运行测试](#运行测试)
- [测试规范](#测试规范)
- [常见问题](#常见问题)

## 测试环境

我们使用以下工具进行测试：

- **Jest**: JavaScript 测试框架
- **React Testing Library**: React 组件测试工具
- **User Event**: 模拟用户交互的工具

## 测试工具

我们提供了两个脚本来简化测试流程：

1. `scripts/create-test.sh`: 创建新的组件测试文件
2. `scripts/run-tests.sh`: 运行测试

## 测试脚本

### 创建测试

使用 `create-test.sh` 脚本为组件创建测试文件：

```bash
# 进入前端目录
cd frontend

# 为 SearchBar 组件创建测试
./scripts/create-test.sh -c SearchBar
```

这将在 `__tests__/components/` 目录下创建一个测试文件模板。

### 运行测试

使用 `run-tests.sh` 脚本运行测试：

```bash
# 进入前端目录
cd frontend

# 运行所有测试
./scripts/run-tests.sh -a

# 运行特定组件的测试
./scripts/run-tests.sh -c SearchBar

# 以监视模式运行测试（修改文件时自动重新运行）
./scripts/run-tests.sh -c SearchBar -w
```

## 创建测试

每个组件测试文件应包含以下几种测试：

1. **渲染测试**: 确保组件正确渲染
2. **交互测试**: 测试用户交互（点击、输入等）
3. **状态测试**: 测试组件状态变化
4. **边缘情况**: 测试异常情况

### 测试文件示例

以下是 `SearchBar` 组件测试的示例：

```javascript
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import SearchBar from '../../app/components/SearchBar';
import userEvent from '@testing-library/user-event';

describe('SearchBar组件', () => {
  test('正确渲染搜索栏', () => {
    const handleSearch = jest.fn();
    render(<SearchBar onSearch={handleSearch} placeholder="搜索地址或交易" />);

    // 检查输入框和按钮是否正确渲染
    const input = screen.getByPlaceholderText('搜索地址或交易');
    expect(input).toBeInTheDocument();

    // 验证搜索按钮
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  test('点击搜索按钮触发回调', async () => {
    const handleSearch = jest.fn();
    render(<SearchBar onSearch={handleSearch} placeholder="搜索..." />);

    // 输入搜索词
    const input = screen.getByPlaceholderText('搜索...');
    await userEvent.type(input, 'test query');

    // 点击搜索按钮
    fireEvent.click(screen.getByRole('button'));

    // 验证回调被调用，且参数正确
    expect(handleSearch).toHaveBeenCalledTimes(1);
    expect(handleSearch).toHaveBeenCalledWith('test query');
  });

  test('搜索为空时不触发回调', () => {
    const handleSearch = jest.fn();
    render(<SearchBar onSearch={handleSearch} />);

    // 点击搜索按钮但没有输入内容
    fireEvent.click(screen.getByRole('button'));

    // 验证回调没有被调用
    expect(handleSearch).not.toHaveBeenCalled();
  });
});
```

## 测试规范

编写测试时，请遵循以下规范：

1. **测试文件命名**: 使用 `组件名.test.js` 格式
2. **测试描述**: 使用清晰的描述说明测试内容
3. **测试隔离**: 每个测试应该独立，不依赖其他测试
4. **模拟函数**: 使用 `jest.fn()` 模拟回调函数
5. **用户交互**: 使用 `userEvent` 模拟用户交互，而不是直接使用 `fireEvent`
6. **断言**: 使用明确的断言验证结果

## 常见问题

### 找不到元素

如果测试中找不到元素，可以尝试以下方法：

1. 使用 `screen.debug()` 查看渲染的 DOM
2. 使用不同的查询方法，如 `getByText`、`getByPlaceholderText` 等
3. 检查组件是否正确渲染

### 测试失败

如果测试失败，请检查：

1. 组件实现是否正确
2. 测试代码是否正确
3. 是否有异步操作需要等待

### 异步测试

对于异步操作，请使用 `async/await` 或 `act` 确保测试正确等待操作完成：

```javascript
test('异步操作', async () => {
  // 使用 async/await
  await userEvent.type(input, 'text');

  // 或使用 act
  await act(async () => {
    // 异步操作
  });
});
```

---

如有任何问题，请联系项目负责人。

## 前端测试改进

### 测试文件结构

现在前端测试文件结构已经更加完善，包括：

1. **Jest 组件测试**：`frontend/tests/components/` 目录下
2. **Cypress 组件测试**：`frontend/cypress/component/` 目录下
3. **Cypress E2E测试**：`frontend/cypress/e2e/` 目录下
4. **Mock 服务**：`frontend/tests/mocks/` 目录下

### 已完成的改进

我们已经完成了以下测试改进：

1. **修复了Jest测试配置**

   - 更新了`jest.config.js`测试匹配模式
   - 修复了TypeScript配置，添加了`@types/jest`支持
   - 增强了`jest.setup.js`，提供全局测试函数和模拟对象

2. **添加了更多组件测试**

   - 核心组件测试：`TransactionList`, `VirtualizedList`, `GlobalSearch`, `RiskBadge`等
   - 高级组件测试：`ErrorBoundary`, `OptimizedForm`, `MemoizedComponent`, `LazyImage`等
   - 统计和图表组件：`StatisticsCards`等

3. **增强了无障碍测试**

   - 添加了`cypress-axe`集成
   - 创建了专门的无障碍测试文件
   - 添加了自动运行无障碍测试的功能
   - 测试了ARIA属性和键盘导航

4. **添加了CI/CD支持**
   - 创建了GitHub Actions工作流文件
   - 配置了自动化测试流程
   - 添加了测试覆盖率检查
   - 配置了无障碍测试自动运行

### 如何运行测试

#### 运行Jest组件测试

```bash
# 运行所有测试
yarn test

# 运行特定组件测试
yarn test TransactionList

# 运行带有覆盖率报告的测试
yarn test --coverage
```

#### 运行Cypress组件测试

```bash
# 打开Cypress组件测试UI
yarn cypress:component

# 运行无头模式下的组件测试
yarn cypress:component:headless
```

#### 运行Cypress E2E测试

```bash
# 启动应用并打开Cypress UI
yarn e2e

# 运行无头模式下的E2E测试
yarn e2e:headless
```

#### 运行无障碍测试

```bash
# 运行所有无障碍测试
CYPRESS_autoRunA11y=true yarn cypress run --spec "cypress/e2e/a11y-tests.cy.js"

# 运行组件无障碍测试
CYPRESS_autoRunA11y=true yarn cypress run --component
```

### 无障碍测试指南

我们添加了全面的无障碍测试支持，确保应用符合WCAG 2.1 AA标准：

1. **自动检查**

   - 颜色对比度
   - 键盘导航
   - ARIA属性
   - 语义化HTML

2. **手动测试项目**

   - 屏幕阅读器兼容性
   - 键盘导航流程
   - 高对比度模式

3. **CI/CD集成**
   - 在每个PR中自动运行无障碍测试
   - 测试失败会阻止合并

### 自定义Cypress命令

我们添加了多个自定义Cypress命令来简化测试编写：

```javascript
// 检查无障碍
cy.checkA11y();

// 检查颜色对比度
cy.checkColorContrast();

// 测试不同设备尺寸的无障碍性
cy.checkResponsiveA11y();

// 测试完整的键盘导航流程
cy.checkCompleteKeyboardNavigation(['selector1', 'selector2']);

// 检查图片alt文本
cy.checkImagesHaveAlt();

// 检查表单标签
cy.checkFormLabels();
```

### 测试覆盖率目标

我们的测试覆盖率目标如下：

- **语句覆盖率**: 90%
- **分支覆盖率**: 80%
- **函数覆盖率**: 90%
- **行覆盖率**: 90%

当前覆盖率可以通过运行 `yarn test --coverage` 查看。

### CI/CD流程

我们的CI/CD流程现在包括以下步骤：

1. **代码检查和单元测试**

   - 运行ESLint
   - 运行TypeScript类型检查
   - 运行Jest单元测试
   - 生成覆盖率报告

2. **Cypress测试**

   - 运行Cypress组件测试
   - 运行Cypress E2E测试
   - 检查测试是否通过

3. **无障碍测试**

   - 运行自动化无障碍测试
   - 检查WCAG 2.1 AA合规性

4. **预览环境部署**
   - 构建前端应用
   - 部署到预览环境
   - 在PR中添加预览链接

所有测试必须通过才能将代码合并到主分支。

### 后续改进计划

1. **提高测试覆盖率**

   - 为更多组件添加测试，特别是复杂的图表组件
   - 为路由和页面组件添加测试

2. **端到端测试**

   - 扩展Cypress E2E测试场景
   - 添加更多用户流程测试

3. **性能测试**

   - 使用React性能工具测试组件渲染性能
   - 使用Lighthouse自动化测试页面性能指标

4. **可访问性测试**
   - 扩展自动化无障碍测试
   - 添加更多WCAG 2.1 AA标准检查

## 性能测试

我们添加了全面的性能测试功能，用于监控和优化前端组件和页面的性能表现。

### 性能测试工具

性能测试包括以下关键功能：

1. **页面加载性能测试**：测量不同页面的加载时间，包括总加载时间、DOM加载时间和网络延迟。
2. **组件渲染性能测试**：测量核心组件的渲染时间，特别是在大数据集下的性能表现。
3. **交互性能测试**：测量用户交互（如筛选、表单提交）的响应时间。
4. **资源加载性能**：分析资源加载情况，识别大尺寸或加载缓慢的资源。

### 运行性能测试

使用以下命令运行性能测试：

```bash
# 运行性能测试
yarn test:performance

# 在开发服务器上运行
yarn dev
# 另一个终端中：
cypress run --spec 'cypress/e2e/performance-tests.cy.js'
```

### 性能测试页面

我们创建了专门的性能测试页面（`/performance-test`），用于测试组件在不同数据量和场景下的性能表现。这个页面提供了以下功能：

- 测试交易列表组件在大数据集（1000项）下的渲染性能
- 测试全局搜索组件的响应性能
- 测试图表组件的渲染性能
- 测试风险评分卡组件的渲染性能
- 测试滚动性能

### 性能基准和目标

我们定义了以下性能目标：

- 页面总加载时间: < 5000ms
- DOM加载时间: < 3000ms
- 组件渲染时间: < 3000ms
- 交互响应时间: < 500ms
- 搜索响应时间: < 300ms

## 自动化无障碍测试报告

我们开发了一个完整的无障碍测试报告系统，提供详细的无障碍合规分析。

### 运行无障碍测试并生成报告

使用以下命令运行测试并生成HTML格式的报告：

```bash
# 运行无障碍测试并生成报告
yarn test:a11y:report

# 查看生成的报告
yarn view:a11y-report
```

### 报告内容

生成的报告包含以下内容：

1. **合规评分**：基于违规数量和严重程度计算的总体评分（0-100分）。
2. **页面分析**：每个测试页面的无障碍问题摘要。
3. **组件分析**：各组件的无障碍合规情况。
4. **常见违规**：最常见的无障碍问题，按出现频率排序。
5. **违规详情**：每个违规的详细说明和修复建议。

### 无障碍测试覆盖范围

我们的无障碍测试覆盖以下内容：

1. **WCAG 2.1 AA合规性**：测试是否符合WCAG 2.1 AA标准。
2. **键盘导航**：测试完整的键盘导航流程。
3. **ARIA属性**：验证元素是否有正确的ARIA角色和属性。
4. **颜色对比度**：检查文本和背景的颜色对比度是否符合标准。
5. **响应式设计**：在不同设备尺寸下测试无障碍性。
6. **深色模式**：测试深色模式下的无障碍性。

### 自动化集成

无障碍测试和报告生成已集成到CI/CD流程中：

```yaml
# .github/workflows/test.yml中的无障碍测试任务
- name: 运行无障碍测试
  run: |
    yarn start &
    sleep 5
    CYPRESS_autoRunA11y=true yarn cypress run --spec "cypress/e2e/a11y-tests.cy.js" --browser chrome
```

## 全面用户流程测试

我们添加了全面的用户流程测试，模拟真实用户在应用中的完整交互流程。

### 用户流程测试内容

用户流程测试涵盖以下核心场景：

1. **用户登录与风险检测**：测试登录、查看风险提醒、搜索地址、查看交易历史等完整流程。
2. **交易监控**：测试交易列表页面的筛选、排序、时间范围选择和导出功能。
3. **分析和报告**：测试分析页面的图表交互、报告生成和分享功能。
4. **移动端响应式**：测试移动设备上的用户交互流程。

### 运行用户流程测试

使用以下命令运行用户流程测试：

```bash
# 运行用户流程测试
yarn test:user-flow
```

## 定期测试监控

我们添加了定期测试监控功能，可以在开发过程中持续监控测试状态：

```bash
# 监控组件测试
yarn monitor:test
```

这个命令会每60秒自动运行一次测试，当检测到代码变化时，帮助开发者及时发现和修复问题。

## 测试运行自动化

为了简化测试流程，我们提供了以下集成命令：

```bash
# 运行所有测试（覆盖率、无障碍和性能测试）
yarn test:ci

# 运行单个组件的所有测试
yarn test -- -t "ComponentName"
```

## 最佳实践

1. **定期运行性能测试**，特别是在添加新功能或修改现有组件后。
2. **修复所有严重和重要的无障碍问题**，保持至少90分以上的无障碍合规评分。
3. **确保所有核心用户流程始终通过测试**，不要在没有测试覆盖的情况下部署新功能。
4. **在代码审查过程中审查测试报告**，确保没有性能退化或无障碍问题。
5. **在不同设备和浏览器上测试**，特别是移动设备视图。
