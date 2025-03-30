# 无障碍设计规范

## 介绍

本文档提供了 ChainIntelAI 项目的无障碍设计规范。遵循这些规范可以确保我们的应用程序对所有用户都是可访问的，包括使用辅助技术的用户。

## 基本原则

我们的无障碍设计基于以下原则：

1. **感知性** - 信息和界面组件必须以用户可以感知的方式呈现
2. **可操作性** - 界面组件和导航必须可操作
3. **可理解性** - 信息和操作必须可理解
4. **健壮性** - 内容必须足够健壮，以便能够被各种用户代理可靠地解释

## 实现标准

我们遵循 [WCAG 2.1 AA 级标准](https://www.w3.org/TR/WCAG21/)，这是全球公认的无障碍标准。

## 设计和开发指南

### 1. 颜色对比

- 文本和背景之间的对比度必须至少为 4.5:1（AA 级标准）
- 大文本（18pt+）和背景之间的对比度必须至少为 3:1
- 不要仅使用颜色来传达信息，必须辅以其他提示（如图标、文本等）

#### 推荐颜色组合

```css
/* 文本和背景 */
.text-gray-800 /* #1a202c */ + .bg-white /* #ffffff */ /* 对比度: 16:1 */
.text-white /* #ffffff */ + .bg-blue-700 /* #2563eb */ /* 对比度: 5.76:1 */
.text-blue-800 /* #1e40af */ + .bg-gray-100 /* #f3f4f6 */ /* 对比度: 10.86:1 */
```

### 2. 键盘导航

- 所有交互元素必须可通过键盘操作（Tab、Enter、Space、方向键等）
- 页面应有逻辑的焦点顺序
- 可视焦点指示器必须明显可见
- 没有"键盘陷阱"（无法使用键盘离开某个元素）

#### 实现方式

使用我们的 `useKeyboardNavigation` Hook：

```tsx
import { useKeyboardNavigation } from '../hooks/useKeyboardNavigation';

const { handleEnterAndSpace, handleArrowKeys, manageFocusTrap } = useKeyboardNavigation();

// 为按钮添加键盘支持
<button onClick={handleClick} onKeyDown={handleEnterAndSpace(handleClick)}>
  按钮文本
</button>;
```

### 3. 表单和输入

- 所有表单控件必须有关联的标签
- 错误消息必须与相关表单控件关联
- 必填字段必须清楚标明
- 使用 `aria-invalid` 标记无效输入
- 使用 `aria-describedby` 关联错误消息和表单控件

#### 实现方式

使用我们的 `A11yFormInput` 组件：

```tsx
import A11yFormInput from '../components/a11y/A11yFormInput';

<A11yFormInput
  id="email"
  name="email"
  type="email"
  label="电子邮件"
  value={email}
  onChange={handleEmailChange}
  required
  error={errors.email}
  description="我们不会与任何人分享您的电子邮件"
/>;
```

### 4. 图像和媒体

- 所有非装饰性图像必须有替代文本（alt 属性）
- 装饰性图像应使用 `alt=""` 或 `aria-hidden="true"`
- 视频必须提供字幕和文字脚本
- 音频内容必须提供文字脚本

### 5. ARIA 属性与角色

- 使用正确的 ARIA 角色标记内容结构和用途
- 使用 ARIA 属性时必须遵循 ARIA 规范
- 优先使用原生 HTML 元素，只在必要时才使用 ARIA 角色

#### 常用 ARIA 角色和属性

```html
<!-- 导航 -->
<nav role="navigation" aria-label="主导航">
  <!-- 警告/通知 -->
  <div role="alert">重要提示</div>

  <!-- 标记必填字段 -->
  <input aria-required="true" />

  <!-- 标记无效输入 -->
  <input aria-invalid="true" />

  <!-- 关联错误消息 -->
  <input aria-describedby="error-message" />
  <p id="error-message">错误信息</p>
</nav>
```

### 6. 头部结构

- 使用正确的标题层次结构 (h1-h6)
- 每个页面应只有一个 h1
- 不跳过标题级别（例如，不要从 h2 直接跳到 h4）

### 7. 可复用组件

我们提供以下无障碍组件，请优先使用：

- `FocusableItem` - 可键盘操作的交互元素
- `AccessibleMenu` - 支持完整键盘导航的菜单
- `A11yFormInput` - 无障碍表单输入组件

## 测试与验证

在开发过程中，定期运行以下测试：

```bash
# 运行无障碍检查脚本
yarn a11y:check
```

## 常见问题与解决方案

### Q: 如何处理焦点管理？

A: 使用 `useKeyboardNavigation` Hook 中的 `manageFocusTrap` 函数来管理模态框和弹出菜单中的焦点。

### Q: 如何正确实现 ARIA 属性？

A: 参考 [WAI-ARIA 实践](https://www.w3.org/TR/wai-aria-practices-1.1/) 文档，或使用我们提供的组件库。

### Q: 颜色对比度不足怎么办？

A: 使用颜色对比检查工具（如 [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)）来验证对比度，并调整颜色以达到至少 4.5:1 的对比度。

## 参考资源

- [WCAG 2.1 指南](https://www.w3.org/TR/WCAG21/)
- [WAI-ARIA 实践](https://www.w3.org/TR/wai-aria-practices-1.1/)
- [MDN 无障碍指南](https://developer.mozilla.org/zh-CN/docs/Web/Accessibility)
