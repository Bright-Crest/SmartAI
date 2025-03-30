#!/usr/bin/env node

/**
 * 无障碍问题修复报告生成器
 *
 * 此脚本分析最新的无障碍审计结果，并生成优先级排序的修复建议报告。
 * 它按组件和问题类型分组，提供具体的修复建议。
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// 配置路径
const REPORTS_DIR = path.resolve(__dirname, '..', 'a11y-reports');
const LATEST_REPORT = path.join(REPORTS_DIR, 'latest.json');
const FIX_REPORT_PATH = path.join(REPORTS_DIR, 'a11y-fix-report.md');

// 确保目录存在
if (!fs.existsSync(REPORTS_DIR)) {
  fs.mkdirSync(REPORTS_DIR, { recursive: true });
}

// 检查最新报告是否存在
if (!fs.existsSync(LATEST_REPORT)) {
  console.error('错误: 找不到最新的无障碍审计报告。请先运行 yarn a11y:check');
  process.exit(1);
}

// 读取最新报告
const reportData = JSON.parse(fs.readFileSync(LATEST_REPORT, 'utf8'));

/**
 * 生成修复建议
 * @param {Object} issue 问题数据
 * @returns {string} 修复建议
 */
function generateFixSuggestion(issue) {
  const typeMap = {
    'keyboard-focus': '键盘焦点问题',
    'color-contrast': '颜色对比度问题',
    'aria-attributes': 'ARIA属性问题',
    'heading-structure': '标题结构问题',
    'form-label': '表单标签问题',
    'image-alt': '图片替代文本问题',
  };

  const type = issue.type || 'unknown';
  const typeName = typeMap[type] || '未知问题类型';

  let suggestion = '';

  switch (type) {
    case 'keyboard-focus':
      if (issue.details.includes('missing-tabindex')) {
        suggestion = '添加tabIndex属性使元素可聚焦，推荐使用FocusableItem组件替代原生元素';
      } else if (issue.details.includes('missing-keyboard-handler')) {
        suggestion =
          '添加onKeyDown处理函数，处理Enter和Space键，使用useKeyboardNavigation钩子简化实现';
      } else if (issue.details.includes('missing-focus-style')) {
        suggestion = '添加焦点样式，确保元素在获得焦点时有明显的视觉提示';
      } else {
        suggestion =
          '确保所有交互元素都支持键盘导航，使用FocusableItem组件或实现正确的键盘事件处理';
      }
      break;

    case 'color-contrast':
      const contrastMatch = issue.details.match(/对比度为 (\d+\.\d+)/);
      const currentContrast = contrastMatch ? contrastMatch[1] : '未知';
      suggestion = `提高文本与背景的对比度（当前为${currentContrast}，需要至少4.5:1），使用更深的文本颜色或更浅的背景颜色`;
      break;

    case 'aria-attributes':
      if (issue.details.includes('未识别的ARIA属性')) {
        suggestion = '检查ARIA属性名称是否正确，参考WAI-ARIA规范';
      } else if (issue.details.includes('aria-hidden="true"')) {
        suggestion = '不要在可聚焦元素上使用aria-hidden="true"，会导致键盘陷阱';
      } else {
        suggestion = '确保ARIA属性使用正确，验证属性值的有效性';
      }
      break;

    case 'form-label':
      suggestion = '确保每个表单控件都有关联的标签，使用A11yFormInput组件或正确关联label和input';
      break;

    default:
      suggestion = issue.recommendation || '审核组件的无障碍实现';
  }

  return suggestion;
}

/**
 * 按组件和问题类型分组问题
 * @param {Array} issues 问题列表
 * @returns {Object} 分组后的问题
 */
function groupIssuesByComponentAndType(issues) {
  const result = {};

  // 合并关键和主要问题
  const allIssues = [...(reportData.issues.critical || []), ...(reportData.issues.major || [])];

  // 按文件分组
  allIssues.forEach(issue => {
    if (!issue.file) return;

    const fileName = issue.file;
    if (!result[fileName]) {
      result[fileName] = {};
    }

    const issueType = issue.type || 'unknown';
    if (!result[fileName][issueType]) {
      result[fileName][issueType] = [];
    }

    result[fileName][issueType].push(issue);
  });

  return result;
}

/**
 * 生成修复报告内容
 * @returns {string} 报告内容
 */
function generateReportContent() {
  // 获取日期和时间
  const now = new Date();
  const dateStr = now.toISOString().split('T')[0];
  const timeStr = now.toTimeString().split(' ')[0];

  // 获取汇总信息
  const { total, critical, major, minor } = reportData.summary;

  // 按组件和类型分组问题
  const groupedIssues = groupIssuesByComponentAndType(reportData.issues);

  // 构建报告头部
  let content = `# 无障碍问题修复报告

生成时间: ${dateStr} ${timeStr}

## 汇总信息

* **总问题数**: ${total}
* **严重问题**: ${critical}
* **主要问题**: ${major}
* **次要问题**: ${minor}

## 优先修复组件

按问题数量排序的前3个组件:

`;

  // 添加优先修复的组件
  const recommendations = reportData.recommendations || [];
  const topComponents = recommendations.slice(0, 3);

  topComponents.forEach((comp, index) => {
    content += `${index + 1}. **${comp.file}** - ${comp.issueCount}个问题 (主要是${comp.primaryType})\n`;
    content += `   - 建议: ${comp.recommendation}\n`;
  });

  content += `\n## 按组件的详细修复建议\n\n`;

  // 添加每个组件的详细建议
  Object.keys(groupedIssues)
    .sort()
    .forEach(fileName => {
      content += `### ${fileName}\n\n`;

      const componentIssues = groupedIssues[fileName];
      Object.keys(componentIssues).forEach(issueType => {
        const issues = componentIssues[issueType];
        const typeDisplayName =
          {
            'keyboard-focus': '键盘焦点问题',
            'color-contrast': '颜色对比度问题',
            'aria-attributes': 'ARIA属性问题',
            'heading-structure': '标题结构问题',
            'form-label': '表单标签问题',
            'image-alt': '图片替代文本问题',
          }[issueType] || issueType;

        content += `#### ${typeDisplayName} (${issues.length}个问题)\n\n`;

        // 对每种类型，只显示前3个问题的详细信息
        const sampleIssues = issues.slice(0, 3);
        sampleIssues.forEach(issue => {
          content += `- **第${issue.line || '未知'}行**: \`${issue.code || '未提供代码'}\`\n`;
          content += `  - 问题: ${issue.details || '未提供详情'}\n`;
          content += `  - 修复建议: ${generateFixSuggestion(issue)}\n\n`;
        });

        if (issues.length > 3) {
          content += `... 以及${issues.length - 3}个其他相似问题\n\n`;
        }
      });

      content += `---\n\n`;
    });

  // 添加一般修复指南
  content += `## 常见问题修复指南

### 键盘焦点问题

1. 使用\`FocusableItem\`组件替代自定义点击元素
   \`\`\`tsx
   <FocusableItem
     onClick={handleClick}
     className="your-class"
     role="button"
   >
     按钮内容
   </FocusableItem>
   \`\`\`

2. 使用\`useKeyboardNavigation\`钩子处理键盘事件
   \`\`\`tsx
   const { handleEnterAndSpace } = useKeyboardNavigation();
   
   <div
     role="button"
     tabIndex={0}
     onClick={handleClick}
     onKeyDown={handleEnterAndSpace(handleClick)}
   >
     内容
   </div>
   \`\`\`

### 颜色对比度问题

1. 提高文本与背景的对比度
   - 使用更深的文本颜色（例如从\`text-gray-400\`改为\`text-gray-700\`）
   - 确保所有文本都符合WCAG AA标准（普通文本4.5:1，大文本3:1）

2. 使用Tailwind的替代类
   - 用\`text-blue-800\`替代\`text-blue-500\`
   - 用\`bg-gray-100\`替代\`bg-gray-200\`

### 表单标签问题

1. 使用\`A11yFormInput\`组件
   \`\`\`tsx
   <A11yFormInput
     id="search-input"
     name="search"
     type="search"
     label="搜索"
     value={searchTerm}
     onChange={handleSearchChange}
   />
   \`\`\`

2. 确保标签正确关联
   \`\`\`tsx
   <label htmlFor="username">用户名</label>
   <input id="username" name="username" type="text" />
   \`\`\`

### ARIA属性问题

1. 验证ARIA属性名称
   - 使用标准的ARIA属性（例如\`aria-label\`、\`aria-labelledby\`）
   - 检查拼写错误

2. 正确使用ARIA角色
   - 表格：\`role="table"\`, \`role="row"\`, \`role="cell"\`
   - 列表：\`role="list"\`, \`role="listitem"\`

## 后续步骤

1. 使用\`yarn a11y:check:component <组件路径>\`检查特定组件
2. 修复问题后，再次运行\`yarn a11y:check\`验证改进
3. 使用屏幕阅读器测试关键功能（参考\`docs/screen-reader-testing-guide.md\`）

`;

  return content;
}

// 生成并保存报告
try {
  const reportContent = generateReportContent();
  fs.writeFileSync(FIX_REPORT_PATH, reportContent);
  console.log(`✅ 无障碍修复报告已生成: ${FIX_REPORT_PATH}`);

  // 尝试打开报告
  try {
    if (process.platform === 'darwin') {
      execSync(`open "${FIX_REPORT_PATH}"`);
    } else if (process.platform === 'win32') {
      execSync(`start "" "${FIX_REPORT_PATH}"`);
    } else if (process.platform === 'linux') {
      execSync(`xdg-open "${FIX_REPORT_PATH}"`);
    }
  } catch (err) {
    console.log(`请手动打开报告: ${FIX_REPORT_PATH}`);
  }
} catch (error) {
  console.error('生成报告时出错:', error.message);
  process.exit(1);
}
