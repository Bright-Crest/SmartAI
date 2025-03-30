#!/usr/bin/env node

/**
 * 无障碍审计工具
 *
 * 此脚本集成了多个无障碍检查工具，运行全面的无障碍审计并生成详细报告。
 * 它会检查颜色对比度、键盘焦点管理、标题结构等问题。
 *
 * 用法: node scripts/a11y-audit.js [组件路径]
 *
 * 例如:
 * - 全量检查: node scripts/a11y-audit.js
 * - 组件检查: node scripts/a11y-audit.js components/TransactionList.tsx
 */

const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');
const glob = require('glob');

// 获取命令行参数（可选的组件路径）
const componentPath = process.argv[2];

// 确保输出目录存在
const REPORT_DIR = path.resolve(__dirname, '..', 'a11y-reports');
if (!fs.existsSync(REPORT_DIR)) {
  fs.mkdirSync(REPORT_DIR, { recursive: true });
}

// 生成HTML报告文件名
const reportDate = new Date().toISOString().replace(/:/g, '-').split('.')[0];
const htmlReportPath = path.join(REPORT_DIR, `a11y-report-${reportDate}.html`);
const jsonReportPath = path.join(REPORT_DIR, `a11y-report-${reportDate}.json`);

// 创建报告的快捷链接
const latestJsonReportPath = path.join(REPORT_DIR, 'latest.json');

// 收集所有问题
const allIssues = {
  colorContrast: [],
  keyboardFocus: [],
  headingStructure: [],
  ariaAttributes: [],
  imageAlts: [],
  formLabels: [],
};

// 运行颜色对比度检查
console.log('正在检查颜色对比度问题...');
try {
  const colorCheckScript = path.resolve(__dirname, 'check-color-contrast.js');
  const colorCheckOutput = execSync(`node ${colorCheckScript}`, { encoding: 'utf8' });

  // 解析输出并提取问题
  const colorIssues = colorCheckOutput
    .split('\n\n')
    .filter(block => block.includes('问题 #'))
    .map(block => {
      const lines = block.split('\n');
      const file = lines
        .find(line => line.includes('文件:'))
        ?.replace('  文件:', '')
        .trim();
      const line = lines
        .find(line => line.includes('行号:'))
        ?.replace('  行号:', '')
        .trim();
      const textColor = lines
        .find(line => line.includes('文本颜色:'))
        ?.replace('  文本颜色:', '')
        .trim();
      const bgColor = lines
        .find(line => line.includes('背景颜色:'))
        ?.replace('  背景颜色:', '')
        .trim();
      const contrast = lines
        .find(line => line.includes('对比度:'))
        ?.replace('  对比度:', '')
        .trim();
      const recommendation = lines
        .find(line => line.includes('建议:'))
        ?.replace('  建议:', '')
        .trim();

      return {
        type: 'color-contrast',
        file,
        line,
        details: `文本颜色(${textColor}) 与背景颜色(${bgColor}) 的对比度为 ${contrast}`,
        recommendation,
      };
    });

  if (colorIssues.length > 0) {
    allIssues.colorContrast = colorIssues;
  }
} catch (error) {
  console.error('颜色对比度检查失败:', error.message);
}

// 运行键盘焦点检查
console.log('正在检查键盘焦点和导航问题...');
try {
  const keyboardCheckScript = path.resolve(__dirname, 'check-keyboard-focus.js');
  const keyboardCheckOutput = execSync(`node ${keyboardCheckScript}`, { encoding: 'utf8' });

  // 解析输出并提取问题
  let currentFile = '';
  const keyboardIssues = [];

  keyboardCheckOutput.split('\n').forEach(line => {
    if (line.startsWith('文件:')) {
      currentFile = line.replace('文件:', '').trim();
    } else if (line.includes('问题 #') && line.includes('行')) {
      const lineMatch = line.match(/行 (\d+)/);
      const lineNumber = lineMatch ? lineMatch[1] : 'unknown';

      const typeMatch = keyboardCheckOutput
        .split('\n')
        .find(
          l =>
            l.includes('类型:') &&
            keyboardCheckOutput.indexOf(l) > keyboardCheckOutput.indexOf(line)
        );
      const type = typeMatch ? typeMatch.replace('  类型:', '').trim() : '';

      const descMatch = keyboardCheckOutput
        .split('\n')
        .find(
          l =>
            l.includes('描述:') &&
            keyboardCheckOutput.indexOf(l) > keyboardCheckOutput.indexOf(line)
        );
      const description = descMatch ? descMatch.replace('  描述:', '').trim() : '';

      const codeMatch = keyboardCheckOutput
        .split('\n')
        .find(
          l =>
            l.includes('代码:') &&
            keyboardCheckOutput.indexOf(l) > keyboardCheckOutput.indexOf(line)
        );
      const code = codeMatch ? codeMatch.replace('  代码:', '').trim() : '';

      keyboardIssues.push({
        type: 'keyboard-focus',
        file: currentFile,
        line: lineNumber,
        details: `${type}: ${description}`,
        code,
        recommendation: '确保所有交互元素都支持键盘导航和焦点管理',
      });
    }
  });

  if (keyboardIssues.length > 0) {
    allIssues.keyboardFocus = keyboardIssues;
  }
} catch (error) {
  console.error('键盘焦点检查失败:', error.message);
}

// 运行标题结构检查
console.log('正在检查标题结构问题...');
try {
  const headingCheckScript = path.resolve(__dirname, 'check-heading-structure.js');
  const headingCheckOutput = execSync(`node ${headingCheckScript}`, { encoding: 'utf8' });

  // 解析输出并提取问题
  let currentFile = '';
  const headingIssues = [];

  headingCheckOutput.split('\n').forEach(line => {
    if (line.startsWith('文件:')) {
      currentFile = line.replace('文件:', '').trim();
    } else if (line.includes('问题 #') && line.includes('行')) {
      const lineMatch = line.match(/行 (\d+)/);
      const lineNumber = lineMatch ? lineMatch[1] : 'unknown';

      const typeMatch = headingCheckOutput
        .split('\n')
        .find(
          l =>
            l.includes('类型:') && headingCheckOutput.indexOf(l) > headingCheckOutput.indexOf(line)
        );
      const type = typeMatch ? typeMatch.replace('  类型:', '').trim() : '';

      const descMatch = headingCheckOutput
        .split('\n')
        .find(
          l =>
            l.includes('描述:') && headingCheckOutput.indexOf(l) > headingCheckOutput.indexOf(line)
        );
      const description = descMatch ? descMatch.replace('  描述:', '').trim() : '';

      headingIssues.push({
        type: 'heading-structure',
        file: currentFile,
        line: lineNumber,
        details: `${type}: ${description}`,
        recommendation: '确保页面标题遵循正确的层次结构，从h1开始依次使用',
      });
    }
  });

  if (headingIssues.length > 0) {
    allIssues.headingStructure = headingIssues;
  }
} catch (error) {
  console.error('标题结构检查失败:', error.message);
}

// 检查ARIA属性
console.log('正在检查ARIA属性问题...');
try {
  // 使用grep查找所有ARIA属性
  const ariaGrepCommand =
    'grep -r "aria-" --include="*.jsx" --include="*.tsx" ./components ./pages';
  const ariaOutput = execSync(ariaGrepCommand, {
    encoding: 'utf8',
    cwd: path.resolve(__dirname, '..'),
  });

  // 检查一些常见的ARIA错误
  const ariaIssues = [];

  ariaOutput.split('\n').forEach(line => {
    if (!line) return;

    const [filePath, code] = line.split(':', 2);
    const file = filePath.replace('./', '');

    // 检查未识别的aria属性
    if (
      code.match(/aria-[a-z]+=/) &&
      !code.match(
        /aria-(label|labelledby|describedby|hidden|expanded|haspopup|controls|pressed|checked|selected|invalid|required|disabled|level|current|live|atomic|busy|relevant)=/
      )
    ) {
      ariaIssues.push({
        type: 'aria-attributes',
        file,
        details: '可能使用了未识别的ARIA属性',
        code: code.trim(),
        recommendation: '请验证ARIA属性名称是否正确，参考WAI-ARIA规范',
      });
    }

    // 检查aria-hidden="true"与可聚焦元素的组合（这是不推荐的）
    if (
      code.includes('aria-hidden="true"') &&
      (code.includes('tabIndex') ||
        code.includes('<button') ||
        code.includes('<a ') ||
        code.includes('<input'))
    ) {
      ariaIssues.push({
        type: 'aria-attributes',
        file,
        details: 'aria-hidden="true"不应用于可聚焦元素',
        code: code.trim(),
        recommendation: '从可聚焦元素中移除aria-hidden="true"属性，或使元素不可聚焦',
      });
    }
  });

  if (ariaIssues.length > 0) {
    allIssues.ariaAttributes = ariaIssues;
  }
} catch (error) {
  console.error('ARIA属性检查失败:', error.message);
}

// 检查图片alt属性
console.log('正在检查图片alt属性...');
try {
  // 使用grep查找所有图片标签
  const imgGrepCommand = 'grep -r "<img" --include="*.jsx" --include="*.tsx" ./components ./pages';
  const imgOutput = execSync(imgGrepCommand, {
    encoding: 'utf8',
    cwd: path.resolve(__dirname, '..'),
  });

  const imgIssues = [];

  imgOutput.split('\n').forEach(line => {
    if (!line) return;

    const [filePath, code] = line.split(':', 2);
    const file = filePath.replace('./', '');

    // 检查缺少alt属性的图片
    if (!code.includes('alt=')) {
      imgIssues.push({
        type: 'image-alts',
        file,
        details: '图片缺少alt属性',
        code: code.trim(),
        recommendation: '添加描述性alt属性，或对于装饰性图片使用alt=""',
      });
    }

    // 检查空的alt属性但不是装饰性图片（有onClick或role）
    if (
      (code.includes('alt=""') || code.includes('alt={""} ')) &&
      (code.includes('onClick') || code.includes('role='))
    ) {
      imgIssues.push({
        type: 'image-alts',
        file,
        details: '交互性图片不应有空的alt属性',
        code: code.trim(),
        recommendation: '为交互性图片添加描述性alt属性',
      });
    }
  });

  if (imgIssues.length > 0) {
    allIssues.imageAlts = imgIssues;
  }
} catch (error) {
  console.error('图片alt属性检查失败:', error.message);
}

// 检查表单标签关联
console.log('正在检查表单标签关联...');
try {
  // 查找所有表单输入元素
  const formGrepCommand =
    'grep -r "<input\\|<select\\|<textarea" --include="*.jsx" --include="*.tsx" ./components ./pages';
  const formOutput = execSync(formGrepCommand, {
    encoding: 'utf8',
    cwd: path.resolve(__dirname, '..'),
  });

  const formIssues = [];

  formOutput.split('\n').forEach(line => {
    if (!line) return;

    const [filePath, code] = line.split(':', 2);
    const file = filePath.replace('./', '');

    // 检查缺少id属性的输入元素
    if (!code.includes('id=')) {
      formIssues.push({
        type: 'form-labels',
        file,
        details: '表单元素缺少id属性，可能导致标签关联失败',
        code: code.trim(),
        recommendation: '添加唯一id属性并确保使用htmlFor关联标签',
      });
    }

    // 检查缺少aria-label的无标签输入元素
    if (
      !code.includes('aria-label') &&
      !code.includes('aria-labelledby') &&
      !code.includes('type="hidden"')
    ) {
      // 读取整个文件检查是否有关联的label
      const fileContent = fs.readFileSync(path.resolve(__dirname, '..', file), 'utf8');
      const idMatch = code.match(/id=["']([^"']+)["']/);

      if (idMatch && idMatch[1]) {
        const id = idMatch[1];
        const hasLabel =
          fileContent.includes(`htmlFor="${id}"`) || fileContent.includes(`htmlFor={'${id}'}`);

        if (!hasLabel) {
          formIssues.push({
            type: 'form-labels',
            file,
            details: `表单元素(id="${id}")缺少关联的标签`,
            code: code.trim(),
            recommendation: '使用<label htmlFor="elementId">或aria-label属性为表单元素添加标签',
          });
        }
      }
    }
  });

  if (formIssues.length > 0) {
    allIssues.formLabels = formIssues;
  }
} catch (error) {
  console.error('表单标签检查失败:', error.message);
}

// 汇总所有问题
const totalIssues = Object.values(allIssues).reduce((acc, issues) => acc + issues.length, 0);

// 按严重性对问题进行分类
const criticalIssues = [];
const majorIssues = [];
const minorIssues = [];

// 整理所有问题
Object.entries(allIssues).forEach(([category, issues]) => {
  issues.forEach(issue => {
    // 根据问题类型确定严重性
    if (
      category === 'keyboardFocus' ||
      (category === 'ariaAttributes' && issue.details.includes('aria-hidden="true"')) ||
      (category === 'formLabels' && !issue.details.includes('关联的标签'))
    ) {
      criticalIssues.push({ ...issue, severity: 'critical', category });
    } else if (
      category === 'colorContrast' ||
      category === 'headingStructure' ||
      category === 'imageAlts'
    ) {
      majorIssues.push({ ...issue, severity: 'major', category });
    } else {
      minorIssues.push({ ...issue, severity: 'minor', category });
    }
  });
});

// 运行特定组件测试（新增功能）
const runComponentLevelTests = componentPath => {
  console.log(`正在测试组件: ${componentPath}...`);

  const issues = [];
  const componentDir = path.dirname(componentPath);
  const componentFile = path.basename(componentPath);

  try {
    // 分析组件的导入项
    const componentContent = fs.readFileSync(path.resolve(__dirname, '..', componentPath), 'utf8');

    // 检查是否使用了无障碍组件
    const a11yComponentCheck = {
      usesFocusableItem:
        componentContent.includes('import FocusableItem') ||
        componentContent.includes('from "./a11y/FocusableItem"'),
      usesA11yFormInput:
        componentContent.includes('import A11yFormInput') ||
        componentContent.includes('from "./a11y/A11yFormInput"'),
      usesKeyboardNavigation: componentContent.includes('useKeyboardNavigation'),
    };

    // 如果组件包含交互元素但未使用无障碍组件
    if (
      (componentContent.includes('onClick={') || componentContent.includes('onSubmit={')) &&
      !a11yComponentCheck.usesFocusableItem &&
      !a11yComponentCheck.usesKeyboardNavigation
    ) {
      issues.push({
        type: 'component-structure',
        file: componentPath,
        details: '组件包含交互元素但未使用无障碍组件',
        recommendation: '考虑使用FocusableItem替换自定义点击元素，或使用useKeyboardNavigation钩子',
      });
    }

    // 检查表单元素
    if (
      (componentContent.includes('<input') ||
        componentContent.includes('<textarea') ||
        componentContent.includes('<select')) &&
      !a11yComponentCheck.usesA11yFormInput
    ) {
      issues.push({
        type: 'component-structure',
        file: componentPath,
        details: '组件包含表单元素但未使用无障碍表单组件',
        recommendation: '考虑使用A11yFormInput替换原生表单控件以确保正确的标签关联',
      });
    }

    // 检查aria-*属性使用情况
    const ariaMatches = componentContent.match(/aria-[a-zA-Z]+/g) || [];
    const uniqueAriaAttributes = [...new Set(ariaMatches)];

    // 检查是否有可能的ARIA属性错误
    uniqueAriaAttributes.forEach(attr => {
      const validAriaAttributes = [
        'aria-label',
        'aria-labelledby',
        'aria-describedby',
        'aria-details',
        'aria-hidden',
        'aria-expanded',
        'aria-haspopup',
        'aria-controls',
        'aria-pressed',
        'aria-checked',
        'aria-selected',
        'aria-invalid',
        'aria-required',
        'aria-disabled',
        'aria-level',
        'aria-current',
        'aria-live',
        'aria-atomic',
        'aria-busy',
        'aria-relevant',
        'aria-rowcount',
        'aria-rowindex',
        'aria-colcount',
        'aria-colindex',
        'aria-roledescription',
      ];

      if (!validAriaAttributes.includes(attr)) {
        issues.push({
          type: 'aria-attributes',
          file: componentPath,
          details: `可能使用了未识别的ARIA属性: ${attr}`,
          recommendation: '请验证ARIA属性名称是否正确，参考WAI-ARIA规范',
        });
      }
    });

    return issues;
  } catch (error) {
    console.error(`组件测试失败: ${error.message}`);
    return [];
  }
};

// 添加严重性分类统计
const categorizeBySeverity = issues => {
  const categoryMap = {
    'keyboard-focus': 'keyboardFocus',
    'color-contrast': 'colorContrast',
    'aria-attributes': 'ariaAttributes',
    'heading-structure': 'headingStructure',
    'image-alt': 'imageAlts',
    'form-label': 'formLabels',
    'component-structure': 'componentStructure',
  };

  const stats = {
    keyboard: { critical: 0, major: 0, minor: 0 },
    contrast: { critical: 0, major: 0, minor: 0 },
    aria: { critical: 0, major: 0, minor: 0 },
    semantics: { critical: 0, major: 0, minor: 0 },
    form: { critical: 0, major: 0, minor: 0 },
    component: { critical: 0, major: 0, minor: 0 },
  };

  // 统计每种类型的严重程度分布
  Object.keys(issues).forEach(issueType => {
    issues[issueType].forEach(issue => {
      const category =
        issueType === 'keyboardFocus'
          ? 'keyboard'
          : issueType === 'colorContrast'
            ? 'contrast'
            : issueType === 'ariaAttributes'
              ? 'aria'
              : issueType === 'headingStructure'
                ? 'semantics'
                : issueType === 'formLabels'
                  ? 'form'
                  : issueType === 'componentStructure'
                    ? 'component'
                    : 'semantics';

      const severity =
        issue.severity === 'critical' ? 'critical' : issue.severity === 'major' ? 'major' : 'minor';

      stats[category][severity]++;
    });
  });

  return stats;
};

// 生成优化建议报告
const generateRecommendations = issues => {
  const allIssuesList = Object.values(issues).flat();

  // 按文件分组
  const fileGroups = {};
  allIssuesList.forEach(issue => {
    if (!issue.file) return;

    if (!fileGroups[issue.file]) {
      fileGroups[issue.file] = [];
    }
    fileGroups[issue.file].push(issue);
  });

  // 生成每个文件的建议
  const recommendations = Object.keys(fileGroups).map(file => {
    const fileIssues = fileGroups[file];

    // 识别文件中最常见的问题类型
    const typeCounts = {};
    fileIssues.forEach(issue => {
      const type = issue.type || 'unknown';
      typeCounts[type] = (typeCounts[type] || 0) + 1;
    });

    // 找出最常见的问题类型
    let mostCommonType = '';
    let highestCount = 0;
    Object.keys(typeCounts).forEach(type => {
      if (typeCounts[type] > highestCount) {
        highestCount = typeCounts[type];
        mostCommonType = type;
      }
    });

    // 生成针对性建议
    let primaryRecommendation = '';
    switch (mostCommonType) {
      case 'keyboard-focus':
        primaryRecommendation = '重点优化键盘导航，使用FocusableItem组件并实现焦点管理';
        break;
      case 'color-contrast':
        primaryRecommendation = '调整颜色对比度，确保文本清晰可读';
        break;
      case 'aria-attributes':
        primaryRecommendation = '审核并修正ARIA属性使用，确保语义正确';
        break;
      case 'form-label':
        primaryRecommendation = '确保所有表单元素都有关联的标签，使用A11yFormInput组件';
        break;
      default:
        primaryRecommendation = '审核组件的整体无障碍实现';
    }

    return {
      file,
      issueCount: fileIssues.length,
      primaryType: mostCommonType,
      recommendation: primaryRecommendation,
      details: fileIssues.slice(0, 3).map(i => i.details), // 只显示前3个问题的详情
    };
  });

  // 按问题数量排序
  return recommendations.sort((a, b) => b.issueCount - a.issueCount);
};

// 如果指定了组件路径，则运行组件级测试
if (componentPath) {
  console.log(`正在对组件 ${componentPath} 进行无障碍测试...`);
  const componentIssues = runComponentLevelTests(componentPath);

  // 将组件测试结果添加到总体问题中
  componentIssues.forEach(issue => {
    const category = issue.type.includes('aria')
      ? 'ariaAttributes'
      : issue.type.includes('keyboard')
        ? 'keyboardFocus'
        : issue.type.includes('component')
          ? 'componentStructure'
          : 'ariaAttributes';

    if (!allIssues[category]) {
      allIssues[category] = [];
    }

    allIssues[category].push({
      ...issue,
      severity: 'major', // 默认设为主要问题
    });
  });
}

// 生成HTML报告
const htmlReport = `
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>无障碍检查报告 - ${new Date().toLocaleDateString('zh-CN')}</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      line-height: 1.5;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      color: #333;
    }
    h1 {
      border-bottom: 2px solid #0066cc;
      padding-bottom: 8px;
      color: #0055aa;
    }
    h2 {
      margin-top: 30px;
      color: #0066cc;
    }
    h3 {
      margin-top: 25px;
      color: #444;
    }
    .summary {
      background-color: #f8f9fa;
      border-radius: 6px;
      padding: 15px;
      margin: 20px 0;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 15px;
      margin-top: 15px;
    }
    .summary-card {
      background: white;
      border-radius: 6px;
      padding: 15px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .issue {
      background-color: white;
      border-left: 4px solid #aaa;
      border-radius: 4px;
      padding: 15px;
      margin-bottom: 15px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .issue.critical {
      border-left-color: #dc3545;
    }
    .issue.major {
      border-left-color: #fd7e14;
    }
    .issue.minor {
      border-left-color: #ffc107;
    }
    .issue-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    .issue-title {
      font-weight: bold;
      font-size: 1.1em;
    }
    .issue-severity {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.85em;
      color: white;
    }
    .severity-critical {
      background-color: #dc3545;
    }
    .severity-major {
      background-color: #fd7e14;
    }
    .severity-minor {
      background-color: #ffc107;
      color: #333;
    }
    .issue-location {
      font-family: monospace;
      margin-bottom: 8px;
      color: #666;
    }
    .issue-details {
      margin-bottom: 8px;
    }
    .issue-recommendation {
      background-color: #e9f5ff;
      padding: 8px;
      border-radius: 4px;
    }
    .code-block {
      background-color: #f8f9fa;
      padding: 8px;
      border-radius: 4px;
      font-family: monospace;
      overflow-x: auto;
      margin: 8px 0;
      white-space: pre-wrap;
    }
    .tabs {
      display: flex;
      margin-bottom: 15px;
      border-bottom: 1px solid #ddd;
    }
    .tab {
      padding: 8px 15px;
      cursor: pointer;
      border: 1px solid transparent;
      border-bottom: none;
      margin-right: 5px;
      border-radius: 4px 4px 0 0;
    }
    .tab.active {
      border-color: #ddd;
      background-color: white;
      margin-bottom: -1px;
      border-bottom: 1px solid white;
    }
    .tab-content {
      display: none;
    }
    .tab-content.active {
      display: block;
    }
    .issue-count {
      font-weight: bold;
      font-size: 1.5em;
    }
    @media (max-width: 768px) {
      .summary-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <h1>前端无障碍检查报告</h1>
  <p>生成时间: ${new Date().toLocaleString('zh-CN')}</p>
  
  <div class="summary">
    <h2>摘要</h2>
    <p>共检查出 <strong>${totalIssues}</strong> 个无障碍问题：</p>
    <div class="summary-grid">
      <div class="summary-card">
        <div>严重问题</div>
        <div class="issue-count" style="color: #dc3545;">${criticalIssues.length}</div>
      </div>
      <div class="summary-card">
        <div>主要问题</div>
        <div class="issue-count" style="color: #fd7e14;">${majorIssues.length}</div>
      </div>
      <div class="summary-card">
        <div>次要问题</div>
        <div class="issue-count" style="color: #ffc107;">${minorIssues.length}</div>
      </div>
    </div>
  </div>
  
  <div class="tabs">
    <div class="tab active" onclick="switchTab('all')">所有问题 (${totalIssues})</div>
    <div class="tab" onclick="switchTab('critical')">严重问题 (${criticalIssues.length})</div>
    <div class="tab" onclick="switchTab('major')">主要问题 (${majorIssues.length})</div>
    <div class="tab" onclick="switchTab('minor')">次要问题 (${minorIssues.length})</div>
  </div>
  
  <div class="tab-content active" id="tab-all">
    <h2>所有问题</h2>
    ${
      criticalIssues.length === 0 && majorIssues.length === 0 && minorIssues.length === 0
        ? '<p>没有发现问题，太棒了！</p>'
        : ''
    }
    
    ${[...criticalIssues, ...majorIssues, ...minorIssues]
      .map(
        issue => `
      <div class="issue ${issue.severity}">
        <div class="issue-header">
          <div class="issue-title">${getCategoryTitle(issue.category)}</div>
          <div class="issue-severity severity-${issue.severity}">${getSeverityLabel(issue.severity)}</div>
        </div>
        <div class="issue-location">${issue.file}${issue.line ? `:${issue.line}` : ''}</div>
        <div class="issue-details">${issue.details}</div>
        ${issue.code ? `<div class="code-block">${issue.code}</div>` : ''}
        <div class="issue-recommendation">${issue.recommendation}</div>
      </div>
    `
      )
      .join('')}
  </div>
  
  <div class="tab-content" id="tab-critical">
    <h2>严重问题</h2>
    ${criticalIssues.length === 0 ? '<p>没有发现严重问题，太棒了！</p>' : ''}
    
    ${criticalIssues
      .map(
        issue => `
      <div class="issue critical">
        <div class="issue-header">
          <div class="issue-title">${getCategoryTitle(issue.category)}</div>
          <div class="issue-severity severity-critical">${getSeverityLabel(issue.severity)}</div>
        </div>
        <div class="issue-location">${issue.file}${issue.line ? `:${issue.line}` : ''}</div>
        <div class="issue-details">${issue.details}</div>
        ${issue.code ? `<div class="code-block">${issue.code}</div>` : ''}
        <div class="issue-recommendation">${issue.recommendation}</div>
      </div>
    `
      )
      .join('')}
  </div>
  
  <div class="tab-content" id="tab-major">
    <h2>主要问题</h2>
    ${majorIssues.length === 0 ? '<p>没有发现主要问题，太棒了！</p>' : ''}
    
    ${majorIssues
      .map(
        issue => `
      <div class="issue major">
        <div class="issue-header">
          <div class="issue-title">${getCategoryTitle(issue.category)}</div>
          <div class="issue-severity severity-major">${getSeverityLabel(issue.severity)}</div>
        </div>
        <div class="issue-location">${issue.file}${issue.line ? `:${issue.line}` : ''}</div>
        <div class="issue-details">${issue.details}</div>
        ${issue.code ? `<div class="code-block">${issue.code}</div>` : ''}
        <div class="issue-recommendation">${issue.recommendation}</div>
      </div>
    `
      )
      .join('')}
  </div>
  
  <div class="tab-content" id="tab-minor">
    <h2>次要问题</h2>
    ${minorIssues.length === 0 ? '<p>没有发现次要问题，太棒了！</p>' : ''}
    
    ${minorIssues
      .map(
        issue => `
      <div class="issue minor">
        <div class="issue-header">
          <div class="issue-title">${getCategoryTitle(issue.category)}</div>
          <div class="issue-severity severity-minor">${getSeverityLabel(issue.severity)}</div>
        </div>
        <div class="issue-location">${issue.file}${issue.line ? `:${issue.line}` : ''}</div>
        <div class="issue-details">${issue.details}</div>
        ${issue.code ? `<div class="code-block">${issue.code}</div>` : ''}
        <div class="issue-recommendation">${issue.recommendation}</div>
      </div>
    `
      )
      .join('')}
  </div>
  
  <script>
    function switchTab(tabId) {
      // 隐藏所有标签内容
      document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
      });
      
      // 取消所有标签的活动状态
      document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
      });
      
      // 显示选定的标签内容
      document.getElementById('tab-' + tabId).classList.add('active');
      
      // 激活选定的标签
      document.querySelectorAll('.tab').forEach(tab => {
        if (tab.textContent.toLowerCase().includes(tabId)) {
          tab.classList.add('active');
        }
      });
    }
    
    // 默认选中"所有问题"标签
    document.addEventListener('DOMContentLoaded', () => {
      switchTab('all');
    });
  </script>
</body>
</html>
`;

// 添加优化建议和统计数据
const stats = categorizeBySeverity(allIssues);
const recommendations = generateRecommendations(allIssues);

// 将结果保存为JSON
const jsonReport = {
  summary: {
    date: new Date().toISOString(),
    total: totalIssues,
    critical: criticalIssues.length,
    major: majorIssues.length,
    minor: minorIssues.length,
    stats,
  },
  issues: {
    critical: criticalIssues,
    major: majorIssues,
    minor: minorIssues,
  },
  recommendations,
};

fs.writeFileSync(jsonReportPath, JSON.stringify(jsonReport, null, 2));
fs.writeFileSync(latestJsonReportPath, JSON.stringify(jsonReport, null, 2));

// 保存HTML报告
fs.writeFileSync(htmlReportPath, htmlReport);

console.log(`\n无障碍审计完成!`);
console.log(
  `共发现 ${totalIssues} 个问题（${criticalIssues.length} 严重，${majorIssues.length} 主要，${minorIssues.length} 次要）`
);
console.log(`HTML报告已保存到: ${htmlReportPath}`);
console.log(`JSON报告已保存到: ${jsonReportPath}`);

// 辅助函数

// 根据严重性获取标签文本
function getSeverityLabel(severity) {
  switch (severity) {
    case 'critical':
      return '严重';
    case 'major':
      return '主要';
    case 'minor':
      return '次要';
    default:
      return '未知';
  }
}

// 根据问题类别获取标题
function getCategoryTitle(category) {
  switch (category) {
    case 'colorContrast':
      return '颜色对比度问题';
    case 'keyboardFocus':
      return '键盘焦点与导航问题';
    case 'headingStructure':
      return '标题结构问题';
    case 'ariaAttributes':
      return 'ARIA属性问题';
    case 'imageAlts':
      return '图片替代文本问题';
    case 'formLabels':
      return '表单标签问题';
    default:
      return '无障碍问题';
  }
}
