#!/usr/bin/env node

/**
 * 键盘焦点和导航问题检查工具
 *
 * 此脚本用于检查项目中的React组件，找出潜在的键盘导航和焦点管理问题。
 * 主要检查：
 * - 没有键盘事件处理程序的可点击元素
 * - 缺少tabIndex的交互元素
 * - 没有焦点样式的可聚焦元素
 * - 错误使用div作为按钮
 *
 * 用法: node scripts/check-keyboard-focus.js
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// 可能的键盘事件处理程序
const KEYBOARD_EVENT_HANDLERS = ['onKeyDown', 'onKeyUp', 'onKeyPress'];

// 需要键盘事件处理的角色
const INTERACTIVE_ROLES = [
  'button',
  'link',
  'checkbox',
  'radio',
  'tab',
  'menuitem',
  'option',
  'switch',
];

// 检查组件文件中的键盘导航问题
function checkFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const issues = [];

  // 检查每一行代码
  lines.forEach((line, index) => {
    // 检查可点击但没有键盘事件的元素
    if (line.includes('onClick') || line.includes('onDoubleClick')) {
      const hasKeyboardHandler = KEYBOARD_EVENT_HANDLERS.some(handler => line.includes(handler));

      if (
        !hasKeyboardHandler &&
        !line.includes('<button') &&
        !line.includes('<a ') &&
        !line.includes('role="button"') &&
        !line.includes('role="link"')
      ) {
        issues.push({
          line: index + 1,
          code: line.trim(),
          type: 'missing-keyboard-handler',
          description: '可点击元素没有键盘事件处理程序（onKeyDown/onKeyUp/onKeyPress）',
        });
      }
    }

    // 检查使用div模拟交互元素但没有正确的角色属性
    if (
      (line.includes('<div') || line.includes('<span')) &&
      (line.includes('onClick') || line.includes('onDoubleClick'))
    ) {
      const hasProperRole = INTERACTIVE_ROLES.some(role => line.includes(`role="${role}"`));

      if (!hasProperRole) {
        issues.push({
          line: index + 1,
          code: line.trim(),
          type: 'missing-role',
          description: '使用div或span作为交互元素但没有适当的角色属性',
        });
      }
    }

    // 检查自定义交互元素是否有tabIndex
    if (
      (line.includes('role="button"') ||
        line.includes('role="link"') ||
        INTERACTIVE_ROLES.some(role => line.includes(`role="${role}"`))) &&
      !line.includes('tabIndex') &&
      !line.includes('<button') &&
      !line.includes('<a ') &&
      !line.includes('<input')
    ) {
      issues.push({
        line: index + 1,
        code: line.trim(),
        type: 'missing-tabindex',
        description: '自定义交互元素没有tabIndex属性',
      });
    }

    // 检查可能缺少焦点样式的元素
    if (
      line.includes('tabIndex') &&
      !content.includes(':focus') &&
      !content.includes('focus-visible') &&
      !content.includes('focus-ring')
    ) {
      issues.push({
        line: index + 1,
        code: line.trim(),
        type: 'missing-focus-style',
        description: '可聚焦元素可能缺少焦点样式',
      });
    }

    // 检查完全阻止事件传播的处理程序（可能阻止键盘事件）
    if (line.includes('stopPropagation()') || line.includes('preventDefault()')) {
      if (
        !line.includes('keyCode') &&
        !line.includes('key ===') &&
        !line.includes('keyboardEvent')
      ) {
        issues.push({
          line: index + 1,
          code: line.trim(),
          type: 'event-blocking',
          description: '事件处理程序阻止事件传播，可能影响键盘导航',
        });
      }
    }
  });

  return issues;
}

// 主函数
function main() {
  // 获取组件文件列表
  const componentFiles = glob.sync('components/**/*.{jsx,tsx}', {
    cwd: path.resolve(__dirname, '..'),
  });

  let allIssues = [];

  // 分析每个文件
  componentFiles.forEach(file => {
    const filePath = path.resolve(__dirname, '..', file);
    try {
      const issues = checkFile(filePath);

      if (issues.length > 0) {
        allIssues.push({
          file,
          issues,
        });
      }
    } catch (error) {
      console.error(`处理文件 ${file} 时出错:`, error.message);
    }
  });

  // 输出结果
  if (allIssues.length > 0) {
    console.log(`在 ${allIssues.length} 个文件中发现键盘导航问题:\n`);

    allIssues.forEach(fileIssues => {
      console.log(`文件: ${fileIssues.file}`);
      console.log(`问题数量: ${fileIssues.issues.length}`);

      fileIssues.issues.forEach((issue, index) => {
        console.log(`  问题 #${index + 1} (行 ${issue.line})`);
        console.log(`  类型: ${issue.type}`);
        console.log(`  描述: ${issue.description}`);
        console.log(`  代码: ${issue.code}`);
        console.log('  修复建议: ');

        switch (issue.type) {
          case 'missing-keyboard-handler':
            console.log('    添加键盘事件处理程序，例如:');
            console.log('    onKeyDown={(e) => e.key === "Enter" && handleClick()}');
            break;

          case 'missing-role':
            console.log('    添加适当的角色属性:');
            console.log('    role="button" aria-pressed="false"');
            console.log('    或考虑使用语义化HTML元素，如<button>或<a>');
            break;

          case 'missing-tabindex':
            console.log('    添加tabIndex属性:');
            console.log('    tabIndex={0}');
            break;

          case 'missing-focus-style':
            console.log('    添加焦点样式:');
            console.log('    className={`${styles.element} ${isFocused ? styles.focused : ""}`}');
            console.log('    或在CSS中添加:focus和:focus-visible样式');
            break;

          case 'event-blocking':
            console.log('    确保只在必要时阻止事件传播:');
            console.log('    if (特定条件) { e.stopPropagation(); }');
            console.log('    或使用e.stopPropagation仅阻止特定键盘事件:');
            console.log('    if (e.key === "Escape") { e.stopPropagation(); }');
            break;
        }

        console.log('');
      });

      console.log('-------------------------------------------\n');
    });

    console.log('综合建议:');
    console.log('1. 对于可点击的div/span，确保添加role="button"、tabIndex={0}和键盘事件处理');
    console.log('2. 尽可能使用语义化HTML元素，如<button>、<a>、<input>');
    console.log('3. 为所有可聚焦元素添加清晰的焦点视觉指示器');
    console.log('4. 在全局样式中添加:focus-visible样式以确保键盘用户的可访问性');
    console.log('5. 避免在没有条件检查的情况下阻止事件传播');
  } else {
    console.log('未发现键盘导航问题，太棒了！');
  }
}

main();
