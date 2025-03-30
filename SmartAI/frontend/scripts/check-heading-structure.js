#!/usr/bin/env node

/**
 * 标题结构检查工具
 *
 * 此脚本用于检查项目中React组件的标题结构，确保标题有正确的层次结构和语义性。
 * 主要检查：
 * - 标题层次是否正确（不跳级）
 * - 是否使用了正确的标题元素（h1-h6）
 * - 页面是否有主标题（h1）
 * - 是否正确使用了ARIA标题属性
 *
 * 用法: node scripts/check-heading-structure.js
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// 检查组件文件中的标题结构
function checkFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const issues = [];

  // 提取所有标题元素
  const headings = [];
  let inJSX = false;

  lines.forEach((line, index) => {
    // 标记JSX区域
    if (line.includes('return (') || line.includes('return(')) {
      inJSX = true;
    }

    if (inJSX) {
      // 检查h1-h6标签
      for (let i = 1; i <= 6; i++) {
        if (line.includes(`<h${i}`) || line.includes(`<H${i}`)) {
          headings.push({
            level: i,
            line: index + 1,
            text: line.trim(),
          });
        }
      }

      // 检查aria-level属性
      const ariaLevelMatch = line.match(/aria-level=["'](\d+)["']/);
      if (ariaLevelMatch) {
        headings.push({
          level: parseInt(ariaLevelMatch[1]),
          line: index + 1,
          text: line.trim(),
          isAria: true,
        });
      }
    }

    if (line.includes(');') && inJSX) {
      inJSX = false;
    }
  });

  // 如果是页面组件（通常名称为Page或者View结尾），检查是否有h1
  const isPageComponent = path.basename(filePath).match(/(?:Page|View|Screen)\.(jsx|tsx)$/);
  if (isPageComponent && headings.length > 0) {
    const hasH1 = headings.some(h => h.level === 1);
    if (!hasH1) {
      issues.push({
        line: 1,
        type: 'missing-h1',
        description: '页面组件缺少主标题(h1)元素',
      });
    }
  }

  // 检查标题层次是否跳级
  for (let i = 0; i < headings.length - 1; i++) {
    if (
      headings[i + 1].level > headings[i].level &&
      headings[i + 1].level - headings[i].level > 1
    ) {
      issues.push({
        line: headings[i + 1].line,
        type: 'skipped-heading-level',
        description: `标题层次从h${headings[i].level}跳到h${headings[i + 1].level}，缺少中间层次`,
        text: headings[i + 1].text,
      });
    }
  }

  // 检查ARIA标题级别是否合适
  headings
    .filter(h => h.isAria)
    .forEach(heading => {
      if (heading.level < 1 || heading.level > 6) {
        issues.push({
          line: heading.line,
          type: 'invalid-aria-level',
          description: `无效的aria-level值: ${heading.level}（有效范围是1-6）`,
          text: heading.text,
        });
      }
    });

  return {
    issues,
    hasHeadings: headings.length > 0,
    headings,
  };
}

// 主函数
function main() {
  // 获取组件文件列表
  const componentFiles = glob.sync('components/**/*.{jsx,tsx}', {
    cwd: path.resolve(__dirname, '..'),
  });
  const pageFiles = glob.sync('pages/**/*.{jsx,tsx}', {
    cwd: path.resolve(__dirname, '..'),
  });

  const allFiles = [...componentFiles, ...pageFiles];

  let allIssues = [];
  let componentsWithHeadings = 0;

  // 分析每个文件
  allFiles.forEach(file => {
    const filePath = path.resolve(__dirname, '..', file);
    try {
      const { issues, hasHeadings, headings } = checkFile(filePath);

      if (hasHeadings) {
        componentsWithHeadings++;
      }

      if (issues.length > 0) {
        allIssues.push({
          file,
          issues,
          headings,
        });
      }
    } catch (error) {
      console.error(`处理文件 ${file} 时出错:`, error.message);
    }
  });

  // 输出结果
  console.log(
    `分析了 ${allFiles.length} 个文件，其中 ${componentsWithHeadings} 个文件包含标题元素.\n`
  );

  if (allIssues.length > 0) {
    console.log(`在 ${allIssues.length} 个文件中发现标题结构问题:\n`);

    allIssues.forEach(fileIssues => {
      console.log(`文件: ${fileIssues.file}`);
      console.log(`问题数量: ${fileIssues.issues.length}`);

      if (fileIssues.headings && fileIssues.headings.length > 0) {
        console.log('当前标题结构:');
        fileIssues.headings.forEach(h => {
          console.log(`  ${h.isAria ? 'ARIA 标题' : 'HTML 标题'} (级别 ${h.level}) 在行 ${h.line}`);
        });
      }

      fileIssues.issues.forEach((issue, index) => {
        console.log(`  问题 #${index + 1} (行 ${issue.line})`);
        console.log(`  类型: ${issue.type}`);
        console.log(`  描述: ${issue.description}`);
        if (issue.text) {
          console.log(`  代码: ${issue.text}`);
        }
        console.log('  修复建议: ');

        switch (issue.type) {
          case 'missing-h1':
            console.log('    添加一个h1元素作为页面的主标题:');
            console.log('    <h1>页面标题</h1>');
            break;

          case 'skipped-heading-level':
            console.log('    不要跳过标题级别，确保标题级别依次递增:');
            console.log('    例如，h1 -> h2 -> h3，而不是 h1 -> h3');
            break;

          case 'invalid-aria-level':
            console.log('    使用有效的aria-level值 (1-6):');
            console.log('    aria-level="2"');
            break;
        }

        console.log('');
      });

      console.log('-------------------------------------------\n');
    });

    console.log('综合建议:');
    console.log('1. 确保每个页面都有一个h1作为主标题');
    console.log('2. 标题层次应从h1开始，按顺序递增，不要跳级');
    console.log('3. 在使用aria-level时确保值在有效范围内(1-6)');
    console.log('4. 考虑使用语义化标题（h1-h6）代替带有aria-level的非标题元素');
  } else {
    console.log('未发现标题结构问题，太棒了！');
  }

  // 检查是否存在没有标题的页面组件
  const pageFilesWithoutHeadings = pageFiles.filter(file => {
    const filePath = path.resolve(__dirname, '..', file);
    try {
      const { hasHeadings } = checkFile(filePath);
      return !hasHeadings;
    } catch (error) {
      return false;
    }
  });

  if (pageFilesWithoutHeadings.length > 0) {
    console.log(`\n警告: 发现 ${pageFilesWithoutHeadings.length} 个页面组件没有任何标题元素:`);
    pageFilesWithoutHeadings.forEach(file => {
      console.log(`  - ${file}`);
    });
    console.log('\n建议: 每个页面应该至少有一个标题元素，以提高内容的可访问性和结构化。');
  }
}

main();
