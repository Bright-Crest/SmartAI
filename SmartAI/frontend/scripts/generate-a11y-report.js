#!/usr/bin/env node

/**
 * 无障碍测试报告生成器
 *
 * 这个脚本从Cypress测试生成的JSON格式测试结果生成HTML格式的无障碍测试报告
 *
 * 用法: node generate-a11y-report.js
 */

const fs = require('fs');
const path = require('path');

// 配置路径
const A11Y_RESULTS_DIR = path.resolve(__dirname, '../cypress/a11y-results');
const REPORT_TEMPLATE = path.resolve(__dirname, '../cypress/fixtures/a11y-report-template.html');
const OUTPUT_REPORT = path.resolve(__dirname, '../cypress/a11y-results/accessibility-report.html');

// 确保结果目录存在
if (!fs.existsSync(A11Y_RESULTS_DIR)) {
  console.log('创建无障碍测试结果目录...');
  fs.mkdirSync(A11Y_RESULTS_DIR, { recursive: true });
}

// 读取所有测试结果文件
console.log('读取测试结果文件...');
const resultFiles = fs
  .readdirSync(A11Y_RESULTS_DIR)
  .filter(file => file.endsWith('.json'))
  .map(file => path.join(A11Y_RESULTS_DIR, file));

// 如果没有测试结果，退出
if (resultFiles.length === 0) {
  console.error('错误: 未找到测试结果文件。请先运行无障碍测试。');
  process.exit(1);
}

// 查找摘要报告
const summaryFile = resultFiles.find(file => file.includes('summary-report'));

if (!summaryFile) {
  console.error('错误: 未找到摘要报告文件。请确保测试完成。');
  process.exit(1);
}

console.log(`找到摘要报告: ${summaryFile}`);

// 读取摘要报告
const summaryData = JSON.parse(fs.readFileSync(summaryFile, 'utf8'));

// 读取各页面报告
const pageReports = [];
for (const file of resultFiles) {
  if (file !== summaryFile && file.includes('a11y-report-')) {
    try {
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      pageReports.push(data);
    } catch (err) {
      console.warn(`警告: 无法解析文件 ${file}: ${err.message}`);
    }
  }
}

console.log(`找到 ${pageReports.length} 个页面报告`);

// 处理组件级别违规
console.log('分析组件违规情况...');
const componentIssues = {};

pageReports.forEach(report => {
  report.violations.forEach(violation => {
    violation.nodes.forEach(node => {
      // 尝试从HTML中提取组件名
      const componentMatch = node.html.match(/data-testid="([^"]+)"/);
      const componentName = componentMatch ? componentMatch[1] : '其他';

      // 初始化组件数据
      if (!componentIssues[componentName]) {
        componentIssues[componentName] = {
          total: 0,
          critical: 0,
          serious: 0,
          moderate: 0,
          minor: 0,
          violations: [],
        };
      }

      // 累计违规
      componentIssues[componentName].total++;
      componentIssues[componentName][violation.impact]++;

      // 存储违规详情
      if (!componentIssues[componentName].violations.find(v => v.id === violation.id)) {
        componentIssues[componentName].violations.push({
          id: violation.id,
          impact: violation.impact,
          description: violation.description,
          help: violation.help,
          helpUrl: violation.helpUrl,
        });
      }
    });
  });
});

// 提取常见违规
console.log('分析常见违规...');
const allViolations = pageReports.flatMap(report => report.violations);
const violationCounts = {};

allViolations.forEach(violation => {
  if (!violationCounts[violation.id]) {
    violationCounts[violation.id] = {
      count: 0,
      impact: violation.impact,
      description: violation.description,
      help: violation.help,
      helpUrl: violation.helpUrl,
    };
  }
  violationCounts[violation.id].count += violation.nodes.length;
});

// 排序常见违规（按计数降序）
const commonViolations = Object.entries(violationCounts)
  .map(([id, data]) => ({ id, ...data }))
  .sort((a, b) => b.count - a.count);

// 准备报告数据
console.log('准备报告数据...');
const reportData = {
  ...summaryData,
  componentIssues,
  commonViolations: commonViolations.slice(0, 10), // 取前10个常见违规
};

// 读取报告模板
console.log('读取报告模板...');
let templateHtml = fs.readFileSync(REPORT_TEMPLATE, 'utf8');

// 注入数据到模板
console.log('生成HTML报告...');
templateHtml = templateHtml.replace(
  /<script>\s*\/\/[^<]*loadReportData\(exampleData\);\s*<\/script>/s,
  `<script>
    // 报告数据
    const reportData = ${JSON.stringify(reportData, null, 2)};
    
    // 加载报告数据
    loadReportData(reportData);
    
    // 渲染组件分析
    const componentsTable = document.getElementById('components-table');
    const componentData = ${JSON.stringify(Object.entries(componentIssues), null, 2)};
    
    componentData
      .sort((a, b) => b[1].total - a[1].total)
      .forEach(([name, data]) => {
        const row = document.createElement('tr');
        
        const nameCell = document.createElement('td');
        nameCell.className = 'component-name';
        nameCell.textContent = name;
        
        const totalCell = document.createElement('td');
        totalCell.textContent = data.total;
        
        const criticalCell = document.createElement('td');
        criticalCell.textContent = data.critical;
        if (data.critical > 0) criticalCell.style.color = '#dc3545';
        
        const seriousCell = document.createElement('td');
        seriousCell.textContent = data.serious;
        if (data.serious > 0) seriousCell.style.color = '#fd7e14';
        
        const moderateCell = document.createElement('td');
        moderateCell.textContent = data.moderate;
        
        const minorCell = document.createElement('td');
        minorCell.textContent = data.minor;
        
        row.appendChild(nameCell);
        row.appendChild(totalCell);
        row.appendChild(criticalCell);
        row.appendChild(seriousCell);
        row.appendChild(moderateCell);
        row.appendChild(minorCell);
        
        componentsTable.appendChild(row);
      });
    
    // 渲染常见违规
    const commonViolationsContainer = document.getElementById('common-violations');
    const commonViolationsData = ${JSON.stringify(commonViolations.slice(0, 10), null, 2)};
    
    commonViolationsData.forEach(violation => {
      const container = document.createElement('div');
      container.className = 'violation-details';
      
      const header = document.createElement('div');
      header.style.display = 'flex';
      header.style.justifyContent = 'space-between';
      header.style.alignItems = 'center';
      
      const title = document.createElement('h3');
      title.style.margin = '0';
      title.style.display = 'flex';
      title.style.alignItems = 'center';
      
      const impact = document.createElement('span');
      impact.className = \`violation-tag tag-\${violation.impact}\`;
      impact.textContent = violation.impact.toUpperCase();
      
      const id = document.createElement('span');
      id.style.marginLeft = '10px';
      id.textContent = violation.description;
      
      title.appendChild(impact);
      title.appendChild(id);
      
      const count = document.createElement('span');
      count.className = 'violation-count';
      count.textContent = \`\${violation.count} 处\`;
      
      header.appendChild(title);
      header.appendChild(count);
      
      const description = document.createElement('p');
      description.innerHTML = \`\${violation.help} <a href="\${violation.helpUrl}" target="_blank">了解更多</a>\`;
      
      container.appendChild(header);
      container.appendChild(description);
      
      commonViolationsContainer.appendChild(container);
    });
  </script>`
);

// 写入最终报告
fs.writeFileSync(OUTPUT_REPORT, templateHtml);

console.log(`报告已生成: ${OUTPUT_REPORT}`);

// 输出摘要信息
console.log('\n无障碍测试摘要:');
console.log(`总体合规评分: ${summaryData.complianceScore.toFixed(2)}/100 (${summaryData.status})`);
console.log(`测试页面数: ${summaryData.pagesTestedCount}`);
console.log(`总违规数: ${summaryData.totalViolations}`);
console.log(
  `严重违规: ${summaryData.criticalViolations}, 重要违规: ${summaryData.seriousViolations}`
);

// 如果合规分数低于期望，给出警告
if (summaryData.complianceScore < 90) {
  console.warn('\n⚠️  警告: 合规评分低于90分。请查看报告详情并解决关键问题。');
}

// 提示打开报告
console.log('\n要查看完整报告，请在浏览器中打开以下文件:');
console.log(OUTPUT_REPORT);
