// 无障碍测试报告生成
// 这个测试将对整个应用进行无障碍检查并生成报告

// 定义要测试的页面路径
const pagesToTest = [
  { path: '/', name: '首页' },
  { path: '/dashboard', name: '仪表盘' },
  { path: '/transactions', name: '交易列表' },
  { path: '/analytics', name: '分析页面' },
  { path: '/feedback', name: '反馈页面' },
  { path: '/address/0x1234567890abcdef1234567890abcdef12345678', name: '地址详情' },
  { path: '/profile', name: '用户资料' },
  { path: '/settings', name: '设置页面' },
];

// 无障碍规则集
const a11yRuleSets = [
  { name: 'WCAG 2.1 A', rules: ['wcag2a'] },
  { name: 'WCAG 2.1 AA', rules: ['wcag2aa'] },
  { name: 'WCAG 2.1 AAA', rules: ['wcag2aaa'] },
];

// 要详细检查的关键组件
const keyComponents = [
  { selector: '[data-testid="transaction-list"]', name: '交易列表' },
  { selector: '[data-testid="risk-alerts"]', name: '风险提醒' },
  { selector: '[data-testid="global-search"]', name: '全局搜索' },
  { selector: '[data-testid="risk-score-card"]', name: '风险评分卡' },
  { selector: '[data-testid="chart-container"]', name: '图表组件' },
  { selector: '[data-testid="pagination"]', name: '分页控件' },
  { selector: 'form', name: '表单组件' },
  { selector: 'nav', name: '导航菜单' },
];

// 自定义报告函数
const generateA11yReport = (pageName, violations) => {
  const timestamp = new Date().toISOString().replace(/:/g, '-');
  const reportFileName = `a11y-report-${pageName.replace(/\s+/g, '-').toLowerCase()}-${timestamp}.json`;

  const report = {
    page: pageName,
    timestamp: new Date().toISOString(),
    violations: violations.map(violation => ({
      id: violation.id,
      impact: violation.impact,
      description: violation.description,
      help: violation.help,
      helpUrl: violation.helpUrl,
      nodes: violation.nodes.map(node => ({
        html: node.html,
        target: node.target,
        failureSummary: node.failureSummary,
      })),
    })),
    summary: {
      total: violations.length,
      critical: violations.filter(v => v.impact === 'critical').length,
      serious: violations.filter(v => v.impact === 'serious').length,
      moderate: violations.filter(v => v.impact === 'moderate').length,
      minor: violations.filter(v => v.impact === 'minor').length,
    },
  };

  // 保存报告到文件
  cy.writeFile(`cypress/a11y-results/${reportFileName}`, report);

  // 在控制台输出报告摘要
  cy.task('log', `🔍 无障碍测试报告 - ${pageName}`);
  cy.task('log', `总违规数: ${report.summary.total}`);
  cy.task(
    'log',
    `严重: ${report.summary.critical}, 重要: ${report.summary.serious}, 中等: ${report.summary.moderate}, 轻微: ${report.summary.minor}`
  );

  if (violations.length > 0) {
    cy.task('log', '前5个关键违规:');
    violations.slice(0, 5).forEach(violation => {
      cy.task(
        'log',
        `- ${violation.impact.toUpperCase()}: ${violation.description} (${violation.nodes.length}个元素)`
      );
    });
  }

  // 返回摘要数据用于最终报告
  return report.summary;
};

describe('自动无障碍测试报告生成', () => {
  // 确保报告目录存在
  before(() => {
    cy.task('log', '创建无障碍测试报告目录');
    cy.exec('mkdir -p cypress/a11y-results').then(result => {
      cy.task('log', '清理旧的测试报告');
      cy.exec('rm -f cypress/a11y-results/*.json');
    });
  });

  // 存储所有页面的报告摘要
  const pageSummaries = {};

  // 测试每个页面
  pagesToTest.forEach(page => {
    it(`测试 ${page.name} 的无障碍性 (WCAG 2.1 AA)`, () => {
      cy.visit(page.path);
      cy.injectAxe();

      // 等待页面加载完成
      cy.get('body', { timeout: 10000 }).should('be.visible');

      // 使用AA标准进行测试
      cy.checkA11y(
        null,
        {
          runOnly: {
            type: 'tag',
            values: ['wcag2aa'],
          },
        },
        violations => {
          pageSummaries[page.name] = generateA11yReport(page.name, violations);
        }
      );

      // 深入检查关键组件
      keyComponents.forEach(component => {
        cy.get(component.selector).then($el => {
          if ($el.length) {
            cy.task('log', `检查组件: ${component.name}`);
            cy.checkA11y(
              component.selector,
              {
                runOnly: {
                  type: 'tag',
                  values: ['wcag2aa'],
                },
              },
              violations => {
                if (violations.length > 0) {
                  cy.task('log', `组件 ${component.name} 有 ${violations.length} 个无障碍问题`);
                }
              }
            );
          }
        });
      });
    });
  });

  // 为每个规则集测试首页
  a11yRuleSets.forEach(ruleSet => {
    it(`使用 ${ruleSet.name} 标准测试首页`, () => {
      cy.visit('/');
      cy.injectAxe();

      cy.checkA11y(
        null,
        {
          runOnly: {
            type: 'tag',
            values: ruleSet.rules,
          },
        },
        violations => {
          generateA11yReport(`首页 (${ruleSet.name})`, violations);
        }
      );
    });
  });

  // 测试移动视图
  it('在移动设备视图下测试无障碍性', () => {
    cy.viewport('iphone-x');
    cy.visit('/');
    cy.injectAxe();

    cy.checkA11y(
      null,
      {
        runOnly: {
          type: 'tag',
          values: ['wcag2aa'],
        },
      },
      violations => {
        generateA11yReport('首页 (移动视图)', violations);
      }
    );

    // 测试菜单展开状态
    cy.get('[data-testid="mobile-menu-button"]').click();
    cy.get('[data-testid="mobile-menu"]').should('be.visible');

    cy.checkA11y(
      '[data-testid="mobile-menu"]',
      {
        runOnly: {
          type: 'tag',
          values: ['wcag2aa'],
        },
      },
      violations => {
        generateA11yReport('移动菜单', violations);
      }
    );
  });

  // 测试不同颜色模式
  it('在深色模式下测试无障碍性', () => {
    cy.visit('/');
    cy.injectAxe();

    // 切换到深色模式
    cy.get('[data-testid="theme-toggle"]').click();

    cy.checkA11y(
      null,
      {
        runOnly: {
          type: 'tag',
          values: ['wcag2aa'],
        },
      },
      violations => {
        generateA11yReport('首页 (深色模式)', violations);
      }
    );
  });

  // 测试键盘导航
  it('测试页面键盘导航', () => {
    cy.visit('/');

    // 焦点应该在第一个可聚焦元素上
    cy.focused().should('exist');

    // 测试Tab键导航
    cy.checkCompleteKeyboardNavigation([
      'nav a:first-child',
      '[data-testid="global-search"] input',
      '[data-testid="login-button"]',
    ]);

    // 测试表单键盘导航
    cy.visit('/feedback');

    cy.checkCompleteKeyboardNavigation([
      'input[name="name"]',
      'input[name="email"]',
      'select[name="category"]',
      'textarea[name="message"]',
      'button[type="submit"]',
    ]);
  });

  // 生成最终总结报告
  after(() => {
    cy.task('log', '生成无障碍测试总结报告');

    const summaryReport = {
      timestamp: new Date().toISOString(),
      pagesTestedCount: pagesToTest.length,
      pageSummaries: pageSummaries,
      totalViolations: Object.values(pageSummaries).reduce((sum, page) => sum + page.total, 0),
      criticalViolations: Object.values(pageSummaries).reduce(
        (sum, page) => sum + page.critical,
        0
      ),
      seriousViolations: Object.values(pageSummaries).reduce((sum, page) => sum + page.serious, 0),
    };

    // 计算总体合规评分 (100 - 加权违规数)
    const weightedViolations =
      summaryReport.criticalViolations * 5 +
      summaryReport.seriousViolations * 3 +
      Object.values(pageSummaries).reduce((sum, page) => sum + page.moderate, 0) +
      Object.values(pageSummaries).reduce((sum, page) => sum + page.minor, 0) * 0.5;

    summaryReport.complianceScore = Math.max(0, Math.min(100, 100 - weightedViolations));

    // 根据得分确定状态
    if (summaryReport.complianceScore >= 90) {
      summaryReport.status = '良好';
    } else if (summaryReport.complianceScore >= 70) {
      summaryReport.status = '需要改进';
    } else {
      summaryReport.status = '不符合标准';
    }

    // 保存总结报告
    cy.writeFile('cypress/a11y-results/summary-report.json', summaryReport);

    // 输出报告摘要
    cy.task('log', '📊 无障碍测试总结');
    cy.task(
      'log',
      `总体合规评分: ${summaryReport.complianceScore.toFixed(2)}/100 (${summaryReport.status})`
    );
    cy.task('log', `测试页面数: ${summaryReport.pagesTestedCount}`);
    cy.task('log', `总违规数: ${summaryReport.totalViolations}`);
    cy.task(
      'log',
      `严重违规: ${summaryReport.criticalViolations}, 重要违规: ${summaryReport.seriousViolations}`
    );

    // 输出每个页面的摘要
    cy.task('log', '\n各页面无障碍状况:');
    Object.entries(pageSummaries).forEach(([pageName, summary]) => {
      cy.task(
        'log',
        `- ${pageName}: ${summary.total}个问题 (严重: ${summary.critical}, 重要: ${summary.serious})`
      );
    });
  });
});
