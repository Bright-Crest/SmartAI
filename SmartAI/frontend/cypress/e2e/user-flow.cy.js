// 用户流程端到端测试
// 测试用户在应用中的完整交互流程

describe('用户完整流程测试', () => {
  beforeEach(() => {
    // 模拟API响应
    cy.intercept('GET', '/api/stats', { fixture: 'stats.json' }).as('getStats');
    cy.intercept('GET', '/api/risk', { fixture: 'risk.json' }).as('getRisk');
    cy.intercept('GET', '/api/alerts', { fixture: 'alerts.json' }).as('getAlerts');
    cy.intercept('GET', '/api/addresses', { fixture: 'addresses.json' }).as('getAddresses');
    cy.intercept('GET', '/api/transactions', { fixture: 'transactions.json' }).as(
      'getTransactions'
    );
    cy.intercept('GET', '/api/trends', { fixture: 'trends.json' }).as('getTrends');
    cy.intercept('POST', '/api/address/analyze', { fixture: 'analysis.json' }).as('analyzeAddress');
    cy.intercept('POST', '/api/feedback', { fixture: 'feedback-success.json' }).as(
      'submitFeedback'
    );

    // 访问首页
    cy.visit('/');
  });

  it('用户登录与风险检测流程', () => {
    // 1. 检查首页正确加载
    cy.get('h1').should('be.visible');
    cy.get('nav').should('exist');

    // 2. 登录流程
    cy.contains('登录').click();
    cy.url().should('include', '/login');
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();

    // 3. 验证登录成功并跳转到仪表盘
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="user-info"]').should('contain', 'test@example.com');

    // 4. 查看风险提醒
    cy.get('[data-testid="risk-alerts"]').within(() => {
      cy.get('[data-testid="alert-item"]').first().click();
    });

    // 5. 验证风险详情页面
    cy.url().should('include', '/alerts/');
    cy.get('h1').should('contain', '风险详情');
    cy.get('[data-testid="risk-score"]').should('be.visible');

    // 6. 返回仪表盘
    cy.contains('返回').click();
    cy.url().should('include', '/dashboard');

    // 7. 测试搜索功能
    const testAddress = '0x1234567890abcdef1234567890abcdef12345678';
    cy.get('[data-testid="global-search"]').within(() => {
      cy.get('input').type(testAddress);
      cy.get('button').click();
    });

    // 8. 验证地址分析页面
    cy.url().should('include', `/address/${testAddress}`);
    cy.wait('@analyzeAddress');
    cy.get('h1').should('contain', '地址分析');
    cy.get('[data-testid="address-details"]').should('contain', testAddress);

    // 9. 查看交易历史
    cy.contains('交易历史').click();
    cy.get('[data-testid="transaction-list"]').should('be.visible');
    cy.get('[data-testid="transaction-item"]').should('have.length.at.least', 1);

    // 10. 测试筛选功能
    cy.get('[data-testid="filter-dropdown"]').click();
    cy.get('[data-testid="filter-option"]').contains('高风险').click();
    cy.get('[data-testid="transaction-item"]').should('have.length.at.least', 1);
    cy.get('[data-testid="risk-badge"]').should('contain', '高');

    // 11. 查看单个交易详情
    cy.get('[data-testid="transaction-item"]').first().click();
    cy.url().should('include', '/transaction/');
    cy.get('[data-testid="transaction-details"]').should('be.visible');

    // 12. 返回地址分析页面
    cy.go('back');

    // 13. 查看关联地址
    cy.contains('关联地址').click();
    cy.get('[data-testid="related-addresses"]').should('be.visible');

    // 14. 导出报告
    cy.contains('导出报告').click();
    cy.get('[data-testid="export-options"]').should('be.visible');
    cy.get('[data-testid="export-pdf"]').click();

    // 15. 提交反馈
    cy.visit('/feedback');
    cy.get('input[name="name"]').type('测试用户');
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('select[name="category"]').select('功能建议');
    cy.get('textarea[name="message"]').type('这是一条测试反馈信息');
    cy.get('button[type="submit"]').click();
    cy.wait('@submitFeedback');
    cy.get('[data-testid="success-message"]').should('be.visible');

    // 16. 登出
    cy.get('[data-testid="user-menu"]').click();
    cy.contains('退出登录').click();
    cy.url().should('eq', Cypress.config().baseUrl + '/');
    cy.get('nav').contains('登录').should('be.visible');
  });

  it('交易监控流程', () => {
    // 1. 登录
    cy.contains('登录').click();
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();

    // 2. 访问交易监控页面
    cy.contains('交易监控').click();
    cy.url().should('include', '/transactions');

    // 3. 验证交易列表
    cy.get('[data-testid="transaction-list"]').should('be.visible');
    cy.get('[data-testid="transaction-item"]').should('have.length.at.least', 5);

    // 4. 测试排序功能
    cy.get('[data-testid="sort-by"]').select('日期');
    cy.get('[data-testid="sort-direction"]').select('降序');
    cy.get('[data-testid="transaction-item"]').should('have.length.at.least', 5);

    // 5. 测试时间范围筛选
    cy.get('[data-testid="date-range-picker"]').click();
    cy.get('[data-testid="date-range-last-week"]').click();
    cy.get('[data-testid="transaction-item"]').should('have.length.at.least', 1);

    // 6. 测试金额筛选
    cy.get('[data-testid="amount-filter"]').click();
    cy.get('[data-testid="amount-min"]').type('1');
    cy.get('[data-testid="amount-max"]').type('10');
    cy.get('[data-testid="apply-filter"]').click();
    cy.get('[data-testid="transaction-item"]').should('have.length.at.least', 1);

    // 7. 测试导出功能
    cy.get('[data-testid="export-button"]').click();
    cy.get('[data-testid="export-csv"]').click();

    // 8. 测试实时更新
    cy.intercept('GET', '/api/transactions/live', {
      body: {
        newTransaction: {
          id: 'tx-new-1',
          from: '0xabcdef1234567890abcdef1234567890abcdef12',
          to: '0x1234567890abcdef1234567890abcdef12345678',
          amount: '5.4321',
          timestamp: new Date().toISOString(),
          status: 'confirmed',
          riskScore: 85,
        },
      },
    }).as('liveTransaction');

    cy.get('[data-testid="enable-live-updates"]').click();
    cy.wait('@liveTransaction');

    // 验证新交易已添加到列表
    cy.get('[data-testid="transaction-item"]').first().should('contain', 'tx-new-1');

    // 9. 返回仪表盘
    cy.contains('仪表盘').click();
    cy.url().should('include', '/dashboard');
  });

  it('分析和报告流程', () => {
    // 1. 登录
    cy.contains('登录').click();
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('input[name="password"]').type('password123');
    cy.get('button[type="submit"]').click();

    // 2. 访问分析页面
    cy.contains('分析').click();
    cy.url().should('include', '/analytics');

    // 3. 验证分析图表
    cy.get('[data-testid="analytics-dashboard"]').should('be.visible');
    cy.get('[data-testid="trend-chart"]').should('be.visible');
    cy.get('[data-testid="risk-distribution-chart"]').should('be.visible');

    // 4. 选择自定义时间范围
    cy.get('[data-testid="date-range"]').click();
    cy.get('[data-testid="date-range-custom"]').click();
    cy.get('[data-testid="date-start"]').type('2023-01-01');
    cy.get('[data-testid="date-end"]').type('2023-12-31');
    cy.get('[data-testid="apply-date-range"]').click();

    // 5. 等待图表更新
    cy.wait('@getTrends');
    cy.get('[data-testid="trend-chart"]').should('be.visible');

    // 6. 切换图表类型
    cy.get('[data-testid="chart-type"]').select('柱状图');
    cy.get('[data-testid="bar-chart"]').should('be.visible');

    // 7. 生成报告
    cy.get('[data-testid="generate-report"]').click();
    cy.get('[data-testid="report-options"]').should('be.visible');
    cy.get('[data-testid="include-charts"]').check();
    cy.get('[data-testid="include-raw-data"]').check();
    cy.get('[data-testid="report-format"]').select('PDF');
    cy.get('[data-testid="create-report"]').click();

    // 8. 验证报告已生成
    cy.get('[data-testid="report-success"]').should('be.visible');
    cy.get('[data-testid="download-report"]').should('be.visible');

    // 9. 共享报告
    cy.get('[data-testid="share-report"]').click();
    cy.get('[data-testid="share-email"]').type('colleague@example.com');
    cy.get('[data-testid="send-report"]').click();
    cy.get('[data-testid="share-success"]').should('be.visible');

    // 10. 保存分析配置
    cy.get('[data-testid="save-analysis"]').click();
    cy.get('[data-testid="analysis-name"]').type('年度风险分析');
    cy.get('[data-testid="save-analysis-confirm"]').click();
    cy.get('[data-testid="save-success"]').should('be.visible');

    // 11. 查看保存的分析
    cy.get('[data-testid="saved-analyses"]').click();
    cy.contains('年度风险分析').should('be.visible');

    // 12. 加载保存的分析
    cy.contains('年度风险分析').click();
    cy.url().should('include', '/analytics');
    cy.get('[data-testid="date-range"]').should('contain', '2023-01-01');
  });

  it('移动设备响应式流程', () => {
    // 设置移动视图
    cy.viewport('iphone-x');

    // 1. 检查导航菜单
    cy.get('[data-testid="mobile-menu-button"]').should('be.visible');
    cy.get('[data-testid="mobile-menu-button"]').click();
    cy.get('[data-testid="mobile-menu"]').should('be.visible');

    // 2. 访问仪表盘
    cy.contains('仪表盘').click();
    cy.url().should('include', '/dashboard');

    // 3. 验证响应式布局
    cy.get('[data-testid="stats-cards"]').should('be.visible');
    cy.get('[data-testid="risk-alerts"]').should('be.visible');

    // 4. 测试移动搜索功能
    cy.get('[data-testid="mobile-search-button"]').click();
    cy.get('[data-testid="search-overlay"]').should('be.visible');
    cy.get('[data-testid="search-input"]').type('0x1234');
    cy.get('[data-testid="search-submit"]').click();

    // 5. 验证搜索结果页面
    cy.url().should('include', '/search');
    cy.get('[data-testid="search-results"]').should('be.visible');

    // 6. 点击搜索结果
    cy.get('[data-testid="search-result-item"]').first().click();
    cy.url().should('include', '/address/');

    // 7. 返回上一页
    cy.get('[data-testid="back-button"]').click();
    cy.url().should('include', '/search');

    // 8. 测试菜单折叠
    cy.get('[data-testid="mobile-menu-button"]').click();
    cy.get('[data-testid="mobile-menu"]').should('be.visible');
    cy.contains('分析').click();
    cy.url().should('include', '/analytics');

    // 9. 验证图表响应式布局
    cy.get('[data-testid="trend-chart"]').should('be.visible');
    cy.get('[data-testid="chart-container"]').should('have.css', 'width', '100%');
  });
});
