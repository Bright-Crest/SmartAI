// 前端性能测试文件
// 使用Cypress测试页面和组件的性能

/**
 * 性能测试辅助函数
 */
// 测量组件加载时间
const measureLoadTime = (selector, name) => {
  const startTime = performance.now();
  cy.get(selector, { timeout: 10000 })
    .should('be.visible')
    .then(() => {
      const endTime = performance.now();
      const loadTime = endTime - startTime;
      cy.task('log', `${name} 加载时间: ${loadTime.toFixed(2)}ms`);

      // 记录性能指标
      cy.window().then(win => {
        // 如果性能数据对象不存在，则创建
        win.appPerfMetrics = win.appPerfMetrics || {};
        win.appPerfMetrics[name] = loadTime;
      });

      // 验证加载时间是否低于阈值
      expect(loadTime).to.be.lessThan(3000); // 3秒阈值，可根据需要调整
    });
};

// 测量页面渲染性能
const measurePagePerformance = (url, name) => {
  cy.visit(url);
  cy.window().then(win => {
    // 等待页面完全加载
    cy.wrap(null).then(() => {
      // 获取性能时间
      const perfData = win.performance.timing;
      const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
      const domLoadTime = perfData.domComplete - perfData.domLoading;
      const networkLatency = perfData.responseEnd - perfData.requestStart;

      // 记录性能数据
      cy.task('log', `--- ${name} 性能指标 ---`);
      cy.task('log', `总加载时间: ${pageLoadTime}ms`);
      cy.task('log', `DOM加载时间: ${domLoadTime}ms`);
      cy.task('log', `网络延迟: ${networkLatency}ms`);

      // 验证性能指标
      expect(pageLoadTime).to.be.lessThan(5000);
      expect(domLoadTime).to.be.lessThan(3000);
    });
  });
};

/**
 * 页面性能测试
 */
describe('页面加载性能测试', () => {
  // 模拟慢速网络连接进行测试
  beforeEach(() => {
    // 模拟API响应，避免网络延迟干扰测试
    cy.intercept('GET', '/api/stats', { fixture: 'stats.json' }).as('getStats');
    cy.intercept('GET', '/api/risk', { fixture: 'risk.json' }).as('getRisk');
    cy.intercept('GET', '/api/alerts', { fixture: 'alerts.json' }).as('getAlerts');
    cy.intercept('GET', '/api/transactions', { fixture: 'transactions.json' }).as(
      'getTransactions'
    );
  });

  it('首页加载性能', () => {
    measurePagePerformance('/', '首页');
  });

  it('仪表盘页面加载性能', () => {
    measurePagePerformance('/dashboard', '仪表盘');
  });

  it('交易页面加载性能', () => {
    measurePagePerformance('/transactions', '交易列表');
  });

  it('分析页面加载性能', () => {
    measurePagePerformance('/analytics', '分析页面');
  });
});

/**
 * 组件性能测试
 */
describe('组件渲染性能测试', () => {
  beforeEach(() => {
    cy.visit('/performance-test');
  });

  it('测试TransactionList组件渲染性能', () => {
    // 大数据集渲染性能测试
    cy.window().then(win => {
      // 调用预设的性能测试函数
      win.renderTestComponent('TransactionList', 1000); // 渲染1000条数据

      // 测量渲染时间
      measureLoadTime(
        '[data-testid="transaction-list"] [data-testid="list-item"]',
        'TransactionList (1000项)'
      );

      // 测试滚动性能
      cy.get('[data-testid="transaction-list"]')
        .scrollTo('bottom', { duration: 1000 })
        .then(() => {
          const scrollTime = win.lastScrollTime;
          cy.task('log', `滚动性能: ${scrollTime}ms`);
          expect(scrollTime).to.be.lessThan(500); // 滚动应该小于500ms
        });
    });
  });

  it('测试GlobalSearch组件性能', () => {
    cy.window().then(win => {
      win.renderTestComponent('GlobalSearch');

      // 测试搜索输入响应性能
      const startTime = performance.now();
      cy.get('[data-testid="search-input"]').type('test query', { delay: 0 });

      // 等待搜索结果显示
      cy.get('[data-testid="search-results"]', { timeout: 5000 })
        .should('be.visible')
        .then(() => {
          const endTime = performance.now();
          const responseTime = endTime - startTime;
          cy.task('log', `搜索响应时间: ${responseTime.toFixed(2)}ms`);
          expect(responseTime).to.be.lessThan(300); // 应该小于300ms
        });
    });
  });

  it('测试RiskScoreCard组件渲染性能', () => {
    cy.window().then(win => {
      win.renderTestComponent('RiskScoreCard');
      measureLoadTime('[data-testid="risk-score-card"]', 'RiskScoreCard');
    });
  });

  it('测试图表组件渲染性能', () => {
    cy.window().then(win => {
      win.renderTestComponent('Charts');
      measureLoadTime('[data-testid="chart-container"] svg', 'Chart组件');
    });
  });
});

/**
 * 交互性能测试
 */
describe('用户交互性能测试', () => {
  beforeEach(() => {
    cy.visit('/dashboard');
  });

  it('测试筛选操作性能', () => {
    cy.get('[data-testid="filter-button"]').click();

    const startTime = performance.now();

    // 选择筛选器选项
    cy.get('[data-testid="filter-option"]').first().click();

    // 验证列表已更新
    cy.get('[data-testid="transaction-list"]')
      .should('be.visible')
      .then(() => {
        const endTime = performance.now();
        const filterTime = endTime - startTime;
        cy.task('log', `筛选操作响应时间: ${filterTime.toFixed(2)}ms`);
        expect(filterTime).to.be.lessThan(500); // 应该小于500ms
      });
  });

  it('测试表单提交性能', () => {
    cy.visit('/feedback');

    // 填写表单
    cy.get('input[name="name"]').type('测试用户');
    cy.get('input[name="email"]').type('test@example.com');
    cy.get('textarea[name="feedback"]').type('这是一条测试反馈信息');

    const startTime = performance.now();

    // 提交表单
    cy.get('button[type="submit"]').click();

    // 等待响应
    cy.get('[data-testid="success-message"]', { timeout: 5000 })
      .should('be.visible')
      .then(() => {
        const endTime = performance.now();
        const submitTime = endTime - startTime;
        cy.task('log', `表单提交响应时间: ${submitTime.toFixed(2)}ms`);
        expect(submitTime).to.be.lessThan(2000); // 应该小于2秒
      });
  });
});

/**
 * 资源加载性能测试
 */
describe('资源加载性能', () => {
  it('分析资源加载性能', () => {
    cy.visit('/', {
      onBeforeLoad(win) {
        cy.spy(win.performance, 'getEntriesByType').as('getEntries');
      },
    });

    cy.get('@getEntries').then(getEntries => {
      const resourceEntries = getEntries('resource');

      // 分析资源加载情况
      let totalSize = 0;
      let slowResources = [];

      resourceEntries.forEach(entry => {
        totalSize += entry.transferSize || 0;

        if (entry.duration > 1000) {
          slowResources.push({
            name: entry.name,
            duration: entry.duration,
            size: entry.transferSize,
          });
        }
      });

      // 记录资源指标
      cy.task('log', `资源总大小: ${(totalSize / 1024).toFixed(2)}KB`);
      cy.task('log', `慢资源数量: ${slowResources.length}`);

      if (slowResources.length > 0) {
        cy.task('log', '慢资源列表:');
        slowResources.forEach(res => {
          cy.task(
            'log',
            `- ${res.name}: ${res.duration.toFixed(2)}ms (${(res.size / 1024).toFixed(2)}KB)`
          );
        });
      }

      // 验证资源性能
      expect(totalSize).to.be.lessThan(5 * 1024 * 1024); // 总大小应小于5MB
      expect(slowResources.length).to.be.lessThan(5); // 慢资源应少于5个
    });
  });
});

// 在每次测试运行后记录收集的性能数据
after(() => {
  cy.window().then(win => {
    if (win.appPerfMetrics) {
      cy.task('log', '性能测试摘要:');
      Object.entries(win.appPerfMetrics).forEach(([name, time]) => {
        cy.task('log', `${name}: ${time.toFixed(2)}ms`);
      });
    }
  });
});
