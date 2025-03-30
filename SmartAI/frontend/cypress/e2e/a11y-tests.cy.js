// 自动化无障碍测试文件

// 主页无障碍测试
describe('主页无障碍测试', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.injectAxe();
  });

  it('应符合WCAG 2.1 AA标准', () => {
    cy.checkA11y();
  });

  it('测试页面键盘导航', () => {
    cy.checkCompleteKeyboardNavigation(['nav a:first-child', 'main a', 'main button', 'footer a']);
  });

  it('图片应有替代文本', () => {
    cy.checkImagesHaveAlt();
  });

  it('页面应有正确的标题层次结构', () => {
    cy.get('h1').should('exist');
    cy.get('h1 + h2, h1 + div h2, h1 + section h2').should('exist');
  });
});

// 交易页面无障碍测试
describe('交易页面无障碍测试', () => {
  beforeEach(() => {
    cy.visit('/transactions');
    cy.injectAxe();
  });

  it('应符合WCAG 2.1 AA标准', () => {
    cy.checkA11y();
  });

  it('表格应有正确的ARIA角色', () => {
    cy.get('[role="table"]').should('exist');
    cy.get('[role="columnheader"]').should('exist');
    cy.get('[role="rowgroup"]').should('exist');
  });

  it('分页控件应可通过键盘访问', () => {
    cy.get('[role="navigation"][aria-label*="分页"]').should('exist');
    cy.checkKeyboardNavigation('[role="navigation"][aria-label*="分页"] button:first-child');
  });

  it('搜索功能应可访问', () => {
    cy.get('[role="search"]').should('exist');
    cy.checkFormLabels();
  });
});

// 地址详情页面无障碍测试
describe('地址详情页面无障碍测试', () => {
  beforeEach(() => {
    // 模拟测试地址
    cy.visit('/address/0x1234567890abcdef1234567890abcdef12345678');
    cy.injectAxe();
  });

  it('应符合WCAG 2.1 AA标准', () => {
    cy.checkA11y();
  });

  it('风险评分信息应可访问', () => {
    cy.get('[role="region"][aria-label*="风险"]').should('exist');
  });

  it('数据可视化应有替代文本描述', () => {
    cy.get('svg').should('have.attr', 'aria-label');
  });
});

// 表单页面无障碍测试
describe('表单页面无障碍测试', () => {
  beforeEach(() => {
    cy.visit('/feedback');
    cy.injectAxe();
  });

  it('应符合WCAG 2.1 AA标准', () => {
    cy.checkA11y();
  });

  it('表单字段应有关联标签', () => {
    cy.checkFormLabels();
  });

  it('必填字段应正确标识', () => {
    cy.get('[aria-required="true"]').should('exist');
  });

  it('表单错误应正确通知', () => {
    // 尝试提交空表单
    cy.get('form').submit();
    // 验证错误消息
    cy.get('[role="alert"]').should('exist');
  });
});

// 响应式设计无障碍测试
describe('响应式设计无障碍测试', () => {
  it('主页在各种设备尺寸上应保持可访问性', () => {
    cy.visit('/');
    cy.injectAxe();
    cy.checkResponsiveA11y();
  });
});

// 颜色对比度测试
describe('颜色对比度测试', () => {
  it('风险页面应有足够的颜色对比度', () => {
    cy.visit('/risk-alerts');
    cy.injectAxe();
    cy.checkColorContrast();
  });
});

// 自动运行 - 在每个测试完成后
afterEach(() => {
  // 将违规记录到结果目录
  if (Cypress.env('autoRunA11y')) {
    cy.task('log', '测试完成，生成报告');
  }
});
