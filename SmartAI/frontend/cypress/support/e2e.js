// 导入命令
import './commands';
import '@cypress/code-coverage/support';
import 'cypress-axe';

// 在所有测试开始之前
before(() => {
  // 初始化测试环境或设置全局变量
  cy.log('测试环境已初始化');
});

// 在每个测试之前
beforeEach(() => {
  // 重置状态，确保每个测试有清洁的环境
  cy.log('开始新测试');

  // 删除所有cookie和localStorage，确保干净的测试状态
  cy.clearCookies();
  cy.clearLocalStorage();

  // 记录测试名称以便调试
  const testName = Cypress.currentTest.title;
  cy.log(`执行测试: ${testName}`);
});

// 在所有测试结束后
after(() => {
  // 清理全局状态
  cy.log('测试套件已完成');
});

// 在每个测试之后
afterEach(() => {
  // 如果设置了自动运行无障碍测试，则在每个测试后运行
  if (Cypress.env('autoRunA11y')) {
    cy.autoCheckA11y();
  }
});

// 隐藏 fetch/XHR 请求以减少日志噪音
const app = window.top;
if (!app.document.head.querySelector('[data-hide-command-log-request]')) {
  const style = app.document.createElement('style');
  style.innerHTML = '.command-name-request, .command-name-xhr { display: none }';
  style.setAttribute('data-hide-command-log-request', '');
  app.document.head.appendChild(style);
}
