// ***********************************************
// 自定义命令和覆盖现有命令
// https://on.cypress.io/custom-commands
// ***********************************************

// 导入axe-core和cypress-axe
import 'cypress-axe';
import axe from 'axe-core';

// 模拟登录命令
Cypress.Commands.add('login', (email, password) => {
  cy.session([email, password], () => {
    cy.visit('/login');
    cy.get('input[name=email]').type(email);
    cy.get('input[name=password]').type(password);
    cy.get('button[type=submit]').click();
    cy.url().should('not.include', '/login');
  });
});

// 模拟 API 响应
Cypress.Commands.add('mockApiResponse', (route, fixture) => {
  cy.intercept('GET', route, { fixture }).as('apiRequest');
});

// 全局无障碍检查命令
Cypress.Commands.add('checkA11y', (context, options) => {
  cy.injectAxe();
  cy.checkA11y(
    context,
    options || {
      runOnly: {
        type: 'tag',
        values: ['wcag2a', 'wcag2aa'],
      },
    },
    violations => {
      // 将违规信息输出到控制台，便于调试
      cy.task('log', JSON.stringify(violations, null, 2));
      // 发送到任务以在控制台中创建表格
      if (violations.length > 0) {
        cy.task('axeReport', violations);
      }
    }
  );
});

// 自动执行无障碍测试 - 在每个测试后自动运行
Cypress.Commands.add('autoCheckA11y', () => {
  if (Cypress.env('autoRunA11y')) {
    cy.checkA11y();
  }
});

// 检查颜色对比度
Cypress.Commands.add('checkColorContrast', context => {
  cy.injectAxe();
  cy.checkA11y(context, {
    runOnly: {
      type: 'tag',
      values: ['color-contrast'],
    },
  });
});

// 响应式设计测试
Cypress.Commands.add('checkResponsiveA11y', context => {
  // 移动设备视图
  cy.viewport('iphone-6');
  cy.checkA11y(context);

  // 平板设备视图
  cy.viewport('ipad-2');
  cy.checkA11y(context);

  // 桌面视图
  cy.viewport('macbook-15');
  cy.checkA11y(context);
});

// 等待加载状态消失
Cypress.Commands.add('waitForLoadingToDisappear', () => {
  cy.get('[data-testid="loading"]').should('not.exist');
});

// 完整的键盘导航测试，按元素顺序测试Tab键导航
Cypress.Commands.add('checkCompleteKeyboardNavigation', selectors => {
  // 聚焦第一个元素
  cy.get('body').focus();
  cy.realPress('Tab');

  // 遍历所有预期可聚焦的元素
  selectors.forEach((selector, index) => {
    // 验证当前聚焦元素是否匹配预期选择器
    cy.focused().then($el => {
      const matches = $el.is(selector);
      if (!matches) {
        // 如果不匹配，找到预期元素并输出调试信息
        cy.get(selector).then($expected => {
          cy.log(`键盘导航失败：预期聚焦在 "${selector}"，但实际聚焦在 "${$el.prop('outerHTML')}"`);
          cy.log(`预期元素：${$expected.prop('outerHTML')}`);
        });
      }
      // 使用断言验证聚焦元素
      expect(matches, `元素 #${index + 1} (${selector}) 应该可聚焦`).to.be.true;
    });

    // 按Tab移动到下一个元素
    cy.realPress('Tab');
  });
});

// 检查组件是否可以用键盘操作
Cypress.Commands.add('checkKeyboardNavigation', selector => {
  cy.get(selector).first().focus().should('have.focus');
  cy.tab().should('have.focus');
});

// 检查组件是否有正确的ARIA标签
Cypress.Commands.add('checkAriaLabels', (selector, expectedLabel) => {
  cy.get(selector).should('have.attr', 'aria-label', expectedLabel);
});

// 检查元素是否有适当的角色
Cypress.Commands.add('checkAriaRole', (selector, expectedRole) => {
  cy.get(selector).should('have.attr', 'role', expectedRole);
});

// 检查图片是否有alt文本
Cypress.Commands.add('checkImagesHaveAlt', () => {
  cy.get('img').each($img => {
    cy.wrap($img).should('have.attr', 'alt');
  });
});

// 检查表单字段是否有关联的标签
Cypress.Commands.add('checkFormLabels', () => {
  cy.get('input, select, textarea').each($field => {
    const id = $field.attr('id');
    if (id) {
      cy.get(`label[for="${id}"]`).should('exist');
    }
  });
});

// 检查ARIA属性
Cypress.Commands.add('checkAriaAttributes', (selector, attributes) => {
  cy.get(selector).then($el => {
    Object.entries(attributes).forEach(([attr, value]) => {
      expect($el).to.have.attr(attr, value);
    });
  });
});

// 测试表单验证
Cypress.Commands.add('testFormValidation', (formSelector, fieldsToTest) => {
  cy.get(formSelector).within(() => {
    // 尝试提交空表单
    cy.get('button[type="submit"]').click();

    // 检查每个字段的验证
    fieldsToTest.forEach(field => {
      // 检查错误消息
      if (field.required) {
        cy.get(`[name="${field.name}"]`).invoke('prop', 'validationMessage').should('not.be.empty');
      }

      // 测试无效输入
      if (field.invalidValue) {
        cy.get(`[name="${field.name}"]`).clear().type(field.invalidValue);
        cy.get('button[type="submit"]').click();
        cy.get(`[name="${field.name}"]`)
          .invoke('prop', 'validity')
          .should('not.have.property', 'valid', true);
      }

      // 测试有效输入
      if (field.validValue) {
        cy.get(`[name="${field.name}"]`).clear().type(field.validValue);
        cy.get(`[name="${field.name}"]`)
          .invoke('prop', 'validity')
          .should('have.property', 'valid', true);
      }
    });
  });
});

// 测试动画完成
Cypress.Commands.add('waitForAnimation', selector => {
  cy.get(selector).then($el => {
    return new Cypress.Promise(resolve => {
      const observer = new MutationObserver(() => {
        if (!$el.is(':animated')) {
          observer.disconnect();
          resolve();
        }
      });

      observer.observe($el[0], {
        attributes: true,
        attributeFilter: ['style', 'class'],
      });

      // 设置超时，防止无限等待
      setTimeout(() => {
        observer.disconnect();
        resolve();
      }, 5000);
    });
  });
});

// 测试响应式布局
Cypress.Commands.add('checkResponsiveLayout', (selector, breakpoints) => {
  const element = cy.get(selector);

  breakpoints.forEach(breakpoint => {
    cy.viewport(breakpoint.width, breakpoint.height);
    cy.wait(300); // 等待视图调整

    if (breakpoint.visible === false) {
      element.should('not.be.visible');
    } else {
      element.should('be.visible');
    }

    if (breakpoint.css) {
      Object.entries(breakpoint.css).forEach(([prop, value]) => {
        element.should('have.css', prop, value);
      });
    }
  });
});

// 在每个测试后自动运行无障碍检查
afterEach(() => {
  if (Cypress.env('autoRunA11y')) {
    cy.autoCheckA11y();
  }
});
