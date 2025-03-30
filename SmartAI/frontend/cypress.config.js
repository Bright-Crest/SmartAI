const { defineConfig } = require('cypress');

module.exports = defineConfig({
  component: {
    devServer: {
      framework: 'next',
      bundler: 'webpack',
    },
    setupNodeEvents(on, config) {
      // 使用yarn而不是npm
      require('@cypress/code-coverage/task')(on, config);

      // 注册任务用于记录无障碍违规
      on('task', {
        log(message) {
          console.log(message);
          return null;
        },
        table(message) {
          console.table(message);
          return null;
        },
      });

      return config;
    },
    specPattern: 'cypress/component/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'cypress/support/component.js',
  },
  e2e: {
    baseUrl: 'http://localhost:3000',
    setupNodeEvents(on, config) {
      // 使用yarn而不是npm
      require('@cypress/code-coverage/task')(on, config);

      // 注册任务用于记录无障碍违规
      on('task', {
        log(message) {
          console.log(message);
          return null;
        },
        table(message) {
          console.table(message);
          return null;
        },
        // 记录无障碍违规
        axeReport(violations) {
          console.log('无障碍违规:');
          console.table(
            violations.map(({ id, impact, description, nodes }) => ({
              id,
              impact,
              description,
              nodes: nodes.length,
            }))
          );
          return null;
        },
      });

      return config;
    },
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
  },
  env: {
    codeCoverage: {
      exclude: ['cypress/**/*.*'],
    },
    // 自动运行无障碍测试
    autoRunA11y: true,
  },
  viewportWidth: 1280,
  viewportHeight: 720,
  video: false,
  experimentalMemoryManagement: true,
  numTestsKeptInMemory: 1,
  // 无障碍测试相关配置
  a11y: {
    enabled: true,
    runOnly: {
      type: 'tag',
      values: ['wcag2a', 'wcag2aa'],
    },
    // 配置要检查的规则
    rules: {
      'color-contrast': { enabled: true },
      'page-has-heading-one': { enabled: false },
      'landmark-one-main': { enabled: true },
      region: { enabled: false },
    },
    defaultCommandTimeout: 10000,
    screenshotOnFail: true,
  },
  // 设置测试重试次数
  retries: {
    runMode: 1,
    openMode: 0,
  },
  // 设置测试超时时间
  defaultCommandTimeout: 8000,
  // 确保测试可以顺利运行而不会耗尽内存
  numTestsKeptInMemory: 1,
});
