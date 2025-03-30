// 添加jest-dom的扩展
import '@testing-library/jest-dom';

// 为全局提供测试类型
// @ts-ignore
global.jest = jest;
// @ts-ignore
global.describe = describe;
// @ts-ignore
global.test = test;
// @ts-ignore
global.it = it;
// @ts-ignore
global.expect = expect;
// @ts-ignore
global.beforeEach = beforeEach;
// @ts-ignore
global.afterEach = afterEach;
// @ts-ignore
global.beforeAll = beforeAll;
// @ts-ignore
global.afterAll = afterAll;
// @ts-ignore
global.jest.mock = jest.mock;
// @ts-ignore
global.jest.fn = jest.fn;
// @ts-ignore
global.jest.spyOn = jest.spyOn;

// 设置测试环境
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// 模拟 localStorage
class LocalStorageMock {
  constructor() {
    this.store = {};
  }

  clear() {
    this.store = {};
  }

  getItem(key) {
    return this.store[key] || null;
  }

  setItem(key, value) {
    this.store[key] = String(value);
  }

  removeItem(key) {
    delete this.store[key];
  }
}

global.localStorage = new LocalStorageMock();
global.sessionStorage = new LocalStorageMock();

// 模拟Next.js的路由
jest.mock('next/router', () => ({
  useRouter() {
    return {
      route: '/',
      pathname: '',
      query: {},
      asPath: '',
      push: jest.fn(),
      replace: jest.fn(),
      reload: jest.fn(),
      back: jest.fn(),
      prefetch: jest.fn(),
      beforePopState: jest.fn(),
      events: {
        on: jest.fn(),
        off: jest.fn(),
        emit: jest.fn(),
      },
      isFallback: false,
    };
  },
}));

// 模拟Next.js的navigation
jest.mock('next/navigation', () => ({
  useRouter() {
    return {
      push: jest.fn(),
      replace: jest.fn(),
      refresh: jest.fn(),
      back: jest.fn(),
      forward: jest.fn(),
    };
  },
  usePathname() {
    return '/';
  },
  useSearchParams() {
    return new URLSearchParams();
  },
}));

// 模拟Next.js的图片组件
jest.mock('next/image', () => ({
  __esModule: true,
  default: props => {
    // eslint-disable-next-line jsx-a11y/alt-text
    return <img {...props} />;
  },
}));

// 模拟framer-motion以减少动画相关问题
jest.mock('framer-motion', () => {
  const actual = jest.requireActual('framer-motion');
  return {
    ...actual,
    motion: {
      ...actual.motion,
      div: ({ children, ...props }) => <div {...props}>{children}</div>,
      span: ({ children, ...props }) => <span {...props}>{children}</span>,
      button: ({ children, ...props }) => <button {...props}>{children}</button>,
      ul: ({ children, ...props }) => <ul {...props}>{children}</ul>,
      li: ({ children, ...props }) => <li {...props}>{children}</li>,
    },
    AnimatePresence: ({ children }) => <>{children}</>,
  };
});

// 模拟i18next
jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: key => key,
    i18n: {
      changeLanguage: jest.fn(),
    },
  }),
  Trans: ({ children }) => children,
}));

// 模拟WebGLRenderingContext
class WebGLRenderingContext {
  MAX_TEXTURE_SIZE = 0x0d33;
  UNMASKED_VENDOR_WEBGL = 0x9245;
  UNMASKED_RENDERER_WEBGL = 0x9246;

  getParameter() {}
  getExtension() {}
}

// 添加到全局对象
global.WebGLRenderingContext = WebGLRenderingContext;

// 模拟IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe(element) {
    // 立即触发回调，表示元素在视口内
    this.callback([
      {
        isIntersecting: true,
        target: element,
      },
    ]);
    return null;
  }
  unobserve() {
    return null;
  }
  disconnect() {
    return null;
  }
};

// 模拟ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor(callback) {
    this.callback = callback;
  }
  observe() {
    // 立即触发回调
    this.callback([]);
    return null;
  }
  unobserve() {
    return null;
  }
  disconnect() {
    return null;
  }
};

// 扩展expect
require('@testing-library/jest-dom');

// 取消所有console.error/warn的输出，让测试输出更清晰
global.console = {
  ...console,
  error: jest.fn(),
  warn: jest.fn(),
  log: console.log,
};
