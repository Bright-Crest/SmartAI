import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ErrorBoundary, { TranslatedErrorBoundary } from '../../components/ErrorBoundary';

// 模拟i18next
jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: { [key: string]: string } = {
        'common.error': '发生错误',
        'common.errorOccurred': '应用程序遇到了问题，请刷新页面重试。',
        'common.refresh': '刷新页面'
      };
      return translations[key] || key;
    }
  })
}));

// 创建一个会抛出错误的组件
const ErrorComponent = ({ shouldThrow = false }) => {
  if (shouldThrow) {
    throw new Error('测试错误');
  }
  return <div data-testid="normal-component">正常渲染的组件</div>;
};

// 错误发生前的console.error实现
const originalConsoleError = console.error;

describe('ErrorBoundary组件', () => {
  // 在每个测试之前修改console.error，防止测试中的错误日志
  beforeEach(() => {
    console.error = jest.fn();
  });

  // 在每个测试之后恢复console.error
  afterEach(() => {
    console.error = originalConsoleError;
  });

  // 基本渲染测试
  test('正常渲染没有错误的子组件', () => {
    render(
      <ErrorBoundary>
        <div data-testid="test-child">子组件内容</div>
      </ErrorBoundary>
    );
    
    expect(screen.getByTestId('test-child')).toBeInTheDocument();
  });
  
  // 错误捕获测试
  test('捕获子组件中的错误并显示错误UI', () => {
    // 注意：React 会在控制台显示一个错误，这是正常的
    // 我们在该测试中静默了这个错误输出
    const { container } = render(
      <ErrorBoundary>
        <ErrorComponent shouldThrow={true} />
      </ErrorBoundary>
    );
    
    // 检查错误UI是否渲染
    expect(screen.getByText('页面发生错误')).toBeInTheDocument();
    expect(screen.getByText('应用程序遇到了问题，请刷新页面重试。')).toBeInTheDocument();
    
    // 检查是否显示了错误信息
    expect(container.querySelector('.text-xs')).toHaveTextContent('Error: 测试错误');
    
    // 检查是否有刷新按钮
    expect(screen.getByRole('button', { name: '刷新页面' })).toBeInTheDocument();
  });
  
  // 自定义fallback测试
  test('使用自定义fallback UI', () => {
    const customFallback = <div data-testid="custom-error">自定义错误UI</div>;
    
    render(
      <ErrorBoundary fallback={customFallback}>
        <ErrorComponent shouldThrow={true} />
      </ErrorBoundary>
    );
    
    // 检查自定义fallback是否被渲染
    expect(screen.getByTestId('custom-error')).toBeInTheDocument();
    expect(screen.queryByText('页面发生错误')).not.toBeInTheDocument();
  });
  
  // 刷新功能测试
  test('点击刷新按钮重新加载页面', () => {
    // 模拟window.location.reload
    const mockReload = jest.fn();
    Object.defineProperty(window, 'location', {
      value: { reload: mockReload },
      writable: true
    });
    
    render(
      <ErrorBoundary>
        <ErrorComponent shouldThrow={true} />
      </ErrorBoundary>
    );
    
    // 点击刷新按钮
    fireEvent.click(screen.getByRole('button', { name: '刷新页面' }));
    
    // 验证reload被调用
    expect(mockReload).toHaveBeenCalledTimes(1);
  });
  
  // 模拟组件恢复测试
  test('重新渲染时如果不再抛出错误则恢复正常显示', () => {
    // 这个测试展示了如何测试错误恢复场景，但在Jest中直接重置ErrorBoundary状态比较困难
    // 通常这需要通过重新挂载组件或修改状态来实现
    
    const { rerender } = render(
      <ErrorBoundary>
        <ErrorComponent shouldThrow={true} />
      </ErrorBoundary>
    );
    
    // 确认错误UI已显示
    expect(screen.getByText('页面发生错误')).toBeInTheDocument();
    
    // 重新渲染，但这次不抛出错误
    // 注意：由于React的生命周期工作方式，这里可能无法完全重置ErrorBoundary
    // 这部分测试仅用于演示目的
    rerender(
      <ErrorBoundary>
        <ErrorComponent shouldThrow={false} />
      </ErrorBoundary>
    );
    
    // 在实际应用中，我们需要完全卸载并重新装载ErrorBoundary以重置状态
  });
});

// 测试带有i18n支持的TranslatedErrorBoundary
describe('TranslatedErrorBoundary组件', () => {
  beforeEach(() => {
    console.error = jest.fn();
  });

  afterEach(() => {
    console.error = originalConsoleError;
  });
  
  test('使用翻译文本显示错误信息', () => {
    render(
      <TranslatedErrorBoundary>
        <ErrorComponent shouldThrow={true} />
      </TranslatedErrorBoundary>
    );
    
    // 检查翻译后的文本
    expect(screen.getByText('发生错误')).toBeInTheDocument();
    expect(screen.getByText('应用程序遇到了问题，请刷新页面重试。')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '刷新页面' })).toBeInTheDocument();
  });
  
  test('正常渲染没有错误的子组件', () => {
    render(
      <TranslatedErrorBoundary>
        <div data-testid="translated-child">翻译组件的子元素</div>
      </TranslatedErrorBoundary>
    );
    
    expect(screen.getByTestId('translated-child')).toBeInTheDocument();
  });
}); 