import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import MemoizedComponent from '../../components/MemoizedComponent';

// 模拟console.log以跟踪渲染
const originalConsoleLog = console.log;
jest.spyOn(console, 'log').mockImplementation((...args) => {
  // 只拦截包含"渲染"的日志
  if (typeof args[0] === 'string' && args[0].includes('渲染')) {
    // 不输出到控制台
    return;
  }
  // 其他日志照常输出
  originalConsoleLog(...args);
});

describe('MemoizedComponent组件', () => {
  afterAll(() => {
    // 测试结束后恢复console.log
    console.log = originalConsoleLog;
  });
  
  // 基本渲染测试
  test('正确渲染组件内容', () => {
    render(
      <MemoizedComponent 
        title="测试标题" 
        description="测试描述" 
        important={true} 
      />
    );
    
    expect(screen.getByText('测试标题')).toBeInTheDocument();
    expect(screen.getByText('测试描述')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveClass('bg-yellow-100');
  });
  
  // 测试属性变化
  test('根据important属性更改样式', () => {
    const { rerender } = render(
      <MemoizedComponent 
        title="标题" 
        description="描述" 
        important={true} 
      />
    );
    
    // 重要消息应该有黄色背景
    const alertElement = screen.getByRole('alert');
    expect(alertElement).toHaveClass('bg-yellow-100');
    expect(alertElement).toHaveClass('border-yellow-400');
    
    // 重新渲染，不重要的消息
    rerender(
      <MemoizedComponent 
        title="标题" 
        description="描述" 
        important={false} 
      />
    );
    
    // 不重要消息应该有灰色背景
    expect(alertElement).toHaveClass('bg-gray-100');
    expect(alertElement).toHaveClass('border-gray-300');
  });
  
  // 测试记忆化功能
  test('相同属性不会导致重新渲染', () => {
    // 清除之前的模拟调用
    jest.clearAllMocks();
    
    const { rerender } = render(
      <MemoizedComponent 
        title="记忆化测试" 
        description="这是一个记忆化组件测试" 
        important={true} 
      />
    );
    
    // 第一次渲染会调用console.log
    const firstRenderCalls = console.log.mock.calls.length;
    
    // 使用相同的props重新渲染
    rerender(
      <MemoizedComponent 
        title="记忆化测试" 
        description="这是一个记忆化组件测试" 
        important={true} 
      />
    );
    
    // 由于属性相同，不应该有新的渲染日志
    expect(console.log.mock.calls.length).toBe(firstRenderCalls);
  });
  
  test('不同属性会导致重新渲染', () => {
    // 清除之前的模拟调用
    jest.clearAllMocks();
    
    const { rerender } = render(
      <MemoizedComponent 
        title="原始标题" 
        description="原始描述" 
        important={true} 
      />
    );
    
    // 第一次渲染会调用console.log
    const firstRenderCalls = console.log.mock.calls.length;
    
    // 使用不同的props重新渲染
    rerender(
      <MemoizedComponent 
        title="新标题" 
        description="原始描述" 
        important={true} 
      />
    );
    
    // 由于title变化，应该有新的渲染日志
    expect(console.log.mock.calls.length).toBeGreaterThan(firstRenderCalls);
  });
  
  // 边缘情况测试
  test('处理空描述', () => {
    render(
      <MemoizedComponent 
        title="只有标题" 
        description="" 
        important={false} 
      />
    );
    
    expect(screen.getByText('只有标题')).toBeInTheDocument();
    // 描述为空时不应该渲染描述区域
    expect(screen.queryByTestId('component-description')).not.toBeInTheDocument();
  });
  
  test('处理长文本', () => {
    const longTitle = '这是一个非常长的标题'.repeat(10);
    const longDescription = '这是一个非常长的描述文本，用于测试组件处理长文本的能力。'.repeat(10);
    
    render(
      <MemoizedComponent 
        title={longTitle} 
        description={longDescription} 
        important={true} 
      />
    );
    
    expect(screen.getByText(longTitle)).toBeInTheDocument();
    expect(screen.getByText(longDescription)).toBeInTheDocument();
  });
  
  // 无障碍测试
  test('具有适当的无障碍属性', () => {
    render(
      <MemoizedComponent 
        title="无障碍测试" 
        description="测试ARIA属性" 
        important={true} 
      />
    );
    
    const alertElement = screen.getByRole('alert');
    
    // 重要消息应该有正确的aria属性
    expect(alertElement).toHaveAttribute('aria-live', 'polite');
    
    // 标题应该有正确的角色
    expect(screen.getByText('无障碍测试')).toHaveClass('font-bold');
  });
  
  // 自定义样式测试
  test('应用自定义className', () => {
    render(
      <MemoizedComponent 
        title="样式测试" 
        description="测试自定义样式" 
        important={true}
        className="custom-test-class" 
      />
    );
    
    const alertElement = screen.getByRole('alert');
    expect(alertElement).toHaveClass('custom-test-class');
  });
}); 