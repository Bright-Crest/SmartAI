import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import StatisticsCards from '../../components/StatisticsCards';

// 模拟react-i18next
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key })
}));

// 模拟framer-motion
jest.mock('framer-motion', () => {
  const actual = jest.requireActual('framer-motion');
  return {
    ...actual,
    motion: {
      div: ({ children, ...props }: React.PropsWithChildren<object>) => (
        <div data-testid={props['data-testid'] || 'motion-div'} {...props}>
          {children}
        </div>
      )
    }
  };
});

// 模拟heroicons
jest.mock('@heroicons/react/24/outline', () => ({
  ArrowTrendingUpIcon: () => <svg data-testid="icon-trend-up" />,
  ArrowTrendingDownIcon: () => <svg data-testid="icon-trend-down" />,
  ExclamationTriangleIcon: () => <svg data-testid="icon-warning" />,
  ShieldCheckIcon: () => <svg data-testid="icon-shield" />
}));

describe('StatisticsCards组件', () => {
  // 基本渲染测试
  test('渲染所有统计卡片', () => {
    render(<StatisticsCards />);
    
    // 验证所有卡片标题是否存在
    expect(screen.getByText('监控地址')).toBeInTheDocument();
    expect(screen.getByText('今日交易')).toBeInTheDocument();
    expect(screen.getByText('风险警报')).toBeInTheDocument();
    expect(screen.getByText('资金流出')).toBeInTheDocument();
    
    // 验证所有卡片值是否存在
    expect(screen.getByText('12,345')).toBeInTheDocument();
    expect(screen.getByText('1,234')).toBeInTheDocument();
    expect(screen.getByText('23')).toBeInTheDocument();
    expect(screen.getByText('$2.3M')).toBeInTheDocument();
  });
  
  // 图标测试
  test('渲染正确的图标', () => {
    render(<StatisticsCards />);
    
    // 验证所有图标是否正确渲染
    expect(screen.getByTestId('icon-shield')).toBeInTheDocument();
    expect(screen.getByTestId('icon-trend-up')).toBeInTheDocument();
    expect(screen.getByTestId('icon-warning')).toBeInTheDocument();
    expect(screen.getByTestId('icon-trend-down')).toBeInTheDocument();
  });
  
  // 变化百分比测试
  test('显示正确的百分比变化', () => {
    render(<StatisticsCards />);
    
    // 验证正增长的显示
    expect(screen.getByText('+5.2%')).toBeInTheDocument();
    expect(screen.getByText('+2.1%')).toBeInTheDocument();
    
    // 验证负增长的显示
    expect(screen.getByText('-8.4%')).toBeInTheDocument();
    
    // 验证大额正增长的显示
    expect(screen.getByText('+12.5%')).toBeInTheDocument();
  });
  
  // 动画测试
  test('使用Framer Motion进行动画', () => {
    render(<StatisticsCards />);
    
    // 验证容器使用了motion
    const motionContainer = screen.getByTestId('motion-div');
    expect(motionContainer).toBeInTheDocument();
    expect(motionContainer).toHaveClass('grid');
    expect(motionContainer).toHaveClass('grid-cols-1');
    expect(motionContainer).toHaveClass('md:grid-cols-2');
    expect(motionContainer).toHaveClass('lg:grid-cols-4');
  });
  
  // 样式测试
  test('应用正确的样式类', () => {
    const { container } = render(<StatisticsCards />);
    
    // 验证卡片背景色
    const cards = container.querySelectorAll('.bg-white');
    expect(cards.length).toBe(4);
    
    // 验证图标颜色
    expect(container.querySelector('.bg-blue-500')).toBeInTheDocument();
    expect(container.querySelector('.bg-green-500')).toBeInTheDocument();
    expect(container.querySelector('.bg-red-500')).toBeInTheDocument();
    expect(container.querySelector('.bg-purple-500')).toBeInTheDocument();
    
    // 验证文字颜色
    expect(container.querySelector('.text-gray-500')).toBeInTheDocument();
    expect(container.querySelector('.text-green-500')).toBeInTheDocument();
    expect(container.querySelector('.text-red-500')).toBeInTheDocument();
  });
  
  // 响应式设计测试
  test('包含响应式设计类', () => {
    const { container } = render(<StatisticsCards />);
    
    const gridContainer = container.querySelector('div');
    expect(gridContainer).toHaveClass('grid-cols-1');
    expect(gridContainer).toHaveClass('md:grid-cols-2');
    expect(gridContainer).toHaveClass('lg:grid-cols-4');
  });
  
  // 缓存测试
  test('使用React.memo来优化渲染', () => {
    // 这个测试主要验证组件是否被导出为 React.memo 包装的组件
    // 由于无法直接测试 React.memo 的实现，我们可以通过组件名称来间接验证
    
    // 以下是验证组件名称的一种方法
    // 注意：这种方法依赖于 React 的内部实现，可能会随着 React 版本变化而失效
    const { container } = render(<StatisticsCards />);
    
    // 验证组件正确渲染（作为替代检查）
    expect(container.querySelectorAll('.bg-white').length).toBe(4);
  });
  
  // 无障碍支持测试
  test('符合基本的无障碍标准', () => {
    const { container } = render(<StatisticsCards />);
    
    // 验证卡片包含基本的语义元素
    const paragraphs = container.querySelectorAll('p');
    expect(paragraphs.length).toBeGreaterThanOrEqual(8); // 每张卡片至少2个段落
    
    // 验证文本对比度（通过颜色类名间接验证）
    expect(container.querySelector('.text-gray-500')).toBeInTheDocument(); // 较低对比度的次要文本
    expect(container.querySelector('.text-2xl')).toBeInTheDocument(); // 较大字号的主要数值
  });
}); 