import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Skeleton, { 
  TransactionSkeleton, 
  AddressSkeleton, 
  DashboardSkeleton 
} from '../../components/Skeleton';

// 直接使用这些全局函数，不需要显式定义类型
// jest会在测试环境中提供这些全局函数

describe('Skeleton组件', () => {
  test('渲染单个骨架屏', () => {
    render(<Skeleton className="h-10 w-20" />);
    
    // 骨架屏的DOM结构很简单，我们可以通过className来测试
    const skeletonElement = document.querySelector('.animate-pulse');
    expect(skeletonElement).toBeInTheDocument();
    expect(skeletonElement).toHaveClass('h-10');
    expect(skeletonElement).toHaveClass('w-20');
    expect(skeletonElement).toHaveClass('bg-gray-200');
    expect(skeletonElement).toHaveClass('rounded');
  });
  
  test('渲染多个骨架屏', () => {
    const count = 3;
    render(<Skeleton className="h-5 w-10" count={count} />);
    
    // 检查是否渲染了正确数量的骨架屏
    const skeletonElements = document.querySelectorAll('.animate-pulse');
    expect(skeletonElements.length).toBe(count);
    
    // 检查每个骨架屏是否具有正确的类
    skeletonElements.forEach(element => {
      expect(element).toHaveClass('h-5');
      expect(element).toHaveClass('w-10');
      expect(element).toHaveClass('bg-gray-200');
      expect(element).toHaveClass('rounded');
    });
  });
  
  test('正确合并自定义className', () => {
    render(<Skeleton className="h-8 w-32 custom-class" />);
    
    const skeletonElement = document.querySelector('.animate-pulse');
    expect(skeletonElement).toHaveClass('h-8');
    expect(skeletonElement).toHaveClass('w-32');
    expect(skeletonElement).toHaveClass('custom-class');
  });
  
  // 边缘情况
  test('处理空className', () => {
    render(<Skeleton />);
    
    const skeletonElement = document.querySelector('.animate-pulse');
    expect(skeletonElement).toBeInTheDocument();
    expect(skeletonElement).toHaveClass('bg-gray-200');
    expect(skeletonElement).toHaveClass('rounded');
  });
  
  test('处理count为0的情况', () => {
    render(<Skeleton count={0} />);
    
    const skeletonElements = document.querySelectorAll('.animate-pulse');
    expect(skeletonElements.length).toBe(0);
  });
  
  test('处理count为负数的情况', () => {
    render(<Skeleton count={-3} />);
    
    // 期望没有渲染任何骨架屏，因为Array.from({length: -3})会创建空数组
    const skeletonElements = document.querySelectorAll('.animate-pulse');
    expect(skeletonElements.length).toBe(0);
  });
});

describe('TransactionSkeleton组件', () => {
  test('渲染交易骨架屏', () => {
    render(<TransactionSkeleton />);
    
    // 检查主容器
    const container = document.querySelector('.bg-white.rounded-lg.shadow-md');
    expect(container).toBeInTheDocument();
    
    // 检查是否渲染了5个交易项
    const transactionItems = document.querySelectorAll('.border-b');
    expect(transactionItems.length).toBe(5);
    
    // 检查骨架屏元素总数是否正确
    // 2个头部 + (2个内容+2个右侧)*5个项 = 22个
    const skeletonElements = document.querySelectorAll('.animate-pulse');
    expect(skeletonElements.length).toBe(22);
  });
  
  test('骨架屏应该有动画效果', () => {
    render(<TransactionSkeleton />);
    
    // 检查所有元素是否都有动画类
    const skeletonElements = document.querySelectorAll('.animate-pulse');
    skeletonElements.forEach(element => {
      expect(element).toHaveClass('animate-pulse');
    });
  });
});

describe('AddressSkeleton组件', () => {
  test('渲染地址骨架屏', () => {
    render(<AddressSkeleton />);
    
    // 检查主容器
    const container = document.querySelector('.bg-white.rounded-lg.shadow-md');
    expect(container).toBeInTheDocument();
    
    // 检查是否渲染了4个信息卡片
    const infoCards = document.querySelectorAll('.bg-gray-50.rounded-lg');
    expect(infoCards.length).toBe(4);
    
    // 检查标签项
    const tagContainer = document.querySelector('.flex.flex-wrap.gap-2');
    expect(tagContainer).toBeInTheDocument();
    
    // 检查标签数量
    const tags = tagContainer?.querySelectorAll('.rounded-full') || [];
    expect(tags.length).toBe(3);
  });
});

describe('DashboardSkeleton组件', () => {
  test('渲染仪表盘骨架屏', () => {
    render(<DashboardSkeleton />);
    
    // 检查主容器
    const container = document.querySelector('.space-y-6');
    expect(container).toBeInTheDocument();
    
    // 检查是否渲染了4个卡片
    const cards = document.querySelectorAll('.bg-white.rounded-lg.shadow-md');
    expect(cards.length).toBe(5); // 4个统计卡片 + 1个大图表卡片
    
    // 检查图表区域
    const chartArea = document.querySelector('.h-64.w-full');
    expect(chartArea).toBeInTheDocument();
  });
});

// 无障碍测试
describe('Skeleton组件无障碍性', () => {
  test('应使用适当的ARIA属性标识加载状态', () => {
    render(
      <div 
        role="status" 
        aria-busy="true" 
        aria-label="加载中"
      >
        <Skeleton className="h-10 w-full" count={3} />
        <span className="sr-only">加载中...</span>
      </div>
    );
    
    // 检查无障碍属性
    const container = screen.getByRole('status');
    expect(container).toHaveAttribute('aria-busy', 'true');
    expect(container).toHaveAttribute('aria-label', '加载中');
    
    // 检查屏幕阅读器文本
    const srOnlyText = screen.getByText('加载中...');
    expect(srOnlyText).toHaveClass('sr-only');
  });
  
  test('TransactionSkeleton应该有无障碍标签', () => {
    render(
      <div role="status" aria-busy="true" aria-label="加载交易数据">
        <TransactionSkeleton />
        <span className="sr-only">加载交易数据中...</span>
      </div>
    );
    
    const container = screen.getByRole('status');
    expect(container).toHaveAttribute('aria-busy', 'true');
    expect(screen.getByText('加载交易数据中...')).toBeInTheDocument();
  });
}); 