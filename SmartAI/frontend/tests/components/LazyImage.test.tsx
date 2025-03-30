import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import LazyImage from '../../components/LazyImage';

// 模拟IntersectionObserver
const mockIntersectionObserver = jest.fn();
mockIntersectionObserver.mockImplementation(callback => {
  return {
    observe: jest.fn().mockImplementation(() => {
      // 立即调用回调，表示元素已经进入视口
      callback([{ isIntersecting: true }]);
    }),
    unobserve: jest.fn(),
    disconnect: jest.fn(),
  };
});

// 保存原始的Image实现
const originalImage = global.Image;

describe('LazyImage组件', () => {
  beforeAll(() => {
    // 替换全局IntersectionObserver
    global.IntersectionObserver = mockIntersectionObserver;
  });

  afterAll(() => {
    // 还原全局IntersectionObserver
    global.IntersectionObserver = window.IntersectionObserver;
    // 还原全局Image
    global.Image = originalImage;
  });

  beforeEach(() => {
    // 重置模拟函数
    jest.clearAllMocks();
  });

  // 基本渲染测试
  test('初始状态下显示占位符', () => {
    render(
      <LazyImage 
        src="https://example.com/image.jpg" 
        alt="测试图片"
        width={200}
        height={150}
      />
    );
    
    // 检查占位符是否存在
    const placeholder = screen.getByTestId('image-placeholder');
    expect(placeholder).toBeInTheDocument();
    
    // 图片应该存在但处于隐藏状态
    const image = screen.getByAltText('测试图片');
    expect(image).toHaveStyle({ opacity: '0' });
  });

  // 加载测试
  test('图片加载完成后应隐藏占位符并显示图片', async () => {
    // 模拟Image对象
    const mockImage = {
      onload: null,
      onerror: null,
      src: '',
    };
    
    // 替换全局Image实现
    global.Image = jest.fn().mockImplementation(() => mockImage);
    
    const { container } = render(
      <LazyImage 
        src="https://example.com/image.jpg" 
        alt="测试图片加载"
        width={200}
        height={150}
      />
    );
    
    // 触发加载完成事件
    mockImage.onload?.();
    
    // 等待动画完成
    await waitFor(() => {
      const image = screen.getByAltText('测试图片加载');
      expect(image).toHaveStyle({ opacity: '1' });
    });
    
    // 占位符应该隐藏
    const placeholder = screen.queryByTestId('image-placeholder');
    expect(placeholder).not.toBeInTheDocument();
  });

  // 错误处理测试
  test('图片加载失败时应显示错误状态', async () => {
    // 模拟Image对象
    const mockImage = {
      onload: null,
      onerror: null,
      src: '',
    };
    
    // 替换全局Image实现
    global.Image = jest.fn().mockImplementation(() => mockImage);
    
    render(
      <LazyImage 
        src="https://example.com/invalid-image.jpg" 
        alt="错误图片"
        width={200}
        height={150}
        fallback={<div data-testid="error-fallback">加载失败</div>}
      />
    );
    
    // 触发错误事件
    mockImage.onerror?.();
    
    // 验证错误回退组件是否显示
    await waitFor(() => {
      const errorFallback = screen.getByTestId('error-fallback');
      expect(errorFallback).toBeInTheDocument();
    });
    
    // 占位符和图片应该不可见
    const placeholder = screen.queryByTestId('image-placeholder');
    expect(placeholder).not.toBeInTheDocument();
    
    const image = screen.queryByAltText('错误图片');
    expect(image).not.toBeVisible();
  });

  // 默认错误回退测试
  test('当未提供fallback时显示默认错误图标', async () => {
    // 模拟Image对象
    const mockImage = {
      onload: null,
      onerror: null,
      src: '',
    };
    
    // 替换全局Image实现
    global.Image = jest.fn().mockImplementation(() => mockImage);
    
    render(
      <LazyImage 
        src="https://example.com/invalid-image.jpg" 
        alt="无fallback错误图片"
        width={200}
        height={150}
      />
    );
    
    // 触发错误事件
    mockImage.onerror?.();
    
    // 验证默认错误图标是否显示
    await waitFor(() => {
      const defaultErrorIcon = screen.getByRole('img', { name: /图片加载失败/ });
      expect(defaultErrorIcon).toBeInTheDocument();
    });
  });

  // 自定义样式测试
  test('应用自定义className', () => {
    render(
      <LazyImage 
        src="https://example.com/image.jpg" 
        alt="自定义样式图片"
        width={200}
        height={150}
        className="custom-image-class"
      />
    );
    
    const imageContainer = screen.getByTestId('lazy-image-container');
    expect(imageContainer).toHaveClass('custom-image-class');
  });

  // 无障碍支持测试
  test('具有适当的无障碍属性', () => {
    render(
      <LazyImage 
        src="https://example.com/image.jpg" 
        alt="无障碍测试图片"
        width={200}
        height={150}
        aria-label="装饰性图片"
      />
    );
    
    const image = screen.getByAltText('无障碍测试图片');
    expect(image).toHaveAttribute('aria-label', '装饰性图片');
  });

  // 动画效果测试
  test('应用渐入动画效果', async () => {
    // 模拟Image对象
    const mockImage = {
      onload: null,
      onerror: null,
      src: '',
    };
    
    // 替换全局Image实现
    global.Image = jest.fn().mockImplementation(() => mockImage);
    
    render(
      <LazyImage 
        src="https://example.com/image.jpg" 
        alt="动画测试图片"
        width={200}
        height={150}
      />
    );
    
    // 初始状态图片应该是透明的
    const image = screen.getByAltText('动画测试图片');
    expect(image).toHaveStyle({ opacity: '0' });
    
    // 触发加载完成事件
    mockImage.onload?.();
    
    // 等待动画完成
    await waitFor(() => {
      expect(image).toHaveStyle({ opacity: '1' });
      expect(image).toHaveStyle({ transition: 'opacity 0.3s ease-in-out' });
    });
  });
}); 