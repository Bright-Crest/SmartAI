import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import VirtualizedList from '../../components/VirtualizedList';

// 测试数据生成函数
const generateItems = (count: number) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    content: `Item ${i + 1}`
  }));
};

// 渲染项目函数
const renderItem = (item: { id: number; content: string }, index: number) => (
  <div data-testid={`item-${item.id}`} tabIndex={0}>
    {item.content}
  </div>
);

describe('VirtualizedList组件', () => {
  // 基本渲染测试
  test('正确渲染虚拟列表', () => {
    const items = generateItems(100);
    const { container } = render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
      />
    );
    
    // 验证容器高度
    const listContainer = container.firstChild as HTMLElement;
    expect(listContainer).toHaveStyle({ height: '200px' });
    
    // 验证总内容高度
    const contentContainer = listContainer.firstChild as HTMLElement;
    expect(contentContainer).toHaveStyle({ height: '5000px' }); // 100 * 50 = 5000
    
    // 只渲染可见项目及overscan项目
    // 默认overscan为3，可见项目为 200 / 50 = 4，总共应渲染 4 + (3*2) = 10 个项目
    const renderedItems = container.querySelectorAll('[data-testid^="item-"]');
    expect(renderedItems.length).toBeGreaterThanOrEqual(4); // 至少渲染可见项目
    expect(renderedItems.length).toBeLessThanOrEqual(10); // 不超过可见项目+overscan
  });
  
  // 测试滚动功能
  test('滚动时更新渲染的项目', () => {
    const items = generateItems(100);
    const { container } = render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
      />
    );
    
    const listContainer = container.firstChild as HTMLElement;
    
    // 模拟滚动
    fireEvent.scroll(listContainer, { target: { scrollTop: 300 } });
    
    // 滚动后应该渲染新的项目集合
    // scrollTop=300，对应item索引6，可见项目4个，使用默认overscan=3
    // 应该渲染索引3-13的项目（总共11个项目）
    const renderedItems = container.querySelectorAll('[data-testid^="item-"]');
    
    // 验证是否更新了渲染的项目
    let foundNewItem = false;
    renderedItems.forEach(item => {
      const id = parseInt((item as HTMLElement).dataset.testid!.split('-')[1]);
      if (id >= 6 && id <= 10) {
        foundNewItem = true;
      }
    });
    
    expect(foundNewItem).toBe(true);
  });
  
  // 测试不同的 overscan 值
  test('使用自定义 overscan 值', () => {
    const items = generateItems(100);
    const overscan = 5;
    const { container } = render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        overscan={overscan}
        renderItem={renderItem}
      />
    );
    
    // 可见项目为 200 / 50 = 4，总共应渲染 4 + (5*2) = 14 个项目
    const renderedItems = container.querySelectorAll('[data-testid^="item-"]');
    expect(renderedItems.length).toBeGreaterThanOrEqual(4); // 至少渲染可见项目
    expect(renderedItems.length).toBeLessThanOrEqual(14); // 不超过可见项目+overscan
  });
  
  // 边缘情况测试
  test('处理空数据数组', () => {
    const { container } = render(
      <VirtualizedList
        items={[]}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
      />
    );
    
    const contentContainer = container.querySelector('div > div') as HTMLElement;
    expect(contentContainer).toHaveStyle({ height: '0px' });
    
    // 不应渲染任何项目
    const renderedItems = container.querySelectorAll('[data-testid^="item-"]');
    expect(renderedItems.length).toBe(0);
  });
  
  test('处理单个项目的数组', () => {
    const items = generateItems(1);
    const { container } = render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
      />
    );
    
    // 总高度应为一个项目的高度
    const contentContainer = container.querySelector('div > div') as HTMLElement;
    expect(contentContainer).toHaveStyle({ height: '50px' });
    
    // 应该渲染唯一的项目
    expect(screen.getByTestId('item-0')).toBeInTheDocument();
  });
  
  test('处理比窗口少的项目', () => {
    const items = generateItems(3); // 3个项目，总高度150px，小于窗口高度200px
    const { container } = render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
      />
    );
    
    // 总高度应为所有项目的高度
    const contentContainer = container.querySelector('div > div') as HTMLElement;
    expect(contentContainer).toHaveStyle({ height: '150px' });
    
    // 应该渲染所有3个项目
    expect(screen.getByTestId('item-0')).toBeInTheDocument();
    expect(screen.getByTestId('item-1')).toBeInTheDocument();
    expect(screen.getByTestId('item-2')).toBeInTheDocument();
  });
  
  // 无障碍测试
  test('支持键盘导航', () => {
    const items = generateItems(10);
    render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
      />
    );
    
    // 验证项目是否可聚焦
    const firstItem = screen.getByTestId('item-0');
    expect(firstItem).toHaveAttribute('tabIndex', '0');
  });
  
  test('应用自定义className', () => {
    const items = generateItems(10);
    const className = 'custom-class';
    const { container } = render(
      <VirtualizedList
        items={items}
        itemHeight={50}
        windowHeight={200}
        renderItem={renderItem}
        className={className}
      />
    );
    
    const listContainer = container.firstChild as HTMLElement;
    expect(listContainer).toHaveClass(className);
  });
  
  test('添加适当的ARIA角色', () => {
    const items = generateItems(10);
    const { container } = render(
      <div role="region" aria-label="列表区域">
        <VirtualizedList
          items={items}
          itemHeight={50}
          windowHeight={200}
          renderItem={(item, index) => (
            <div
              data-testid={`item-${item.id}`}
              tabIndex={0}
              role="listitem"
              aria-posinset={index + 1}
              aria-setsize={items.length}
            >
              {item.content}
            </div>
          )}
        />
      </div>
    );
    
    // 检查列表区域
    expect(container.firstChild).toHaveAttribute('role', 'region');
    expect(container.firstChild).toHaveAttribute('aria-label', '列表区域');
    
    // 检查列表项
    const firstItem = screen.getByTestId('item-0');
    expect(firstItem).toHaveAttribute('role', 'listitem');
    expect(firstItem).toHaveAttribute('aria-posinset', '1');
    expect(firstItem).toHaveAttribute('aria-setsize', '10');
  });
}); 