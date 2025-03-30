import React, { useState, useEffect, useRef, useCallback } from 'react';

interface VirtualListProps<T> {
  items: T[];
  height: number;
  itemHeight: number;
  renderItem: (item: T, index: number, isVisible: boolean) => React.ReactNode;
  renderPlaceholder?: () => React.ReactNode;
  overscan?: number;
  onItemsRendered?: (startIndex: number, endIndex: number) => void;
  onScroll?: (scrollTop: number) => void;
  className?: string;
  itemKey?: (item: T, index: number) => string | number;
}

/**
 * 性能优化的虚拟列表组件
 * 只渲染可视区域内的元素，实现高效渲染大量数据
 */
function OptimizedVirtualList<T>({
  items,
  height,
  itemHeight,
  renderItem,
  renderPlaceholder = () => <div style={{ height: `${itemHeight}px` }}></div>,
  overscan = 3,
  onItemsRendered,
  onScroll,
  className = '',
  itemKey = (_, index) => index,
}: VirtualListProps<T>) {
  // 容器引用
  const containerRef = useRef<HTMLDivElement>(null);
  // 滚动位置
  const [scrollTop, setScrollTop] = useState(0);
  // 记录上一次渲染的索引范围
  const lastRenderedIndicesRef = useRef({ startIndex: 0, endIndex: 0 });
  // 存储项目测量的高度（如果实际高度与预期不同）
  const [measuredItemHeights, setMeasuredItemHeights] = useState<Record<string | number, number>>({});

  // 处理滚动事件
  const handleScroll = useCallback(() => {
    if (containerRef.current) {
      const { scrollTop } = containerRef.current;
      setScrollTop(scrollTop);
      onScroll?.(scrollTop);
    }
  }, [onScroll]);

  // 计算需要渲染的项目范围
  const getVisibleRange = useCallback(() => {
    if (!containerRef.current) {
      return { startIndex: 0, endIndex: overscan * 2 };
    }

    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const endIndex = Math.min(
      items.length - 1,
      Math.ceil((scrollTop + height) / itemHeight) + overscan
    );

    return { startIndex, endIndex };
  }, [scrollTop, height, itemHeight, items.length, overscan]);

  // 测量项目的实际高度
  const measureItem = useCallback((key: string | number, height: number) => {
    setMeasuredItemHeights(prev => {
      if (prev[key] === height) return prev;
      return { ...prev, [key]: height };
    });
  }, []);

  // 当需要渲染的范围变化时通知父组件
  useEffect(() => {
    const { startIndex, endIndex } = getVisibleRange();
    
    // 仅当范围变化时才触发回调
    if (startIndex !== lastRenderedIndicesRef.current.startIndex || 
        endIndex !== lastRenderedIndicesRef.current.endIndex) {
      lastRenderedIndicesRef.current = { startIndex, endIndex };
      onItemsRendered?.(startIndex, endIndex);
    }
  }, [getVisibleRange, onItemsRendered]);

  // 计算虚拟列表总高度
  const totalHeight = items.length * itemHeight;

  // 计算每个可见项的位置和样式
  const { startIndex, endIndex } = getVisibleRange();
  const visibleItems = [];

  // 生成可见项
  for (let i = startIndex; i <= endIndex && i < items.length; i++) {
    const item = items[i];
    const key = itemKey(item, i);
    const itemTop = i * itemHeight;
    const itemStyle = {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      transform: `translateY(${itemTop}px)`,
      // 使用测量高度或默认高度
      height: `${measuredItemHeights[key] || itemHeight}px`,
    };

    visibleItems.push(
      <div key={key} style={itemStyle as React.CSSProperties} 
           ref={el => {
             if (el) {
               const actualHeight = el.getBoundingClientRect().height;
               if (actualHeight !== measuredItemHeights[key] && actualHeight > 0) {
                 measureItem(key, actualHeight);
               }
             }
           }}>
        {renderItem(item, i, true)}
      </div>
    );
  }

  // 渲染超出可视区域的占位符（可选）
  if (startIndex > 0 && renderPlaceholder) {
    visibleItems.unshift(
      <div key="top-placeholder" style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: `${startIndex * itemHeight}px`,
      }}>
        {renderPlaceholder()}
      </div>
    );
  }

  // 容器样式
  const containerStyle: React.CSSProperties = {
    position: 'relative',
    height: `${height}px`,
    overflow: 'auto',
    willChange: 'transform',
  };

  // 内容样式
  const contentStyle: React.CSSProperties = {
    position: 'relative',
    height: `${totalHeight}px`,
    width: '100%',
  };

  return (
    <div
      ref={containerRef}
      style={containerStyle}
      className={className}
      onScroll={handleScroll}
    >
      <div style={contentStyle}>
        {visibleItems}
      </div>
    </div>
  );
}

export default OptimizedVirtualList; 