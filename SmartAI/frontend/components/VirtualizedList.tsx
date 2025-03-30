"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { throttle, measurePerformance } from '../utils/performance';

// 列表项类型
interface ListItem {
  id: string | number;
  content: string;
  height?: number; // 添加可选的高度属性，支持动态高度
}

// 虚拟列表属性类型
interface VirtualizedListProps {
  items: ListItem[];
  itemHeight: number | ((item: ListItem, index: number) => number); // 允许动态计算高度
  windowHeight?: number;
  overscan?: number;
  renderItem: (item: ListItem, index: number) => React.ReactNode;
  className?: string;
  onScroll?: (scrollTop: number) => void;
  scrollToIndex?: number;
  onItemsRendered?: (startIndex: number, endIndex: number) => void;
  enableResizeObserver?: boolean; // 是否启用容器大小自动监测
  enablePerformanceMetrics?: boolean; // 是否启用性能指标收集
  onPerformanceReport?: (metrics: {
    fps: number;
    renderTime: number;
    visibleItems: number;
    totalItems: number;
  }) => void;
}

/**
 * 高性能虚拟化列表组件
 * 只渲染可视区域内的列表项，大幅提高长列表的性能
 * 支持动态高度、自动调整大小和性能监控
 */
const VirtualizedList: React.FC<VirtualizedListProps> = ({
  items,
  itemHeight,
  windowHeight,
  overscan = 5,
  renderItem,
  className = '',
  onScroll,
  scrollToIndex,
  onItemsRendered,
  enableResizeObserver = true,
  enablePerformanceMetrics = false,
  onPerformanceReport
}) => {
  // 滚动容器引用
  const containerRef = useRef<HTMLDivElement>(null);
  
  // 滚动位置状态
  const [scrollTop, setScrollTop] = useState(0);
  
  // 容器高度状态（用于自适应容器大小）
  const [containerHeight, setContainerHeight] = useState(windowHeight || 400);
  
  // 记录渲染次数用于性能分析
  const renderCountRef = useRef(0);
  const lastRenderTimeRef = useRef(0);
  
  // 性能监测器
  const performanceMonitorRef = useRef<ReturnType<typeof measurePerformance> | null>(null);
  
  // 计算每个项目的高度和位置（使用记忆化以提高性能）
  const { itemOffsets, totalHeight } = useMemo(() => {
    // 初始化偏移数组和高度
    const offsets: number[] = [0]; // 第一项的起始偏移为0
    let currentOffset = 0;
    let currentIndex = 0;
    
    // 计算每个项目的偏移位置
    for (const item of items) {
      const height = typeof itemHeight === 'function' 
        ? itemHeight(item, currentIndex) 
        : itemHeight;
      
      currentOffset += height;
      offsets.push(currentOffset);
      currentIndex++;
    }
    
    return {
      itemOffsets: offsets,
      totalHeight: currentOffset
    };
  }, [items, itemHeight]);
  
  // 二分查找确定当前滚动位置对应的项目索引
  const findStartIndex = useCallback((scrollTop: number): number => {
    let low = 0;
    let high = items.length;
    
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (itemOffsets[mid] <= scrollTop) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    
    return Math.max(0, low - 1);
  }, [items.length, itemOffsets]);
  
  // 计算可视区域内的起始和结束索引
  const startIndex = Math.max(0, findStartIndex(scrollTop) - overscan);
  
  // 计算结束索引（考虑动态高度）
  const endIndex = useMemo(() => {
    let index = startIndex;
    let heightSum = 0;
    
    while (index < items.length && heightSum < containerHeight + (overscan * (typeof itemHeight === 'number' ? itemHeight : 50))) {
      const height = typeof itemHeight === 'function' 
        ? itemHeight(items[index], index) 
        : itemHeight;
      
      heightSum += height;
      index++;
    }
    
    return Math.min(items.length - 1, index);
  }, [startIndex, items, containerHeight, overscan, itemHeight]);
  
  // 计算可视区域内的列表项
  const visibleItems = useMemo(() => {
    // 性能监测：开始记录渲染时间
    const startTime = performance.now();
    
    // 获取可视项
    const result = items.slice(startIndex, endIndex + 1);
    
    // 性能监测：结束记录渲染时间
    lastRenderTimeRef.current = performance.now() - startTime;
    renderCountRef.current++;
    
    if (enablePerformanceMetrics && performanceMonitorRef.current) {
      performanceMonitorRef.current.recordRenderTime(lastRenderTimeRef.current);
    }
    
    // 如果存在渲染回调，调用它
    if (onItemsRendered) {
      onItemsRendered(startIndex, endIndex);
    }
    
    return result;
  }, [items, startIndex, endIndex, enablePerformanceMetrics, onItemsRendered]);
  
  // 使用节流优化滚动事件处理
  const handleScroll = useCallback(
    throttle(() => {
      if (containerRef.current) {
        const newScrollTop = containerRef.current.scrollTop;
        setScrollTop(newScrollTop);
        
        if (onScroll) {
          onScroll(newScrollTop);
        }
      }
    }, 16), // 约60fps
    [onScroll]
  );
  
  // 添加滚动事件监听
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      return () => {
        container.removeEventListener('scroll', handleScroll);
      };
    }
  }, [handleScroll]);
  
  // 处理容器大小变化（如果启用ResizeObserver）
  useEffect(() => {
    if (!enableResizeObserver) return;
    
    const container = containerRef.current;
    if (!container) return;
    
    let resizeObserver: ResizeObserver | null = null;
    
    // 使用ResizeObserver监控容器大小变化
    try {
      resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
          if (entry.target === container) {
            setContainerHeight(entry.contentRect.height);
          }
        }
      });
      
      resizeObserver.observe(container);
    } catch (error) {
      console.warn('ResizeObserver not supported:', error);
    }
    
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [enableResizeObserver]);
  
  // 处理滚动到指定索引
  useEffect(() => {
    if (scrollToIndex !== undefined && containerRef.current) {
      containerRef.current.scrollTop = itemOffsets[scrollToIndex];
    }
  }, [scrollToIndex, itemOffsets]);
  
  // 性能监测
  useEffect(() => {
    if (!enablePerformanceMetrics) return;
    
    // 使用性能监测工具
    const performanceMonitor = measurePerformance(metrics => {
      if (onPerformanceReport) {
        onPerformanceReport({
          ...metrics,
          visibleItems: visibleItems.length,
          totalItems: items.length
        });
      }
    });
    
    performanceMonitorRef.current = performanceMonitor;
    performanceMonitor.start();
    
    return () => {
      performanceMonitor.stop();
      performanceMonitorRef.current = null;
    };
  }, [enablePerformanceMetrics, onPerformanceReport, visibleItems.length, items.length]);
  
  // 渲染优化：使用useMemo缓存渲染结果
  const renderedItems = useMemo(() => {
    return visibleItems.map((item, index) => {
      const actualIndex = startIndex + index;
      const top = itemOffsets[actualIndex];
      const height = typeof itemHeight === 'function' 
        ? itemHeight(item, actualIndex) 
        : itemHeight;
      
      return (
        <div
          key={item.id}
          style={{
            position: 'absolute',
            top,
            height,
            left: 0,
            right: 0
          }}
          data-index={actualIndex}
        >
          {renderItem(item, actualIndex)}
        </div>
      );
    });
  }, [visibleItems, startIndex, itemOffsets, itemHeight, renderItem]);
  
  return (
    <div
      ref={containerRef}
      className={`overflow-auto relative ${className}`}
      style={{ height: windowHeight || '100%' }}
      data-testid="virtualized-list-container"
    >
      {/* 创建一个占位容器，确保滚动条高度正确 */}
      <div 
        style={{ height: totalHeight, position: 'relative' }} 
        data-testid="virtualized-list-inner"
      >
        {renderedItems}
      </div>
    </div>
  );
};

// 使用示例组件
export const VirtualizedListExample: React.FC = () => {
  // 生成大量测试数据（带随机高度）
  const generateItems = (count: number): ListItem[] => {
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      content: `列表项 ${i + 1} - item${i}`,
      height: Math.floor(Math.random() * 50) + 50 // 随机高度 50-100px
    }));
  };
  
  // 生成10000个列表项
  const items = useMemo(() => generateItems(10000), []);
  
  // 性能指标状态
  const [performanceMetrics, setPerformanceMetrics] = useState({
    fps: 0,
    renderTime: 0,
    visibleItems: 0,
    totalItems: 0
  });
  
  // 使用动态高度
  const getItemHeight = (item: ListItem) => item.height || 72;
  
  // 渲染单个列表项
  const renderListItem = (item: ListItem, index: number) => (
    <div
      className={`p-4 border-b ${
        index % 2 === 0 ? 'bg-gray-50' : 'bg-white'
      } hover:bg-blue-50 transition-colors`}
      style={{ height: '100%' }}
    >
      <div className="font-medium">{item.content}</div>
      <div className="text-sm text-gray-500">
        ID: {item.id} | 高度: {item.height}px
      </div>
    </div>
  );
  
  return (
    <div className="p-4 max-w-2xl mx-auto">
      <h2 className="text-xl font-bold mb-4">优化版虚拟化列表示例 (10,000 项)</h2>
      <p className="mb-4 text-gray-600">
        此列表使用高级虚拟化技术，支持动态高度和容器自适应。
      </p>
      
      {/* 性能指标显示 */}
      <div className="mb-4 p-3 bg-gray-50 rounded-md border text-sm">
        <div className="font-medium mb-1">性能指标:</div>
        <div className="grid grid-cols-4 gap-2 text-xs">
          <div>FPS: <span className="font-mono">{performanceMetrics.fps}</span></div>
          <div>渲染时间: <span className="font-mono">{performanceMetrics.renderTime.toFixed(2)}ms</span></div>
          <div>可视项: <span className="font-mono">{performanceMetrics.visibleItems}</span></div>
          <div>总项数: <span className="font-mono">{performanceMetrics.totalItems}</span></div>
        </div>
      </div>
      
      <div className="border rounded-md shadow-sm">
        <VirtualizedList
          items={items}
          itemHeight={getItemHeight} // 使用动态高度
          windowHeight={400} // 可视窗口的高度
          overscan={10} // 额外渲染的项目数量
          renderItem={renderListItem}
          enableResizeObserver={true}
          enablePerformanceMetrics={true}
          onPerformanceReport={setPerformanceMetrics}
        />
      </div>
      
      <div className="mt-4 p-4 bg-gray-50 rounded-md border">
        <h3 className="font-medium mb-2">性能优化说明:</h3>
        <ul className="list-disc pl-5 space-y-1 text-sm text-gray-600">
          <li>支持动态高度项目，使用更精确的偏移计算</li>
          <li>使用二分查找算法快速定位滚动位置</li>
          <li>通过ResizeObserver自动适应容器大小变化</li>
          <li>使用useMemo缓存计算结果，避免不必要的重新计算</li>
          <li>内置性能监测，实时显示FPS和渲染时间</li>
          <li>支持滚动到指定索引的项目</li>
          <li>优化的渲染算法，确保只渲染必要的内容</li>
        </ul>
      </div>
    </div>
  );
};

export default VirtualizedList; 