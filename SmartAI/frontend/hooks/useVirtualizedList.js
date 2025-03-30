import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * 虚拟列表Hook，优化大量数据渲染性能
 *
 * @param {Object} options - 配置选项
 * @param {number} [options.itemHeight=40] - 单个项目的高度(px)
 * @param {number} [options.overscan=5] - 可视区域外额外渲染的行数
 * @param {number} [options.scrollThreshold=200] - 滚动防抖阈值(ms)
 * @param {boolean} [options.dynamicItemHeight=false] - 是否使用动态项目高度
 * @returns {Object} 虚拟列表属性和方法
 */
function useVirtualizedList(options = {}) {
  const {
    itemHeight = 40,
    overscan = 5,
    scrollThreshold = 200,
    dynamicItemHeight = false,
  } = options;

  // 容器ref
  const containerRef = useRef(null);

  // 滚动相关状态
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  // 记录各项目的实际高度，用于动态高度模式
  const itemHeightsRef = useRef({});
  // 记录累计高度，用于计算动态高度模式的位置
  const cumulativeHeightsRef = useRef([]);

  // 记录防抖定时器
  const scrollTimerRef = useRef(null);
  // 记录上次测量高度的时间
  const lastMeasureTimeRef = useRef(0);

  // 更新容器尺寸的函数
  const updateContainerHeight = useCallback(() => {
    if (containerRef.current) {
      setContainerHeight(containerRef.current.clientHeight);
    }
  }, []);

  // 测量项目高度
  const measureItemHeight = useCallback(
    (index, element) => {
      if (!element || !dynamicItemHeight) return;

      // 限制测量频率，避免重排
      const now = Date.now();
      if (now - lastMeasureTimeRef.current < 100) return;
      lastMeasureTimeRef.current = now;

      const height = element.getBoundingClientRect().height;

      if (height > 0 && height !== itemHeightsRef.current[index]) {
        itemHeightsRef.current[index] = height;

        // 更新累计高度
        let currentHeight = 0;
        const newCumulativeHeights = [0]; // 第一个元素是0

        for (let i = 0; i < index + 1; i++) {
          currentHeight += itemHeightsRef.current[i] || itemHeight;
          newCumulativeHeights.push(currentHeight);
        }

        cumulativeHeightsRef.current = newCumulativeHeights;
      }
    },
    [dynamicItemHeight, itemHeight]
  );

  // 根据索引获取项目偏移量（用于绝对定位）
  const getItemOffset = useCallback(
    index => {
      if (dynamicItemHeight) {
        // 动态高度模式
        return cumulativeHeightsRef.current[index] || index * itemHeight;
      }
      // 固定高度模式
      return index * itemHeight;
    },
    [dynamicItemHeight, itemHeight]
  );

  // 根据数据总量计算内容总高度
  const getContentHeight = useCallback(
    totalItems => {
      if (dynamicItemHeight && cumulativeHeightsRef.current.length > 0) {
        // 已知项目的累计高度加上未测量项目的估计高度
        const measuredItems = cumulativeHeightsRef.current.length - 1;
        const measuredHeight = cumulativeHeightsRef.current[measuredItems];
        return measuredHeight + (totalItems - measuredItems) * itemHeight;
      }
      // 固定高度模式
      return totalItems * itemHeight;
    },
    [dynamicItemHeight, itemHeight]
  );

  // 获取可见范围内的项目索引
  const getVisibleRange = useCallback(
    totalItems => {
      if (!containerRef.current) {
        return { startIndex: 0, endIndex: 20 };
      }

      if (dynamicItemHeight) {
        // 动态高度模式 - 查找最接近的项目
        let startIndex = 0;
        let endIndex = totalItems - 1;

        // 二分查找起始索引
        if (cumulativeHeightsRef.current.length > 0) {
          let low = 0;
          let high = cumulativeHeightsRef.current.length - 1;

          while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            const midOffset = cumulativeHeightsRef.current[mid];

            if (midOffset < scrollTop) {
              low = mid + 1;
            } else {
              high = mid - 1;
            }
          }

          startIndex = Math.max(0, high);
        } else {
          startIndex = Math.floor(scrollTop / itemHeight);
        }

        // 计算可见范围内的结束索引
        const visibleHeight = scrollTop + containerHeight;
        if (cumulativeHeightsRef.current.length > 0) {
          let low = 0;
          let high = cumulativeHeightsRef.current.length - 1;

          while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            const midOffset = cumulativeHeightsRef.current[mid];

            if (midOffset < visibleHeight) {
              low = mid + 1;
            } else {
              high = mid - 1;
            }
          }

          endIndex = low < totalItems ? low : totalItems - 1;
        } else {
          endIndex = Math.ceil(visibleHeight / itemHeight);
        }

        // 添加overscan
        startIndex = Math.max(0, startIndex - overscan);
        endIndex = Math.min(totalItems - 1, endIndex + overscan);

        return { startIndex, endIndex };
      }

      // 固定高度模式
      const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
      const endIndex = Math.min(
        totalItems - 1,
        Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
      );

      return { startIndex, endIndex };
    },
    [scrollTop, containerHeight, itemHeight, overscan, dynamicItemHeight]
  );

  // 处理滚动事件
  const handleScroll = useCallback(
    event => {
      // 获取滚动位置
      const newScrollTop = event.target.scrollTop;

      // 防抖更新
      if (scrollTimerRef.current) {
        clearTimeout(scrollTimerRef.current);
      }

      // 如果滚动距离较大，立即更新
      if (Math.abs(newScrollTop - scrollTop) > 100) {
        setScrollTop(newScrollTop);
      } else {
        // 否则使用防抖
        scrollTimerRef.current = setTimeout(() => {
          setScrollTop(newScrollTop);
        }, scrollThreshold);
      }
    },
    [scrollTop, scrollThreshold]
  );

  // 滚动到指定索引
  const scrollToIndex = useCallback(
    (index, behavior = 'auto', offset = 0) => {
      if (!containerRef.current) return;

      const top = getItemOffset(index);

      containerRef.current.scrollTo({
        top: top - offset,
        behavior,
      });
    },
    [getItemOffset]
  );

  // 初始化和清理
  useEffect(() => {
    // 初始化容器高度
    updateContainerHeight();

    // 监听容器大小变化
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      updateContainerHeight();
    });

    resizeObserver.observe(container);

    // 添加滚动监听
    container.addEventListener('scroll', handleScroll);

    // 组件销毁时清理
    return () => {
      if (scrollTimerRef.current) {
        clearTimeout(scrollTimerRef.current);
      }

      container.removeEventListener('scroll', handleScroll);
      resizeObserver.disconnect();
    };
  }, [updateContainerHeight, handleScroll]);

  // 返回虚拟列表需要的工具和属性
  return {
    containerRef,
    containerHeight,
    scrollTop,
    getVisibleRange,
    getItemOffset,
    getContentHeight,
    measureItemHeight,
    scrollToIndex,
  };
}

export default useVirtualizedList;
