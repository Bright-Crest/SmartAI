import { useState, useEffect, useRef } from 'react';

/**
 * 使用ResizeObserver API监听元素尺寸变化的Hook
 *
 * @param {Object} options - 配置选项
 * @param {number} [options.debounceDelay=200] - 防抖延迟(毫秒)
 * @param {boolean} [options.observeSize=true] - 是否观察尺寸变化
 * @param {boolean} [options.observePosition=false] - 是否观察位置变化
 * @returns {[React.MutableRefObject<HTMLElement>, { width: number, height: number, top: number, left: number }]}
 *          返回一个ref和包含尺寸信息的对象
 */
function useResizeObserver(options = {}) {
  const { debounceDelay = 200, observeSize = true, observePosition = false } = options;

  // 创建一个ref来存储DOM元素
  const ref = useRef(null);

  // 状态存储元素的尺寸和位置
  const [dimensions, setDimensions] = useState({
    width: 0,
    height: 0,
    top: 0,
    left: 0,
  });

  // 监听器的防抖定时器
  const timerRef = useRef(null);

  // ResizeObserver实例
  const observerRef = useRef(null);

  useEffect(() => {
    // 获取当前元素
    const element = ref.current;
    if (!element) return;

    // 更新尺寸的函数
    const updateDimensions = entries => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      timerRef.current = setTimeout(() => {
        if (!Array.isArray(entries) || !entries.length) return;

        // 获取元素的矩形区域信息
        const entry = entries[0];
        const rect = entry.target.getBoundingClientRect();

        setDimensions(prevDimensions => {
          const newDimensions = { ...prevDimensions };

          // 只有在需要时才更新尺寸
          if (observeSize) {
            newDimensions.width = rect.width;
            newDimensions.height = rect.height;
          }

          // 只有在需要时才更新位置
          if (observePosition) {
            newDimensions.top = rect.top;
            newDimensions.left = rect.left;
          }

          return newDimensions;
        });
      }, debounceDelay);
    };

    // 初始化时手动获取一次尺寸
    const rect = element.getBoundingClientRect();
    setDimensions({
      width: observeSize ? rect.width : 0,
      height: observeSize ? rect.height : 0,
      top: observePosition ? rect.top : 0,
      left: observePosition ? rect.left : 0,
    });

    // 创建观察器
    if (typeof ResizeObserver !== 'undefined') {
      observerRef.current = new ResizeObserver(updateDimensions);
      observerRef.current.observe(element);
    } else {
      // 降级处理：如果ResizeObserver不可用，使用window.resize事件
      console.warn('ResizeObserver API不可用，降级使用window.resize');

      const handleResize = () => {
        const entries = [{ target: element }];
        updateDimensions(entries);
      };

      window.addEventListener('resize', handleResize);

      // 返回清理函数
      return () => {
        if (timerRef.current) {
          clearTimeout(timerRef.current);
        }
        window.removeEventListener('resize', handleResize);
      };
    }

    // 返回清理函数
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [debounceDelay, observeSize, observePosition]);

  return [ref, dimensions];
}

export default useResizeObserver;
