import { useState, useEffect, RefObject } from 'react';

interface DimensionObject {
  width: number;
  height: number;
}

/**
 * 观察元素大小变化的自定义钩子
 * 使用ResizeObserver API监听DOM元素尺寸变化
 */
function useResizeObserver<T extends HTMLElement = HTMLElement>(
  ref: RefObject<T>
): DimensionObject | undefined {
  const [dimensions, setDimensions] = useState<DimensionObject>();
  
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    
    // 处理尺寸变化
    const resizeCallback: ResizeObserverCallback = (entries) => {
      if (!Array.isArray(entries) || !entries.length) return;
      
      const entry = entries[0];
      
      // 使用内容盒模型的尺寸
      if (entry.contentBoxSize) {
        // 确保兼容不同浏览器
        const contentBoxSize = Array.isArray(entry.contentBoxSize) 
          ? entry.contentBoxSize[0] 
          : entry.contentBoxSize;
          
        setDimensions({
          width: contentBoxSize.inlineSize,
          height: contentBoxSize.blockSize,
        });
      } 
      // 回退到旧的 API
      else {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    };
    
    // 创建观察器
    let resizeObserver: ResizeObserver;
    
    try {
      resizeObserver = new ResizeObserver(resizeCallback);
      resizeObserver.observe(element);
      
      // 立即设置初始尺寸
      setDimensions({
        width: element.clientWidth,
        height: element.clientHeight,
      });
    } catch (error) {
      console.error('ResizeObserver 不可用:', error);
      
      // 回退到窗口事件监听
      const handleResize = () => {
        if (ref.current) {
          setDimensions({
            width: ref.current.clientWidth,
            height: ref.current.clientHeight,
          });
        }
      };
      
      window.addEventListener('resize', handleResize);
      handleResize(); // 设置初始尺寸
      
      // 清理函数
      return () => {
        window.removeEventListener('resize', handleResize);
      };
    }
    
    // 清理函数
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [ref]);
  
  return dimensions;
}

export default useResizeObserver; 