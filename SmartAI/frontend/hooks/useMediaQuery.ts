import { useState, useEffect } from 'react';

/**
 * 自定义钩子用于响应式设计
 * 接受一个media query字符串，返回是否匹配该查询
 * 
 * @param {string} query - CSS媒体查询字符串，如 '(max-width: 640px)'
 * @returns {boolean} 是否匹配该媒体查询
 */
function useMediaQuery(query: string): boolean {
  // 确保我们在客户端环境下
  const getMatches = (): boolean => {
    // 服务器端渲染时，假设不匹配
    if (typeof window !== 'undefined') {
      return window.matchMedia(query).matches;
    }
    return false;
  };

  const [matches, setMatches] = useState<boolean>(getMatches());

  useEffect(() => {
    // 初始化匹配状态
    setMatches(getMatches());

    // 创建一个媒体查询列表
    const mediaQuery = window.matchMedia(query);
    
    // 定义事件处理函数
    const handleChange = (): void => {
      setMatches(mediaQuery.matches);
    };

    // 添加事件监听器
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
    } else {
      // 兼容旧版浏览器
      mediaQuery.addListener(handleChange);
    }

    // 清理函数
    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleChange);
      } else {
        // 兼容旧版浏览器
        mediaQuery.removeListener(handleChange);
      }
    };
  }, [query]);

  return matches;
}

export default useMediaQuery; 