/**
 * 客户端渲染Hook
 * 解决Next.js中的水合错误问题，确保组件只在客户端渲染
 */
import { useState, useEffect } from 'react';

/**
 * 客户端渲染检测钩子
 * 用于确保组件只在客户端渲染，避免服务端渲染不兼容问题
 *
 * @returns {boolean} 是否为客户端渲染
 */
function clientOnlyHook() {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return isClient;
}

// 命名导出
export const useClientOnly = clientOnlyHook;

// 默认导出保持一致性
export default clientOnlyHook;
