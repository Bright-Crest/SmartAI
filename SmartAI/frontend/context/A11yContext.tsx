import React, { createContext, useContext, useState, useEffect } from 'react';

interface A11yContextType {
  // 用户偏好
  highContrast: boolean;
  setHighContrast: (value: boolean) => void;
  reduceMotion: boolean;
  setReduceMotion: (value: boolean) => void;
  fontSize: 'normal' | 'large' | 'x-large';
  setFontSize: (size: 'normal' | 'large' | 'x-large') => void;
  
  // 状态通知
  announce: (message: string, type?: 'polite' | 'assertive') => void;
}

const A11yContext = createContext<A11yContextType | null>(null);

/**
 * 无障碍上下文提供器 - 管理全局无障碍设置和通知
 */
export const A11yProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  // 状态管理
  const [highContrast, setHighContrast] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const [fontSize, setFontSize] = useState<'normal' | 'large' | 'x-large'>('normal');
  
  // 屏幕阅读器通知
  const announce = (message: string, type: 'polite' | 'assertive' = 'polite') => {
    const announcer = document.getElementById(
      type === 'assertive' ? 'a11y-assertive-announcer' : 'a11y-polite-announcer'
    );
    
    if (announcer) {
      announcer.textContent = '';
      // 强制重绘
      setTimeout(() => {
        announcer.textContent = message;
      }, 50);
    }
  };
  
  // 系统偏好检测
  useEffect(() => {
    // 检测减少动画偏好
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduceMotion(prefersReducedMotion.matches);
    
    const handleReduceMotionChange = (e: MediaQueryListEvent) => {
      setReduceMotion(e.matches);
    };
    
    prefersReducedMotion.addEventListener('change', handleReduceMotionChange);
    
    // 检测高对比度模式（近似）
    const prefersContrast = window.matchMedia('(prefers-contrast: more)');
    setHighContrast(prefersContrast.matches);
    
    const handleContrastChange = (e: MediaQueryListEvent) => {
      setHighContrast(e.matches);
    };
    
    prefersContrast.addEventListener('change', handleContrastChange);
    
    return () => {
      prefersReducedMotion.removeEventListener('change', handleReduceMotionChange);
      prefersContrast.removeEventListener('change', handleContrastChange);
    };
  }, []);
  
  // 应用全局样式
  useEffect(() => {
    document.documentElement.classList.toggle('high-contrast', highContrast);
    document.documentElement.classList.toggle('reduce-motion', reduceMotion);
    document.documentElement.dataset.fontSize = fontSize;
  }, [highContrast, reduceMotion, fontSize]);
  
  return (
    <A11yContext.Provider
      value={{
        highContrast,
        setHighContrast,
        reduceMotion,
        setReduceMotion,
        fontSize,
        setFontSize,
        announce
      }}
    >
      {/* 添加屏幕阅读器通知区域 */}
      <div
        id="a11y-polite-announcer"
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
      />
      <div
        id="a11y-assertive-announcer"
        className="sr-only"
        aria-live="assertive"
        aria-atomic="true"
      />
      {children}
    </A11yContext.Provider>
  );
};

/**
 * 使用无障碍上下文
 * @returns 无障碍上下文值
 */
export const useA11y = () => {
  const context = useContext(A11yContext);
  if (!context) {
    throw new Error('useA11y必须在A11yProvider内部使用');
  }
  return context;
};

export default A11yProvider; 