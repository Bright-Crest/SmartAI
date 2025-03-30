import React, { useRef, useEffect } from 'react';
import useKeyboardNavigation from '../../hooks/useKeyboardNavigation';

interface FocusTrapProps {
  /**
   * 是否激活焦点捕获
   */
  active: boolean;
  
  /**
   * 初始焦点元素的选择器
   */
  initialFocus?: string;
  
  /**
   * 子元素
   */
  children: React.ReactNode;
  
  /**
   * 当焦点尝试离开时触发
   */
  onEscapeKeyDown?: () => void;
  
  /**
   * 当组件卸载时恢复焦点
   */
  restoreFocus?: boolean;
  
  /**
   * 自定义类名
   */
  className?: string;
}

/**
 * 焦点陷阱组件 - 将键盘焦点限制在组件内部
 * 用于模态对话框、下拉菜单等临时UI元素
 */
const FocusTrap: React.FC<FocusTrapProps> = ({
  active,
  initialFocus,
  children,
  onEscapeKeyDown,
  restoreFocus = true,
  className = ''
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const { manageFocusTrap } = useKeyboardNavigation();
  
  // 保存之前的焦点
  useEffect(() => {
    if (active && restoreFocus) {
      previousFocusRef.current = document.activeElement as HTMLElement;
    }
  }, [active, restoreFocus]);
  
  // 恢复之前的焦点
  useEffect(() => {
    return () => {
      if (restoreFocus && previousFocusRef.current) {
        if ('focus' in previousFocusRef.current) {
          previousFocusRef.current.focus();
        }
      }
    };
  }, [restoreFocus]);
  
  // 设置初始焦点
  useEffect(() => {
    if (active && containerRef.current) {
      let focusElement: HTMLElement | null = null;
      
      if (initialFocus) {
        focusElement = containerRef.current.querySelector(initialFocus) as HTMLElement;
      }
      
      if (!focusElement) {
        // 找到第一个可聚焦元素
        const focusables = containerRef.current.querySelectorAll(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        
        if (focusables.length > 0) {
          focusElement = focusables[0] as HTMLElement;
        }
      }
      
      if (focusElement && 'focus' in focusElement) {
        setTimeout(() => {
          focusElement?.focus();
        }, 100);
      }
    }
  }, [active, initialFocus]);
  
  // 管理键盘事件
  useEffect(() => {
    if (active && containerRef.current) {
      const handleKeyDown = manageFocusTrap(containerRef, onEscapeKeyDown);
      
      containerRef.current.addEventListener('keydown', handleKeyDown as any);
      return () => {
        containerRef.current?.removeEventListener('keydown', handleKeyDown as any);
      };
    }
  }, [active, manageFocusTrap, onEscapeKeyDown]);
  
  return (
    <div ref={containerRef} className={className}>
      {children}
    </div>
  );
};

export default FocusTrap; 