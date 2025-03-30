import { useCallback, useRef } from 'react';

/**
 * 键盘导航Hook，提供统一的键盘导航支持
 * 简化组件中处理键盘事件的逻辑
 */
interface KeyboardNavOptions {
  /**
   * 是否阻止默认行为
   * @default true
   */
  preventDefault?: boolean;
  
  /**
   * 当按键被触发时是否停止事件传播
   * @default false
   */
  stopPropagation?: boolean;
}

interface ArrowKeyHandlers {
  up?: () => void;
  down?: () => void;
  left?: () => void;
  right?: () => void;
  home?: () => void;
  end?: () => void;
}

/**
 * 列表导航选项接口
 */
interface ListNavigationOptions {
  /**
   * 当前活动索引
   */
  currentIndex: number;
  
  /**
   * 列表项总数
   */
  itemCount: number;
  
  /**
   * 索引变化时的回调
   */
  onIndexChange: (newIndex: number) => void;
  
  /**
   * 是否支持垂直导航
   * @default true
   */
  vertical?: boolean;
  
  /**
   * 是否支持水平导航
   * @default false
   */
  horizontal?: boolean;
  
  /**
   * 是否循环导航（从末尾回到开头）
   * @default true
   */
  circular?: boolean;
}

/**
 * 通用键盘导航Hook
 * @returns 键盘事件处理工具函数
 */
const useKeyboardNavigation = () => {
  // 存储上一次焦点元素的引用
  const lastFocusedElementRef = useRef<HTMLElement | null>(null);

  /**
   * 处理Enter和Space键按下，执行点击操作
   * @param callback 要执行的回调函数
   * @returns 键盘事件处理函数
   */
  const handleEnterAndSpace = useCallback(
    (callback: () => void) => (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        callback();
      }
    },
    []
  );

  /**
   * 处理方向键导航
   * @param handlers 方向键处理函数对象
   * @returns 键盘事件处理函数
   */
  const handleArrowKeys = useCallback(
    (handlers: ArrowKeyHandlers) => (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowUp':
          if (handlers.up) {
            e.preventDefault();
            handlers.up();
          }
          break;
        case 'ArrowDown':
          if (handlers.down) {
            e.preventDefault();
            handlers.down();
          }
          break;
        case 'ArrowLeft':
          if (handlers.left) {
            e.preventDefault();
            handlers.left();
          }
          break;
        case 'ArrowRight':
          if (handlers.right) {
            e.preventDefault();
            handlers.right();
          }
          break;
        case 'Home':
          if (handlers.home) {
            e.preventDefault();
            handlers.home();
          }
          break;
        case 'End':
          if (handlers.end) {
            e.preventDefault();
            handlers.end();
          }
          break;
      }
    },
    []
  );

  /**
   * 保存当前焦点元素
   */
  const saveFocus = useCallback(() => {
    lastFocusedElementRef.current = document.activeElement as HTMLElement;
  }, []);

  /**
   * 恢复上一次的焦点
   */
  const restoreFocus = useCallback(() => {
    if (lastFocusedElementRef.current) {
      lastFocusedElementRef.current.focus();
    }
  }, []);

  /**
   * 管理焦点陷阱，用于模态对话框等
   * @param containerRef 容器元素的引用
   * @param onEscape Escape键处理函数
   * @returns 键盘事件处理函数
   */
  const manageFocusTrap = useCallback(
    (containerRef: React.RefObject<HTMLElement>, onEscape?: () => void) => (e: React.KeyboardEvent) => {
      if (!containerRef.current) return;

      const container = containerRef.current;
      const focusableElements = container.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      
      const firstElement = focusableElements[0] as HTMLElement;
      const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

      // 处理Escape键
      if (e.key === 'Escape' && onEscape) {
        e.preventDefault();
        onEscape();
        return;
      }

      // 处理Tab键循环
      if (e.key === 'Tab') {
        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    },
    []
  );

  /**
   * 处理列表键盘导航
   * @param options 列表导航选项
   * @returns 键盘事件处理函数
   */
  const handleListNavigation = useCallback(
    (options: ListNavigationOptions) => (e: React.KeyboardEvent) => {
      const {
        currentIndex,
        itemCount,
        onIndexChange,
        vertical = true,
        horizontal = false,
        circular = true,
      } = options;

      if (itemCount === 0) return;

      let newIndex = currentIndex;

      // 垂直导航处理
      if (vertical) {
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          newIndex = currentIndex - 1;
          if (newIndex < 0) {
            newIndex = circular ? itemCount - 1 : 0;
          }
          onIndexChange(newIndex);
          return;
        }

        if (e.key === 'ArrowDown') {
          e.preventDefault();
          newIndex = currentIndex + 1;
          if (newIndex >= itemCount) {
            newIndex = circular ? 0 : itemCount - 1;
          }
          onIndexChange(newIndex);
          return;
        }
      }

      // 水平导航处理
      if (horizontal) {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          newIndex = currentIndex - 1;
          if (newIndex < 0) {
            newIndex = circular ? itemCount - 1 : 0;
          }
          onIndexChange(newIndex);
          return;
        }

        if (e.key === 'ArrowRight') {
          e.preventDefault();
          newIndex = currentIndex + 1;
          if (newIndex >= itemCount) {
            newIndex = circular ? 0 : itemCount - 1;
          }
          onIndexChange(newIndex);
          return;
        }
      }

      // Home 和 End 按键处理
      if (e.key === 'Home') {
        e.preventDefault();
        onIndexChange(0);
        return;
      }

      if (e.key === 'End') {
        e.preventDefault();
        onIndexChange(itemCount - 1);
        return;
      }
    },
    []
  );

  return {
    handleEnterAndSpace,
    handleArrowKeys,
    saveFocus,
    restoreFocus,
    manageFocusTrap,
    handleListNavigation
  };
};

export default useKeyboardNavigation; 