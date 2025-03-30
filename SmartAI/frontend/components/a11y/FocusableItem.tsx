import React, { forwardRef } from 'react';
import useKeyboardNavigation from '../../hooks/useKeyboardNavigation';

export interface FocusableItemProps extends React.HTMLAttributes<HTMLDivElement> {
  /**
   * 组件内容
   */
  children: React.ReactNode;
  
  /**
   * 点击处理函数
   */
  onClick: () => void;
  
  /**
   * 是否禁用
   */
  disabled?: boolean;
  
  /**
   * ARIA 角色
   */
  role?: string;
  
  /**
   * 标签索引（-1 表示不可通过 Tab 导航）
   */
  tabIndex?: number;
  
  /**
   * 自定义类名
   */
  className?: string;
  
  /**
   * ARIA 扩展属性
   */
  [key: `aria-${string}`]: any;
}

/**
 * 可聚焦项组件
 * 提供一致的键盘导航和焦点管理
 * 所有交互元素必须可通过键盘访问并有视觉焦点指示器
 */
const FocusableItem = forwardRef<HTMLDivElement, FocusableItemProps>(
  ({ 
    children, 
    onClick, 
    disabled = false, 
    role = 'button', 
    tabIndex = 0, 
    className = '',
    onKeyDown,
    ...props 
  }, ref) => {
    // 使用键盘导航钩子
    const { handleEnterAndSpace } = useKeyboardNavigation();

    // 合并自定义键盘事件和默认的回车/空格处理
    const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
      // 如果禁用，不处理任何键盘事件
      if (disabled) return;

      // 如果提供了自定义键盘事件处理函数，先调用它
      if (onKeyDown) {
        onKeyDown(e);
        // 如果事件已被处理（阻止默认行为），不再继续
        if (e.defaultPrevented) return;
      }

      // 默认的回车/空格键处理
      handleEnterAndSpace(onClick)(e);
    };

    return (
      <div
        ref={ref}
        role={role}
        tabIndex={disabled ? -1 : tabIndex}
        className={`${className} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onClick={disabled ? undefined : onClick}
        onKeyDown={handleKeyDown}
        aria-disabled={disabled}
        {...props}
      >
        {children}
      </div>
    );
  }
);

FocusableItem.displayName = 'FocusableItem';

export default FocusableItem; 