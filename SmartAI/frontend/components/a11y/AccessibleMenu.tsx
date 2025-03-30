import React, { useState, useRef, useEffect } from 'react';
import FocusableItem from './FocusableItem';
import useKeyboardNavigation from '../../hooks/useKeyboardNavigation';

export interface MenuItem {
  id: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

interface AccessibleMenuProps {
  items: MenuItem[];
  trigger: React.ReactNode;
  className?: string;
  menuClassName?: string;
  itemClassName?: string;
  menuId?: string;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  onClose?: () => void;
}

/**
 * 无障碍菜单组件 - 提供完全的键盘导航支持
 * 遵循WAI-ARIA菜单设计模式
 */
const AccessibleMenu: React.FC<AccessibleMenuProps> = ({
  items,
  trigger,
  className = '',
  menuClassName = '',
  itemClassName = '',
  menuId,
  placement = 'bottom',
  onClose
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);
  const menuItemsRef = useRef<Array<HTMLDivElement | null>>([]);
  
  // 使用键盘导航钩子
  const { handleArrowKeys, manageFocusTrap, saveFocus, restoreFocus } = useKeyboardNavigation();
  
  // 生成唯一ID
  const uniqueId = useRef(`menu-${Math.random().toString(36).substring(2, 9)}`);
  const menuUniqueId = menuId || uniqueId.current;
  
  // 切换菜单状态
  const toggleMenu = () => {
    if (isOpen) {
      closeMenu();
    } else {
      openMenu();
    }
  };
  
  // 打开菜单
  const openMenu = () => {
    setIsOpen(true);
    saveFocus();
    // 在下一个周期聚焦第一个菜单项
    setTimeout(() => {
      if (menuItemsRef.current[0]) {
        menuItemsRef.current[0].focus();
        setActiveIndex(0);
      }
    }, 10);
  };
  
  // 关闭菜单
  const closeMenu = () => {
    setIsOpen(false);
    setActiveIndex(-1);
    // 恢复焦点到触发器
    setTimeout(() => {
      if (triggerRef.current) {
        triggerRef.current.focus();
      }
    }, 10);
    // 调用关闭回调
    if (onClose) {
      onClose();
    }
  };
  
  // 处理项目点击
  const handleItemClick = (callback: () => void) => {
    return () => {
      callback();
      closeMenu();
    };
  };
  
  // 处理键盘导航
  const handleMenuKeyDown = (e: React.KeyboardEvent) => {
    // 使用上下键导航
    handleArrowKeys({
      up: () => {
        const newIndex = activeIndex <= 0 ? items.length - 1 : activeIndex - 1;
        setActiveIndex(newIndex);
        menuItemsRef.current[newIndex]?.focus();
      },
      down: () => {
        const newIndex = activeIndex >= items.length - 1 ? 0 : activeIndex + 1;
        setActiveIndex(newIndex);
        menuItemsRef.current[newIndex]?.focus();
      }
    })(e);

    // 使用ESC键关闭菜单
    if (e.key === 'Escape') {
      closeMenu();
    }
  };
  
  // 点击外部关闭菜单
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        menuRef.current && 
        !menuRef.current.contains(event.target as Node) &&
        triggerRef.current && 
        !triggerRef.current.contains(event.target as Node)
      ) {
        if (isOpen) {
          closeMenu();
        }
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);
  
  // 确定菜单位置样式
  const getPlacementStyle = () => {
    switch (placement) {
      case 'top':
        return 'bottom-full left-0 mb-1';
      case 'left':
        return 'right-full top-0 mr-1';
      case 'right':
        return 'left-full top-0 ml-1';
      case 'bottom':
      default:
        return 'top-full left-0 mt-1';
    }
  };
  
  return (
    <div className={`relative inline-block ${className}`}>
      {/* 菜单触发器 */}
      <div 
        ref={triggerRef}
        onClick={toggleMenu}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ' || e.key === 'ArrowDown') {
            e.preventDefault();
            if (!isOpen) {
              openMenu();
            }
          }
        }}
        role="button"
        tabIndex={0}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        aria-controls={menuUniqueId}
      >
        {trigger}
      </div>
      
      {/* 菜单内容 */}
      {isOpen && (
        <div 
          ref={menuRef}
          id={menuUniqueId}
          role="menu"
          aria-orientation="vertical"
          className={`absolute z-10 min-w-[10rem] py-1 bg-white border border-gray-200 rounded-md shadow-lg ${getPlacementStyle()} ${menuClassName}`}
          onKeyDown={(e) => {
            handleMenuKeyDown(e);
            manageFocusTrap(menuRef, closeMenu)(e);
          }}
        >
          {items.map((item, index) => (
            <FocusableItem
              key={item.id}
              ref={(el: HTMLDivElement | null) => {
                menuItemsRef.current[index] = el;
              }}
              role="menuitem"
              tabIndex={activeIndex === index ? 0 : -1}
              onClick={handleItemClick(item.onClick)}
              className={`block w-full px-4 py-2 text-left text-gray-800 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none ${
                activeIndex === index ? 'bg-gray-50' : ''
              } ${item.disabled ? 'opacity-50 cursor-not-allowed' : ''} ${itemClassName}`}
              aria-disabled={item.disabled}
              disabled={item.disabled}
            >
              {item.label}
            </FocusableItem>
          ))}
        </div>
      )}
    </div>
  );
};

export default AccessibleMenu; 