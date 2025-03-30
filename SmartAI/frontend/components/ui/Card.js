import React from 'react';

/**
 * 卡片组件
 * @param {Object} props - 组件属性
 * @param {React.ReactNode} props.children - 子元素
 * @param {string} props.className - 额外的CSS类名
 * @param {boolean} props.hasShadow - 是否有阴影
 * @param {boolean} props.hasBorder - 是否有边框
 * @param {boolean} props.isInteractive - 是否有交互效果
 * @param {Function} props.onClick - 点击事件处理函数
 * @returns {JSX.Element} 卡片组件
 */
export default function Card({
  children,
  className = '',
  hasShadow = true,
  hasBorder = true,
  isInteractive = false,
  onClick,
  ...rest
}) {
  // 基础样式
  const baseStyle = 'bg-white dark:bg-gray-800 rounded-lg overflow-hidden';

  // 阴影样式
  const shadowStyle = hasShadow ? 'shadow-md' : '';

  // 边框样式
  const borderStyle = hasBorder ? 'border border-gray-200 dark:border-gray-700' : '';

  // 交互样式
  const interactiveStyle = isInteractive
    ? 'transition-all duration-200 hover:shadow-lg cursor-pointer transform hover:-translate-y-1'
    : '';

  // 组合所有样式
  const cardStyle = `${baseStyle} ${shadowStyle} ${borderStyle} ${interactiveStyle} ${className}`;

  return (
    <div className={cardStyle} onClick={onClick} {...rest}>
      {children}
    </div>
  );
}
