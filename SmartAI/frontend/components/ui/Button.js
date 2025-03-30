import React from 'react';

/**
 * 按钮组件
 * @param {Object} props - 组件属性
 * @param {string} props.variant - 按钮变体，可选值：primary, secondary, outline, danger, success
 * @param {string} props.size - 按钮大小，可选值：sm, md, lg
 * @param {boolean} props.isLoading - 是否显示加载状态
 * @param {boolean} props.isFullWidth - 是否占满容器宽度
 * @param {boolean} props.disabled - 是否禁用
 * @param {React.ReactNode} props.children - 子组件
 * @param {Function} props.onClick - 点击事件处理函数
 * @param {string} props.type - 按钮类型，可选值：button, submit, reset
 * @param {string} props.className - 额外的CSS类名
 * @returns {JSX.Element} 按钮组件
 */
export default function Button({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  isFullWidth = false,
  disabled = false,
  children,
  onClick,
  type = 'button',
  className = '',
  ...rest
}) {
  // 基础样式
  const baseStyle =
    'inline-flex items-center justify-center rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200';

  // 变体样式
  const variantStyles = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
    secondary:
      'bg-gray-200 text-gray-800 hover:bg-gray-300 focus:ring-gray-500 dark:bg-gray-700 dark:text-gray-100 dark:hover:bg-gray-600',
    outline:
      'bg-transparent border border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-500 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800',
    danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
    success: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500',
  };

  // 尺寸样式
  const sizeStyles = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  };

  // 宽度样式
  const widthStyle = isFullWidth ? 'w-full' : '';

  // 禁用和加载状态
  const stateStyle = disabled || isLoading ? 'opacity-70 cursor-not-allowed' : 'cursor-pointer';

  // 组合所有样式
  const buttonStyle = `${baseStyle} ${variantStyles[variant]} ${sizeStyles[size]} ${widthStyle} ${stateStyle} ${className}`;

  return (
    <button
      type={type}
      className={buttonStyle}
      onClick={onClick}
      disabled={disabled || isLoading}
      {...rest}
    >
      {isLoading && (
        <svg
          className="animate-spin -ml-1 mr-2 h-4 w-4 text-current"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          ></path>
        </svg>
      )}
      {children}
    </button>
  );
}
