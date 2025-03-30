import React from 'react';

/**
 * 卡片组件
 */
export const Card = ({ children, className = '', ...props }) => {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow ${className}`} {...props}>
      {children}
    </div>
  );
};

/**
 * 按钮组件
 */
export const Button = ({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  className = '',
  ...props
}) => {
  // 样式映射
  const variantStyles = {
    primary: 'bg-blue-500 hover:bg-blue-600 text-white',
    secondary:
      'bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200',
    danger: 'bg-red-500 hover:bg-red-600 text-white',
  };

  const sizeStyles = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  };

  return (
    <button
      onClick={onClick}
      className={`rounded font-medium transition-colors ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

/**
 * 提示框组件
 */
export const Alert = ({ children, variant = 'info', className = '', ...props }) => {
  // 样式映射
  const variantStyles = {
    info: 'bg-blue-50 text-blue-800 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800',
    success:
      'bg-green-50 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-800',
    warning:
      'bg-yellow-50 text-yellow-800 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-300 dark:border-yellow-800',
    error:
      'bg-red-50 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800',
  };

  return (
    <div className={`p-4 rounded-md border ${variantStyles[variant]} ${className}`} {...props}>
      {children}
    </div>
  );
};
