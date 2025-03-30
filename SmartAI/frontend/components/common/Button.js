import React from 'react';

/**
 * 通用按钮组件
 *
 * @param {Object} props - 组件属性
 * @param {React.ReactNode} props.children - 子组件内容
 * @param {function} props.onClick - 点击事件处理函数
 * @param {string} props.type - 按钮类型 ('primary', 'secondary', 'danger', 'success')
 * @param {boolean} props.disabled - 是否禁用按钮
 * @param {Object} props.style - 自定义样式
 * @param {string} props.className - 自定义CSS类名
 */
export default function Button({
  children,
  onClick,
  type = 'primary',
  disabled = false,
  style = {},
  className = '',
}) {
  // 根据类型确定颜色
  const getTypeStyles = () => {
    switch (type) {
      case 'primary':
        return { backgroundColor: '#3498db', color: 'white' };
      case 'secondary':
        return { backgroundColor: '#95a5a6', color: 'white' };
      case 'danger':
        return { backgroundColor: '#e74c3c', color: 'white' };
      case 'success':
        return { backgroundColor: '#2ecc71', color: 'white' };
      default:
        return { backgroundColor: '#3498db', color: 'white' };
    }
  };

  return (
    <button
      className={`button ${type} ${className}`}
      onClick={onClick}
      disabled={disabled}
      style={{
        display: 'inline-block',
        padding: '8px 16px',
        border: 'none',
        borderRadius: '5px',
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontWeight: '500',
        transition: 'background-color 0.3s, opacity 0.3s',
        opacity: disabled ? 0.6 : 1,
        ...getTypeStyles(),
        ...style,
      }}
    >
      {children}
    </button>
  );
}
