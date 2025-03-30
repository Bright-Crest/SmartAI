import React from 'react';

/**
 * 通用卡片组件
 *
 * @param {Object} props - 组件属性
 * @param {React.ReactNode} props.children - 子组件内容
 * @param {string} props.title - 卡片标题
 * @param {Object} props.style - 自定义样式
 * @param {function} props.onClick - 点击事件处理函数
 * @param {string} props.className - 自定义CSS类名
 */
export default function Card({ children, title, style = {}, onClick, className = '' }) {
  return (
    <div
      className={`card ${className}`}
      style={{
        border: '1px solid #ddd',
        borderRadius: '10px',
        padding: '20px',
        backgroundColor: '#fff',
        boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
        transition: 'transform 0.3s ease, box-shadow 0.3s ease',
        cursor: onClick ? 'pointer' : 'default',
        ...style,
      }}
      onClick={onClick}
    >
      {title && <h3 style={{ color: '#2c3e50', marginTop: 0 }}>{title}</h3>}
      {children}
    </div>
  );
}
