import React, { useEffect, useRef } from 'react';

/**
 * 上下文菜单组件
 * 用于显示节点右键操作菜单
 *
 * @param {Object} props - 组件属性
 * @param {boolean} props.isOpen - 菜单是否打开
 * @param {Object} props.position - 菜单位置 {x, y}
 * @param {Object} props.node - 当前操作的节点
 * @param {Function} props.onClose - 关闭菜单的回调函数
 * @param {Object} props.actions - 菜单操作项
 * @returns {JSX.Element} 上下文菜单组件
 */
export default function ContextMenu({ isOpen, position, node, onClose, actions = {} }) {
  const menuRef = useRef(null);

  // 处理点击外部关闭菜单
  useEffect(() => {
    const handleClickOutside = e => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        onClose();
      }
    };

    // 添加全局点击事件监听
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  // 处理ESC键关闭菜单
  useEffect(() => {
    const handleKeyDown = e => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // 如果菜单没有打开，不渲染任何内容
  if (!isOpen || !node) return null;

  // 默认操作
  const defaultActions = {
    view: {
      label: '查看详情',
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
          />
        </svg>
      ),
      handler: actions.onView || (() => console.log('查看详情', node)),
    },
    expand: {
      label: '展开/收起',
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"
          />
        </svg>
      ),
      handler: actions.onToggleExpand || (() => console.log('展开/收起', node)),
    },
    highlight: {
      label: '高亮相关节点',
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 10V3L4 14h7v7l9-11h-7z"
          />
        </svg>
      ),
      handler: actions.onHighlight || (() => console.log('高亮相关节点', node)),
    },
    track: {
      label: '关注此地址',
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"
          />
        </svg>
      ),
      handler: actions.onTrack || (() => console.log('关注此地址', node)),
    },
    analyze: {
      label: '深度分析',
      icon: (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          />
        </svg>
      ),
      handler: actions.onAnalyze || (() => console.log('深度分析', node)),
    },
  };

  // 合并默认操作和自定义操作
  const menuActions = { ...defaultActions, ...actions };

  return (
    <div
      ref={menuRef}
      className="absolute bg-white dark:bg-gray-800 shadow-lg rounded-md py-2 z-50 w-48 text-sm"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
      }}
    >
      <div className="px-3 py-1 border-b border-gray-200 dark:border-gray-700 font-semibold text-gray-700 dark:text-gray-300 mb-1">
        {node.label || '节点操作'}
      </div>
      <div className="py-1">
        {Object.entries(menuActions).map(([key, action]) => (
          <button
            key={key}
            className="w-full text-left px-4 py-2 flex items-center hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
            onClick={() => {
              action.handler(node);
              onClose();
            }}
          >
            <span className="mr-2 text-gray-500 dark:text-gray-400">{action.icon}</span>
            {action.label}
          </button>
        ))}
      </div>
    </div>
  );
}
