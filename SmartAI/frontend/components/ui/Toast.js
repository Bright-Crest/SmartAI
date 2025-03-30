import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';
import {
  FaCheckCircle,
  FaExclamationCircle,
  FaInfoCircle,
  FaExclamationTriangle,
  FaTimes,
} from 'react-icons/fa';

/**
 * Toast类型
 */
export const ToastTypes = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
};

/**
 * Toast组件
 *
 * @param {Object} props - 组件属性
 * @param {string} props.id - Toast ID
 * @param {string} props.message - 消息内容
 * @param {string} props.type - Toast类型
 * @param {Function} props.onClose - 关闭回调
 * @param {number} props.duration - 显示时长（毫秒）
 * @returns {JSX.Element} Toast组件
 */
const Toast = ({ id, message, type = ToastTypes.INFO, onClose, duration = 5000 }) => {
  // 状态
  const [isVisible, setIsVisible] = useState(true);
  const [progress, setProgress] = useState(100);
  const [intervalId, setIntervalId] = useState(null);

  // 图标映射
  const icons = {
    [ToastTypes.SUCCESS]: <FaCheckCircle className="text-green-500 dark:text-green-400" />,
    [ToastTypes.ERROR]: <FaExclamationCircle className="text-red-500 dark:text-red-400" />,
    [ToastTypes.WARNING]: (
      <FaExclamationTriangle className="text-yellow-500 dark:text-yellow-400" />
    ),
    [ToastTypes.INFO]: <FaInfoCircle className="text-blue-500 dark:text-blue-400" />,
  };

  // 背景颜色映射
  const bgColors = {
    [ToastTypes.SUCCESS]: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800',
    [ToastTypes.ERROR]: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800',
    [ToastTypes.WARNING]:
      'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800',
    [ToastTypes.INFO]: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800',
  };

  // 进度条颜色映射
  const progressColors = {
    [ToastTypes.SUCCESS]: 'bg-green-500 dark:bg-green-400',
    [ToastTypes.ERROR]: 'bg-red-500 dark:bg-red-400',
    [ToastTypes.WARNING]: 'bg-yellow-500 dark:bg-yellow-400',
    [ToastTypes.INFO]: 'bg-blue-500 dark:bg-blue-400',
  };

  // 处理关闭
  const handleClose = () => {
    setIsVisible(false);

    // 清除进度条计时器
    if (intervalId) {
      clearInterval(intervalId);
    }

    // 通知父组件
    if (onClose) {
      setTimeout(() => onClose(id), 300); // 等待动画完成
    }
  };

  // 设置自动关闭计时器
  useEffect(() => {
    if (duration) {
      // 自动关闭计时器
      const closeTimeout = setTimeout(handleClose, duration);

      // 进度条更新
      const interval = 100; // 每100毫秒更新一次进度条
      const stepSize = 100 / (duration / interval);

      const progressInterval = setInterval(() => {
        setProgress(prevProgress => {
          const newProgress = prevProgress - stepSize;
          return newProgress > 0 ? newProgress : 0;
        });
      }, interval);

      setIntervalId(progressInterval);

      return () => {
        clearTimeout(closeTimeout);
        clearInterval(progressInterval);
      };
    }
  }, [duration, id]);

  return (
    <div
      className={`max-w-md w-full rounded-lg shadow-lg overflow-hidden border ${bgColors[type]} transform transition-all duration-300 ${
        isVisible ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0'
      }`}
      role="alert"
    >
      <div className="p-4 flex items-start">
        <div className="flex-shrink-0 mr-3">{icons[type]}</div>
        <div className="flex-1 mr-2">
          <p className="text-sm font-medium text-gray-800 dark:text-gray-200">{message}</p>
        </div>
        <button
          type="button"
          className="text-gray-400 hover:text-gray-500 focus:outline-none dark:text-gray-500 dark:hover:text-gray-400"
          onClick={handleClose}
          aria-label="关闭"
        >
          <FaTimes className="h-4 w-4" />
        </button>
      </div>

      {/* 进度条 */}
      {duration > 0 && (
        <div className="w-full h-1 bg-gray-200 dark:bg-gray-700">
          <div
            className={`h-full ${progressColors[type]} transition-all duration-100 ease-linear`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
};

/**
 * Toast容器组件
 */
export const ToastContainer = ({ toasts, removeToast }) => {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  // 服务器端渲染时，不挂载组件
  if (!isMounted || typeof window === 'undefined') {
    return null;
  }

  // 使用Portal渲染到body
  return ReactDOM.createPortal(
    <div className="fixed inset-0 z-50 flex flex-col items-end justify-start pointer-events-none p-4 space-y-4 overflow-hidden">
      {toasts.map(toast => (
        <Toast
          key={toast.id}
          id={toast.id}
          message={toast.message}
          type={toast.type}
          duration={toast.duration}
          onClose={removeToast}
        />
      ))}
    </div>,
    document.body
  );
};

/**
 * 创建并管理Toast的钩子函数
 *
 * @returns {Object} Toast管理对象
 */
export const useToast = () => {
  const [toasts, setToasts] = useState([]);

  // 添加Toast
  const addToast = (message, type = ToastTypes.INFO, duration = 5000) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setToasts(prevToasts => [...prevToasts, { id, message, type, duration }]);
    return id;
  };

  // 移除Toast
  const removeToast = id => {
    setToasts(prevToasts => prevToasts.filter(toast => toast.id !== id));
  };

  // 快捷方法
  const success = (message, duration) => addToast(message, ToastTypes.SUCCESS, duration);
  const error = (message, duration) => addToast(message, ToastTypes.ERROR, duration);
  const warning = (message, duration) => addToast(message, ToastTypes.WARNING, duration);
  const info = (message, duration) => addToast(message, ToastTypes.INFO, duration);
  const clear = () => setToasts([]);

  return {
    toasts,
    addToast,
    removeToast,
    success,
    error,
    warning,
    info,
    clear,
  };
};

export default Toast;
