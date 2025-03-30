/**
 * 错误处理工具
 * 提供统一的错误处理和日志记录功能
 */

// 全局错误处理回调
let globalErrorCallback = null;

// 错误类型
export const ErrorTypes = {
  NETWORK: 'network',
  API: 'api',
  VALIDATION: 'validation',
  AUTHENTICATION: 'authentication',
  AUTHORIZATION: 'authorization',
  UNKNOWN: 'unknown',
  TIMEOUT: 'timeout',
  CACHE: 'cache',
};

/**
 * 自定义应用错误类
 */
export class AppError extends Error {
  /**
   * 构造函数
   * @param {string} message - 错误消息
   * @param {string} type - 错误类型
   * @param {any} originalError - 原始错误对象
   * @param {Object} metadata - 额外的元数据
   */
  constructor(message, type = ErrorTypes.UNKNOWN, originalError = null, metadata = {}) {
    super(message);
    this.name = 'AppError';
    this.type = type;
    this.originalError = originalError;
    this.metadata = metadata;
    this.timestamp = new Date().toISOString();
  }

  /**
   * 获取友好的错误消息
   */
  getFriendlyMessage() {
    switch (this.type) {
      case ErrorTypes.NETWORK:
        return '网络连接错误，请检查您的网络连接后重试';
      case ErrorTypes.API:
        return '服务器响应错误，请稍后重试';
      case ErrorTypes.VALIDATION:
        return '输入数据验证失败，请检查输入';
      case ErrorTypes.AUTHENTICATION:
        return '认证错误，请重新登录';
      case ErrorTypes.AUTHORIZATION:
        return '您没有权限执行此操作';
      case ErrorTypes.TIMEOUT:
        return '请求超时，请稍后重试';
      case ErrorTypes.CACHE:
        return '数据缓存错误';
      default:
        return this.message || '发生未知错误，请稍后重试';
    }
  }

  /**
   * 获取日志信息
   */
  getLogInfo() {
    return {
      message: this.message,
      type: this.type,
      timestamp: this.timestamp,
      stack: this.stack,
      originalError: this.originalError
        ? {
            message: this.originalError.message,
            stack: this.originalError.stack,
          }
        : null,
      metadata: this.metadata,
    };
  }
}

/**
 * 处理API响应错误
 * @param {Error} error - 错误对象
 * @param {Object} options - 选项
 * @returns {AppError} 应用错误对象
 */
export function handleApiError(error, options = {}) {
  // 检查是否是网络错误
  if (!error.response) {
    return new AppError('网络连接错误', ErrorTypes.NETWORK, error, { ...options, statusCode: 0 });
  }

  const { status, data } = error.response;
  let errorType = ErrorTypes.API;
  let message = '服务器响应错误';

  // 根据状态码确定错误类型
  if (status === 400) {
    errorType = ErrorTypes.VALIDATION;
    message = data?.message || '请求参数无效';
  } else if (status === 401) {
    errorType = ErrorTypes.AUTHENTICATION;
    message = '身份验证失败，请重新登录';
  } else if (status === 403) {
    errorType = ErrorTypes.AUTHORIZATION;
    message = '您没有权限执行此操作';
  } else if (status === 404) {
    message = '请求的资源不存在';
  } else if (status === 408 || status === 504) {
    errorType = ErrorTypes.TIMEOUT;
    message = '请求超时，请稍后重试';
  } else if (status >= 500) {
    message = '服务器内部错误，请稍后重试';
  }

  return new AppError(message, errorType, error, {
    ...options,
    statusCode: status,
    responseData: data,
  });
}

/**
 * 处理验证错误
 * @param {string} message - 错误消息
 * @param {Object} fields - 字段错误
 * @returns {AppError} 应用错误对象
 */
export function handleValidationError(message, fields = {}) {
  return new AppError(message || '输入数据验证失败', ErrorTypes.VALIDATION, null, { fields });
}

/**
 * 设置全局错误处理回调
 * @param {Function} callback - 错误处理回调函数
 */
export function setGlobalErrorHandler(callback) {
  if (typeof callback !== 'function') {
    throw new Error('错误处理回调必须是一个函数');
  }
  globalErrorCallback = callback;
}

/**
 * 捕获并处理错误
 * @param {Error} error - 错误对象
 * @param {Object} options - 选项
 * @returns {AppError} 处理后的错误对象
 */
export function captureError(error, options = {}) {
  // 转换为AppError
  const appError =
    error instanceof AppError
      ? error
      : new AppError(error.message || '发生未知错误', ErrorTypes.UNKNOWN, error, options);

  // 记录错误
  console.error('[错误捕获]', appError.getLogInfo());

  // 如果有全局处理器，调用它
  if (globalErrorCallback) {
    globalErrorCallback(appError);
  }

  return appError;
}

/**
 * 使用错误处理封装异步函数
 * @param {Function} fn - 要封装的异步函数
 * @param {Object} options - 错误处理选项
 * @returns {Function} 封装后的函数
 */
export function withErrorHandling(fn, options = {}) {
  return async (...args) => {
    try {
      return await fn(...args);
    } catch (error) {
      const appError = captureError(error, {
        functionName: fn.name,
        arguments: args,
        ...options,
      });

      if (options.throwError !== false) {
        throw appError;
      }

      return options.fallbackValue;
    }
  };
}

/**
 * 防抖函数
 * @param {Function} fn - 要防抖的函数
 * @param {number} wait - 等待时间（毫秒）
 * @returns {Function} 防抖后的函数
 */
export function debounce(fn, wait = 300) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn.apply(this, args), wait);
  };
}

export default {
  AppError,
  ErrorTypes,
  handleApiError,
  handleValidationError,
  setGlobalErrorHandler,
  captureError,
  withErrorHandling,
  debounce,
};
