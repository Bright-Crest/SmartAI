/**
 * API客户端
 * 用于处理与后端API的通信
 */

import { ERROR_MESSAGES } from './constants';

// 获取API基础URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3999/api';

/**
 * 通用请求函数
 * @param {string} endpoint - API端点
 * @param {Object} options - 请求选项
 * @returns {Promise<any>} - 响应数据
 */
export async function fetchAPI(endpoint, options = {}) {
  const defaultHeaders = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  };

  const config = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  try {
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, config);

    // 检查响应状态
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `请求失败: ${response.status} ${response.statusText}`);
    }

    // 解析JSON响应
    return await response.json();
  } catch (error) {
    console.error('API请求错误:', error);

    // 处理不同类型的错误
    if (error.name === 'AbortError') {
      throw new Error('请求已取消');
    }

    if (error.message === 'Failed to fetch') {
      throw new Error(ERROR_MESSAGES.NETWORK_ERROR);
    }

    throw error;
  }
}

/**
 * GET请求
 * @param {string} endpoint - API端点
 * @param {Object} params - URL参数
 * @param {Object} options - 请求选项
 * @returns {Promise<any>} - 响应数据
 */
export async function get(endpoint, params = {}, options = {}) {
  // 构建URL参数
  const queryParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      queryParams.append(key, value);
    }
  });

  const queryString = queryParams.toString();
  const url = queryString ? `${endpoint}?${queryString}` : endpoint;

  return fetchAPI(url, { method: 'GET', ...options });
}

/**
 * POST请求
 * @param {string} endpoint - API端点
 * @param {Object} data - 请求体数据
 * @param {Object} options - 请求选项
 * @returns {Promise<any>} - 响应数据
 */
export async function post(endpoint, data = {}, options = {}) {
  return fetchAPI(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
    ...options,
  });
}

/**
 * PUT请求
 * @param {string} endpoint - API端点
 * @param {Object} data - 请求体数据
 * @param {Object} options - 请求选项
 * @returns {Promise<any>} - 响应数据
 */
export async function put(endpoint, data = {}, options = {}) {
  return fetchAPI(endpoint, {
    method: 'PUT',
    body: JSON.stringify(data),
    ...options,
  });
}

/**
 * DELETE请求
 * @param {string} endpoint - API端点
 * @param {Object} options - 请求选项
 * @returns {Promise<any>} - 响应数据
 */
export async function del(endpoint, options = {}) {
  return fetchAPI(endpoint, {
    method: 'DELETE',
    ...options,
  });
}

/**
 * 带有abortController的可取消请求
 * @param {Function} fetchFn - 请求函数
 * @param {Array} args - 请求函数的参数
 * @returns {Object} - 请求结果和取消函数
 */
export function createCancellableRequest(fetchFn, ...args) {
  const abortController = new AbortController();
  const signal = abortController.signal;

  // 添加signal到最后一个参数(options)
  const lastArgIndex = args.length - 1;
  if (lastArgIndex >= 0) {
    if (typeof args[lastArgIndex] === 'object') {
      args[lastArgIndex] = { ...args[lastArgIndex], signal };
    } else {
      args.push({ signal });
    }
  } else {
    args.push({ signal });
  }

  const promise = fetchFn(...args);

  return {
    promise,
    cancel: () => abortController.abort(),
  };
}

// 默认导出所有API方法
export default {
  get,
  post,
  put,
  delete: del,
  fetchAPI,
  createCancellableRequest,
};
