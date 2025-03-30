/**
 * API客户端工具
 * 提供所有API请求的基础功能和辅助函数
 */

// API基础URL，根据环境不同自动选择
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api';

// 导入拦截器
import interceptApiRequest from '../utils/apiMockInterceptor';

/**
 * 通用请求错误类
 */
export class ApiError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/**
 * 通用API请求函数
 *
 * @param {string} endpoint - API端点路径
 * @param {Object} options - 请求选项
 * @returns {Promise<any>} - 请求结果
 * @throws {ApiError} 当请求失败时抛出
 */
export async function request(endpoint, options = {}) {
  // 尝试使用模拟拦截器
  try {
    const mockResponse = await interceptApiRequest(endpoint, options);
    if (mockResponse !== null) {
      console.log(`[API] 使用模拟数据: ${endpoint}`);
      return mockResponse;
    }
  } catch (mockError) {
    console.error('[API] 模拟数据错误:', mockError);
    throw mockError;
  }

  const { method = 'GET', params = {}, data = null, headers = {}, timeout = 30000 } = options;

  // 判断是否使用完整URL或基础URL
  const isFullUrl = endpoint.startsWith('http://') || endpoint.startsWith('https://');
  const baseUrl = isFullUrl ? '' : API_BASE_URL;

  // 构建完整URL
  const url = new URL(
    endpoint.startsWith('/') && !isFullUrl ? endpoint : `/${endpoint}`.replace('//', '/'),
    isFullUrl ? endpoint : baseUrl
  );

  // 添加查询参数
  if (params && Object.keys(params).length > 0) {
    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== null) {
        url.searchParams.append(key, params[key]);
      }
    });
  }

  // 请求配置
  const config = {
    method,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...headers,
    },
    credentials: isFullUrl ? 'omit' : 'include', // 外部API不发送cookie
    mode: isFullUrl ? 'cors' : 'same-origin', // 外部API使用CORS
  };

  // 添加请求体
  if (data && ['POST', 'PUT', 'PATCH'].includes(method)) {
    config.body = JSON.stringify(data);
  }

  try {
    const finalUrl = isFullUrl ? endpoint : url.toString();
    console.log(`[API] 发送请求: ${method} ${finalUrl}`);

    const response = await fetch(finalUrl, config);

    // 检查响应状态
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      throw new ApiError(
        errorData?.message || errorData?.detail || `请求失败: ${response.status}`,
        response.status,
        errorData
      );
    }

    // 解析响应
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      return await response.json();
    } else {
      return await response.text();
    }
  } catch (error) {
    // 如果已经是ApiError则直接抛出
    if (error instanceof ApiError) {
      throw error;
    }

    // 其他网络错误
    throw new ApiError(`网络请求失败: ${error.message}`, 0, { originalError: error });
  }
}

// 导出通用工具
export const utils = {
  /**
   * 检查请求是否失败
   */
  async handleRequest(promise) {
    try {
      const data = await promise;
      return [null, data];
    } catch (error) {
      return [error, null];
    }
  },

  /**
   * 获取错误消息
   */
  getErrorMessage(error) {
    if (error instanceof ApiError) {
      return error.message;
    }
    return error?.message || '未知错误';
  },
};
