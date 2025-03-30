/**
 * API工具
 * 提供HTTP请求和缓存的统一封装
 */

import axios from 'axios';
import { memCache, storageCache, withCache } from './cache';
import { handleApiError } from './errorHandler';

// 创建基础axios实例
const createApiClient = baseURL => {
  const apiClient = axios.create({
    baseURL,
    timeout: 15000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // 响应拦截器
  apiClient.interceptors.response.use(
    response => response.data,
    error => {
      // 抛出处理后的API错误
      throw handleApiError(error);
    }
  );

  return apiClient;
};

// 创建API客户端实例
const apiClient = createApiClient(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api');
const blockchainApiClient = createApiClient(
  process.env.NEXT_PUBLIC_BLOCKCHAIN_API_URL || 'http://localhost:8001'
);
const riskApiClient = createApiClient(
  process.env.NEXT_PUBLIC_RISK_API_URL || 'http://localhost:8002'
);

/**
 * 构建缓存键
 * @param {string} endpoint - API端点
 * @param {Object} params - 查询参数
 * @returns {string} 缓存键
 */
const buildCacheKey = (endpoint, params) => {
  const sortedParams = params ? JSON.stringify(Object.entries(params).sort()) : '{}';
  return `${endpoint}:${sortedParams}`;
};

/**
 * 清除指定端点的缓存
 * @param {string} endpoint - API端点
 */
export const clearCache = endpoint => {
  // 为了安全起见，我们只清除具有指定前缀的缓存
  const memoryCacheKeys = Array.from(memoryCache.keys()).filter(key => key.startsWith(endpoint));

  memoryCacheKeys.forEach(key => memCache.delete(key));

  // 如果在浏览器环境，也清除localStorage中的缓存
  if (typeof window !== 'undefined') {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(`cache_${endpoint}`)) {
        storageCache.delete(key.replace('cache_', ''));
      }
    }
  }
};

/**
 * 执行带缓存的GET请求
 * @param {Object} client - API客户端
 * @param {string} endpoint - API端点
 * @param {Object} options - 选项
 * @returns {Promise<any>} 响应数据
 */
const cachedGet = (client, endpoint, options = {}) => {
  const {
    params = {},
    useCache = true,
    ttl = 5 * 60 * 1000, // 默认5分钟缓存
    useStorage = false,
  } = options;

  // 如果不使用缓存，直接执行请求
  if (!useCache) {
    return client.get(endpoint, { params });
  }

  // 使用缓存包装请求
  const cacheKey = buildCacheKey(endpoint, params);
  const cache = useStorage ? storageCache : memCache;

  // 尝试从缓存获取
  const cachedData = cache.get(cacheKey);
  if (cachedData !== null) {
    return Promise.resolve(cachedData);
  }

  // 缓存未命中，执行请求
  return client.get(endpoint, { params }).then(data => {
    // 存入缓存
    cache.set(cacheKey, data, ttl);
    return data;
  });
};

/**
 * API工具对象
 */
const api = {
  /**
   * 执行GET请求
   * @param {string} endpoint - API端点
   * @param {Object} options - 选项
   * @returns {Promise<any>} 响应数据
   */
  get: (endpoint, options = {}) => cachedGet(apiClient, endpoint, options),

  /**
   * 执行POST请求
   * @param {string} endpoint - API端点
   * @param {Object} data - 请求数据
   * @param {Object} config - 额外配置
   * @returns {Promise<any>} 响应数据
   */
  post: (endpoint, data = {}, config = {}) => apiClient.post(endpoint, data, config),

  /**
   * 执行PUT请求
   * @param {string} endpoint - API端点
   * @param {Object} data - 请求数据
   * @param {Object} config - 额外配置
   * @returns {Promise<any>} 响应数据
   */
  put: (endpoint, data = {}, config = {}) => apiClient.put(endpoint, data, config),

  /**
   * 执行DELETE请求
   * @param {string} endpoint - API端点
   * @param {Object} config - 额外配置
   * @returns {Promise<any>} 响应数据
   */
  delete: (endpoint, config = {}) => apiClient.delete(endpoint, config),

  /**
   * 执行区块链API的GET请求
   * @param {string} endpoint - API端点
   * @param {Object} options - 选项
   * @returns {Promise<any>} 响应数据
   */
  blockchain: {
    get: (endpoint, options = {}) => cachedGet(blockchainApiClient, endpoint, options),
    post: (endpoint, data = {}, config = {}) => blockchainApiClient.post(endpoint, data, config),
  },

  /**
   * 执行风险分析API的GET请求
   * @param {string} endpoint - API端点
   * @param {Object} options - 选项
   * @returns {Promise<any>} 响应数据
   */
  risk: {
    get: (endpoint, options = {}) => cachedGet(riskApiClient, endpoint, options),
    post: (endpoint, data = {}, config = {}) => riskApiClient.post(endpoint, data, config),
  },

  /**
   * 清除指定端点的缓存
   * @param {string} endpoint - API端点
   */
  clearCache,
};

export default api;
