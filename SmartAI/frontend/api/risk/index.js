/**
 * 区块链风险分析API客户端
 */
import { request, utils } from '../index';
import adapter from './adapter';

// API配置
const BASE_PATH = '/risk';
const API_CONFIG = {
  // 从环境变量获取API URL，如果不存在则使用默认值
  fastapi: {
    baseUrl: process.env.NEXT_PUBLIC_RISK_API_URL || 'http://localhost:8002',
    path: BASE_PATH,
  },
  express: {
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000',
    path: '/api/risk',
  },
};

// 默认服务类型
let defaultServiceType = 'express';

/**
 * 设置默认服务类型
 * @param {string} serviceType - 'express' 或 'fastapi'
 */
export function setDefaultServiceType(serviceType) {
  if (['express', 'fastapi'].includes(serviceType)) {
    defaultServiceType = serviceType;
  } else {
    console.warn(`无效的服务类型: ${serviceType}，将使用默认值: ${defaultServiceType}`);
  }
}

/**
 * 获取API URL
 * @param {string} endpoint - API端点
 * @param {string} serviceType - 服务类型
 * @returns {string} 完整的API URL
 */
function getApiUrl(endpoint, serviceType = defaultServiceType) {
  const config = API_CONFIG[serviceType];
  return `${config.baseUrl}${config.path}/${endpoint}`.replace(/\/\//g, '/');
}

/**
 * 获取地址风险评分
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 请求选项
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Object>} 风险评分和风险因素
 */
export async function getAddressRiskScore(address, options = {}) {
  try {
    const { serviceType = defaultServiceType } = options;
    const apiUrl = getApiUrl(`score/${address}`, serviceType);

    console.log(`获取地址风险评分, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 使用适配器处理响应
    return adapter.adaptRiskScoreResponse(response);
  } catch (error) {
    console.error(`获取地址风险评分失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址风险指标数据
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 请求选项
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Object>} 维度指标数据
 */
export async function getAddressMetrics(address, options = {}) {
  try {
    // 维度指标目前只支持Express后端
    const serviceType = 'express';
    const apiUrl = getApiUrl(`metrics/${address}`, serviceType);

    console.log(`获取地址维度指标, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 使用适配器处理响应
    return adapter.adaptRiskMetricsResponse(response);
  } catch (error) {
    console.error(`获取地址维度指标失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址详情
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 请求选项
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Object>} 地址详情数据
 */
export async function getAddressDetails(address, options = {}) {
  try {
    const { serviceType = defaultServiceType } = options;
    const apiUrl = getApiUrl(`address/${address}`, serviceType);

    console.log(`获取地址详情, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 使用适配器处理响应
    return adapter.adaptAddressDetailsResponse(response);
  } catch (error) {
    console.error(`获取地址详情失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址交易历史
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 请求选项
 * @param {number} [options.page=1] - 页码
 * @param {number} [options.pageSize=20] - 每页数量
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Object>} 交易历史数据
 */
export async function getAddressTransactions(address, options = {}) {
  try {
    const { page = 1, pageSize = 20, serviceType = defaultServiceType } = options;

    const apiUrl = getApiUrl(`transactions/${address}`, serviceType);

    console.log(`获取地址交易历史, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      params: { page, pageSize },
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 使用适配器处理响应
    return adapter.adaptTransactionsResponse(response);
  } catch (error) {
    console.error(`获取地址交易历史失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 批量获取地址风险评分
 *
 * @param {string[]} addresses - 区块链地址数组
 * @param {Object} options - 请求选项
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Object>} 批量风险评分结果
 */
export async function getBatchRiskScores(addresses, options = {}) {
  try {
    const { serviceType = defaultServiceType } = options;
    const apiUrl = getApiUrl('batch', serviceType);

    console.log(`批量获取地址风险评分, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      method: 'POST',
      data: { addresses },
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 从响应中提取results数组并适配每个结果
    if (response.success && Array.isArray(response.data.results)) {
      return {
        results: response.data.results.map(adapter.adaptRiskScoreResponse),
        count: response.data.count,
        processedAt: response.data.processedAt,
      };
    } else if (response.results && Array.isArray(response.results)) {
      return {
        results: response.results.map(adapter.adaptRiskScoreResponse),
        count: response.count,
        processedAt: response.processedAt,
      };
    }

    return response;
  } catch (error) {
    console.error(`批量获取地址风险评分失败:`, error);
    throw error;
  }
}

/**
 * 获取地址相似地址
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 请求选项
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Array>} 相似地址列表
 */
export async function getSimilarAddresses(address, options = {}) {
  try {
    const { serviceType = defaultServiceType } = options;
    const apiUrl = getApiUrl(`similar/${address}`, serviceType);

    console.log(`获取相似地址, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 从响应中提取数据
    if (response.success && response.data) {
      return response.data;
    }

    return response;
  } catch (error) {
    console.error(`获取相似地址失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址标签
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 请求选项
 * @param {string} [options.serviceType] - 使用的服务类型，'express' 或 'fastapi'
 * @returns {Promise<Array>} 地址标签列表
 */
export async function getAddressTags(address, options = {}) {
  try {
    const { serviceType = defaultServiceType } = options;
    const apiUrl = getApiUrl(`tags/${address}`, serviceType);

    console.log(`获取地址标签, 使用${serviceType}服务: ${apiUrl}`);
    const response = await request(apiUrl, {
      headers: {
        Origin: typeof window !== 'undefined' ? window.location.origin : '',
      },
    });

    // 使用适配器处理响应
    return adapter.adaptAddressTagsResponse(response);
  } catch (error) {
    console.error(`获取地址标签失败 (${address}):`, error);
    throw error;
  }
}

// 导出API方法集合
export default {
  getAddressRiskScore,
  getAddressMetrics,
  getAddressDetails,
  getAddressTransactions,
  getBatchRiskScores,
  getSimilarAddresses,
  getAddressTags,
  setDefaultServiceType,
};
