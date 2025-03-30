/**
 * 地址分析API客户端
 */
import { request, utils } from '../index';

// API基础路径
const BASE_PATH = '/addresses';

/**
 * 获取地址详情
 *
 * @param {string} address - 区块链地址
 * @param {boolean} [includeTransactions=false] - 是否包含相关交易
 * @returns {Promise<Object>} 地址详情
 */
export async function getAddressDetails(address, includeTransactions = false) {
  try {
    const response = await request(`${BASE_PATH}/${address}`, {
      params: { includeTransactions },
    });
    return response;
  } catch (error) {
    console.error(`获取地址详情失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址交易记录
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 查询选项
 * @param {number} [options.limit=20] - 返回的交易数量限制
 * @param {number} [options.offset=0] - 分页偏移量
 * @param {string} [options.sortBy='timestamp'] - 排序字段
 * @param {string} [options.sortDirection='desc'] - 排序方向 (asc, desc)
 * @returns {Promise<Object>} 包含交易列表和分页信息
 */
export async function getAddressTransactions(address, options = {}) {
  try {
    const response = await request(`${BASE_PATH}/${address}/transactions`, {
      params: options,
    });
    return response;
  } catch (error) {
    console.error(`获取地址交易记录失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址相关网络
 *
 * @param {string} address - 区块链地址
 * @param {Object} options - 查询选项
 * @param {number} [options.depth=1] - 关系网络深度
 * @param {number} [options.limit=50] - 返回的节点数量限制
 * @returns {Promise<Object>} 包含nodes和links的网络数据
 */
export async function getAddressNetwork(address, options = {}) {
  try {
    const response = await request(`${BASE_PATH}/${address}/network`, {
      params: options,
    });
    return response;
  } catch (error) {
    console.error(`获取地址网络关系失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取地址风险评分
 *
 * @param {string} address - 区块链地址
 * @returns {Promise<Object>} 风险评分和风险因素
 */
export async function getAddressRiskScore(address) {
  try {
    const response = await request(`${BASE_PATH}/${address}/risk`);
    return response;
  } catch (error) {
    console.error(`获取地址风险评分失败 (${address}):`, error);
    throw error;
  }
}

/**
 * 获取高风险地址列表
 *
 * @param {Object} options - 查询选项
 * @param {number} [options.limit=20] - 返回的地址数量限制
 * @param {number} [options.offset=0] - 分页偏移量
 * @returns {Promise<Object>} 包含地址列表和分页信息
 */
export async function getHighRiskAddresses(options = {}) {
  try {
    const response = await request(`${BASE_PATH}/high-risk`, {
      params: options,
    });
    return response;
  } catch (error) {
    console.error('获取高风险地址列表失败:', error);
    throw error;
  }
}

/**
 * 搜索地址
 *
 * @param {string} query - 搜索关键词（地址、标签等）
 * @param {number} [limit=10] - 返回结果数量限制
 * @returns {Promise<Array>} 搜索结果
 */
export async function searchAddresses(query, limit = 10) {
  try {
    const response = await request(`${BASE_PATH}/search`, {
      params: { query, limit },
    });
    return response;
  } catch (error) {
    console.error('搜索地址失败:', error);
    throw error;
  }
}

// 导出单个函数和整体对象
export default {
  getAddressDetails,
  getAddressTransactions,
  getAddressNetwork,
  getAddressRiskScore,
  getHighRiskAddresses,
  searchAddresses,
};
