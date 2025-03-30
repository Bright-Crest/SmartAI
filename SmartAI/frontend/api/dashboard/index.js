/**
 * 仪表盘API客户端
 */
import { request, utils } from '../index';

// API基础路径
const BASE_PATH = '/dashboard';

/**
 * 获取总览数据
 *
 * @param {string} [timeRange='24h'] - 时间范围 (1h, 24h, 7d, 30d)
 * @returns {Promise<Object>} 仪表盘总览数据
 */
export async function getOverview(timeRange = '24h') {
  try {
    const response = await request(`${BASE_PATH}/overview`, {
      params: { timeRange },
    });
    return response;
  } catch (error) {
    console.error('获取仪表盘总览数据失败:', error);
    throw error;
  }
}

/**
 * 获取风险趋势数据
 *
 * @param {string} [timeRange='7d'] - 时间范围 (1d, 7d, 30d, 90d)
 * @param {string} [interval='day'] - 间隔 (hour, day, week, month)
 * @returns {Promise<Array>} 风险趋势数据
 */
export async function getRiskTrends(timeRange = '7d', interval = 'day') {
  try {
    const response = await request(`${BASE_PATH}/risk-trends`, {
      params: { timeRange, interval },
    });
    return response;
  } catch (error) {
    console.error('获取风险趋势数据失败:', error);
    throw error;
  }
}

/**
 * 获取热门地址
 *
 * @param {number} [limit=5] - 返回数量限制
 * @param {string} [sortBy='activity'] - 排序字段 (activity, value, risk)
 * @returns {Promise<Array>} 热门地址列表
 */
export async function getTopAddresses(limit = 5, sortBy = 'activity') {
  try {
    const response = await request(`${BASE_PATH}/top-addresses`, {
      params: { limit, sortBy },
    });
    return response;
  } catch (error) {
    console.error('获取热门地址失败:', error);
    throw error;
  }
}

/**
 * 获取热门交易
 *
 * @param {number} [limit=5] - 返回数量限制
 * @param {string} [sortBy='value'] - 排序字段 (value, time, risk)
 * @returns {Promise<Array>} 热门交易列表
 */
export async function getTopTransactions(limit = 5, sortBy = 'value') {
  try {
    const response = await request(`${BASE_PATH}/top-transactions`, {
      params: { limit, sortBy },
    });
    return response;
  } catch (error) {
    console.error('获取热门交易失败:', error);
    throw error;
  }
}

/**
 * 获取网络活动统计
 *
 * @param {string} [timeRange='24h'] - 时间范围 (1h, 24h, 7d, 30d)
 * @param {string} [interval='hour'] - 间隔 (minute, hour, day, week)
 * @returns {Promise<Object>} 网络活动统计数据
 */
export async function getNetworkActivity(timeRange = '24h', interval = 'hour') {
  try {
    const response = await request(`${BASE_PATH}/network-activity`, {
      params: { timeRange, interval },
    });
    return response;
  } catch (error) {
    console.error('获取网络活动统计失败:', error);
    throw error;
  }
}

// 导出单个函数和整体对象
export default {
  getOverview,
  getRiskTrends,
  getTopAddresses,
  getTopTransactions,
  getNetworkActivity,
};
