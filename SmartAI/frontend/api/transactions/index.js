/**
 * 交易数据API接口
 */

// 导入axios或自定义的API客户端
import axios from 'axios';

// 导入示例数据
import { sampleTransactions } from '../../types/transaction';

// API基础URL，可以通过环境变量设置
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api';

/**
 * 获取交易列表
 * @param {Object} params - 查询参数
 * @param {number} [params.page=1] - 页码
 * @param {number} [params.pageSize=10] - 每页数量
 * @param {string} [params.fromAddress] - 发送方地址过滤
 * @param {string} [params.toAddress] - 接收方地址过滤
 * @param {string} [params.currency] - 币种过滤
 * @param {string} [params.status] - 状态过滤
 * @param {number} [params.minAmount] - 最小金额
 * @param {number} [params.maxAmount] - 最大金额
 * @param {number} [params.startTime] - 开始时间戳
 * @param {number} [params.endTime] - 结束时间戳
 * @param {string} [params.sortBy='timestamp'] - 排序字段
 * @param {string} [params.sortOrder='desc'] - 排序方向 (asc/desc)
 * @param {boolean} [params.useMockData=false] - 是否使用模拟数据
 * @returns {Promise<import('../../types/transaction').TransactionListResponse>} 交易列表响应
 */
export async function fetchTransactions(params = {}) {
  const { page = 1, pageSize = 10, useMockData = false, ...filters } = params;

  // 开发环境可使用模拟数据
  if (useMockData || process.env.NODE_ENV === 'development') {
    // 模拟API延迟
    await new Promise(resolve => setTimeout(resolve, 600));

    // 过滤交易
    let filteredTxs = [...sampleTransactions];

    // 应用过滤条件
    if (filters.fromAddress) {
      filteredTxs = filteredTxs.filter(tx =>
        tx.fromAddress.toLowerCase().includes(filters.fromAddress.toLowerCase())
      );
    }

    if (filters.toAddress) {
      filteredTxs = filteredTxs.filter(tx =>
        tx.toAddress.toLowerCase().includes(filters.toAddress.toLowerCase())
      );
    }

    if (filters.currency) {
      filteredTxs = filteredTxs.filter(tx => tx.currency === filters.currency);
    }

    if (filters.status) {
      filteredTxs = filteredTxs.filter(tx => tx.status === filters.status);
    }

    if (filters.minAmount !== undefined) {
      filteredTxs = filteredTxs.filter(tx => tx.amount >= filters.minAmount);
    }

    if (filters.maxAmount !== undefined) {
      filteredTxs = filteredTxs.filter(tx => tx.amount <= filters.maxAmount);
    }

    if (filters.startTime !== undefined) {
      filteredTxs = filteredTxs.filter(tx => tx.timestamp >= filters.startTime);
    }

    if (filters.endTime !== undefined) {
      filteredTxs = filteredTxs.filter(tx => tx.timestamp <= filters.endTime);
    }

    // 应用排序
    const sortBy = filters.sortBy || 'timestamp';
    const sortOrder = filters.sortOrder || 'desc';

    filteredTxs.sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    // 应用分页
    const total = filteredTxs.length;
    const startIndex = (page - 1) * pageSize;
    const paginatedTxs = filteredTxs.slice(startIndex, startIndex + pageSize);

    return {
      transactions: paginatedTxs,
      total,
      page,
      pageSize,
      totalPages: Math.ceil(total / pageSize),
    };
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/transactions`, {
      params: { page, pageSize, ...filters },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching transactions:', error);
    throw new Error(error.response?.data?.message || '获取交易数据失败');
  }
}

/**
 * 获取交易详情
 * @param {string} txId - 交易ID
 * @param {boolean} [useMockData=false] - 是否使用模拟数据
 * @returns {Promise<import('../../types/transaction').Transaction>} 交易详情
 */
export async function fetchTransactionDetails(txId, useMockData = false) {
  if (!txId) {
    throw new Error('交易ID不能为空');
  }

  // 开发环境可使用模拟数据
  if (useMockData || process.env.NODE_ENV === 'development') {
    // 模拟API延迟
    await new Promise(resolve => setTimeout(resolve, 500));

    // 查找交易
    const tx = sampleTransactions.find(t => t.id === txId);
    if (!tx) {
      throw new Error('找不到指定交易');
    }

    // 为模拟数据添加更多详细信息
    return {
      ...tx,
      confirmations: tx.status === 'confirmed' ? Math.floor(Math.random() * 100) + 1 : 0,
      gasPrice: tx.currency === 'ETH' ? Math.random() * 100 + 10 : undefined,
      gasLimit: tx.currency === 'ETH' ? Math.floor(Math.random() * 21000) + 21000 : undefined,
      nonce: Math.floor(Math.random() * 1000),
      input:
        '0x' +
        Array(10)
          .fill(0)
          .map(() => Math.floor(Math.random() * 16).toString(16))
          .join(''),
    };
  }

  try {
    const response = await axios.get(`${API_BASE_URL}/transactions/${txId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching transaction details for ${txId}:`, error);
    throw new Error(error.response?.data?.message || '获取交易详情失败');
  }
}

/**
 * 获取地址相关的交易
 * @param {string} address - 钱包地址
 * @param {Object} [params] - 其他查询参数
 * @param {number} [params.page=1] - 页码
 * @param {number} [params.pageSize=10] - 每页数量
 * @param {boolean} [params.useMockData=false] - 是否使用模拟数据
 * @returns {Promise<import('../../types/transaction').TransactionListResponse>} 交易列表响应
 */
export async function fetchAddressTransactions(address, params = {}) {
  if (!address) {
    throw new Error('地址不能为空');
  }

  return fetchTransactions({
    ...params,
    fromAddress: address,
    toAddress: address,
    useAddressQuery: true, // 特殊标志，表示这是一个OR查询（来自地址或发送到地址）
  });
}

/**
 * 交易分析API客户端
 */
import { request, utils } from '../index';

// API基础路径
const BASE_PATH = '/transactions';

/**
 * 获取交易列表
 *
 * @param {Object} options - 查询选项
 * @param {string} [options.address] - 过滤特定地址相关的交易
 * @param {string} [options.riskLevel] - 过滤风险级别 (high, medium, low)
 * @param {number} [options.limit=20] - 返回的交易数量限制
 * @param {number} [options.offset=0] - 分页偏移量
 * @param {string} [options.sortBy='timestamp'] - 排序字段
 * @param {string} [options.sortDirection='desc'] - 排序方向 (asc, desc)
 * @returns {Promise<Object>} 包含交易列表和分页信息
 */
export async function getTransactions(options = {}) {
  try {
    const response = await request(BASE_PATH, { params: options });
    return response;
  } catch (error) {
    console.error('获取交易列表失败:', error);
    throw error;
  }
}

/**
 * 获取交易详情
 *
 * @param {string} txHash - 交易哈希
 * @param {boolean} [includeRelated=false] - 是否包含相关交易
 * @returns {Promise<Object>} 交易详情
 */
export async function getTransactionDetails(txHash, includeRelated = false) {
  try {
    const response = await request(`${BASE_PATH}/${txHash}`, {
      params: { includeRelated },
    });
    return response;
  } catch (error) {
    console.error(`获取交易详情失败 (${txHash}):`, error);
    throw error;
  }
}

/**
 * 获取异常交易列表
 *
 * @param {Object} options - 查询选项
 * @param {number} [options.limit=10] - 返回的交易数量限制
 * @param {number} [options.timeRange='24h'] - 时间范围 (1h, 24h, 7d, 30d)
 * @returns {Promise<Array>} 异常交易列表
 */
export async function getAnomalousTransactions(options = {}) {
  try {
    const response = await request(`${BASE_PATH}/anomalies`, { params: options });
    return response;
  } catch (error) {
    console.error('获取异常交易失败:', error);
    throw error;
  }
}

/**
 * 获取交易统计信息
 *
 * @param {Object} [filters] - 过滤条件
 * @param {string} [filters.timeRange='24h'] - 时间范围 (1h, 24h, 7d, 30d, all)
 * @returns {Promise<Object>} 交易统计信息
 */
export async function getTransactionStats(filters = {}) {
  try {
    const response = await request(`${BASE_PATH}/stats`, { params: filters });
    return response;
  } catch (error) {
    console.error('获取交易统计信息失败:', error);
    throw error;
  }
}

/**
 * 搜索交易
 *
 * @param {string} query - 搜索关键词（交易哈希、地址等）
 * @param {number} [limit=10] - 返回结果数量限制
 * @returns {Promise<Array>} 搜索结果
 */
export async function searchTransactions(query, limit = 10) {
  try {
    const response = await request(`${BASE_PATH}/search`, {
      params: { query, limit },
    });
    return response;
  } catch (error) {
    console.error('搜索交易失败:', error);
    throw error;
  }
}

// 导出单个函数和整体对象
export default {
  getTransactions,
  getTransactionDetails,
  getAnomalousTransactions,
  getTransactionStats,
  searchTransactions,
};
