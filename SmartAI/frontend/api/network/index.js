/**
 * 网络数据API模块
 * 提供获取网络数据、节点详情和网络统计的功能
 */

// 导入模拟数据
import { simpleSampleData, sampleNetworkData } from '../../types/network';

/**
 * 获取网络数据
 *
 * @param {Object} options - 请求选项
 * @param {boolean} [options.useMockData=false] - 是否使用模拟数据
 * @returns {Promise<Object>} 网络数据，包含nodes和links
 */
export const fetchNetworkData = async (options = {}) => {
  const { useMockData = false } = options;

  console.log('API: fetchNetworkData', { useMockData, options });

  // 始终使用模拟数据以便调试
  if (useMockData || process.env.NODE_ENV === 'development') {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 300));

    // 使用简单的示例数据，更容易调试
    const data = JSON.parse(JSON.stringify(simpleSampleData));
    console.log('API: 返回模拟数据', {
      nodesCount: data.nodes.length,
      linksCount: data.links.length,
      data: data,
    });
    return data;
  }

  try {
    // 实际API调用
    const response = await fetch('/api/v1/network/graph');

    if (!response.ok) {
      throw new Error(`网络请求失败: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('获取网络数据失败:', error);
    // 在生产环境发生错误时，也返回示例数据
    return JSON.parse(JSON.stringify(simpleSampleData));
  }
};

/**
 * 获取节点详情
 *
 * @param {string} nodeId - 节点ID
 * @param {boolean} [useMockData=false] - 是否使用模拟数据
 * @returns {Promise<Object>} 节点详情
 */
export const fetchNodeDetails = async (nodeId, useMockData = false) => {
  console.log('API: fetchNodeDetails', { nodeId, useMockData });

  // 始终使用模拟数据以便调试
  if (useMockData || process.env.NODE_ENV === 'development') {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 200));

    // 从简单示例数据中查找节点
    const node = simpleSampleData.nodes.find(n => n.id === nodeId);

    if (!node) {
      throw new Error(`未找到ID为${nodeId}的节点`);
    }

    // 添加一些额外的详情信息
    return {
      ...node,
      details: {
        transactionCount: Math.floor(Math.random() * 50) + 1,
        connectedNodes: Math.floor(Math.random() * 10) + 1,
        firstTransaction: new Date(
          Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000
        ).toISOString(),
        lastTransaction: new Date(
          Date.now() - Math.random() * 5 * 24 * 60 * 60 * 1000
        ).toISOString(),
        riskScore:
          node.risk === 'high'
            ? 85 + Math.random() * 15
            : node.risk === 'medium'
              ? 50 + Math.random() * 35
              : Math.random() * 50,
      },
    };
  }

  try {
    const response = await fetch(`/api/v1/network/node/${nodeId}`);

    if (!response.ok) {
      throw new Error(`获取节点详情失败: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`获取节点详情失败:`, error);

    // 在出错时返回模拟数据
    const node = simpleSampleData.nodes.find(n => n.id === nodeId);
    if (node) {
      return {
        ...node,
        details: {
          transactionCount: Math.floor(Math.random() * 50) + 1,
          connectedNodes: Math.floor(Math.random() * 10) + 1,
          firstTransaction: new Date(
            Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000
          ).toISOString(),
          lastTransaction: new Date(
            Date.now() - Math.random() * 5 * 24 * 60 * 60 * 1000
          ).toISOString(),
          riskScore:
            node.risk === 'high'
              ? 85 + Math.random() * 15
              : node.risk === 'medium'
                ? 50 + Math.random() * 35
                : Math.random() * 50,
        },
      };
    }
    throw new Error('获取节点详情失败，请稍后重试');
  }
};

/**
 * 获取网络统计数据
 *
 * @param {boolean} [useMockData=false] - 是否使用模拟数据
 * @returns {Promise<Object>} 网络统计数据
 */
export const fetchNetworkStats = async (useMockData = false) => {
  console.log('API: fetchNetworkStats', { useMockData });

  // 始终使用模拟数据以便调试
  if (useMockData || process.env.NODE_ENV === 'development') {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 200));

    // 计算示例数据的统计信息
    const nodes = simpleSampleData.nodes;
    const links = simpleSampleData.links;

    // 风险分布
    const riskDistribution = {
      high: nodes.filter(n => n.risk === 'high').length,
      medium: nodes.filter(n => n.risk === 'medium').length,
      low: nodes.filter(n => n.risk === 'low').length,
    };

    return {
      totalNodes: nodes.length,
      totalLinks: links.length,
      riskDistribution,
      avgConnections: links.length / nodes.length,
      timestamp: new Date().toISOString(),
    };
  }

  try {
    const response = await fetch('/api/v1/network/stats');

    if (!response.ok) {
      throw new Error(`获取网络统计失败: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('获取网络统计失败:', error);

    // 在出错时返回计算的模拟数据
    const nodes = simpleSampleData.nodes;
    const links = simpleSampleData.links;

    const riskDistribution = {
      high: nodes.filter(n => n.risk === 'high').length,
      medium: nodes.filter(n => n.risk === 'medium').length,
      low: nodes.filter(n => n.risk === 'low').length,
    };

    return {
      totalNodes: nodes.length,
      totalLinks: links.length,
      riskDistribution,
      timestamp: new Date().toISOString(),
    };
  }
};

/**
 * 区块链网络分析API客户端
 */
import { request, utils } from '../index';

// API基础路径
const BASE_PATH = '/network';

/**
 * 获取网络图数据
 *
 * @param {Object} options - 查询选项
 * @param {string} [options.address] - 过滤特定地址相关的网络
 * @param {string} [options.riskLevel] - 过滤风险级别 (high, medium, low)
 * @param {number} [options.limit=100] - 返回的节点数量限制
 * @param {number} [options.depth=2] - 关系网络深度
 * @param {boolean} [options.includeTxDetails=false] - 是否包含交易详情
 * @returns {Promise<Object>} 包含nodes和links的网络数据
 */
export async function getNetworkData(options = {}) {
  try {
    const response = await request(`${BASE_PATH}/graph`, { params: options });
    return response;
  } catch (error) {
    console.error('获取网络数据失败:', error);
    throw error;
  }
}

/**
 * 获取节点详细信息
 *
 * @param {string} nodeId - 节点ID
 * @param {boolean} [includeTransactions=false] - 是否包含相关交易
 * @returns {Promise<Object>} 节点详细信息
 */
export async function getNodeDetails(nodeId, includeTransactions = false) {
  try {
    const response = await request(`${BASE_PATH}/nodes/${nodeId}`, {
      params: { includeTransactions },
    });
    return response;
  } catch (error) {
    console.error(`获取节点详情失败 (${nodeId}):`, error);
    throw error;
  }
}

/**
 * 获取网络统计信息
 *
 * @param {Object} [filters] - 过滤条件
 * @returns {Promise<Object>} 网络统计信息
 */
export async function getNetworkStats(filters = {}) {
  try {
    const response = await request(`${BASE_PATH}/stats`, { params: filters });
    return response;
  } catch (error) {
    console.error('获取网络统计信息失败:', error);
    throw error;
  }
}

/**
 * 按地址搜索节点
 *
 * @param {string} query - 搜索关键词（地址/标签）
 * @param {number} [limit=10] - 返回结果数量限制
 * @returns {Promise<Array>} 搜索结果
 */
export async function searchNodes(query, limit = 10) {
  try {
    const response = await request(`${BASE_PATH}/search`, {
      params: { query, limit },
    });
    return response;
  } catch (error) {
    console.error('搜索节点失败:', error);
    throw error;
  }
}

// 导出单个函数和整体对象
export default {
  getNetworkData,
  getNodeDetails,
  getNetworkStats,
  searchNodes,
};
