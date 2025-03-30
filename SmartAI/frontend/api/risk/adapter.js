/**
 * 风险分析API适配器
 * 用于处理不同后端服务响应格式的统一化
 */

/**
 * 统一化风险分数响应格式
 * 将Express和FastAPI的不同响应格式转换为统一的前端数据结构
 *
 * @param {Object} response 原始响应数据
 * @returns {Object} 标准化的响应数据
 */
export function adaptRiskScoreResponse(response) {
  if (!response) return null;

  // 检查是否是Express响应格式
  if (response.success !== undefined && response.data) {
    // Express格式: { success: true, data: {...} }
    response = response.data;
  }

  // 标准化风险级别
  const riskLevel = response.riskLevel || response.risk_level || 'unknown';
  const extendedRiskLevel = response.extendedRiskLevel || response.extended_risk_level || null;

  // 标准化风险评分
  const score = response.score || response.risk_score || 0;

  // 标准化风险因素
  let factors = [];

  if (Array.isArray(response.factors)) {
    factors = response.factors;
  } else if (Array.isArray(response.risk_factors)) {
    factors = response.risk_factors;
  }

  // 构建统一的响应格式
  return {
    address: response.address,
    score: score,
    riskLevel: riskLevel,
    extendedRiskLevel: extendedRiskLevel,
    factors: factors,
    updatedAt: response.updatedAt || response.updated_at || new Date().toISOString(),
  };
}

/**
 * 统一化风险维度指标响应格式
 *
 * @param {Object} response 原始响应数据
 * @returns {Object} 标准化的响应数据
 */
export function adaptRiskMetricsResponse(response) {
  if (!response) return null;

  // 检查是否是Express响应格式
  if (response.success !== undefined && response.data) {
    // Express格式: { success: true, data: {...} }
    response = response.data;
  }

  // 标准化维度数据
  const dimensions = {};

  // 处理dimensions属性
  if (response.dimensions) {
    Object.keys(response.dimensions).forEach(dim => {
      dimensions[dim] = {
        score: response.dimensions[dim].score || 0,
        label: response.dimensions[dim].label || '',
        factors: response.dimensions[dim].factors || [],
      };
    });
  }

  // 构建统一的响应格式
  return {
    address: response.address,
    score: response.score || response.risk_score || 0,
    riskLevel: response.riskLevel || response.risk_level || 'unknown',
    extendedRiskLevel: response.extendedRiskLevel || response.extended_risk_level || null,
    dimensions: dimensions,
    updatedAt: response.updatedAt || response.updated_at || new Date().toISOString(),
  };
}

/**
 * 统一化地址详情响应格式
 *
 * @param {Object} response 原始响应数据
 * @returns {Object} 标准化的响应数据
 */
export function adaptAddressDetailsResponse(response) {
  if (!response) return null;

  // 检查是否是Express响应格式
  if (response.success !== undefined && response.data) {
    // Express格式: { success: true, data: {...} }
    response = response.data;
  }

  // 构建统一的响应格式
  return {
    address: response.address,
    balance: response.balance || {},
    transactions: response.transactions || {},
    tags: response.tags || [],
    contracts: response.contracts || {},
    tokens: response.tokens || {},
  };
}

/**
 * 统一化地址标签响应格式
 *
 * @param {Object} response 原始响应数据
 * @returns {Array} 标准化的标签数组
 */
export function adaptAddressTagsResponse(response) {
  if (!response) return [];

  // 检查是否是Express响应格式
  if (response.success !== undefined && response.data) {
    // Express格式: { success: true, data: {...} }
    response = response.data;
  }

  // 如果响应已经是数组，直接返回
  if (Array.isArray(response)) {
    return response;
  }

  // 如果响应是包含tags属性的对象
  if (response.tags && Array.isArray(response.tags)) {
    return response.tags;
  }

  return [];
}

/**
 * 统一化交易历史响应格式
 *
 * @param {Object} response 原始响应数据
 * @returns {Object} 标准化的响应数据
 */
export function adaptTransactionsResponse(response) {
  if (!response) return { items: [], pagination: { page: 1, totalPages: 0, totalItems: 0 } };

  // 检查是否是Express响应格式
  if (response.success !== undefined && response.data) {
    // Express格式: { success: true, data: {...} }
    response = response.data;
  }

  // 提取交易项
  let items = [];

  if (Array.isArray(response)) {
    items = response;
  } else if (response.items && Array.isArray(response.items)) {
    items = response.items;
  } else if (response.transactions && Array.isArray(response.transactions)) {
    items = response.transactions;
  }

  // 提取分页信息
  let pagination = {
    page: 1,
    totalPages: 1,
    totalItems: items.length,
  };

  if (response.pagination) {
    pagination = {
      ...pagination,
      ...response.pagination,
    };
  }

  return {
    items,
    pagination,
  };
}

/**
 * 检测API响应来源
 *
 * @param {Object} response 原始响应数据
 * @returns {string} 'express' 或 'fastapi'
 */
export function detectResponseSource(response) {
  if (!response) return 'unknown';

  // Express响应特征
  if (response.success !== undefined && response.data !== undefined) {
    return 'express';
  }

  // FastAPI响应特征 (基于命名约定)
  if (
    (response.risk_score !== undefined || response.risk_level !== undefined) &&
    !response.riskLevel &&
    !response.score
  ) {
    return 'fastapi';
  }

  return 'unknown';
}

// 导出统一的适配器对象
export default {
  adaptRiskScoreResponse,
  adaptRiskMetricsResponse,
  adaptAddressDetailsResponse,
  adaptAddressTagsResponse,
  adaptTransactionsResponse,
  detectResponseSource,
};
