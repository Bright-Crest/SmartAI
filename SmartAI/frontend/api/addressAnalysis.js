import { generateMockData } from '../utils/mockData';

/**
 * 获取地址分析数据
 * 开发环境使用模拟数据，生产环境调用后端API
 *
 * @param {string} address - 以太坊钱包地址
 * @returns {Promise<Object>} 分析结果
 */
export async function getAddressAnalysis(address) {
  // 验证地址格式
  if (!address || !/^0x[a-fA-F0-9]{40}$/.test(address)) {
    throw new Error('无效的以太坊地址');
  }

  // 开发环境使用模拟数据
  if (process.env.NODE_ENV === 'development') {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 1000));
    return generateMockData(address);
  }

  // 生产环境调用后端API
  try {
    const response = await fetch(`/api/address/${address}/analysis`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `请求失败: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('获取地址分析失败:', error);
    throw error;
  }
}

/**
 * 获取地址关系图谱
 *
 * @param {string} address - 以太坊钱包地址
 * @param {Object} options - 图谱选项
 * @returns {Promise<Object>} 图谱数据
 */
export async function getAddressGraph(address, options = {}) {
  // 验证地址格式
  if (!address || !/^0x[a-fA-F0-9]{40}$/.test(address)) {
    throw new Error('无效的以太坊地址');
  }

  // 开发环境使用模拟数据
  if (process.env.NODE_ENV === 'development') {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 1500));
    const mockData = generateMockData(address);
    return mockData.networkData;
  }

  // 生产环境调用后端API
  try {
    const queryParams = new URLSearchParams(options).toString();
    const response = await fetch(`/api/address/${address}/graph?${queryParams}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `请求失败: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('获取地址图谱失败:', error);
    throw error;
  }
}

/**
 * 验证以太坊地址格式
 *
 * @param {string} address - 以太坊钱包地址
 * @returns {boolean} 是否为有效地址
 */
export function isValidEthereumAddress(address) {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}
