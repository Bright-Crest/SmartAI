import axios from 'axios';
import { API_BASE_URL } from '../config/constants';

// 聪明钱API接口
const SMART_MONEY_API = {
  // 分析地址是否为聪明钱
  analyzeAddress: async (address, options = { useCache: true }) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/smart-money/analyze/${address}`, {
        params: { useCache: options.useCache },
      });
      return response.data;
    } catch (error) {
      console.error('分析地址失败', error);
      throw error;
    }
  },

  // 批量分析地址
  batchAnalyzeAddresses: async addresses => {
    try {
      const response = await axios.post(`${API_BASE_URL}/smart-money/batch-analyze`, { addresses });
      return response.data;
    } catch (error) {
      console.error('批量分析地址失败', error);
      throw error;
    }
  },

  // 获取聪明钱排行榜
  getLeaderboard: async (options = {}) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/smart-money/leaderboard`, {
        params: options,
      });
      return response.data;
    } catch (error) {
      console.error('获取排行榜失败', error);
      throw error;
    }
  },

  // 获取投资者类型分布
  getInvestorTypeDistribution: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/smart-money/investor-types`);
      return response.data;
    } catch (error) {
      console.error('获取投资者类型分布失败', error);
      throw error;
    }
  },

  // 获取ROI分布
  getROIDistribution: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/smart-money/roi-distribution`);
      return response.data;
    } catch (error) {
      console.error('获取ROI分布失败', error);
      throw error;
    }
  },

  // 获取最近活跃的聪明钱
  getRecentlyActive: async (days = 7, limit = 10) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/smart-money/recently-active`, {
        params: { days, limit },
      });
      return response.data;
    } catch (error) {
      console.error('获取最近活跃的聪明钱失败', error);
      throw error;
    }
  },

  // 获取热门代币和新兴代币
  getTrendingTokens: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/smart-money/trending-tokens`);
      return response.data;
    } catch (error) {
      console.error('获取热门代币失败', error);
      throw error;
    }
  },
};

export default SMART_MONEY_API;
