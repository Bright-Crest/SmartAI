/**
 * 常量定义文件
 */

// 风险级别
export const RISK_LEVELS = ['high', 'medium', 'low'];

// 风险级别对应的颜色
export const RISK_COLORS = {
  high: 'red-500',
  medium: 'yellow-500',
  low: 'green-500',
  unknown: 'gray-400',
};

// 节点类型
export const NODE_TYPES = {
  EXCHANGE: 'exchange',
  WALLET: 'wallet',
  CONTRACT: 'contract',
  MIXER: 'mixer',
  UNKNOWN: 'unknown',
};

// 交易类型
export const TRANSACTION_TYPES = {
  TRANSFER: 'transfer',
  SWAP: 'swap',
  DEPOSIT: 'deposit',
  WITHDRAW: 'withdraw',
  CONTRACT_CALL: 'contract_call',
  UNKNOWN: 'unknown',
};

// API基础URL
export const API_BASE_URL = 'http://localhost:3001/api';

// Etherscan API密钥
export const ETHERSCAN_API_KEY = 'W6DDB16I381Z1UVDTDAP4AV21XTRPW7AVS';

// Etherscan API基础URL
export const ETHERSCAN_API_URL = 'https://api.etherscan.io/api';

// 备用API端点 - 空数组，强制仅使用主端点
export const ETHERSCAN_API_BACKUP_URLS = [];

// 已知的Etherscan API真实IP地址（美国服务器）
// 13.107.42.14 是etherscan.io的一个真实IP地址
export const ETHERSCAN_API_IPS = {
  'api.etherscan.io': '13.107.42.14',
};

// Etherscan API设置
export const ETHERSCAN_API_CONFIG = {
  timeout: 20000, // curl命令执行超时时间
};

// 图表颜色
export const CHART_COLORS = [
  '#3b82f6', // blue-500
  '#ef4444', // red-500
  '#10b981', // green-500
  '#f59e0b', // amber-500
  '#8b5cf6', // violet-500
  '#ec4899', // pink-500
  '#14b8a6', // teal-500
  '#f97316', // orange-500
  '#6366f1', // indigo-500
  '#06b6d4', // cyan-500
];

// 图表默认配置
export const CHART_CONFIG = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'bottom',
      labels: {
        usePointStyle: true,
        boxWidth: 6,
      },
    },
    tooltip: {
      enabled: true,
      mode: 'index',
      intersect: false,
    },
  },
};

// API路径
export const API_PATHS = {
  NETWORK: '/api/network',
  TRANSACTIONS: '/api/transactions',
  ADDRESSES: '/api/addresses',
  DASHBOARD: '/api/dashboard',
  SMART_MONEY: '/api/smart-money',
};

// 分页默认值
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 10,
  PAGE_SIZE_OPTIONS: [10, 25, 50, 100],
};

// 本地存储键
export const STORAGE_KEYS = {
  THEME: 'chain_intel_theme',
  USER_SETTINGS: 'chain_intel_settings',
  AUTH_TOKEN: 'chain_intel_auth_token',
};

// 防抖延迟（毫秒）
export const DEBOUNCE_DELAY = 300;

export default {
  RISK_LEVELS,
  RISK_COLORS,
  NODE_TYPES,
  TRANSACTION_TYPES,
  CHART_COLORS,
  CHART_CONFIG,
  API_PATHS,
  PAGINATION,
  STORAGE_KEYS,
  DEBOUNCE_DELAY,
  API_BASE_URL,
  ETHERSCAN_API_KEY,
  ETHERSCAN_API_URL,
  ETHERSCAN_API_BACKUP_URLS,
  ETHERSCAN_API_IPS,
  ETHERSCAN_API_CONFIG,
};
