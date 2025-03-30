/**
 * API客户端统一导出
 *
 * 本文件集中导出所有API接口，方便统一引用
 */
import * as networkApi from './network';
import * as transactionsApi from './transactions';
import * as addressesApi from './addresses';
import * as dashboardApi from './dashboard';
import { request, utils, ApiError } from './index';

// 导出所有API
export const api = {
  network: networkApi,
  transactions: transactionsApi,
  addresses: addressesApi,
  dashboard: dashboardApi,

  // 通用请求方法
  request,
  utils,
};

// 导出API错误类
export { ApiError };

// 默认导出
export default api;
