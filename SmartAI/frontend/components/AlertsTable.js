import React from 'react';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

/**
 * 风险等级徽章组件
 * @param {Object} props - 组件属性
 * @param {string} props.level - 风险等级 (high, medium, low)
 * @returns {JSX.Element} 风险等级徽章组件
 */
function RiskBadge({ level }) {
  const colors = {
    high: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    low: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
  };

  const riskText = {
    high: '高风险',
    medium: '中风险',
    low: '低风险',
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[level]}`}
    >
      {riskText[level]}
    </span>
  );
}

/**
 * 状态徽章组件
 * @param {Object} props - 组件属性
 * @param {string} props.status - 状态 (pending, processing, resolved)
 * @returns {JSX.Element} 状态徽章组件
 */
function StatusBadge({ status }) {
  const colors = {
    pending: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    processing: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    resolved: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
  };

  const statusText = {
    pending: '待处理',
    processing: '处理中',
    resolved: '已解决',
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors[status]}`}
    >
      {statusText[status]}
    </span>
  );
}

/**
 * 预警表格组件
 * @param {Object} props - 组件属性
 * @param {Array} props.alerts - 预警数据数组
 * @param {string} props.language - 语言设置
 * @returns {JSX.Element} 预警表格组件
 */
export default function AlertsTable({ alerts, language = 'zh-CN' }) {
  // 格式化时间戳
  const formatTimestamp = timestamp => {
    try {
      const date = new Date(timestamp);
      // 根据当前语言选择日期格式化的区域设置
      const locale = language === 'zh-CN' ? zhCN : undefined;
      return format(date, 'yyyy-MM-dd HH:mm:ss', { locale });
    } catch (error) {
      console.error('时间格式化错误:', error);
      return timestamp;
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-800">
          <tr>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              时间
            </th>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              风险类型
            </th>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              交易哈希
            </th>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              风险等级
            </th>
            <th
              scope="col"
              className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              状态
            </th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
          {alerts.map((alert, index) => (
            <tr
              key={alert.id || index}
              className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors duration-150"
            >
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                {formatTimestamp(alert.timestamp)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                {alert.riskType}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400 font-mono">
                {alert.txHash}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <RiskBadge level={alert.riskLevel} />
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <StatusBadge status={alert.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
