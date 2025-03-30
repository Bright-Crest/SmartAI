import React from 'react';
import { FaUserAlt, FaChartLine, FaWallet, FaExchangeAlt, FaHistory } from 'react-icons/fa';

/**
 * 地址行为总结组件 - 显示地址的行为标签和洞察
 * 
 * @param {Object} props
 * @param {Object} props.data - 行为分析数据
 * @param {string} props.data.behaviorType - 主要行为类型(如"套利型"/"价值持有者"等)
 * @param {string[]} props.data.behaviorTags - 行为标签列表
 * @param {string} props.data.summary - 行为总结文本
 * @param {Object[]} props.data.insights - 洞察点列表
 * @param {string} props.data.insights[].title - 洞察点标题
 * @param {string} props.data.insights[].description - 洞察点描述
 * @param {string} props.className - 额外的CSS类名
 * @returns {JSX.Element}
 */
export default function AddressBehaviorSummary({ data, className = '' }) {
  // 根据行为类型返回对应的图标
  const getBehaviorIcon = (type) => {
    switch (type?.toLowerCase()) {
      case 'arbitrage':
      case '套利型':
        return <FaExchangeAlt className="text-yellow-500" />;
      case 'holder':
      case '价值持有者':
        return <FaWallet className="text-blue-500" />;
      case 'trader':
      case '交易者':
        return <FaChartLine className="text-green-500" />;
      case 'whale':
      case '鲸鱼':
        return <FaWallet className="text-purple-500" />;
      default:
        return <FaUserAlt className="text-gray-500" />;
    }
  };

  if (!data) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 ${className}`}>
        <div className="text-gray-500 dark:text-gray-400 text-center py-4">
          无行为分析数据
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 ${className}`}>
      <h3 className="text-lg font-medium text-gray-800 dark:text-white mb-4">行为定性与洞察</h3>
      
      {/* 主要行为类型 */}
      {data.behaviorType && (
        <div className="flex items-center mb-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="mr-3 text-xl">
            {getBehaviorIcon(data.behaviorType)}
          </div>
          <div>
            <div className="font-medium text-gray-900 dark:text-white">
              {data.behaviorType}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400">
              主要行为类型
            </div>
          </div>
        </div>
      )}
      
      {/* 行为标签 */}
      {data.behaviorTags && data.behaviorTags.length > 0 && (
        <div className="mb-4">
          <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
            行为标签
          </div>
          <div className="flex flex-wrap gap-2">
            {data.behaviorTags.map((tag, index) => (
              <span 
                key={index} 
                className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800 dark:bg-gray-600 dark:text-gray-200"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {/* 行为总结 */}
      {data.summary && (
        <div className="mb-4">
          <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
            行为总结
          </div>
          <div className="text-gray-700 dark:text-gray-300 text-sm">
            {data.summary}
          </div>
        </div>
      )}
      
      {/* 关键洞察 */}
      {data.insights && data.insights.length > 0 && (
        <div>
          <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
            关键洞察
          </div>
          <div className="space-y-3">
            {data.insights.map((insight, index) => (
              <div key={index} className="border-l-2 border-primary-500 pl-3 py-1">
                <div className="font-medium text-gray-800 dark:text-white text-sm">
                  {insight.title}
                </div>
                <div className="text-gray-600 dark:text-gray-400 text-xs">
                  {insight.description}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 