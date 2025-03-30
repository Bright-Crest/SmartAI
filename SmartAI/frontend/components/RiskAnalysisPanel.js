import React, { useState, useEffect } from 'react';
import { getAddressRiskAnalysis } from '../api/risk';
import { FaExclamationTriangle, FaUserAlt, FaChartLine } from 'react-icons/fa';

const riskLevelClasses = {
  low: 'bg-green-100 text-green-800 border-green-200',
  medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  high: 'bg-red-100 text-red-800 border-red-200',
  unknown: 'bg-gray-100 text-gray-800 border-gray-200',
};

/**
 * 风险分析面板组件
 * 展示地址的风险评分、用户画像和相关风险信息
 *
 * @param {Object} props - 组件属性
 * @param {string} props.address - 区块链地址
 * @param {boolean} props.loading - 是否正在加载
 * @param {Object} props.data - 预加载的风险分析数据
 * @param {Function} props.onDataLoaded - 数据加载完成后的回调函数
 * @returns {JSX.Element} 风险分析面板组件
 */
export default function RiskAnalysisPanel({
  address,
  loading: externalLoading,
  data: externalData,
  onDataLoaded,
}) {
  const [data, setData] = useState(externalData || null);
  const [loading, setLoading] = useState(externalLoading || !externalData);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (externalData) {
      setData(externalData);
      setLoading(false);
      return;
    }

    if (!address) {
      setError(new Error('未提供地址'));
      setLoading(false);
      return;
    }

    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        console.log(`获取地址 ${address} 的风险分析数据`);
        const response = await getAddressRiskAnalysis(address);

        setData(response);
        if (onDataLoaded) onDataLoaded(response);
      } catch (err) {
        console.error('获取风险分析数据失败:', err);
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [address, externalData, onDataLoaded]);

  // 如果正在加载，显示加载状态
  if (loading) {
    return (
      <div className="p-4 border rounded-lg shadow-sm bg-white dark:bg-gray-800 animate-pulse">
        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-3"></div>
        <div className="h-24 bg-gray-200 dark:bg-gray-700 rounded mb-3"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2"></div>
        <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mb-2"></div>
      </div>
    );
  }

  // 如果出现错误，显示错误信息
  if (error) {
    return (
      <div className="p-4 border rounded-lg shadow-sm bg-red-50 text-red-800 dark:bg-gray-800 dark:text-red-400">
        <h3 className="text-lg font-medium mb-2">无法加载风险分析</h3>
        <p className="text-sm">{error.message || '发生未知错误'}</p>
      </div>
    );
  }

  // 如果没有数据，显示空状态
  if (!data) {
    return (
      <div className="p-4 border rounded-lg shadow-sm bg-white dark:bg-gray-800">
        <p className="text-gray-500 dark:text-gray-400">无风险分析数据可用</p>
      </div>
    );
  }

  // 获取风险等级对应的样式
  const riskLevel = data.risk_analysis?.risk_level || 'unknown';
  const riskLevelClass = riskLevelClasses[riskLevel] || riskLevelClasses.unknown;

  return (
    <div className="border rounded-lg shadow-sm bg-white dark:bg-gray-800 overflow-hidden">
      {/* 风险评分 */}
      <div className="p-4 border-b dark:border-gray-700">
        <h3 className="text-lg font-medium mb-3 flex items-center dark:text-white">
          <FaExclamationTriangle className="mr-2 text-yellow-500" />
          风险评分
        </h3>

        <div className="flex items-center mb-3">
          <div className="relative w-full h-4 bg-gray-200 rounded-full dark:bg-gray-700 overflow-hidden">
            <div
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
              style={{ width: `${data.risk_analysis?.risk_score || 0}%` }}
            ></div>
          </div>
          <span className="ml-3 font-bold text-lg dark:text-white">
            {Math.round(data.risk_analysis?.risk_score || 0)}
          </span>
        </div>

        <div
          className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${riskLevelClass}`}
        >
          {data.risk_analysis?.risk_description || '未知风险'}
        </div>

        <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
          {data.risk_analysis?.risk_explanation || '无风险解释'}
        </p>
      </div>

      {/* 用户画像 */}
      <div className="p-4 border-b dark:border-gray-700">
        <h3 className="text-lg font-medium mb-3 flex items-center dark:text-white">
          <FaUserAlt className="mr-2 text-blue-500" />
          用户画像
        </h3>

        <div className="mb-2">
          <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium dark:bg-blue-900 dark:text-blue-200">
            {data.user_profile?.cluster_name || '未分类用户'}
          </span>
        </div>

        <p className="text-sm text-gray-600 dark:text-gray-300">
          {data.user_profile?.cluster_description || '无用户画像描述'}
        </p>
      </div>

      {/* 风险因素 */}
      {data.risk_analysis?.risk_factors && data.risk_analysis.risk_factors.length > 0 && (
        <div className="p-4 border-b dark:border-gray-700">
          <h3 className="text-lg font-medium mb-3 flex items-center dark:text-white">
            <FaChartLine className="mr-2 text-red-500" />
            风险因素
          </h3>

          <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-300">
            {data.risk_analysis.risk_factors.map((factor, index) => (
              <li key={index} className="mb-1">
                {factor}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 关注点 */}
      {data.risk_analysis?.attention_points && data.risk_analysis.attention_points.length > 0 && (
        <div className="p-4">
          <h3 className="text-lg font-medium mb-3 flex items-center dark:text-white">
            <FaExclamationTriangle className="mr-2 text-orange-500" />
            需要关注
          </h3>

          <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-300">
            {data.risk_analysis.attention_points.map((point, index) => (
              <li key={index} className="mb-1">
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
