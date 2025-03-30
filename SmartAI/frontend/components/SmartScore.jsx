import React, { useState, useEffect } from 'react';
import { FaCheckCircle, FaExclamationTriangle, FaChartLine, FaInfoCircle, FaChevronDown, FaChevronUp } from 'react-icons/fa';

/**
 * SmartScore组件 - 显示钱包地址的综合评分和标签
 * 
 * @param {Object} props
 * @param {number} props.score - 范围0-100的综合评分
 * @param {string[]} props.tags - 地址标签数组
 * @param {string} props.behaviorType - 地址行为类型
 * @param {string} props.className - 额外的CSS类名
 * @param {boolean} props.animate - 是否启用动画效果
 * @param {Function} props.onViewDetails - 查看详情回调函数
 * @returns {JSX.Element}
 */
export default function SmartScore({ 
  score, 
  tags = [], 
  behaviorType,
  className = '',
  animate = true,
  onViewDetails
}) {
  const [currentScore, setCurrentScore] = useState(0);
  const [showDetails, setShowDetails] = useState(false);
  
  // 动画效果
  useEffect(() => {
    if (!animate) {
      setCurrentScore(score);
      return;
    }
    
    if (currentScore < score) {
      const timer = setTimeout(() => {
        setCurrentScore(prev => Math.min(prev + 1, score));
      }, 20);
      return () => clearTimeout(timer);
    }
  }, [currentScore, score, animate]);

  // 获取分数颜色
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-success-500';
    if (score >= 60) return 'text-warning-500';
    return 'text-danger-500';
  };

  // 获取进度条颜色
  const getProgressColor = (score) => {
    if (score >= 80) return 'bg-success-500';
    if (score >= 60) return 'bg-warning-500';
    return 'bg-danger-500';
  };

  // 获取分数等级
  const getScoreLevel = (score) => {
    if (score >= 80) return '优秀';
    if (score >= 60) return '良好';
    if (score >= 40) return '一般';
    return '风险';
  };
  
  // 获取等级图标
  const getLevelIcon = (score) => {
    if (score >= 60) {
      return <FaCheckCircle className="inline-block mr-1 text-success-500" />;
    }
    return <FaExclamationTriangle className="inline-block mr-1 text-danger-500" />;
  };
  
  // 查看详情按钮点击处理
  const handleViewDetails = () => {
    if (typeof onViewDetails === 'function') {
      onViewDetails();
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 transition-all duration-300 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <FaChartLine className="mr-2 text-primary-500" />
          <h3 className="text-lg font-medium text-gray-800 dark:text-white">SmartScore</h3>
        </div>
        <div className={`text-4xl font-bold ${getScoreColor(currentScore)}`}>
          {currentScore}
        </div>
      </div>
      
      <div className="mb-3">
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
          <div 
            className={`h-2.5 rounded-full transition-all duration-500 ${getProgressColor(currentScore)}`}
            style={{ width: `${currentScore}%` }}
            role="progressbar"
            aria-valuenow={currentScore}
            aria-valuemin="0"
            aria-valuemax="100"
          ></div>
        </div>
        <div className="mt-1 flex justify-between text-sm">
          <div className="text-gray-500 dark:text-gray-400 flex items-center">
            {getLevelIcon(currentScore)}
            {getScoreLevel(currentScore)}
          </div>
          {behaviorType && (
            <div className="text-gray-500 dark:text-gray-400">
              类型: <span className="font-medium">{behaviorType}</span>
            </div>
          )}
        </div>
      </div>
      
      {tags.length > 0 && (
        <div className="mt-4">
          <div className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
            行为标签
          </div>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag, index) => (
              <span 
                key={index} 
                className="px-2 py-1 text-xs rounded-full bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
      
      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
        <button 
          className="flex items-center justify-between w-full text-sm text-gray-500 hover:text-primary-600 dark:text-gray-400 dark:hover:text-primary-500 transition-colors"
          onClick={() => setShowDetails(!showDetails)}
        >
          <span>智能分数说明</span>
          {showDetails ? <FaChevronUp /> : <FaChevronDown />}
        </button>
        
        {showDetails && (
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-300">
            <p className="mb-2">
              SmartScore是基于区块链行为分析的综合评分系统，评分范围0-100，考虑以下几个方面:
            </p>
            <ul className="list-disc pl-5 space-y-1">
              <li>交易历史与模式</li>
              <li>互动协议的类型与多样性</li>
              <li>资金流动情况</li>
              <li>链上活跃度</li>
              <li>资产分布</li>
            </ul>
          </div>
        )}
      </div>
      
      {onViewDetails && (
        <div className="mt-4">
          <button
            onClick={handleViewDetails}
            className="flex items-center justify-center w-full py-2 text-sm text-white bg-primary-600 hover:bg-primary-700 rounded-md transition-colors"
          >
            <FaInfoCircle className="mr-2" />
            查看详细分析
          </button>
        </div>
      )}
    </div>
  );
} 