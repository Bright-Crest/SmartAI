import React, { useState, useRef, useEffect } from 'react';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';
import { FaInfoCircle } from 'react-icons/fa';

// 注册Chart.js组件
ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

// 维度描述信息
const DIMENSION_INFO = {
  activityScore: {
    title: '活跃度',
    description: '评估钱包地址的链上活动频率和规律性。高分表示活跃的交易行为，低分表示持有型用户。',
    examples: ['交易频率', '定期操作', '活动周期性']
  },
  protocolInteractionScore: {
    title: '协议互动',
    description: '衡量与不同DeFi协议的互动广度和深度。高分表示用户熟悉多种协议，低分表示使用单一或少量协议。',
    examples: ['DeFi协议使用', '跨协议操作', '复杂交易']
  },
  profitabilityScore: {
    title: '收益能力',
    description: '分析地址的盈利表现和策略。高分表示交易盈利能力强，低分表示盈利能力有限。',
    examples: ['交易盈亏', '投资回报', '套利能力']
  },
  fundFlowScore: {
    title: '资金流',
    description: '追踪资金流入流出模式和规模。高分表示大额和频繁的资金流动，低分表示小额或稳定的资金流动。',
    examples: ['资金规模', '流动频率', '交易对手多样性']
  },
  distillationScore: {
    title: '蒸馏模型',
    description: '基于机器学习的综合行为评估。高分表示行为模式复杂且专业，低分表示简单或初级的使用模式。',
    examples: ['行为复杂度', '专业性评估', '模式识别']
  }
};

/**
 * 五维雷达图组件 - 显示钱包地址的五个维度分析
 * 
 * @param {Object} props
 * @param {Object} props.data - 五个维度的数据
 * @param {number} props.data.activityScore - 活跃度评分(0-100)
 * @param {number} props.data.protocolInteractionScore - 协议互动评分(0-100)
 * @param {number} props.data.profitabilityScore - 收益能力评分(0-100)
 * @param {number} props.data.fundFlowScore - 资金流评分(0-100)  
 * @param {number} props.data.distillationScore - 蒸馏模型评分(0-100)
 * @param {string} props.className - 额外的CSS类名
 * @param {boolean} props.interactive - 是否启用交互功能
 * @param {boolean} props.showLegend - 是否显示图例
 * @param {boolean} props.showAverageUser - 是否显示平均用户对比数据
 * @returns {JSX.Element}
 */
export default function RadarChart({ 
  data, 
  className = '',
  interactive = true,
  showLegend = false,
  showAverageUser = false 
}) {
  const [selectedDimension, setSelectedDimension] = useState(null);
  const [chartInstance, setChartInstance] = useState(null);
  const chartRef = useRef(null);
  
  // 模拟平均用户数据
  const averageUserData = {
    activityScore: 50,
    protocolInteractionScore: 45,
    profitabilityScore: 50,
    fundFlowScore: 40,
    distillationScore: 45
  };

  // 格式化数据集
  const chartData = {
    labels: ['活跃度', '协议互动', '收益能力', '资金流', '蒸馏模型'],
    datasets: [
      {
        label: '当前钱包',
        data: [
          data?.activityScore || 0,
          data?.protocolInteractionScore || 0,
          data?.profitabilityScore || 0,
          data?.fundFlowScore || 0, 
          data?.distillationScore || 0
        ],
        backgroundColor: 'rgba(99, 102, 241, 0.2)',
        borderColor: 'rgba(99, 102, 241, 1)',
        pointBackgroundColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 2,
        pointBorderColor: '#fff',
        pointHoverRadius: 6,
        pointHoverBackgroundColor: 'rgba(99, 102, 241, 1)',
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2
      },
      ...(showAverageUser ? [{
        label: '平均用户',
        data: [
          averageUserData.activityScore,
          averageUserData.protocolInteractionScore,
          averageUserData.profitabilityScore,
          averageUserData.fundFlowScore,
          averageUserData.distillationScore
        ],
        backgroundColor: 'rgba(249, 115, 22, 0.1)',
        borderColor: 'rgba(249, 115, 22, 0.8)',
        pointBackgroundColor: 'rgba(249, 115, 22, 0.8)',
        borderWidth: 1,
        pointBorderColor: '#fff',
        pointRadius: 3,
        borderDash: [5, 5],
      }] : [])
    ]
  };

  // 获取Chart.js实例
  useEffect(() => {
    if (chartRef.current) {
      setChartInstance(chartRef.current);
    }
  }, []);

  // 获取维度索引
  const getDimensionIndex = (dimensionTitle) => {
    const titles = Object.values(DIMENSION_INFO).map(info => info.title);
    return titles.indexOf(dimensionTitle);
  };

  // 处理维度点击
  const handleDimensionClick = (dimension) => {
    if (!interactive) return;
    
    if (selectedDimension === dimension) {
      setSelectedDimension(null);
    } else {
      setSelectedDimension(dimension);
    }
  };

  // 获取维度键名
  const getDimensionKey = (dimensionTitle) => {
    return Object.keys(DIMENSION_INFO).find(
      key => DIMENSION_INFO[key].title === dimensionTitle
    );
  };

  // 图表选项配置
  const options = {
    scales: {
      r: {
        angleLines: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)',
        },
        suggestedMin: 0,
        suggestedMax: 100,
        ticks: {
          stepSize: 20,
          callback: (value) => value,
          color: 'rgba(0, 0, 0, 0.6)',
          backdropColor: 'rgba(255, 255, 255, 0.75)',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        pointLabels: {
          font: {
            size: 12,
            weight: 'bold'
          },
          color: 'rgba(0, 0, 0, 0.7)',
        }
      }
    },
    plugins: {
      legend: {
        display: showLegend,
        position: 'bottom',
        labels: {
          usePointStyle: true,
          boxWidth: 8,
          padding: 15
        }
      },
      tooltip: {
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        titleColor: '#222',
        bodyColor: '#333',
        borderColor: 'rgba(0, 0, 0, 0.1)',
        borderWidth: 1,
        boxPadding: 6,
        cornerRadius: 8,
        usePointStyle: true,
        callbacks: {
          title: function(context) {
            return context[0].label;
          },
          label: function(context) {
            const dimensionKey = getDimensionKey(context.label);
            const info = DIMENSION_INFO[dimensionKey];
            return [
              `${context.dataset.label}: ${context.raw}/100`,
              `${info?.description?.substring(0, 60)}...`
            ];
          }
        }
      }
    },
    maintainAspectRatio: false,
    responsive: true,
    events: interactive ? ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'] : [],
    onClick: (event, elements) => {
      if (!interactive || elements.length === 0) return;
      
      const { index } = elements[0];
      const dimensionTitle = chartData.labels[index];
      handleDimensionClick(dimensionTitle);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 ${className}`}>
      <h3 className="text-lg font-medium text-gray-800 dark:text-white mb-2 flex items-center justify-between">
        <span>五维行为分析</span>
        {interactive && (
          <span className="text-xs text-gray-500 dark:text-gray-400">
            点击维度查看详情
          </span>
        )}
      </h3>
      
      <div className="h-64 w-full">
        <Radar ref={chartRef} data={chartData} options={options} />
      </div>
      
      <div className="mt-4 grid grid-cols-5 gap-2 text-center text-xs">
        {chartData.labels.map((label, index) => {
          const value = chartData.datasets[0].data[index];
          const dimensionKey = getDimensionKey(label);
          
          return (
            <div 
              key={index} 
              className={`flex flex-col items-center p-1 rounded-md cursor-pointer transition-colors ${
                selectedDimension === label 
                  ? 'bg-primary-50 dark:bg-primary-900/20' 
                  : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
              } ${interactive ? 'cursor-pointer' : ''}`}
              onClick={() => interactive && handleDimensionClick(label)}
            >
              <div className="font-medium text-gray-700 dark:text-gray-300">{label}</div>
              <div className="text-primary-600 dark:text-primary-400 font-bold">
                {value}
              </div>
            </div>
          );
        })}
      </div>

      {selectedDimension && (
        <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600">
          <div className="flex items-start gap-2">
            <FaInfoCircle className="text-primary-500 mt-1 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white">
                {DIMENSION_INFO[getDimensionKey(selectedDimension)]?.title}
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                {DIMENSION_INFO[getDimensionKey(selectedDimension)]?.description}
              </p>
              
              <div className="mt-2">
                <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  评估指标:
                </h5>
                <ul className="mt-1 text-xs text-gray-600 dark:text-gray-300 flex flex-wrap gap-2">
                  {DIMENSION_INFO[getDimensionKey(selectedDimension)]?.examples.map((example, i) => (
                    <li key={i} className="px-2 py-1 bg-gray-100 dark:bg-gray-600 rounded-full">
                      {example}
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="mt-3 flex justify-between items-center">
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  当前评分: <span className="font-bold text-primary-600 dark:text-primary-400">
                    {chartData.datasets[0].data[getDimensionIndex(selectedDimension)]}/100
                  </span>
                </div>
                {showAverageUser && (
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    平均水平: <span className="font-medium">
                      {averageUserData[getDimensionKey(selectedDimension)]}/100
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 