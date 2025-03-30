import React, { useEffect, useRef, useState } from 'react';
import {
  Chart,
  RadarController,
  RadialLinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { FaInfoCircle } from 'react-icons/fa';

// 检测是否为浏览器环境
const isBrowser = typeof window !== 'undefined';

// 只在浏览器环境中注册Chart.js组件
if (isBrowser) {
  Chart.register(
    RadarController,
    RadialLinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
    ChartDataLabels
  );
}

// 平均用户的基准数据
const averageUserData = {
  activityScore: 50,
  protocolInteractionScore: 40,
  profitabilityScore: 45,
  fundFlowScore: 35,
  distillationScore: 50,
};

// 维度标签中文名称
const dimensionLabels = {
  activityScore: '活跃度',
  protocolInteractionScore: '协议互动',
  profitabilityScore: '收益能力',
  fundFlowScore: '资金流',
  distillationScore: '蒸馏模型',
};

// 维度描述
const dimensionTooltips = {
  activityScore: '评估用户在区块链上的活动频率和规律性，反映参与程度',
  protocolInteractionScore: '衡量与不同DeFi协议和智能合约的交互广度和深度',
  profitabilityScore: '分析投资和交易的盈利能力，评估市场表现',
  fundFlowScore: '追踪资金流入流出模式、规模和网络特征',
  distillationScore: '基于机器学习的综合行为评估，识别专业性和复杂度',
};

/**
 * 五维雷达图组件
 * @param {Object} props
 * @param {Object} props.data - 包含五个维度分数的对象
 * @param {boolean} props.interactive - 是否启用交互功能
 * @param {boolean} props.showAverageUser - 是否显示平均用户数据作为比较基准
 * @param {boolean} props.showLegend - 是否显示图例
 * @param {string} props.className - 自定义CSS类名
 * @param {Function} props.onDimensionClick - 点击维度时的回调函数
 */
const RadarChart = ({
  data,
  interactive = false,
  showAverageUser = false,
  showLegend = true,
  className = '',
  onDimensionClick = null,
}) => {
  const chartRef = useRef(null);
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [chartWidth, setChartWidth] = useState(400);
  const [chartHeight, setChartHeight] = useState(400);

  // 添加交互效果
  useEffect(() => {
    // 如果不是交互式或没有图表引用，则跳过这个效果
    if (!interactive || !chartRef.current) return;

    const chart = chartRef.current;

    // 确保chart存在且已初始化
    if (!chart || !chart.canvas) {
      return;
    }

    // 监听鼠标移动事件
    const handleHover = event => {
      if (!chart || !chart.canvas) {
        return;
      }

      const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);

      if (points.length) {
        const index = points[0].index;
        const dimension = Object.keys(data)[index];
        setHoveredPoint(dimension);
        setActiveTooltip({
          dimension,
          x: event.clientX,
          y: event.clientY,
        });
      } else {
        setHoveredPoint(null);
        setActiveTooltip(null);
      }
    };

    // 添加事件监听器
    chart.canvas.addEventListener('mousemove', handleHover);

    // 清理函数
    return () => {
      // 添加安全检查，确保chart和canvas在卸载时仍然存在
      if (chart && chart.canvas) {
        chart.canvas.removeEventListener('mousemove', handleHover);
      }
    };
  }, [interactive, data]);

  // 检查数据是否可用
  if (!data || Object.keys(data).length === 0) {
    return (
      <div
        className={`relative ${className} flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg`}
        style={{ height: '400px' }}
      >
        <div className="text-gray-500 dark:text-gray-400">加载图表中...</div>
      </div>
    );
  }

  // 转换数据格式为Chart.js所需的结构
  const chartData = {
    labels: Object.keys(data).map(key => dimensionLabels[key] || key),
    datasets: [
      {
        label: '当前地址',
        data: Object.values(data),
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(75, 192, 192, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(75, 192, 192, 1)',
        pointRadius: 4,
        pointHoverRadius: 6,
      },
    ],
  };

  // 如果需要显示平均用户数据
  if (showAverageUser) {
    chartData.datasets.push({
      label: '平均用户',
      data: Object.keys(data).map(key => averageUserData[key] || 0),
      backgroundColor: 'rgba(153, 102, 255, 0.2)',
      borderColor: 'rgba(153, 102, 255, 1)',
      borderWidth: 1,
      pointBackgroundColor: 'rgba(153, 102, 255, 1)',
      pointBorderColor: '#fff',
      pointHoverBackgroundColor: '#fff',
      pointHoverBorderColor: 'rgba(153, 102, 255, 1)',
      pointRadius: 3,
      pointHoverRadius: 5,
    });
  }

  // 图表配置选项
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        beginAtZero: true,
        min: 0,
        max: 100,
        ticks: {
          stepSize: 20,
          backdropColor: 'rgba(0, 0, 0, 0)',
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
        pointLabels: {
          font: {
            size: 14,
            weight: 'bold',
          },
          color: 'rgba(0, 0, 0, 0.7)',
          callback: (label, index) => {
            // 将标签包装成多行以避免重叠
            return label;
          },
        },
        angleLines: {
          color: 'rgba(0, 0, 0, 0.1)',
        },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          title: tooltipItems => {
            const index = tooltipItems[0].dataIndex;
            const dimension = Object.keys(data)[index];
            return dimensionLabels[dimension] || dimension;
          },
          label: context => {
            const index = context.dataIndex;
            const dimension = Object.keys(data)[index];
            const score = context.dataset.data[index];
            return `${context.dataset.label}: ${score}/100`;
          },
          afterLabel: context => {
            const index = context.dataIndex;
            const dimension = Object.keys(data)[index];
            return dimensionTooltips[dimension] || '';
          },
        },
      },
      legend: {
        display: showLegend,
        position: 'bottom',
        labels: {
          font: {
            size: 12,
          },
          usePointStyle: true,
          pointStyle: 'circle',
        },
      },
      datalabels: {
        display: context => {
          // 只在当前地址的数据点上显示标签
          return context.datasetIndex === 0;
        },
        formatter: value => {
          return value;
        },
        color: 'rgba(0, 0, 0, 0.7)',
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        borderRadius: 4,
        font: {
          weight: 'bold',
          size: 10,
        },
        padding: 4,
        align: 'end',
        anchor: 'end',
      },
    },
    elements: {
      line: {
        tension: 0.1,
      },
    },
    interaction: {
      mode: 'point',
      intersect: true,
    },
  };

  // 添加点击事件处理器
  if (interactive && onDimensionClick) {
    options.onClick = (event, elements) => {
      if (elements && elements.length > 0) {
        const index = elements[0].index;
        const dimension = Object.keys(data)[index];
        onDimensionClick(dimension);
      }
    };
  }

  return (
    <div className={`relative ${className}`} style={{ height: '400px' }}>
      <Radar
        ref={chartRef}
        data={chartData}
        options={options}
        plugins={[
          {
            id: 'customInteraction',
            beforeDraw: chart => {
              if (hoveredPoint && interactive) {
                const meta = chart.getDatasetMeta(0);
                const index = Object.keys(data).findIndex(key => key === hoveredPoint);

                if (index >= 0 && meta.data[index]) {
                  const ctx = chart.ctx;
                  ctx.save();
                  ctx.beginPath();
                  ctx.arc(meta.data[index].x, meta.data[index].y, 8, 0, Math.PI * 2);
                  ctx.fillStyle = 'rgba(75, 192, 192, 0.3)';
                  ctx.fill();
                  ctx.strokeStyle = 'rgba(75, 192, 192, 1)';
                  ctx.lineWidth = 2;
                  ctx.stroke();
                  ctx.restore();
                }
              }
            },
          },
        ]}
      />

      {/* 自定义工具提示 */}
      {interactive && activeTooltip && (
        <div
          className="absolute bg-white dark:bg-gray-800 shadow-lg rounded-lg p-3 z-10 transition-opacity duration-200"
          style={{
            left: `${activeTooltip.x + 15}px`,
            top: `${activeTooltip.y - 15}px`,
            transform: 'translate(-50%, -100%)',
            minWidth: '200px',
            opacity: 0.9,
          }}
        >
          <div className="flex items-center justify-between mb-1">
            <h4 className="text-sm font-bold text-gray-900 dark:text-white">
              {dimensionLabels[activeTooltip.dimension]}
            </h4>
            <span className="text-xs bg-primary-100 text-primary-800 px-2 py-0.5 rounded-full">
              {data[activeTooltip.dimension]}/100
            </span>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            {dimensionTooltips[activeTooltip.dimension]}
          </p>
          {interactive && onDimensionClick && (
            <div className="text-xs text-primary-600 dark:text-primary-400 mt-2 flex items-center">
              <FaInfoCircle className="mr-1" />
              点击查看详情
            </div>
          )}
        </div>
      )}

      {/* 图例说明 */}
      {interactive && (
        <div className="absolute bottom-0 right-0 text-xs text-gray-500 dark:text-gray-400 bg-white/50 dark:bg-gray-800/50 p-1 rounded">
          {onDimensionClick && <span>点击维度查看详情</span>}
        </div>
      )}
    </div>
  );
};

export default RadarChart;
