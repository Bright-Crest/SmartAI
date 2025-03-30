import React, { useState, useEffect } from 'react';
import MainLayout from '../components/MainLayout';
import Card from '../components/Card';
import { useClientOnly } from '../hooks/useClientOnly';
import { formatDate, formatCurrency, truncateAddress } from '../utils/formatters';
import { RISK_LEVELS, RISK_COLORS } from '../utils/constants';

// 模拟数据生成函数 - 历史趋势数据
const generateTimeSeriesData = (days = 30, baseValue = 100, volatility = 0.2) => {
  const data = [];
  const today = new Date();
  let value = baseValue;

  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);

    // 随机波动
    const change = (Math.random() - 0.5) * volatility;
    value = Math.max(0, value * (1 + change));

    data.push({
      date,
      value: Math.round(value * 100) / 100,
    });
  }

  return data;
};

// 仪表盘页面
export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [riskyAddresses, setRiskyAddresses] = useState([]);
  const [recentTransactions, setRecentTransactions] = useState([]);
  const [timeRange, setTimeRange] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);

  // 使用客户端渲染模式
  const isClient = useClientOnly();

  // 获取仪表盘数据
  useEffect(() => {
    if (!isClient) return;

    const fetchDashboardData = async () => {
      try {
        setIsLoading(true);

        // 生成模拟统计数据
        const mockStats = {
          totalTransactions: Math.floor(Math.random() * 1000000) + 500000,
          totalAddresses: Math.floor(Math.random() * 100000) + 30000,
          activeAddresses: Math.floor(Math.random() * 20000) + 5000,
          totalValue: Math.random() * 10000 + 5000,
          avgTransactionValue: Math.random() * 5 + 0.5,
          riskDistribution: {
            high: Math.floor(Math.random() * 30) + 5,
            medium: Math.floor(Math.random() * 40) + 20,
            low: Math.floor(Math.random() * 50) + 30,
          },
          trends: {
            transactions: generateTimeSeriesData(30, 1000, 0.1),
            activeAddresses: generateTimeSeriesData(30, 200, 0.05),
            value: generateTimeSeriesData(30, 5000, 0.15),
          },
        };

        // 生成模拟高风险地址
        const mockRiskyAddresses = Array(5)
          .fill()
          .map((_, index) => ({
            address: `0x${Math.random().toString(16).substring(2, 42)}`,
            riskScore: Math.floor(Math.random() * 30) + 70,
            transactions: Math.floor(Math.random() * 100) + 20,
            lastActive: new Date(Date.now() - Math.random() * 86400000 * 7),
            value: Math.random() * 50 + 10,
            risk: 'high',
          }));

        // 生成模拟近期交易
        const mockRecentTransactions = Array(10)
          .fill()
          .map((_, index) => ({
            id: `0x${Math.random().toString(16).substring(2, 42)}`,
            hash: `0x${Math.random().toString(16).substring(2, 66)}`,
            from: `0x${Math.random().toString(16).substring(2, 42)}`,
            to: `0x${Math.random().toString(16).substring(2, 42)}`,
            value: Math.random() * 10,
            timestamp: new Date(Date.now() - Math.random() * 86400000),
            risk: RISK_LEVELS[Math.floor(Math.random() * RISK_LEVELS.length)],
          }));

        // 延迟模拟网络请求
        setTimeout(() => {
          setStats(mockStats);
          setRiskyAddresses(mockRiskyAddresses);
          setRecentTransactions(mockRecentTransactions);
          setIsLoading(false);
        }, 800);
      } catch (error) {
        console.error('获取仪表盘数据失败', error);
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, [isClient]);

  // 处理时间范围变更
  const handleTimeRangeChange = range => {
    setTimeRange(range);
  };

  // 获取趋势数据的子集，基于选定的时间范围
  const getTrendData = data => {
    if (!data) return [];

    const cutoffs = {
      '24h': 1,
      '7d': 7,
      '30d': 30,
      '90d': 90,
    };

    const days = cutoffs[timeRange] || 30;
    return data.slice(-days - 1);
  };

  // 渲染统计卡片
  const renderStatCards = () => {
    if (!stats) return null;

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-600 mb-2">总交易数</h3>
          <p className="text-3xl font-bold">{stats.totalTransactions.toLocaleString()}</p>
          <div className="mt-2 text-sm text-green-600">
            +{Math.floor(Math.random() * 5) + 1}% 较上周
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-600 mb-2">注册地址</h3>
          <p className="text-3xl font-bold">{stats.totalAddresses.toLocaleString()}</p>
          <div className="mt-2 text-sm text-green-600">
            +{Math.floor(Math.random() * 3) + 1}% 较上周
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-600 mb-2">活跃地址</h3>
          <p className="text-3xl font-bold">{stats.activeAddresses.toLocaleString()}</p>
          <div className="mt-2 text-sm text-green-600">
            +{Math.floor(Math.random() * 8) + 2}% 较上周
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-medium text-gray-600 mb-2">交易总额</h3>
          <p className="text-3xl font-bold">{formatCurrency(stats.totalValue)}</p>
          <div className="mt-2 text-sm text-green-600">
            +{Math.floor(Math.random() * 10) + 5}% 较上周
          </div>
        </Card>
      </div>
    );
  };

  // 渲染风险分布
  const renderRiskDistribution = () => {
    if (!stats || !stats.riskDistribution) return null;

    const { riskDistribution } = stats;
    const total = riskDistribution.high + riskDistribution.medium + riskDistribution.low;

    return (
      <Card className="p-6 mb-8">
        <h3 className="text-xl font-bold mb-4">风险分布</h3>

        <div className="w-full h-4 bg-gray-200 rounded-full mb-6">
          <div className="flex h-full rounded-full overflow-hidden">
            <div
              className="h-full"
              style={{
                width: `${(riskDistribution.high / total) * 100}%`,
                backgroundColor: RISK_COLORS.high,
              }}
            ></div>
            <div
              className="h-full"
              style={{
                width: `${(riskDistribution.medium / total) * 100}%`,
                backgroundColor: RISK_COLORS.medium,
              }}
            ></div>
            <div
              className="h-full"
              style={{
                width: `${(riskDistribution.low / total) * 100}%`,
                backgroundColor: RISK_COLORS.low,
              }}
            ></div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-2">
          <div>
            <div className="flex items-center mb-1">
              <span
                className="inline-block w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: RISK_COLORS.high }}
              ></span>
              <span className="text-sm font-medium">高风险</span>
            </div>
            <p className="text-2xl font-bold">{riskDistribution.high}</p>
            <p className="text-sm text-gray-500">
              {Math.round((riskDistribution.high / total) * 100)}%
            </p>
          </div>

          <div>
            <div className="flex items-center mb-1">
              <span
                className="inline-block w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: RISK_COLORS.medium }}
              ></span>
              <span className="text-sm font-medium">中风险</span>
            </div>
            <p className="text-2xl font-bold">{riskDistribution.medium}</p>
            <p className="text-sm text-gray-500">
              {Math.round((riskDistribution.medium / total) * 100)}%
            </p>
          </div>

          <div>
            <div className="flex items-center mb-1">
              <span
                className="inline-block w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: RISK_COLORS.low }}
              ></span>
              <span className="text-sm font-medium">低风险</span>
            </div>
            <p className="text-2xl font-bold">{riskDistribution.low}</p>
            <p className="text-sm text-gray-500">
              {Math.round((riskDistribution.low / total) * 100)}%
            </p>
          </div>
        </div>
      </Card>
    );
  };

  // 渲染高风险地址列表
  const renderRiskyAddresses = () => {
    if (!riskyAddresses || riskyAddresses.length === 0) return null;

    return (
      <Card className="mb-8">
        <div className="p-4 border-b">
          <h3 className="text-xl font-bold">高风险地址</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  地址
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  风险评分
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  交易数
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  最近活动
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  交易额
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {riskyAddresses.map(address => (
                <tr key={address.address} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 hover:underline">
                    <a href={`/addresses?address=${address.address}`}>
                      {truncateAddress(address.address)}
                    </a>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <span
                        className="inline-block w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: RISK_COLORS.high }}
                      ></span>
                      <span className="font-medium">{address.riskScore}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {address.transactions}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatDate(address.lastActive)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    {formatCurrency(address.value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    );
  };

  // 渲染最近交易
  const renderRecentTransactions = () => {
    if (!recentTransactions || recentTransactions.length === 0) return null;

    return (
      <Card>
        <div className="p-4 border-b">
          <h3 className="text-xl font-bold">最近交易</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  交易哈希
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  时间
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  发送方
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  接收方
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  金额 (ETH)
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  风险
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {recentTransactions.map(tx => {
                const riskColor = RISK_COLORS[tx.risk] || RISK_COLORS.low;

                return (
                  <tr key={tx.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 hover:underline">
                      <a href={`/transactions?hash=${tx.hash}`}>{truncateAddress(tx.hash)}</a>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(tx.timestamp)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <a
                        href={`/addresses?address=${tx.from}`}
                        className="hover:underline text-blue-600"
                      >
                        {truncateAddress(tx.from)}
                      </a>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <a
                        href={`/addresses?address=${tx.to}`}
                        className="hover:underline text-blue-600"
                      >
                        {truncateAddress(tx.to)}
                      </a>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      {formatCurrency(tx.value)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <span
                          className="inline-block w-3 h-3 rounded-full mr-2"
                          style={{ backgroundColor: riskColor }}
                        ></span>
                        <span className="font-medium capitalize">{tx.risk}</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    );
  };

  // 如果不是客户端渲染，则不显示内容
  if (!isClient) {
    return null;
  }

  return (
    <MainLayout title="仪表盘 | ChainIntelAI" description="区块链网络的整体统计和关键指标">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between mb-8">
          <h1 className="text-3xl font-bold mb-4 md:mb-0">区块链网络仪表盘</h1>

          <div className="flex space-x-2">
            <button
              className={`px-3 py-1 rounded-md text-sm ${
                timeRange === '24h'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => handleTimeRangeChange('24h')}
            >
              24小时
            </button>
            <button
              className={`px-3 py-1 rounded-md text-sm ${
                timeRange === '7d'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => handleTimeRangeChange('7d')}
            >
              7天
            </button>
            <button
              className={`px-3 py-1 rounded-md text-sm ${
                timeRange === '30d'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => handleTimeRangeChange('30d')}
            >
              30天
            </button>
            <button
              className={`px-3 py-1 rounded-md text-sm ${
                timeRange === '90d'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
              onClick={() => handleTimeRangeChange('90d')}
            >
              90天
            </button>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : (
          <>
            {renderStatCards()}
            {renderRiskDistribution()}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <Card className="p-6">
                <h3 className="text-xl font-bold mb-6">交易趋势</h3>
                <div className="h-64 w-full flex items-center justify-center bg-gray-100 rounded">
                  <span className="text-gray-500">图表将在这里显示</span>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-xl font-bold mb-6">交易金额趋势</h3>
                <div className="h-64 w-full flex items-center justify-center bg-gray-100 rounded">
                  <span className="text-gray-500">图表将在这里显示</span>
                </div>
              </Card>
            </div>

            {renderRiskyAddresses()}
            {renderRecentTransactions()}
          </>
        )}
      </div>
    </MainLayout>
  );
}
