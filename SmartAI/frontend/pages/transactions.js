import React, { useState, useEffect } from 'react';
import MainLayout from '../components/MainLayout';
import Card from '../components/Card';
import Button from '../components/Button';
import { useClientOnly } from '../hooks/useClientOnly';
import { formatDate, formatCurrency, truncateAddress } from '../utils/formatters';
import { RISK_LEVELS, RISK_COLORS } from '../utils/constants';

// 交易分析页面
export default function TransactionsPage() {
  const [transactions, setTransactions] = useState([]);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    status: 'all',
    timeRange: '24h',
    riskLevel: 'all',
  });

  // 使用客户端渲染模式
  const isClient = useClientOnly();

  // 获取交易数据
  useEffect(() => {
    if (!isClient) return;

    const fetchTransactions = async () => {
      try {
        setIsLoading(true);

        // 模拟API请求
        const mockTransactions = Array(20)
          .fill()
          .map((_, index) => ({
            id: `0x${Math.random().toString(16).substring(2, 42)}`,
            hash: `0x${Math.random().toString(16).substring(2, 66)}`,
            from: `0x${Math.random().toString(16).substring(2, 42)}`,
            to: `0x${Math.random().toString(16).substring(2, 42)}`,
            value: Math.random() * 10,
            timestamp: new Date(Date.now() - Math.random() * 86400000 * 7),
            status: Math.random() > 0.2 ? 'confirmed' : 'pending',
            gasUsed: Math.floor(Math.random() * 1000000),
            gasPrice: Math.random() * 100,
            blockNumber: Math.floor(Math.random() * 1000000),
            risk: RISK_LEVELS[Math.floor(Math.random() * RISK_LEVELS.length)],
          }));

        // 延迟模拟网络请求
        setTimeout(() => {
          setTransactions(mockTransactions);
          setIsLoading(false);
        }, 800);
      } catch (err) {
        setError('获取交易数据失败，请稍后重试');
        setIsLoading(false);
      }
    };

    fetchTransactions();
  }, [isClient]);

  // 处理搜索
  const handleSearch = e => {
    setSearchTerm(e.target.value);
  };

  // 处理筛选
  const handleFilterChange = (filter, value) => {
    setFilters(prev => ({
      ...prev,
      [filter]: value,
    }));
  };

  // 筛选交易
  const filteredTransactions = transactions.filter(tx => {
    // 搜索条件
    const matchesSearch =
      !searchTerm ||
      tx.hash.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.from.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.to.toLowerCase().includes(searchTerm.toLowerCase());

    // 状态筛选
    const matchesStatus = filters.status === 'all' || tx.status === filters.status;

    // 风险等级筛选
    const matchesRisk = filters.riskLevel === 'all' || tx.risk === filters.riskLevel;

    // 时间范围筛选
    let matchesTimeRange = true;
    const hoursDiff = (Date.now() - tx.timestamp) / (1000 * 60 * 60);

    if (filters.timeRange === '24h') {
      matchesTimeRange = hoursDiff <= 24;
    } else if (filters.timeRange === '7d') {
      matchesTimeRange = hoursDiff <= 24 * 7;
    } else if (filters.timeRange === '30d') {
      matchesTimeRange = hoursDiff <= 24 * 30;
    }

    return matchesSearch && matchesStatus && matchesRisk && matchesTimeRange;
  });

  // 处理交易选择
  const handleSelectTransaction = tx => {
    setSelectedTransaction(tx);
  };

  // 渲染交易列表项
  const renderTransactionItem = tx => {
    const isSelected = selectedTransaction?.id === tx.id;
    const riskColor = RISK_COLORS[tx.risk] || RISK_COLORS.low;

    return (
      <div
        key={tx.id}
        className={`border-b p-4 hover:bg-gray-50 cursor-pointer transition-colors ${isSelected ? 'bg-blue-50' : ''}`}
        onClick={() => handleSelectTransaction(tx)}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center">
            <span
              className="inline-block w-3 h-3 rounded-full mr-2"
              style={{ backgroundColor: riskColor }}
            ></span>
            <span className="font-medium">{truncateAddress(tx.hash)}</span>
          </div>
          <span
            className={`px-2 py-1 rounded text-xs ${
              tx.status === 'confirmed'
                ? 'bg-green-100 text-green-800'
                : 'bg-yellow-100 text-yellow-800'
            }`}
          >
            {tx.status === 'confirmed' ? '已确认' : '待确认'}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          <div className="text-gray-500">发送方:</div>
          <div className="text-right">{truncateAddress(tx.from)}</div>

          <div className="text-gray-500">接收方:</div>
          <div className="text-right">{truncateAddress(tx.to)}</div>

          <div className="text-gray-500">金额:</div>
          <div className="text-right font-medium">{formatCurrency(tx.value)}</div>

          <div className="text-gray-500">时间:</div>
          <div className="text-right">{formatDate(tx.timestamp)}</div>
        </div>
      </div>
    );
  };

  // 渲染交易详情
  const renderTransactionDetails = () => {
    if (!selectedTransaction) {
      return (
        <div className="flex items-center justify-center h-full">
          <p className="text-gray-500">选择一个交易查看详情</p>
        </div>
      );
    }

    const tx = selectedTransaction;
    const riskColor = RISK_COLORS[tx.risk] || RISK_COLORS.low;

    return (
      <div className="p-6">
        <h3 className="text-xl font-bold mb-6">交易详情</h3>

        <div className="mb-6">
          <div className="flex items-center mb-2">
            <span
              className="inline-block w-4 h-4 rounded-full mr-2"
              style={{ backgroundColor: riskColor }}
            ></span>
            <span className="font-medium">风险等级: {tx.risk}</span>
          </div>

          <div className="text-sm bg-gray-50 p-4 rounded-lg break-all">
            <div className="mb-2">
              <span className="text-gray-500 mr-2">交易哈希:</span>
              <span>{tx.hash}</span>
            </div>
            <div className="mb-2">
              <span className="text-gray-500 mr-2">区块:</span>
              <span>{tx.blockNumber}</span>
            </div>
            <div className="mb-2">
              <span className="text-gray-500 mr-2">状态:</span>
              <span className={tx.status === 'confirmed' ? 'text-green-600' : 'text-yellow-600'}>
                {tx.status === 'confirmed' ? '已确认' : '待确认'}
              </span>
            </div>
          </div>
        </div>

        <div className="mb-6">
          <h4 className="font-medium mb-2">地址信息</h4>
          <div className="text-sm bg-gray-50 p-4 rounded-lg">
            <div className="mb-2">
              <span className="text-gray-500 mr-2">发送方:</span>
              <span>{tx.from}</span>
            </div>
            <div>
              <span className="text-gray-500 mr-2">接收方:</span>
              <span>{tx.to}</span>
            </div>
          </div>
        </div>

        <div className="mb-6">
          <h4 className="font-medium mb-2">交易信息</h4>
          <div className="text-sm bg-gray-50 p-4 rounded-lg grid grid-cols-2 gap-4">
            <div>
              <span className="text-gray-500 block">金额:</span>
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-700">交易金额:</span>
                <span className="font-medium">{formatCurrency(tx.value)}</span>
              </div>
            </div>
            <div>
              <span className="text-gray-500 block">Gas 价格:</span>
              <span>{tx.gasPrice.toFixed(2)} Gwei</span>
            </div>
            <div>
              <span className="text-gray-500 block">Gas 使用量:</span>
              <span>{tx.gasUsed.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-gray-500 block">时间:</span>
              <span>{formatDate(tx.timestamp, true)}</span>
            </div>
          </div>
        </div>

        <div className="flex space-x-2">
          <Button
            onClick={() => window.open(`https://etherscan.io/tx/${tx.hash}`, '_blank')}
            variant="outline"
          >
            在区块浏览器中查看
          </Button>
          <Button onClick={() => setSelectedTransaction(null)} variant="text">
            返回列表
          </Button>
        </div>
      </div>
    );
  };

  if (!isClient) {
    return null; // 服务端渲染时不显示内容
  }

  return (
    <MainLayout title="交易分析" description="分析区块链网络中的交易记录和关联信息">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">交易分析</h1>

        <div className="mb-6 flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <input
              type="text"
              placeholder="搜索交易哈希或地址..."
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={searchTerm}
              onChange={handleSearch}
            />
          </div>

          <div className="flex space-x-2">
            <select
              className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={filters.status}
              onChange={e => handleFilterChange('status', e.target.value)}
            >
              <option value="all">所有状态</option>
              <option value="confirmed">已确认</option>
              <option value="pending">待确认</option>
            </select>

            <select
              className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={filters.timeRange}
              onChange={e => handleFilterChange('timeRange', e.target.value)}
            >
              <option value="all">所有时间</option>
              <option value="24h">24小时内</option>
              <option value="7d">7天内</option>
              <option value="30d">30天内</option>
            </select>

            <select
              className="px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={filters.riskLevel}
              onChange={e => handleFilterChange('riskLevel', e.target.value)}
            >
              <option value="all">所有风险</option>
              <option value="high">高风险</option>
              <option value="medium">中风险</option>
              <option value="low">低风险</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-1 h-[calc(100vh-240px)] overflow-hidden flex flex-col">
            <div className="p-4 border-b">
              <h2 className="font-medium">交易列表</h2>
              <p className="text-sm text-gray-500">
                {isLoading ? '加载中...' : `${filteredTransactions.length}个交易`}
              </p>
            </div>

            <div className="flex-1 overflow-y-auto">
              {isLoading ? (
                <div className="flex items-center justify-center h-40">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
                </div>
              ) : error ? (
                <div className="p-4 text-red-500">{error}</div>
              ) : filteredTransactions.length === 0 ? (
                <div className="p-4 text-gray-500">未找到符合条件的交易</div>
              ) : (
                filteredTransactions.map(renderTransactionItem)
              )}
            </div>
          </Card>

          <Card className="lg:col-span-2 h-[calc(100vh-240px)] overflow-auto">
            {renderTransactionDetails()}
          </Card>
        </div>
      </div>
    </MainLayout>
  );
}
