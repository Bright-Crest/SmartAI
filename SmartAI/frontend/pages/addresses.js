import React, { useState, useEffect } from 'react';
import MainLayout from '../components/MainLayout';
import Card from '../components/Card';
import Button from '../components/Button';
import { useClientOnly } from '../hooks/useClientOnly';
import { formatDate, formatCurrency, truncateAddress } from '../utils/formatters';
import { RISK_LEVELS, RISK_COLORS } from '../utils/constants';

// 地址分析页面
export default function AddressesPage() {
  const [address, setAddress] = useState('');
  const [addressData, setAddressData] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [relatedAddresses, setRelatedAddresses] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // 使用客户端渲染模式
  const isClient = useClientOnly();

  // 处理地址查询
  const handleAddressSearch = async e => {
    e.preventDefault();

    if (!address || address.trim().length < 10) {
      setError('请输入有效的区块链地址');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // 模拟API请求获取地址数据
      setTimeout(() => {
        // 生成模拟数据
        const mockAddressData = {
          address: address,
          balance: Math.random() * 100,
          transactionCount: Math.floor(Math.random() * 1000),
          firstSeen: new Date(Date.now() - Math.random() * 86400000 * 365),
          lastActive: new Date(Date.now() - Math.random() * 86400000 * 10),
          risk: RISK_LEVELS[Math.floor(Math.random() * RISK_LEVELS.length)],
          tags: ['Exchange', 'High Volume'].filter(() => Math.random() > 0.5),
          riskScore: Math.floor(Math.random() * 100),
          type: Math.random() > 0.5 ? 'Contract' : 'EOA',
        };

        // 生成模拟交易
        const mockTransactions = Array(15)
          .fill()
          .map((_, index) => ({
            id: `0x${Math.random().toString(16).substring(2, 42)}`,
            hash: `0x${Math.random().toString(16).substring(2, 66)}`,
            from:
              Math.random() > 0.5 ? address : `0x${Math.random().toString(16).substring(2, 42)}`,
            to: Math.random() > 0.5 ? `0x${Math.random().toString(16).substring(2, 42)}` : address,
            value: Math.random() * 10,
            timestamp: new Date(Date.now() - Math.random() * 86400000 * 30),
            status: 'confirmed',
            blockNumber: Math.floor(Math.random() * 1000000),
            direction: Math.random() > 0.5 ? 'in' : 'out',
          }));

        // 生成相关地址
        const mockRelatedAddresses = Array(8)
          .fill()
          .map((_, index) => ({
            address: `0x${Math.random().toString(16).substring(2, 42)}`,
            transactionCount: Math.floor(Math.random() * 100),
            lastTransaction: new Date(Date.now() - Math.random() * 86400000 * 30),
            totalValue: Math.random() * 50,
            risk: RISK_LEVELS[Math.floor(Math.random() * RISK_LEVELS.length)],
          }));

        setAddressData(mockAddressData);
        setTransactions(mockTransactions);
        setRelatedAddresses(mockRelatedAddresses);
        setIsLoading(false);
      }, 800);
    } catch (err) {
      setError('获取地址数据失败，请稍后重试');
      setIsLoading(false);
    }
  };

  // 处理地址输入变化
  const handleAddressChange = e => {
    setAddress(e.target.value);
  };

  // 切换活动标签页
  const handleTabChange = tab => {
    setActiveTab(tab);
  };

  // 渲染地址信息概览
  const renderAddressOverview = () => {
    if (!addressData) return null;

    const { risk } = addressData;
    const riskColor = RISK_COLORS[risk] || RISK_COLORS.low;

    return (
      <>
        <div className="mb-6">
          <h3 className="text-xl font-bold mb-4">地址概览</h3>

          <div className="bg-white rounded-lg shadow p-4 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="mb-4">
                <span className="text-gray-500 text-sm">地址</span>
                <div className="font-medium break-all">{addressData.address}</div>
              </div>

              <div className="flex justify-between mb-2">
                <div className="text-gray-600">余额:</div>
                <div className="font-medium">{formatCurrency(addressData.balance)}</div>
              </div>

              <div className="mb-4">
                <span className="text-gray-500 text-sm">交易数量</span>
                <div className="font-medium">{addressData.transactionCount.toLocaleString()}</div>
              </div>

              <div className="mb-4">
                <span className="text-gray-500 text-sm">地址类型</span>
                <div className="font-medium">{addressData.type}</div>
              </div>
            </div>

            <div>
              <div className="mb-4">
                <span className="text-gray-500 text-sm">风险等级</span>
                <div className="flex items-center">
                  <span
                    className="inline-block w-3 h-3 rounded-full mr-2"
                    style={{ backgroundColor: riskColor }}
                  ></span>
                  <span className="font-medium capitalize">{risk}</span>
                </div>
              </div>

              <div className="mb-4">
                <span className="text-gray-500 text-sm">风险评分</span>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mb-1">
                  <div
                    className="h-2.5 rounded-full"
                    style={{
                      width: `${addressData.riskScore}%`,
                      backgroundColor: riskColor,
                    }}
                  ></div>
                </div>
                <div className="text-xs text-right">{addressData.riskScore}/100</div>
              </div>

              <div className="mb-4">
                <span className="text-gray-500 text-sm">首次交易</span>
                <div className="font-medium">{formatDate(addressData.firstSeen)}</div>
              </div>

              <div className="mb-4">
                <span className="text-gray-500 text-sm">最近活动</span>
                <div className="font-medium">{formatDate(addressData.lastActive)}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="mb-6">
          <h3 className="text-xl font-bold mb-4">标签</h3>

          <div className="bg-white rounded-lg shadow p-4">
            {addressData.tags && addressData.tags.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {addressData.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">暂无标签</p>
            )}
          </div>
        </div>
      </>
    );
  };

  // 渲染交易历史
  const renderTransactions = () => {
    if (!transactions || transactions.length === 0) {
      return (
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-500 text-center">暂无交易记录</p>
        </div>
      );
    }

    return (
      <div className="bg-white rounded-lg shadow overflow-hidden">
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
                  区块
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
                  方向
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  对方地址
                </th>
                <th
                  scope="col"
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  金额 (ETH)
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {transactions.map(tx => {
                const counterpartyAddress = tx.direction === 'in' ? tx.from : tx.to;

                return (
                  <tr key={tx.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600 hover:underline">
                      <a
                        href={`https://etherscan.io/tx/${tx.hash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {truncateAddress(tx.hash)}
                      </a>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {tx.blockNumber.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(tx.timestamp)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
                          tx.direction === 'in'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {tx.direction === 'in' ? '转入' : '转出'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {truncateAddress(counterpartyAddress)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      {formatCurrency(tx.value)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // 渲染相关地址
  const renderRelatedAddresses = () => {
    if (!relatedAddresses || relatedAddresses.length === 0) {
      return (
        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-500 text-center">暂无相关地址</p>
        </div>
      );
    }

    return (
      <div className="bg-white rounded-lg shadow">
        <ul className="divide-y divide-gray-200">
          {relatedAddresses.map(related => {
            const riskColor = RISK_COLORS[related.risk] || RISK_COLORS.low;

            return (
              <li key={related.address} className="p-4 hover:bg-gray-50">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="flex items-center">
                      <span
                        className="inline-block w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: riskColor }}
                      ></span>
                      <a
                        className="font-medium text-blue-600 hover:underline"
                        href={`/addresses?address=${related.address}`}
                      >
                        {truncateAddress(related.address)}
                      </a>
                    </div>
                    <div className="mt-1 text-sm text-gray-500">
                      最近交易: {formatDate(related.lastTransaction)}
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-sm font-medium">{related.transactionCount} 笔交易</div>
                    <div className="mt-1 text-sm text-gray-500">
                      总额: {formatCurrency(related.totalValue)}
                    </div>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    );
  };

  // 处理URL查询参数
  useEffect(() => {
    if (!isClient) return;

    // 从URL获取地址参数
    const queryParams = new URLSearchParams(window.location.search);
    const addressParam = queryParams.get('address');

    if (addressParam) {
      setAddress(addressParam);
      // 自动搜索该地址
      const event = { preventDefault: () => {} };
      handleAddressSearch(event);
    }
  }, [isClient]);

  // 如果不是客户端渲染，则不显示内容
  if (!isClient) {
    return null;
  }

  return (
    <MainLayout title="地址分析" description="分析区块链网络中的地址活动和关联信息">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">地址分析</h1>

        <Card className="mb-6">
          <div className="p-4">
            <form onSubmit={handleAddressSearch}>
              <div className="flex flex-col md:flex-row gap-3">
                <div className="flex-1">
                  <input
                    type="text"
                    placeholder="输入以太坊地址 (0x...)"
                    className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={address}
                    onChange={handleAddressChange}
                  />
                </div>

                <Button type="submit" disabled={isLoading}>
                  {isLoading ? '分析中...' : '分析地址'}
                </Button>
              </div>
            </form>

            {error && <p className="mt-2 text-red-500 text-sm">{error}</p>}
          </div>
        </Card>

        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : addressData ? (
          <div>
            <div className="border-b border-gray-200 mb-6">
              <nav className="-mb-px flex space-x-8">
                <button
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'overview'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => handleTabChange('overview')}
                >
                  概览
                </button>

                <button
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'transactions'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => handleTabChange('transactions')}
                >
                  交易历史 ({transactions.length})
                </button>

                <button
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === 'related'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                  onClick={() => handleTabChange('related')}
                >
                  相关地址 ({relatedAddresses.length})
                </button>
              </nav>
            </div>

            <div className="mb-8">
              {activeTab === 'overview' && renderAddressOverview()}
              {activeTab === 'transactions' && renderTransactions()}
              {activeTab === 'related' && renderRelatedAddresses()}
            </div>

            <div className="flex justify-center mt-4">
              <Button
                variant="outline"
                onClick={() =>
                  window.open(`https://etherscan.io/address/${addressData.address}`, '_blank')
                }
              >
                在区块浏览器中查看
              </Button>
            </div>
          </div>
        ) : (
          <Card className="p-8 text-center">
            <p className="text-gray-500">输入地址开始分析</p>
          </Card>
        )}
      </div>
    </MainLayout>
  );
}
