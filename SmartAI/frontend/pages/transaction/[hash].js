import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import MainLayout from '../../components/MainLayout';
import { useClientOnly } from '../../hooks/useClientOnly';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';

// 风险级别颜色
const getRiskLevelColor = level => {
  switch (level) {
    case 'high':
      return 'text-red-500';
    case 'medium':
      return 'text-yellow-500';
    case 'low':
      return 'text-green-500';
    default:
      return 'text-gray-500';
  }
};

// 风险级别文本
const getRiskLevelText = level => {
  switch (level) {
    case 'high':
      return '高风险';
    case 'medium':
      return '中风险';
    case 'low':
      return '低风险';
    default:
      return '未知';
  }
};

// 生成模拟交易数据
const generateTransactionData = hash => {
  const riskLevel = ['high', 'medium', 'low', 'unknown'][Math.floor(Math.random() * 4)];
  const from = `0x${Math.random().toString(16).substring(2, 42)}`;
  const to = `0x${Math.random().toString(16).substring(2, 42)}`;

  return {
    hash,
    blockNumber: Math.floor(Math.random() * 10000000) + 15000000,
    timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000),
    from,
    to,
    value: Math.random() * 10,
    gasPrice: Math.random() * 100,
    gasUsed: Math.floor(Math.random() * 1000000) + 21000,
    status: Math.random() > 0.1 ? 'success' : 'failed',
    contractInteraction: Math.random() > 0.5,
    method: Math.random() > 0.5 ? 'transfer' : 'approve',
    riskLevel,
    riskFlags:
      riskLevel === 'high'
        ? ['可疑交易模式', '与已标记地址交互', '异常金额']
        : riskLevel === 'medium'
          ? ['新地址交互', '非常规交易时间']
          : [],
    relatedTransactions: Array(Math.floor(Math.random() * 5))
      .fill()
      .map(() => ({
        hash: `0x${Math.random().toString(16).substring(2, 66)}`,
        timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000),
        relation: ['来源', '目标', '相似模式'][Math.floor(Math.random() * 3)],
      })),
    events: [
      {
        name: 'Transfer',
        params: [
          { name: 'from', value: from, type: 'address' },
          { name: 'to', value: to, type: 'address' },
          { name: 'value', value: Math.random() * 1000, type: 'uint256' },
        ],
      },
      ...(Math.random() > 0.7
        ? [
            {
              name: 'Approval',
              params: [
                { name: 'owner', value: from, type: 'address' },
                { name: 'spender', value: to, type: 'address' },
                { name: 'value', value: Math.random() * 10000, type: 'uint256' },
              ],
            },
          ]
        : []),
    ],
  };
};

// 格式化地址显示
const formatAddress = (address, length = 6) => {
  if (!address) return '';
  return `${address.substring(0, length)}...${address.substring(address.length - length)}`;
};

// 格式化日期时间
const formatDateTime = date => {
  if (!date) return '';
  return new Date(date).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

export default function TransactionDetailPage() {
  const router = useRouter();
  const { hash } = router.query;
  const [transactionData, setTransactionData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const isClient = useClientOnly();

  // 获取交易数据
  useEffect(() => {
    if (!isClient || !hash) return;

    const fetchTransactionData = async () => {
      try {
        setIsLoading(true);
        // 模拟API请求延迟
        setTimeout(() => {
          const data = generateTransactionData(hash);
          setTransactionData(data);
          setIsLoading(false);
        }, 800);
      } catch (error) {
        console.error('获取交易数据失败', error);
        setIsLoading(false);
      }
    };

    fetchTransactionData();
  }, [isClient, hash]);

  // 渲染加载状态
  if (isLoading || !transactionData) {
    return (
      <MainLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-8">
        {/* 交易详情头部 */}
        <Card className="p-6 mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">交易详情</h1>
              <p className="text-sm font-mono text-gray-600 dark:text-gray-400 break-all">
                {transactionData.hash}
              </p>
            </div>
            <div className="mt-4 md:mt-0">
              <div className="flex items-center">
                <span
                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    transactionData.status === 'success'
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                  }`}
                >
                  {transactionData.status === 'success' ? '成功' : '失败'}
                </span>

                {transactionData.riskLevel !== 'unknown' && (
                  <span
                    className={`ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      transactionData.riskLevel === 'high'
                        ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                        : transactionData.riskLevel === 'medium'
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                          : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                    }`}
                  >
                    {getRiskLevelText(transactionData.riskLevel)}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* 交易概览 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-y-4 gap-x-6">
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">区块高度</span>
              <span className="text-base">{transactionData.blockNumber}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">交易时间</span>
              <span className="text-base">{formatDateTime(transactionData.timestamp)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">发送方</span>
              <div className="flex items-center text-base">
                <span className="font-mono">{formatAddress(transactionData.from, 12)}</span>
                <Button
                  variant="outline"
                  size="sm"
                  className="ml-2 py-0.5 px-2 text-xs"
                  onClick={() => router.push(`/address/${transactionData.from}`)}
                >
                  查看
                </Button>
              </div>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">接收方</span>
              <div className="flex items-center text-base">
                <span className="font-mono">{formatAddress(transactionData.to, 12)}</span>
                <Button
                  variant="outline"
                  size="sm"
                  className="ml-2 py-0.5 px-2 text-xs"
                  onClick={() => router.push(`/address/${transactionData.to}`)}
                >
                  查看
                </Button>
              </div>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">交易金额</span>
              <span className="text-base">{transactionData.value.toFixed(4)} ETH</span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-gray-500 dark:text-gray-400">Gas价格</span>
              <span className="text-base">{transactionData.gasPrice.toFixed(2)} Gwei</span>
            </div>
            {transactionData.contractInteraction && (
              <>
                <div className="flex flex-col">
                  <span className="text-sm text-gray-500 dark:text-gray-400">交互类型</span>
                  <span className="text-base">合约交互</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-sm text-gray-500 dark:text-gray-400">调用方法</span>
                  <span className="text-base">{transactionData.method}</span>
                </div>
              </>
            )}
          </div>
        </Card>

        {/* 交易事件和风险分析 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* 事件日志 */}
          <Card className="p-6 lg:col-span-2">
            <h2 className="text-lg font-semibold mb-4">事件日志</h2>
            {transactionData.events.length > 0 ? (
              <div className="space-y-4">
                {transactionData.events.map((event, eventIndex) => (
                  <div
                    key={eventIndex}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
                  >
                    <h3 className="text-md font-medium mb-2">事件：{event.name}</h3>
                    <div className="space-y-2">
                      {event.params.map((param, paramIndex) => (
                        <div key={paramIndex} className="grid grid-cols-5 gap-2">
                          <span className="col-span-1 text-sm text-gray-500 dark:text-gray-400">
                            {param.name}:
                          </span>
                          <span className="col-span-3 text-sm font-mono break-all">
                            {param.type === 'address'
                              ? formatAddress(param.value, 8)
                              : param.value.toString()}
                          </span>
                          <span className="col-span-1 text-xs text-gray-400 dark:text-gray-500">
                            {param.type}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 dark:text-gray-400">无事件日志</p>
            )}
          </Card>

          {/* 风险分析 */}
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-4">风险分析</h2>
            <div className="mb-4">
              <div className="flex items-center mb-2">
                <span className="text-gray-600 dark:text-gray-400 mr-2">风险评级:</span>
                <span className={`font-bold ${getRiskLevelColor(transactionData.riskLevel)}`}>
                  {getRiskLevelText(transactionData.riskLevel)}
                </span>
              </div>

              {transactionData.riskFlags.length > 0 ? (
                <div className="mt-4">
                  <h3 className="text-md font-medium mb-2">风险因素:</h3>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    {transactionData.riskFlags.map((flag, index) => (
                      <li key={index} className="text-gray-700 dark:text-gray-300">
                        {flag}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {transactionData.riskLevel === 'low'
                    ? '此交易未检测到明显风险'
                    : '未检测到风险因素'}
                </p>
              )}
            </div>

            {/* 相关交易 */}
            {transactionData.relatedTransactions.length > 0 && (
              <div className="mt-6">
                <h3 className="text-md font-medium mb-2">相关交易:</h3>
                <div className="space-y-2">
                  {transactionData.relatedTransactions.map((tx, index) => (
                    <div
                      key={index}
                      className="border border-gray-200 dark:border-gray-700 p-2 rounded"
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {tx.relation}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {formatDateTime(tx.timestamp)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-mono">{formatAddress(tx.hash, 8)}</span>
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs py-0.5 px-2"
                          onClick={() => router.push(`/transaction/${tx.hash}`)}
                        >
                          查看
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </div>

        {/* 操作按钮 */}
        <div className="flex justify-center space-x-4">
          <Button variant="outline" onClick={() => router.back()}>
            返回
          </Button>
          <Button variant="primary" onClick={() => router.push('/network-analysis')}>
            查看网络分析
          </Button>
        </div>
      </div>
    </MainLayout>
  );
}
