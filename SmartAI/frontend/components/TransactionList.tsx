import React, { useState, useEffect, useCallback, useRef, useMemo, Suspense, lazy } from 'react';
import { format } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import RiskBadge, { RiskLevel } from './RiskBadge';
// 导入无障碍组件
import FocusableItem from './a11y/FocusableItem';
import A11yFormInput from './a11y/A11yFormInput';
import useKeyboardNavigation from '../hooks/useKeyboardNavigation';
// 新导入的钩子，用于响应式设计
import useMediaQuery from '../hooks/useMediaQuery';

// 交易数据类型
interface Transaction {
  id: string;
  from: string;
  to: string;
  amount: string;
  timestamp: number;
  riskLevel: RiskLevel;
  riskScore?: number;
  status?: 'confirmed' | 'pending' | 'failed';
}

// 模拟数据获取函数
const fetchTransactions = async (): Promise<Transaction[]> => {
  // 模拟API调用延迟
  await new Promise(resolve => setTimeout(resolve, 800));
  
  // 模拟数据
  return [
    {
      id: 'tx1',
      from: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
      to: '0x8Ba1f109551bD432803012645Ac136ddd64DBA72',
      amount: '1.5 ETH',
      timestamp: Date.now() - 1000 * 60 * 5, // 5分钟前
      riskLevel: 'low',
      status: 'confirmed'
    },
    {
      id: 'tx2',
      from: '0x2932b7A2355D6fecc4b5c0B6BD44cC31df247a2e',
      to: '0x6B175474E89094C44Da98b954EedeAC495271d0F',
      amount: '5000 USDC',
      timestamp: Date.now() - 1000 * 60 * 15, // 15分钟前
      riskLevel: 'medium',
      riskScore: 45,
      status: 'confirmed'
    },
    {
      id: 'tx3',
      from: '0x5B38Da6a701c568545dCfcB03FcB875f56beddC4',
      to: '0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B',
      amount: '10 ETH',
      timestamp: Date.now() - 1000 * 60 * 30, // 30分钟前
      riskLevel: 'high',
      riskScore: 75,
      status: 'pending'
    },
    {
      id: 'tx4',
      from: '0xdD870fA1b7C4700F2BD7f44238821C26f7392148',
      to: '0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326',
      amount: '50,000 USDT',
      timestamp: Date.now() - 1000 * 60 * 60, // 1小时前
      riskLevel: 'critical',
      riskScore: 95,
      status: 'failed'
    }
  ];
};

// 格式化地址显示
const formatAddress = (address: string): string => {
  return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
};

// 懒加载状态指示组件
const LoadingIndicator = () => (
  <div className="flex justify-center p-4">
    <div className="animate-pulse flex space-x-4">
      <div className="rounded-full bg-gray-200 h-10 w-10"></div>
      <div className="flex-1 space-y-4 py-1">
        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        <div className="space-y-2">
          <div className="h-4 bg-gray-200 rounded"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
        </div>
      </div>
    </div>
  </div>
);

const TransactionList: React.FC = () => {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [selectedTransaction, setSelectedTransaction] = useState<string | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const resultStatusRef = useRef<HTMLDivElement>(null);
  const tableRef = useRef<HTMLDivElement>(null);
  
  // 响应式设计检测
  const isMobile = useMediaQuery('(max-width: 640px)');
  const isTablet = useMediaQuery('(min-width: 641px) and (max-width: 1024px)');
  
  // 使用键盘导航Hook
  const { handleEnterAndSpace, handleArrowKeys } = useKeyboardNavigation();

  // 使用useCallback优化数据加载函数
  const loadTransactions = useCallback(async () => {
    try {
      setLoading(true);
      const data = await fetchTransactions();
      setTransactions(data);
      setFilteredTransactions(data);
      setError(null);
      
      // 更新屏幕阅读器状态消息
      if (resultStatusRef.current) {
        resultStatusRef.current.textContent = `已加载 ${data.length} 条交易记录`;
      }
    } catch (err) {
      setError('加载交易数据失败');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  // 处理搜索
  const handleSearch = useCallback((e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!searchTerm.trim()) {
      setFilteredTransactions(transactions);
      
      if (resultStatusRef.current) {
        resultStatusRef.current.textContent = `显示全部 ${transactions.length} 条交易记录`;
      }
      return;
    }

    const filtered = transactions.filter(tx => 
      tx.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.from.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.to.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tx.amount.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    setFilteredTransactions(filtered);
    
    // 更新屏幕阅读器状态消息
    if (resultStatusRef.current) {
      resultStatusRef.current.textContent = filtered.length > 0 
        ? `找到 ${filtered.length} 条匹配的交易记录` 
        : '没有找到匹配的交易记录';
    }
  }, [searchTerm, transactions]);

  // 处理状态筛选
  const handleStatusFilter = useCallback((status: string | null) => {
    setStatusFilter(status);
    
    if (!status) {
      setFilteredTransactions(transactions);
      
      if (resultStatusRef.current) {
        resultStatusRef.current.textContent = `显示全部 ${transactions.length} 条交易记录`;
      }
      return;
    }
    
    const filtered = transactions.filter(tx => tx.status === status);
    setFilteredTransactions(filtered);
    
    // 更新屏幕阅读器状态消息
    if (resultStatusRef.current) {
      const statusText = status === 'confirmed' ? '已确认' : status === 'pending' ? '处理中' : '失败';
      resultStatusRef.current.textContent = `显示 ${filtered.length} 条${statusText}的交易记录`;
    }
  }, [transactions]);

  // 显示交易详情
  const showTransactionDetails = useCallback((txId: string) => {
    setSelectedTransaction(txId);
    // 此处可添加查看交易详情的逻辑
    console.log('查看交易详情:', txId);
    
    // 通知屏幕阅读器
    if (resultStatusRef.current) {
      resultStatusRef.current.textContent = `查看交易 ${txId} 的详细信息`;
    }
  }, []);

  // 搜索或状态过滤变化时更新过滤后的交易
  useEffect(() => {
    if (statusFilter) {
      handleStatusFilter(statusFilter);
    } else if (searchTerm) {
      handleSearch();
    } else {
      setFilteredTransactions(transactions);
    }
  }, [searchTerm, statusFilter, transactions, handleSearch, handleStatusFilter]);

  // 组件挂载时加载数据
  useEffect(() => {
    loadTransactions();
  }, [loadTransactions]);

  // 处理表格键盘导航
  const handleTableKeyDown = useCallback((e: React.KeyboardEvent, txIndex: number) => {
    handleArrowKeys({
      up: () => {
        const rowElements = document.querySelectorAll('[role="row"][aria-rowindex]');
        if (txIndex > 0) {
          (rowElements[txIndex - 1] as HTMLElement).focus();
        } else {
          // 焦点回到搜索框
          searchInputRef.current?.focus();
        }
      },
      down: () => {
        const rowElements = document.querySelectorAll('[role="row"][aria-rowindex]');
        if (txIndex < rowElements.length - 1) {
          (rowElements[txIndex + 1] as HTMLElement).focus();
        }
      },
      home: () => {
        const rowElements = document.querySelectorAll('[role="row"][aria-rowindex]');
        if (rowElements.length > 0) {
          (rowElements[0] as HTMLElement).focus();
        }
      },
      end: () => {
        const rowElements = document.querySelectorAll('[role="row"][aria-rowindex]');
        if (rowElements.length > 0) {
          (rowElements[rowElements.length - 1] as HTMLElement).focus();
        }
      }
    })(e);
    
    // 处理Enter和Space键
    handleEnterAndSpace(() => {
      showTransactionDetails(filteredTransactions[txIndex].id);
    })(e);
  }, [filteredTransactions, showTransactionDetails, handleArrowKeys, handleEnterAndSpace]);

  // 移动设备视图下简化交易显示
  const renderMobileTransaction = (tx: Transaction, index: number) => (
    <motion.div
      key={tx.id}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="bg-white rounded-lg shadow-sm p-3 mb-2 border-l-4"
      style={{ 
        borderLeftColor: 
          tx.riskLevel === 'critical' ? '#ef4444' : 
          tx.riskLevel === 'high' ? '#f97316' : 
          tx.riskLevel === 'medium' ? '#eab308' : 
          '#22c55e'
      }}
      tabIndex={0}
      role="button"
      aria-label={`交易 ${tx.id}, 从 ${formatAddress(tx.from)} 到 ${formatAddress(tx.to)}, 金额 ${tx.amount}`}
      onClick={() => showTransactionDetails(tx.id)}
      onKeyDown={(e) => handleTableKeyDown(e, index)}
    >
      <div className="flex justify-between items-center">
        <div className="font-medium truncate max-w-[60%]">{formatAddress(tx.id)}</div>
        <RiskBadge level={tx.riskLevel} />
      </div>
      <div className="mt-2 flex justify-between items-center text-sm">
        <div className="text-gray-500">
          {format(new Date(tx.timestamp), 'MM-dd HH:mm')}
        </div>
        <div className="font-medium">{tx.amount}</div>
      </div>
    </motion.div>
  );

  // 加载状态 - 移动设备优化
  if (loading) {
    return (
      <div 
        className="animate-pulse" 
        role="progressbar" 
        aria-label="加载中" 
        aria-busy="true"
      >
        <div className="h-10 bg-gray-200 rounded mb-4"></div>
        {[...Array(isMobile ? 3 : 4)].map((_, i) => (
          <div key={i} className={`h-${isMobile ? '14' : '16'} bg-gray-100 rounded-md mb-2`}></div>
        ))}
      </div>
    );
  }

  // 错误状态
  if (error) {
    return (
      <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded" role="alert">
        <h2 className="font-bold text-lg">加载失败</h2>
        <p>{error}</p>
        <button 
          onClick={() => loadTransactions()}
          className="mt-2 bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 rounded transition-colors"
        >
          重试
        </button>
      </div>
    );
  }

  // 针对移动设备的简化视图
  if (isMobile) {
    return (
      <div className="transaction-list-container">
        <div className="sticky top-0 z-10 bg-white pb-3">
          <div className="flex items-center mb-4">
            <A11yFormInput
              ref={searchInputRef}
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="搜索交易..."
              className="flex-1 p-2 border border-gray-300 rounded-l-md"
              label="搜索交易"
              id="mobile-search-input"
              name="mobile-search"
            />
            <button
              onClick={() => handleSearch()}
              className="bg-blue-500 text-white p-2 rounded-r-md hover:bg-blue-600 transition-colors"
              aria-label="搜索"
            >
              搜索
            </button>
          </div>
          
          <div className="flex space-x-2 overflow-x-auto pb-2 -mx-2 px-2">
            <FocusableItem
              onClick={() => handleStatusFilter(null)}
              className={`px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${
                statusFilter === null
                  ? 'bg-blue-100 text-blue-700 font-medium'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              role="radio"
              aria-checked={statusFilter === null}
            >
              全部
            </FocusableItem>
            <FocusableItem
              onClick={() => handleStatusFilter('confirmed')}
              className={`px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${
                statusFilter === 'confirmed'
                  ? 'bg-green-100 text-green-700 font-medium'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              role="radio"
              aria-checked={statusFilter === 'confirmed'}
            >
              已确认
            </FocusableItem>
            <FocusableItem
              onClick={() => handleStatusFilter('pending')}
              className={`px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${
                statusFilter === 'pending'
                  ? 'bg-yellow-100 text-yellow-700 font-medium'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              role="radio"
              aria-checked={statusFilter === 'pending'}
            >
              处理中
            </FocusableItem>
            <FocusableItem
              onClick={() => handleStatusFilter('failed')}
              className={`px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${
                statusFilter === 'failed'
                  ? 'bg-red-100 text-red-700 font-medium'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              role="radio"
              aria-checked={statusFilter === 'failed'}
            >
              失败
            </FocusableItem>
          </div>
        </div>
        
        {/* 结果状态通知 - 对屏幕阅读器可见，视觉上隐藏 */}
        <div className="sr-only" aria-live="polite" ref={resultStatusRef}>
          {filteredTransactions.length > 0
            ? `显示 ${filteredTransactions.length} 条交易记录`
            : '没有找到交易记录'}
        </div>
        
        {filteredTransactions.length === 0 ? (
          <div className="py-8 text-center text-gray-500">
            <p>没有找到交易记录</p>
          </div>
        ) : (
          <div className="space-y-2">
            <AnimatePresence>
              {filteredTransactions.map((tx, index) => renderMobileTransaction(tx, index))}
            </AnimatePresence>
          </div>
        )}
      </div>
    );
  }

  // 桌面设备标准视图
  return (
    <div className="transaction-list-container">
      {/* 原始的桌面版搜索和筛选UI */}
      <div className="mb-4 flex items-center justify-between flex-wrap gap-2">
        <form onSubmit={handleSearch} className="flex items-center min-w-[300px]">
          <A11yFormInput
            ref={searchInputRef}
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="搜索交易..."
            className="flex-1 p-2 border border-gray-300 rounded-l-md"
            label="搜索交易"
            id="desktop-search-input"
            name="desktop-search"
          />
          <button
            type="submit"
            className="bg-blue-500 text-white p-2 rounded-r-md hover:bg-blue-600 transition-colors"
            aria-label="搜索"
          >
            搜索
          </button>
        </form>
        
        <div className="flex space-x-2" role="radiogroup" aria-label="交易状态筛选">
          <FocusableItem
            onClick={() => handleStatusFilter(null)}
            className={`px-4 py-2 rounded-md ${
              statusFilter === null
                ? 'bg-blue-100 text-blue-700 font-medium'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            role="radio"
            aria-checked={statusFilter === null}
          >
            全部
          </FocusableItem>
          <FocusableItem
            onClick={() => handleStatusFilter('confirmed')}
            className={`px-4 py-2 rounded-md ${
              statusFilter === 'confirmed'
                ? 'bg-green-100 text-green-700 font-medium'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            role="radio"
            aria-checked={statusFilter === 'confirmed'}
          >
            已确认
          </FocusableItem>
          <FocusableItem
            onClick={() => handleStatusFilter('pending')}
            className={`px-4 py-2 rounded-md ${
              statusFilter === 'pending'
                ? 'bg-yellow-100 text-yellow-700 font-medium'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            role="radio"
            aria-checked={statusFilter === 'pending'}
          >
            处理中
          </FocusableItem>
          <FocusableItem
            onClick={() => handleStatusFilter('failed')}
            className={`px-4 py-2 rounded-md ${
              statusFilter === 'failed'
                ? 'bg-red-100 text-red-700 font-medium'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            role="radio"
            aria-checked={statusFilter === 'failed'}
          >
            失败
          </FocusableItem>
        </div>
      </div>
      
      {/* 结果状态通知 - 对屏幕阅读器可见，视觉上隐藏 */}
      <div className="sr-only" aria-live="polite" ref={resultStatusRef}>
        {filteredTransactions.length > 0
          ? `显示 ${filteredTransactions.length} 条交易记录`
          : '没有找到交易记录'}
      </div>
      
      {filteredTransactions.length === 0 ? (
        <div className="py-8 text-center text-gray-500">
          <p>没有找到交易记录</p>
        </div>
      ) : (
        <div className="overflow-x-auto" ref={tableRef}>
          <table className="min-w-full divide-y divide-gray-200" role="grid" aria-label="交易列表">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  交易ID
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  发送方
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  接收方
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  金额
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  时间
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  风险级别
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  状态
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <AnimatePresence>
                {filteredTransactions.map((tx, index) => (
                  <motion.tr
                    key={tx.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className={`cursor-pointer hover:bg-gray-50 ${selectedTransaction === tx.id ? 'bg-blue-50' : ''}`}
                    onClick={() => showTransactionDetails(tx.id)}
                    tabIndex={0}
                    role="row"
                    aria-rowindex={index + 1}
                    aria-selected={selectedTransaction === tx.id}
                    onKeyDown={(e) => handleTableKeyDown(e, index)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {formatAddress(tx.id)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatAddress(tx.from)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatAddress(tx.to)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {tx.amount}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {format(new Date(tx.timestamp), 'yyyy-MM-dd HH:mm')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <RiskBadge level={tx.riskLevel} score={tx.riskScore} />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                        ${tx.status === 'confirmed' ? 'bg-green-100 text-green-800' : 
                          tx.status === 'pending' ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'}`}>
                        {tx.status === 'confirmed' ? '已确认' : 
                         tx.status === 'pending' ? '处理中' : '失败'}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default TransactionList; 