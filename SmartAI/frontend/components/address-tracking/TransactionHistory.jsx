import React, { memo, useState, useCallback, useRef, useEffect } from 'react';
import styles from '../../styles/AddressTracking.module.css';
import { FiClock } from 'react-icons/fi';

// 虚拟列表实现
const VirtualizedTransactionList = ({ transactions }) => {
  const containerRef = useRef(null);
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 });
  const [containerHeight, setContainerHeight] = useState(0);
  const itemHeight = 100; // 每项交易的估计高度
  const buffer = 5; // 缓冲区大小
  
  // 初始化容器高度
  useEffect(() => {
    if (containerRef.current) {
      setContainerHeight(containerRef.current.clientHeight);
    }
  }, []);
  
  // 处理滚动事件，更新可视范围
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    
    const scrollTop = containerRef.current.scrollTop;
    const visibleItems = Math.ceil(containerHeight / itemHeight);
    
    const start = Math.max(0, Math.floor(scrollTop / itemHeight) - buffer);
    const end = Math.min(transactions.length, Math.floor(scrollTop / itemHeight) + visibleItems + buffer);
    
    setVisibleRange({ start, end });
  }, [containerHeight, transactions.length]);
  
  // 监听滚动事件
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      handleScroll(); // 初始化可视范围
      
      return () => {
        container.removeEventListener('scroll', handleScroll);
      };
    }
  }, [handleScroll]);
  
  // 渲染可视区域的交易
  const visibleTransactions = transactions.slice(visibleRange.start, visibleRange.end);
  
  // 计算内容总高度和顶部偏移
  const totalHeight = transactions.length * itemHeight;
  const topPadding = visibleRange.start * itemHeight;
  
  return (
    <div 
      ref={containerRef}
      className={styles.virtualScrollContainer}
      style={{ height: '100%', overflow: 'auto' }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ position: 'absolute', top: topPadding, width: '100%' }}>
          {visibleTransactions.map((tx) => (
            <div key={tx.id} className={styles.transaction}>
              <div className={styles.txDate}>{tx.date}</div>
              <div className={`${styles.txType} ${tx.type === 'in' ? styles.txIn : styles.txOut}`}>
                {tx.type === 'in' ? '入账' : '出账'}
              </div>
              <div className={styles.txValue}>
                {tx.type === 'in' ? '+' : '-'}{tx.value}
              </div>
              <div className={styles.txHash}>
                <span>交易哈希:</span> {tx.hash}
              </div>
              <div className={styles.txDirection}>
                {tx.type === 'in' ? (
                  <span>从 {tx.counterparty} 转入</span>
                ) : (
                  <span>转出至 {tx.counterparty}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// 主交易历史组件
const TransactionHistory = memo(({ transactions }) => {
  return (
    <div className={styles.transactionList} style={{ height: '400px', overflow: 'hidden' }}>
      {transactions.length > 0 ? (
        <VirtualizedTransactionList transactions={transactions} />
      ) : (
        <div className={styles.emptyState}>
          <div className={styles.emptyStateIcon}><FiClock size={30} /></div>
          <h3 className={styles.emptyStateTitle}>无交易记录</h3>
          <p className={styles.emptyStateDescription}>该地址没有交易历史</p>
        </div>
      )}
    </div>
  );
});

TransactionHistory.displayName = 'TransactionHistory';

export default TransactionHistory; 