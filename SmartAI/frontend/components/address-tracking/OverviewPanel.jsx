import React, { memo } from 'react';
import styles from '../../styles/AddressTracking.module.css';

// 使用memo优化渲染性能
const OverviewPanel = memo(({ nodeDetails, transactions, analysisData, getRiskLevelLabel }) => {
  if (!nodeDetails) return null;
  
  // 只显示最近的3笔交易，避免渲染过多数据
  const recentTransactions = transactions?.slice(0, 3) || [];
  const recentEntities = analysisData?.relatedEntities?.slice(0, 3) || [];
  
  return (
    <div className={styles.overview}>
      <p>
        <strong>地址摘要:</strong> 该地址总共有 <strong>{nodeDetails.totalTx || 0}</strong> 笔交易，
        当前余额为 <strong>{nodeDetails.balance || '0 ETH'}</strong>。
        {nodeDetails.riskScore && (
          <>
            {' '}风险评分为 <strong>{nodeDetails.riskScore}/100</strong>，
            被评为<strong>{getRiskLevelLabel(nodeDetails.riskScore).label}</strong>。
          </>
        )}
      </p>
      
      {/* 最近交易预览 */}
      {recentTransactions.length > 0 && (
        <>
          <h3 className={styles.subTitle}>最近交易</h3>
          <div className={styles.transactionsList}>
            {recentTransactions.map((tx) => (
              <div key={tx.id} className={styles.transactionItem}>
                <div className={styles.transactionDirection}>
                  {tx.type === 'in' ? '入账' : '出账'}
                </div>
                <div className={styles.transactionDate}>{tx.date}</div>
                <div className={`${styles.transactionValue} ${tx.type === 'in' ? styles.valueIn : styles.valueOut}`}>
                  {tx.type === 'in' ? '+' : '-'}{tx.value}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
      
      {/* 关联实体预览 */}
      {recentEntities.length > 0 && (
        <>
          <h3 className={styles.subTitle}>主要关联实体</h3>
          <div className={styles.entityList}>
            {recentEntities.map((entity, index) => (
              <div key={index} className={styles.entityItem}>
                <div className={styles.entityBadge} data-type={entity.type}>
                  {entity.type}
                </div>
                <div className={styles.entityName}>{entity.name}</div>
                <div className={styles.entityCount}>{entity.count}笔交易</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
});

OverviewPanel.displayName = 'OverviewPanel';

export default OverviewPanel; 