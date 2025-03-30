import React, { memo } from 'react';
import styles from '../../styles/AddressTracking.module.css';

// 创建单独的小组件以优化渲染
const RiskFactorsSection = memo(({ riskFactors }) => (
  <div className={styles.analysisSection}>
    <h3 className={styles.analysisTitle}>风险因素分析</h3>
    <div className={styles.smartFactors}>
      {riskFactors.map((factor, index) => (
        <div key={index} className={styles.smartFactor}>
          <div className={styles.smartFactorName}>{factor.name}</div>
          <div className={styles.smartFactorBar}>
            <div 
              className={styles.smartFactorFill} 
              style={{ 
                width: `${factor.value}%`,
                backgroundColor: factor.value >= 80 ? '#ef4444' : 
                                factor.value >= 60 ? '#f97316' : 
                                factor.value >= 40 ? '#eab308' : '#22c55e'
              }}
            ></div>
          </div>
          <div className={styles.smartFactorValue}>{factor.value}</div>
        </div>
      ))}
    </div>
  </div>
));

const PatternAnalysisSection = memo(({ tradePatterns }) => (
  <div className={styles.analysisSection}>
    <h3 className={styles.analysisTitle}>交易模式分析</h3>
    <div className={styles.patternList}>
      {tradePatterns.map((pattern, index) => (
        <div key={index} className={styles.patternItem}>
          <div className={styles.patternName}>{pattern.name}</div>
          <div className={styles.patternBar}>
            <div 
              className={styles.patternFill} 
              style={{ 
                width: `${pattern.value}%`,
                backgroundColor: index % 5 === 0 ? '#3b82f6' : 
                                index % 5 === 1 ? '#8b5cf6' : 
                                index % 5 === 2 ? '#ec4899' : 
                                index % 5 === 3 ? '#f97316' : '#14b8a6'
              }}
            ></div>
          </div>
          <div className={styles.patternValue}>{pattern.value}%</div>
        </div>
      ))}
    </div>
  </div>
));

const TimeAnalysisSection = memo(({ timeDistribution }) => (
  <div className={styles.analysisSection}>
    <h3 className={styles.analysisTitle}>交易时间分析</h3>
    <div className={styles.timeList}>
      {timeDistribution.map((time, index) => (
        <div key={index} className={styles.timeItem}>
          <div className={styles.timePeriod}>{time.period}</div>
          <div className={styles.timeValue}>{time.count}笔交易</div>
        </div>
      ))}
    </div>
  </div>
));

const RelatedEntitiesSection = memo(({ relatedEntities }) => (
  <div className={styles.analysisSection}>
    <h3 className={styles.analysisTitle}>相关实体分析</h3>
    <div className={styles.entityList}>
      {relatedEntities.map((entity, index) => (
        <div key={index} className={styles.entityItem}>
          <div className={styles.entityBadge} data-type={entity.type}>
            {entity.type}
          </div>
          <div className={styles.entityName}>{entity.name}</div>
          <div className={styles.entityCount}>{entity.count}笔交易</div>
        </div>
      ))}
    </div>
  </div>
));

// 主分析面板组件
const AnalysisPanel = memo(({ analysisData }) => {
  if (!analysisData) return <div className={styles.emptyState}>无分析数据</div>;
  
  const { riskFactors, timeDistribution, tradePatterns, relatedEntities } = analysisData;
  
  return (
    <div className={styles.analysisContainer}>
      {/* 分解为独立组件以优化渲染 */}
      {riskFactors && <RiskFactorsSection riskFactors={riskFactors} />}
      {tradePatterns && <PatternAnalysisSection tradePatterns={tradePatterns} />}
      {timeDistribution && <TimeAnalysisSection timeDistribution={timeDistribution} />}
      {relatedEntities && <RelatedEntitiesSection relatedEntities={relatedEntities} />}
    </div>
  );
});

// 添加displayName以便于调试
RiskFactorsSection.displayName = 'RiskFactorsSection';
PatternAnalysisSection.displayName = 'PatternAnalysisSection';
TimeAnalysisSection.displayName = 'TimeAnalysisSection';
RelatedEntitiesSection.displayName = 'RelatedEntitiesSection';
AnalysisPanel.displayName = 'AnalysisPanel';

export default AnalysisPanel; 