import React, { useMemo } from 'react';
import { arePropsEqual } from '../utils/performance';

// 风险等级类型
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

// 组件属性类型
interface RiskBadgeProps {
  level: RiskLevel;
  score?: number;
  showScore?: boolean;
  className?: string;
}

/**
 * 风险徽章组件 - 使用React.memo优化渲染性能
 */
const RiskBadge: React.FC<RiskBadgeProps> = React.memo(({ level, score, showScore = false, className = '' }) => {
  // 根据风险级别确定视觉和语义特性
  const { bgColor, textColor, icon, label, description } = useMemo(() => {
    const riskInfo = {
      low: {
        bgColor: 'bg-green-100',
        textColor: 'text-green-800',
        icon: '✓',
        label: '低风险',
        description: '此交易风险级别低，可以放心处理'
      },
      medium: {
        bgColor: 'bg-yellow-100',
        textColor: 'text-yellow-800',
        icon: '!',
        label: '中等风险',
        description: '此交易具有中等风险，建议审核后处理'
      },
      high: {
        bgColor: 'bg-orange-100',
        textColor: 'text-orange-800',
        icon: '!!',
        label: '高风险',
        description: '此交易风险级别高，请谨慎处理并进行额外验证'
      },
      critical: {
        bgColor: 'bg-red-100',
        textColor: 'text-red-800',
        icon: '!!!',
        label: '极高风险',
        description: '此交易风险极高，强烈建议拒绝或进行全面调查'
      }
    };
    
    return riskInfo[level];
  }, [level]);

  return (
    <div 
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${bgColor} ${textColor} ${className}`}
      role="status"
      aria-atomic="true"
      aria-live="polite"
    >
      <span className="sr-only">{description}</span>
      <span aria-hidden="true">{icon}</span>
      <span className="ml-1">{label}</span>
      {showScore && score !== undefined && (
        <span className="ml-1.5 bg-white bg-opacity-25 px-1.5 py-0.5 rounded-sm">
          {score}
        </span>
      )}
    </div>
  );
});

RiskBadge.displayName = 'RiskBadge';

// 使用React.memo包装组件，并使用自定义比较函数
export default RiskBadge; 