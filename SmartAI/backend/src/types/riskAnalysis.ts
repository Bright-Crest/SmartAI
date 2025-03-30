import { RiskLevel, RiskAnalysis } from './events';

export { RiskLevel, RiskAnalysis };

/**
 * 行为标签接口
 * 用于描述地址的行为特征
 */
export interface BehaviorTag {
  /** 标签名称 */
  name: string;
  
  /** 标签置信度 (0-1) */
  confidence: number;
  
  /** 标签类别 */
  category: string;
  
  /** 标签描述 */
  description: string;
  
  /** 标签检测时间 */
  detectedAt?: Date;
  
  /** 来源 */
  source?: string;
  
  /** 相关数据 */
  metadata?: Record<string, any>;
}

export interface BehaviorAnalysis {
  pattern: string;
  confidence: number;
  details: Record<string, any>;
}

export interface GraphAnalysis {
  centrality: number;
  degree: number;
  clustering: number;
  paths: Array<{
    from: string;
    to: string;
    weight: number;
  }>;
}

export interface AIAnalysis {
  behaviorAnalysis: BehaviorAnalysis;
  graphAnalysis: GraphAnalysis;
  summary: string;
}

/**
 * 风险维度定义
 * 表示一个风险评分维度
 */
export interface RiskDimension {
  name: string;
  score: number;
  weight: number;
  tags: string[];
}

/**
 * 增强的风险分析结果
 * 包含基础风险分析结果，以及AI分析和组合分析结果
 */
export interface EnhancedRiskAnalysis extends RiskAnalysis {
  /**
   * 分析时间戳
   */
  timestamp: number;

  /**
   * 行为标签列表
   * 详细描述地址的行为特征
   */
  behaviorTags: BehaviorTag[];

  /**
   * 风险维度列表
   * 按不同维度划分的风险评分
   */
  dimensions: RiskDimension[];

  /**
   * AI分析结果
   */
  aiAnalysis: AIAnalysis;

  /**
   * 组合分析结果
   */
  combinations: Array<{
    /**
     * 风险因素
     */
    factors: string[];
    /**
     * 风险分数
     */
    score: number;
    /**
     * 描述
     */
    description: string;
  }>;

  /**
   * 处理动作
   */
  action: 'block' | 'alert' | 'monitor' | 'none';
}

/**
 * 风险评分接口
 */
export interface RiskScore {
  /** 总体风险分数 (0-100) */
  composite: number;
  
  /** 维度评分 */
  dimensions: {
    /** 资金流动维度分数 */
    flow: number;
    
    /** 行为模式维度分数 */
    behavior: number;
    
    /** 关联分析维度分数 */
    association: number;
    
    /** 历史特征维度分数 */
    historical: number;
    
    /** 技术特征维度分数 */
    technical: number;
  };
  
  /** 置信度 (0-1) */
  confidence: number;
  
  /** 计算时间 */
  calculatedAt: Date;
  
  /** 有效期（天） */
  validityPeriod: number;
}

/**
 * 扩展风险等级枚举
 * 提供比基本风险等级更详细的分级
 */
export enum ExtendedRiskLevel {
  UNKNOWN = 'unknown',
  VERY_LOW = 'very_low',
  LOW = 'low',
  MEDIUM_LOW = 'medium_low',
  MEDIUM = 'medium',
  MEDIUM_HIGH = 'medium_high',
  HIGH = 'high',
  VERY_HIGH = 'very_high',
  CRITICAL = 'critical'
}

/**
 * 地址风险报告接口
 */
export interface AddressRiskReport {
  /** 地址 */
  address: string;
  
  /** 风险评分 */
  riskScore: RiskScore;
  
  /** 风险等级 */
  riskLevel: RiskLevel;
  
  /** 详细风险等级 */
  extendedRiskLevel: ExtendedRiskLevel;
  
  /** 行为标签 */
  behaviorTags: BehaviorTag[];
  
  /** 风险摘要 */
  summary: string;
  
  /** 风险分析详情 */
  details: string;
  
  /** 推荐操作 */
  recommendedActions: string[];
  
  /** 关联高风险地址数量 */
  highRiskConnectionsCount: number;
  
  /** 上次更新时间 */
  updatedAt: Date;
  
  /** 分析者ID */
  analyzerId: string;
  
  /** 警报ID列表 */
  alertIds?: string[];
}

/**
 * 风险分析结果
 */
export interface RiskAnalysisResult {
  /** 风险评分 */
  score: number;
  
  /** 风险等级 */
  level: RiskLevel;
  
  /** 扩展风险等级 */
  extendedLevel?: ExtendedRiskLevel;
  
  /** 检测到的行为标签 */
  tags: BehaviorTag[];
  
  /** 分析维度评分 */
  dimensionScores: Record<string, number>;
  
  /** 风险总结 */
  summary: string;
  
  /** 风险详情 */
  details?: string;
  
  /** 计算置信度 */
  confidence: number;
  
  /** 分析时间 */
  analysisTime: Date;
}

/**
 * 风险追踪配置
 */
export interface TrackingConfig {
  /** 最大追踪深度 */
  maxDepth: number;
  
  /** 最大节点数 */
  maxNodes: number;
  
  /** 最小交易金额 */
  minTransactionValue?: string;
  
  /** 时间范围（天） */
  timeRangeDays?: number;
  
  /** 是否包含合约 */
  includeContracts: boolean;
  
  /** 是否排除交易所 */
  excludeExchanges: boolean;
}

/**
 * 风险模式接口
 */
export interface RiskPattern {
  /** 模式ID */
  id: string;
  
  /** 模式名称 */
  name: string;
  
  /** 模式描述 */
  description: string;
  
  /** 风险级别 */
  riskLevel: RiskLevel;
  
  /** 详细风险级别 */
  extendedRiskLevel?: ExtendedRiskLevel;
  
  /** 模式标签 */
  tags: string[];
  
  /** 检测条件 */
  detectionCriteria: any;
  
  /** 模式权重 */
  weight: number;
}

/**
 * 风险指标接口
 */
export interface RiskIndicator {
  /** 指标名称 */
  name: string;
  
  /** 指标值 */
  value: number;
  
  /** 指标阈值 */
  threshold: number;
  
  /** 指标描述 */
  description: string;
  
  /** 指标权重 */
  weight: number;
  
  /** 异常情况 */
  isAnomaly: boolean;
}
