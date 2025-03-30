/**
 * 聪明钱分析系统 - 核心数据类型定义
 */

// 投资者类型
export enum InvestorType {
  EARLY_INVESTOR = "早期投资者",
  SWING_TRADER = "震荡套利者",
  LONG_TERM_HOLDER = "长期持有者",
  MOMENTUM_TRADER = "趋势交易者",
  ARBITRAGE_TRADER = "套利交易者",
  WHALE = "大户",
  INSTITUTIONAL = "机构投资者",
  DEV_TEAM = "开发团队",
}

// 投资表现数据
export interface InvestmentPerformance {
  // 整体ROI
  overallROI: number;
  // 月度ROI数组
  monthlyROI: number[];
  // 波动率
  volatility: number;
  // 最大回撤
  maxDrawdown: number;
  // 胜率
  winRate: number;
  // 夏普比率
  sharpeRatio: number;
}

// 投资者特征
export interface InvestorTraits {
  // 入场时机准确度 (0-1)
  entryTiming: number;
  // 出场时机准确度 (0-1)
  exitTiming: number;
  // 多样化程度 (0-1)
  diversification: number;
  // 持有韧性 (0-1)
  hodlStrength: number;
  // 反向投资倾向 (0-1)
  contrarian: number;
  // 跟随趋势倾向 (0-1)
  trendFollowing: number;
}

// 成功案例
export interface SuccessCase {
  // 代币地址
  tokenAddress: string;
  // 代币符号
  tokenSymbol: string;
  // 买入价格
  buyPrice: number;
  // 卖出价格
  sellPrice: number;
  // ROI
  roi: number;
  // 持有时间(天)
  holdPeriod: number;
  // 买入日期
  buyDate: string;
  // 卖出日期
  sellDate: string;
  // 交易哈希
  transactions: {
    buy: string;
    sell: string;
  };
}

// 当前持仓
export interface CurrentHolding {
  // 代币地址
  tokenAddress: string;
  // 代币符号
  tokenSymbol: string;
  // 持有数量
  amount: number;
  // 当前价值(USD)
  valueUSD: number;
  // 入场价格
  entryPrice: number;
  // 当前价格
  currentPrice: number;
  // 未实现ROI
  unrealizedROI: number;
  // 持有时间(天)
  holdPeriod: number;
  // 首次购买日期
  entryDate: string;
}

// 社交影响力
export interface SocialInfluence {
  // 跟随者数量
  followerCount: number;
  // 市场影响力 (0-1)
  marketImpact: number;
  // 价格影响效应
  priceEffect: number;
  // 影响网络
  network: {
    // 紧密关联地址
    closelyRelated: string[];
    // 跟随者地址
    followers: string[];
  };
}

// 活动统计
export interface ActivityStats {
  // 首次交易时间
  firstSeen: string;
  // 最后活跃时间
  lastActive: string;
  // 交易频率 (每天平均交易次数)
  tradeFrequency: number;
  // 平均交易规模(USD)
  avgTradeSize: number;
  // 总交易次数
  totalTrades: number;
  // 总交易量(USD)
  totalVolumeUSD: number;
  // 活跃链
  activeChains: number[];
}

// 聪明钱评分
export interface SmartMoneyScore {
  // 总分 (0-100)
  overall: number;
  // 各项评分
  components: {
    // 投资表现分 (0-100)
    performance: number;
    // 时机把握分 (0-100)
    timing: number;
    // 投资组合管理分 (0-100)
    portfolioManagement: number;
    // 抗风险能力分 (0-100)
    riskManagement: number;
    // 洞察力分 (0-100)
    insight: number;
  };
  // 评分置信度 (0-1)
  confidence: number;
  // 历史趋势
  trend: "rising" | "stable" | "declining";
  // 最后更新时间
  lastUpdated: string;
}

// 聪明钱画像 (完整版)
export interface SmartMoneyProfile {
  // 基本信息
  address: string;
  chainId: number;
  ensName?: string;
  
  // 分数评级
  score: SmartMoneyScore;
  
  // 投资表现
  performance: InvestmentPerformance;
  
  // 投资特性
  traits: InvestorTraits;
  
  // 投资类型标签
  investorTypes: InvestorType[];
  
  // 擅长领域
  expertiseAreas: string[];
  
  // 历史成功案例
  successCases: SuccessCase[];
  
  // 当前持仓
  currentHoldings: CurrentHolding[];
  
  // 关联影响力
  influence: SocialInfluence;
  
  // 活动统计
  activityStats: ActivityStats;
  
  // 预测指标
  predictions?: {
    // 可能的下一步操作
    nextMoves: string[];
    // 关注的项目/代币
    watchingTokens: string[];
    // 预测准确度历史
    predictionAccuracy: number;
  };
  
  // 最后更新时间
  updatedAt: string;
  // 创建时间
  createdAt: string;
}

// 交易模式
export interface TradePattern {
  id: string;
  name: string;
  description: string;
  
  // 模式特征
  features: {
    // 时间模式特征
    timePattern: any;
    // 规模模式特征
    sizePattern: any;
    // 序列模式特征
    sequencePattern: any;
  };
  
  // 绩效统计
  performance: {
    // 成功率
    successRate: number;
    // 平均ROI
    avgROI: number;
    // 风险调整后收益
    riskAdjustedReturn: number;
  };
  
  // 适用条件
  conditions: {
    // 适用市场类型
    marketType: string[];
    // 适用资产类别
    assetClass: string[];
    // 适用波动性水平
    volatilityLevel: string;
  };
  
  // 代表性案例
  examples: string[];
  
  // 创建时间
  createdAt: string;
  // 更新时间
  updatedAt: string;
}

// 简化的聪明钱信息 (用于列表显示)
export interface SmartMoneyListItem {
  address: string;
  ensName?: string;
  overallScore: number;
  performanceScore: number;
  roi: number;
  primaryInvestorType: InvestorType;
  successCaseCount: number;
  followerCount: number;
  lastActive: string;
}

// 聪明钱地址活动
export interface SmartMoneyActivity {
  address: string;
  ensName?: string;
  timestamp: string;
  actionType: "buy" | "sell" | "swap" | "add_liquidity" | "remove_liquidity" | "stake" | "unstake" | "transfer";
  tokenAddress: string;
  tokenSymbol: string;
  amount: number;
  valueUSD: number;
  transactionHash: string;
  significance: number; // 0-1, 活动的重要性
}

// 跟踪警报设置
export interface SmartMoneyAlert {
  id: string;
  userId: string;
  name: string;
  
  // 触发条件
  conditions: {
    // 监控的地址
    addresses?: string[];
    // 投资者类型
    investorTypes?: InvestorType[];
    // 最低分数要求
    minScore?: number;
    // 动作类型
    actionTypes?: ("buy" | "sell" | "swap")[];
    // 代币地址
    tokens?: string[];
    // 最小交易价值(USD)
    minValueUSD?: number;
  };
  
  // 通知设置
  notifications: {
    // 启用邮件
    email: boolean;
    // 启用站内信
    inApp: boolean;
    // 启用Webhook
    webhook?: string;
  };
  
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
} 