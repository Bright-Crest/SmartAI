// 前后端共享的API定义
export interface SmartMoneyAPI {
  analyzeAddress(address: string, options?: { useCache?: boolean }): Promise<AddressAnalysisResult>;
  batchAnalyzeAddresses(addresses: string[]): Promise<AddressAnalysisResult[]>;
  getLeaderboard(options?: LeaderboardOptions): Promise<SmartMoneyProfile[]>;
  // 其他API定义...
}

// 定义分页和筛选选项
export interface LeaderboardOptions {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  investorType?: string;
  minScore?: number;
}

// 定义智能钱分析结果接口
export interface AddressAnalysisResult {
  address: string;
  isSmartMoney: boolean;
  reason?: string;
  score: number;
  smartMoneyInfo?: SmartMoneyIdentificationResult;
  portfolio?: any[];
  transactionPatterns?: any;
  analysisTimestamp: string; // ISO格式的日期字符串
  error?: boolean;
  code?: string;
  message?: string;
  details?: any;
}

// 定义SmartMoneyIdentifier返回类型接口
export interface SmartMoneyIdentificationResult {
  isSmartMoney: boolean;
  score: number;
  reason?: string;
  investorType?: string;
  traits?: any;
  tags?: string[];
  expertiseAreas?: string[];
  performanceMetrics?: any;
  scoreComponents?: any;
  confidence?: number;
}

// 定义聪明钱档案接口
export interface SmartMoneyProfile {
  address: string;
  investorTypes: string[];
  tags: string[];
  classification: string;
  expertiseAreas: string[];
  performance: {
    overallROI: number;
    monthlyROI: any[];
    winRate: number;
    sharpeRatio: number;
    volatility: number;
    maxDrawdown: number;
  };
  traits: {
    entryTiming: number;
    exitTiming: number;
    hodlStrength: number;
    diversification: number;
    contrarian: number;
  };
  currentHoldings: any[];
  activityStats: {
    totalTrades: number;
    avgTradeValue: number;
    highestTradeValue: number;
    tradingFrequency: number;
    lastActive: Date;
  };
  score: {
    overall: number;
    components: {
      performance: number;
      timing: number;
      portfolioManagement: number;
      riskManagement: number;
      insight: number;
    };
    confidence: number;
    trend: string;
    lastUpdated: Date;
  };
  metadata: {
    createdAt: Date;
    updatedAt: Date;
    dataQuality: number;
    dataSources: string[];
  };
  relatedAddresses: string[];
  successCases: any[];
  influence: {
    followerCount: number;
    copyTradingValue: number;
    marketImpact: number;
  };
} 