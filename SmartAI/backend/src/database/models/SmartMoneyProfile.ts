import mongoose, { Schema, Document } from 'mongoose';

/**
 * 聪明钱画像接口
 */
export interface ISmartMoneyProfile extends Document {
  // 基本信息
  address: string;                // 区块链地址
  name?: string;                 // 名称/别名（如果已知）
  avatar?: string;               // 头像URL
  tags: string[];                // 标签列表
  classification: string;        // 分类 (个人/机构/项目)
  investorTypes: string[];       // 投资者类型 (鲸鱼/早期投资者/价值投资者等)
  description?: string;          // 文字描述
  
  // 擅长领域
  expertiseAreas: string[];      // 擅长领域 (DeFi, NFT, GameFi等)
  
  // 绩效数据
  performance: {
    overallROI: number;          // 总体ROI
    monthlyROI: number[];        // 月度ROI数组(最近12个月)
    winRate: number;             // 胜率(盈利交易/总交易)
    sharpeRatio: number;         // 夏普比率
    volatility: number;          // 波动率
    maxDrawdown: number;         // 最大回撤
  };
  
  // 交易活动统计
  activityStats: {
    totalTrades: number;         // 总交易次数
    avgTradeValue: number;       // 平均交易金额
    highestTradeValue: number;   // 最高交易金额
    tradingFrequency: number;    // 交易频率(每周平均)
    lastActive: Date;            // 最后活跃时间
  };
  
  // 投资特征
  traits: {
    entryTiming: number;         // 入场时机把握 (0-1)
    exitTiming: number;          // 出场时机把握 (0-1)
    hodlStrength: number;        // 持币韧性 (0-1)
    diversification: number;     // 多样化程度 (0-1)
    contrarian: number;          // 逆势操作倾向 (0-1)
  };
  
  // 当前持仓
  currentHoldings: Array<{
    token: string;               // 代币符号
    amount: number;              // 持有数量
    valueUSD: number;            // 当前价值(USD)
    allocation: number;          // 配置比例(0-1)
    entryPrice?: number;         // 平均入场价格
  }>;
  
  // 成功案例
  successCases: Array<{
    token: string;               // 代币符号
    roi: number;                 // ROI
    entryDate: Date;             // 入场时间
    exitDate?: Date;             // 出场时间(若已出场)
    description?: string;        // 案例描述
  }>;
  
  // 影响力指标
  influence: {
    followerCount: number;       // 追随者数量
    copyTradingValue: number;    // 模仿交易价值(USD)
    marketImpact: number;        // 市场影响力(0-1)
  };
  
  // 聪明钱评分
  score: {
    overall: number;             // 总体评分(0-100)
    components: {                // 评分细分
      performance: number;       // 绩效评分(0-100)
      timing: number;            // 时机评分(0-100)
      portfolioManagement: number; // 组合管理评分(0-100)
      riskManagement: number;    // 风险管理评分(0-100)
      insight: number;           // 洞察力评分(0-100)
    };
    confidence: number;          // 评分置信度(0-1)
    trend: 'rising' | 'stable' | 'declining'; // 评分趋势
    lastUpdated: Date;           // 最后更新时间
  };
  
  // 相关地址
  relatedAddresses: Array<{
    address: string;             // 关联地址
    relationship: string;        // 关系类型(多签成员/团队成员等)
    confidence: number;          // 关联置信度(0-1)
  }>;
  
  // 元数据
  metadata: {
    createdAt: Date;             // 创建时间
    updatedAt: Date;             // 更新时间
    dataQuality: number;         // 数据质量评分(0-1)
    dataSources: string[];       // 数据来源
  };
}

// MongoDB Schema
const SmartMoneyProfileSchema = new Schema<ISmartMoneyProfile>(
  {
    address: { type: String, required: true, unique: true, index: true },
    name: { type: String },
    avatar: { type: String },
    tags: [{ type: String }],
    classification: { type: String, required: true, enum: ['个人', '机构', '项目'] },
    investorTypes: [{ type: String }],
    description: { type: String },
    
    expertiseAreas: [{ type: String }],
    
    performance: {
      overallROI: { type: Number, default: 0 },
      monthlyROI: [{ type: Number }],
      winRate: { type: Number, default: 0 },
      sharpeRatio: { type: Number, default: 0 },
      volatility: { type: Number, default: 0 },
      maxDrawdown: { type: Number, default: 0 }
    },
    
    activityStats: {
      totalTrades: { type: Number, default: 0 },
      avgTradeValue: { type: Number, default: 0 },
      highestTradeValue: { type: Number, default: 0 },
      tradingFrequency: { type: Number, default: 0 },
      lastActive: { type: Date }
    },
    
    traits: {
      entryTiming: { type: Number, default: 0.5 },
      exitTiming: { type: Number, default: 0.5 },
      hodlStrength: { type: Number, default: 0.5 },
      diversification: { type: Number, default: 0.5 },
      contrarian: { type: Number, default: 0.5 }
    },
    
    currentHoldings: [{
      token: { type: String, required: true },
      amount: { type: Number, required: true },
      valueUSD: { type: Number, required: true },
      allocation: { type: Number, required: true },
      entryPrice: { type: Number }
    }],
    
    successCases: [{
      token: { type: String, required: true },
      roi: { type: Number, required: true },
      entryDate: { type: Date, required: true },
      exitDate: { type: Date },
      description: { type: String }
    }],
    
    influence: {
      followerCount: { type: Number, default: 0 },
      copyTradingValue: { type: Number, default: 0 },
      marketImpact: { type: Number, default: 0 }
    },
    
    score: {
      overall: { type: Number, default: 0 },
      components: {
        performance: { type: Number, default: 0 },
        timing: { type: Number, default: 0 },
        portfolioManagement: { type: Number, default: 0 },
        riskManagement: { type: Number, default: 0 },
        insight: { type: Number, default: 0 }
      },
      confidence: { type: Number, default: 0 },
      trend: { type: String, enum: ['rising', 'stable', 'declining'], default: 'stable' },
      lastUpdated: { type: Date, default: Date.now }
    },
    
    relatedAddresses: [{
      address: { type: String, required: true },
      relationship: { type: String, required: true },
      confidence: { type: Number, required: true }
    }],
    
    metadata: {
      createdAt: { type: Date, default: Date.now },
      updatedAt: { type: Date, default: Date.now },
      dataQuality: { type: Number, default: 0.5 },
      dataSources: [{ type: String }]
    }
  },
  { timestamps: true }
);

// 索引优化
SmartMoneyProfileSchema.index({ 'score.overall': -1 });
SmartMoneyProfileSchema.index({ 'activityStats.lastActive': -1 });
SmartMoneyProfileSchema.index({ 'performance.overallROI': -1 });
SmartMoneyProfileSchema.index({ investorTypes: 1 });
SmartMoneyProfileSchema.index({ tags: 1 });
SmartMoneyProfileSchema.index({ expertiseAreas: 1 });

// 模型导出
export const SmartMoneyProfile = mongoose.model<ISmartMoneyProfile>('SmartMoneyProfile', SmartMoneyProfileSchema); 