import { faker } from '@faker-js/faker';
import { nanoid } from 'nanoid';
import Joi from 'joi';
import { RiskLevel, ExtendedRiskLevel } from '../types/riskAnalysis';
import { logger } from '../utils/logger';

// 设置中文语言环境
faker.locale = 'zh_CN';

/**
 * Mock数据服务 - 用于生成测试和演示数据
 */
export class MockDataService {
  private static instance: MockDataService;
  private mockEnabled: boolean;
  
  // Mock数据缓存，避免频繁生成
  private mockAddressCache: Map<string, any> = new Map();
  private mockTransactionsCache: Map<string, any[]> = new Map();
  private mockRiskScoreCache: Map<string, any> = new Map();
  
  private constructor() {
    // 是否启用Mock数据
    this.mockEnabled = process.env.ENABLE_MOCK_DATA === 'true';
  }
  
  /**
   * 获取Mock数据服务实例
   */
  public static getInstance(): MockDataService {
    if (!MockDataService.instance) {
      MockDataService.instance = new MockDataService();
    }
    return MockDataService.instance;
  }
  
  /**
   * 是否启用Mock数据
   */
  public isMockEnabled(): boolean {
    return this.mockEnabled;
  }
  
  /**
   * 设置是否启用Mock数据
   */
  public setMockEnabled(enabled: boolean): void {
    this.mockEnabled = enabled;
  }
  
  /**
   * 生成随机以太坊地址
   */
  public generateEthAddress(): string {
    return `0x${faker.random.alphaNumeric(40).toLowerCase()}`;
  }
  
  /**
   * 生成交易哈希
   */
  public generateTxHash(): string {
    return `0x${faker.random.alphaNumeric(64).toLowerCase()}`;
  }
  
  /**
   * 生成模拟的区块链地址详情
   * @param address 地址，如果未提供则随机生成
   */
  public generateAddressDetails(address?: string): any {
    address = address || this.generateEthAddress();
    
    // 检查缓存
    if (this.mockAddressCache.has(address)) {
      return this.mockAddressCache.get(address);
    }
    
    const addressDetails = {
      address,
      balance: faker.finance.amount(0, 10000, 4),
      transactions: faker.datatype.number({ min: 1, max: 1000 }),
      firstSeen: faker.date.past(2),
      lastSeen: faker.date.recent(30),
      tags: this.generateAddressTags(),
      contracts: faker.datatype.number({ min: 0, max: 10 }),
      tokens: faker.datatype.number({ min: 0, max: 50 }),
      isContract: faker.datatype.boolean(),
      isExchange: faker.datatype.boolean(),
      isScammer: faker.datatype.boolean({ probability: 0.1 }),
      profile: {
        firstSeen: faker.date.past().toISOString(),
        totalVolume: faker.number.float({ min: 100000, max: 1000000, precision: 0.01 }),
        preferredDex: faker.helpers.arrayElement(['Uniswap', 'SushiSwap', 'PancakeSwap']),
        preferredChain: faker.helpers.arrayElement(['Ethereum', 'BSC', 'Polygon']),
        tradingStyle: faker.helpers.arrayElement(['Scalping', 'Swing Trading', 'Position Trading']),
        riskTolerance: faker.helpers.arrayElement(['Conservative', 'Moderate', 'Aggressive'])
      },
      lastUpdated: new Date().toISOString()
    };
    
    // 缓存结果
    this.mockAddressCache.set(address, addressDetails);
    
    return addressDetails;
  }
  
  /**
   * 生成地址标签
   */
  private generateAddressTags(): Array<{ name: string, confidence: number }> {
    const tagCount = faker.datatype.number({ min: 0, max: 5 });
    const tags = [];
    
    const possibleTags = [
      '交易所', 'DeFi', '智能合约', '鲸鱼', '闪电贷', '挖矿', 
      '跨链桥', '代币发行者', '金库', '质押', '黑客', '混币器'
    ];
    
    // 随机选择几个标签
    const selectedTags = faker.random.arrayElements(possibleTags, tagCount);
    
    for (const tag of selectedTags) {
      tags.push({
        name: tag,
        confidence: faker.datatype.float({ min: 0.3, max: 0.95, precision: 0.01 })
      });
    }
    
    return tags;
  }
  
  /**
   * 生成随机风险得分
   * @param address 地址
   */
  public generateRiskScore(address: string): any {
    // 检查缓存
    if (this.mockRiskScoreCache.has(address)) {
      return this.mockRiskScoreCache.get(address);
    }
    
    // 计算风险分数，保持一致性
    const addressNum = parseInt(address.slice(-8), 16);
    const baseScore = (addressNum % 100) || faker.datatype.number({ min: 0, max: 100 });
    
    // 确定风险等级
    let riskLevel: RiskLevel;
    let extendedRiskLevel: ExtendedRiskLevel;
    
    if (baseScore < 20) {
      riskLevel = RiskLevel.LOW;
      extendedRiskLevel = ExtendedRiskLevel.VERY_LOW;
    } else if (baseScore < 40) {
      riskLevel = RiskLevel.LOW;
      extendedRiskLevel = ExtendedRiskLevel.LOW;
    } else if (baseScore < 60) {
      riskLevel = RiskLevel.MEDIUM;
      extendedRiskLevel = ExtendedRiskLevel.MEDIUM;
    } else if (baseScore < 80) {
      riskLevel = RiskLevel.HIGH;
      extendedRiskLevel = ExtendedRiskLevel.HIGH;
    } else {
      riskLevel = RiskLevel.HIGH;
      extendedRiskLevel = ExtendedRiskLevel.VERY_HIGH;
    }
    
    // 生成风险描述
    const riskDescriptions = {
      [RiskLevel.LOW]: '该地址风险较低，未发现明显风险行为',
      [RiskLevel.MEDIUM]: '该地址存在一定风险，建议谨慎交互',
      [RiskLevel.HIGH]: '该地址存在高风险，与多个已知风险地址有交互'
    };
    
    const extendedRiskDescriptions = {
      [ExtendedRiskLevel.VERY_LOW]: '该地址风险非常低，可信度高',
      [ExtendedRiskLevel.LOW]: '该地址风险较低，未发现明显风险行为',
      [ExtendedRiskLevel.MEDIUM]: '该地址存在一定风险，建议谨慎交互',
      [ExtendedRiskLevel.HIGH]: '该地址存在高风险，与已知风险地址有交互',
      [ExtendedRiskLevel.VERY_HIGH]: '该地址风险极高，强烈建议避免任何交互'
    };
    
    // 生成五维指标
    const dimensions = {
      flow: faker.datatype.float({ min: 0, max: 1, precision: 0.01 }),
      behavior: faker.datatype.float({ min: 0, max: 1, precision: 0.01 }),
      association: faker.datatype.float({ min: 0, max: 1, precision: 0.01 }),
      historical: faker.datatype.float({ min: 0, max: 1, precision: 0.01 }),
      technical: faker.datatype.float({ min: 0, max: 1, precision: 0.01 })
    };
    
    // 生成风险因素
    const riskFactors = [];
    if (baseScore > 50) {
      riskFactors.push({
        name: '异常资金流动模式',
        description: '资金流动模式异常，影响风险评分',
        impact: dimensions.flow,
        category: '资金流动'
      });
    }
    
    if (baseScore > 70) {
      riskFactors.push({
        name: '与高风险地址关联',
        description: '与多个已知高风险地址有直接交互',
        impact: dimensions.association,
        category: '关联分析'
      });
    }
    
    // 生成关注点
    const attentionPoints = [];
    if (baseScore > 50) {
      attentionPoints.push('该地址存在一定风险，交互前请充分了解其背景');
    }
    
    if (baseScore > 70) {
      attentionPoints.push('建议避免向该地址发送大额资产');
    }
    
    // 构建完整响应
    const riskScoreData = {
      address,
      risk_score: baseScore.toFixed(2),
      risk_level: riskLevel,
      risk_description: riskDescriptions[riskLevel],
      extended_risk_level: extendedRiskLevel,
      extended_risk_description: extendedRiskDescriptions[extendedRiskLevel],
      risk_explanation: `该地址风险评分为${baseScore.toFixed(2)}，风险等级为${riskLevel}。${riskDescriptions[riskLevel]}`,
      risk_factors: riskFactors,
      attention_points: attentionPoints,
      dimensions,
      features: {
        transaction_count: faker.datatype.number({ min: 1, max: 500 }),
        event_count: faker.datatype.number({ min: 0, max: 100 }),
        age_days: faker.datatype.number({ min: 1, max: 1000 }),
        tag_count: riskFactors.length,
        dimensions
      }
    };
    
    // 缓存结果
    this.mockRiskScoreCache.set(address, riskScoreData);
    
    return riskScoreData;
  }
  
  /**
   * 生成风险指标五维数据
   * @param address 地址
   */
  public generateRiskMetrics(address: string): any {
    // 获取风险分数数据，保持一致性
    const riskScore = this.generateRiskScore(address);
    const dimensions = riskScore.dimensions;
    
    // 资金流动指标
    const flowIndicators = [
      {
        name: '资金流量异常度',
        value: Math.round(dimensions.flow * 100),
        description: '资金流入流出模式的异常程度',
        isAnomaly: dimensions.flow > 0.7
      },
      {
        name: '大额交易比例',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '大额交易在总交易中的比例',
        isAnomaly: faker.datatype.boolean()
      },
      {
        name: '混币器关联度',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '与混币服务的关联程度',
        isAnomaly: faker.datatype.boolean({ probability: 0.3 })
      }
    ];
    
    // 行为特征指标
    const behaviorIndicators = [
      {
        name: '交易频率异常度',
        value: Math.round(dimensions.behavior * 100),
        description: '交易频率的异常程度',
        isAnomaly: dimensions.behavior > 0.7
      },
      {
        name: '交互对象多样性',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '交易对象的多样性程度',
        isAnomaly: faker.datatype.boolean()
      },
      {
        name: '交易时间模式',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '交易时间的规律性',
        isAnomaly: faker.datatype.boolean({ probability: 0.3 })
      }
    ];
    
    // 关联分析指标
    const associationIndicators = [
      {
        name: '高风险地址关联度',
        value: Math.round(dimensions.association * 100),
        description: '与已知高风险地址的关联程度',
        isAnomaly: dimensions.association > 0.7
      },
      {
        name: '关联地址平均风险度',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '关联地址的平均风险程度',
        isAnomaly: faker.datatype.boolean()
      },
      {
        name: '交易网络集中度',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '交易网络的集中程度',
        isAnomaly: faker.datatype.boolean({ probability: 0.3 })
      }
    ];
    
    // 历史记录指标
    const historicalIndicators = [
      {
        name: '历史风险事件',
        value: Math.round(dimensions.historical * 100),
        description: '历史风险事件的严重程度',
        isAnomaly: dimensions.historical > 0.7
      },
      {
        name: '地址活跃度',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '地址的活跃程度',
        isAnomaly: faker.datatype.boolean()
      },
      {
        name: '风险标签数量',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '关联风险标签的数量',
        isAnomaly: faker.datatype.boolean({ probability: 0.3 })
      }
    ];
    
    // 技术特征指标
    const technicalIndicators = [
      {
        name: '合约代码风险',
        value: Math.round(dimensions.technical * 100),
        description: '合约代码的风险程度',
        isAnomaly: dimensions.technical > 0.7
      },
      {
        name: '交易复杂度',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '交易的复杂程度',
        isAnomaly: faker.datatype.boolean()
      },
      {
        name: '交易失败率',
        value: Math.round(faker.datatype.float({ min: 0, max: 100, precision: 1 })),
        description: '交易失败的比率',
        isAnomaly: faker.datatype.boolean({ probability: 0.3 })
      }
    ];
    
    // 构建五维指标数据
    return [
      {
        name: '资金流动',
        key: 'flow',
        description: '资金流动模式分析',
        score: Math.round(dimensions.flow * 100),
        weight: 0.35,
        indicators: flowIndicators
      },
      {
        name: '行为特征',
        key: 'behavior',
        description: '地址行为模式分析',
        score: Math.round(dimensions.behavior * 100),
        weight: 0.25,
        indicators: behaviorIndicators
      },
      {
        name: '关联分析',
        key: 'association',
        description: '地址关联网络分析',
        score: Math.round(dimensions.association * 100),
        weight: 0.20,
        indicators: associationIndicators
      },
      {
        name: '历史记录',
        key: 'historical',
        description: '历史风险事件分析',
        score: Math.round(dimensions.historical * 100),
        weight: 0.10,
        indicators: historicalIndicators
      },
      {
        name: '技术特征',
        key: 'technical',
        description: '技术层面风险分析',
        score: Math.round(dimensions.technical * 100),
        weight: 0.10,
        indicators: technicalIndicators
      }
    ];
  }
  
  /**
   * 生成交易历史
   * @param address 地址
   * @param count 交易数量
   */
  public generateTransactionHistory(address: string, count: number = 10): any[] {
    // 检查缓存
    const cacheKey = `${address}-${count}`;
    if (this.mockTransactionsCache.has(cacheKey)) {
      return this.mockTransactionsCache.get(cacheKey) || [];
    }
    
    const transactions = [];
    
    for (let i = 0; i < count; i++) {
      const isSender = faker.datatype.boolean();
      
      const transaction = {
        hash: this.generateTxHash(),
        blockNumber: faker.datatype.number({ min: 10000000, max: 20000000 }),
        timestamp: faker.date.recent(30).getTime() / 1000,
        from: isSender ? address : this.generateEthAddress(),
        to: isSender ? this.generateEthAddress() : address,
        value: faker.finance.amount(0, 10, 4),
        gasUsed: faker.datatype.number({ min: 21000, max: 1000000 }),
        gasPrice: faker.datatype.number({ min: 1, max: 200 }) * 1e9,
        status: faker.datatype.boolean(0.9),
        method: faker.random.arrayElement(['transfer', 'approve', 'swap', 'mint', 'burn', '']),
        isError: faker.datatype.boolean(0.1),
        riskScore: faker.datatype.number({ min: 0, max: 100 })
      };
      
      transactions.push(transaction);
    }
    
    // 按时间戳排序
    transactions.sort((a, b) => b.timestamp - a.timestamp);
    
    // 缓存结果
    this.mockTransactionsCache.set(cacheKey, transactions);
    
    return transactions;
  }
  
  /**
   * 生成批量风险评分
   * @param addresses 地址列表
   */
  public generateBatchRiskScores(addresses: string[]): any {
    const results = [];
    
    for (const address of addresses) {
      const riskScore = this.generateRiskScore(address);
      
      results.push({
        address,
        risk_score: riskScore.risk_score,
        risk_level: riskScore.risk_level,
        risk_description: riskScore.risk_description,
        success: true
      });
    }
    
    return {
      total: addresses.length,
      successful: addresses.length,
      failed: 0,
      results
    };
  }
  
  /**
   * 验证请求参数有效性
   * @param data 请求数据
   * @param schema Joi验证模式
   */
  public validateRequest(data: any, schema: Joi.Schema): { valid: boolean; error?: string } {
    const validation = schema.validate(data);
    
    if (validation.error) {
      return {
        valid: false,
        error: validation.error.message
      };
    }
    
    return { valid: true };
  }
  
  /**
   * 清除模拟数据缓存
   */
  public clearCache(): void {
    this.mockAddressCache.clear();
    this.mockTransactionsCache.clear();
    this.mockRiskScoreCache.clear();
    
    logger.info('已清除模拟数据缓存');
  }

  // 生成智能投资评分
  generateSmartScore(address: string) {
    return {
      address,
      score: faker.number.int({ min: 0, max: 100 }),
      confidence: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
      lastUpdated: new Date().toISOString(),
      factors: {
        historicalPerformance: faker.number.float({ min: 0, max: 1, precision: 0.01 }),
        tradingStrategy: faker.number.float({ min: 0, max: 1, precision: 0.01 }),
        marketTiming: faker.number.float({ min: 0, max: 1, precision: 0.01 }),
        portfolioDiversity: faker.number.float({ min: 0, max: 1, precision: 0.01 }),
        riskManagement: faker.number.float({ min: 0, max: 1, precision: 0.01 })
      },
      // 新增聪明钱特征
      smartMoneyTraits: {
        isWhale: faker.datatype.boolean({ probability: 0.2 }),
        isInstitutional: faker.datatype.boolean({ probability: 0.3 }),
        isSmartTrader: faker.datatype.boolean({ probability: 0.4 }),
        tradingVolume: faker.number.float({ min: 100000, max: 10000000, precision: 0.01 }),
        averageProfitPerTrade: faker.number.float({ min: 0.05, max: 0.5, precision: 0.01 }),
        preferredTimeframe: faker.helpers.arrayElement(['1h', '4h', '1d', '1w']),
        preferredAssets: faker.helpers.arrayElements(['BTC', 'ETH', 'SOL', 'DOT', 'AVAX'], { min: 2, max: 4 })
      }
    };
  }

  // 生成投资行为指标
  generateInvestmentMetrics(address: string) {
    return {
      address,
      metrics: {
        totalTrades: faker.number.int({ min: 100, max: 1000 }),
        winRate: faker.number.float({ min: 0.4, max: 0.8, precision: 0.01 }),
        averageReturn: faker.number.float({ min: 0.1, max: 0.5, precision: 0.01 }),
        maxDrawdown: faker.number.float({ min: 0.1, max: 0.3, precision: 0.01 }),
        sharpeRatio: faker.number.float({ min: 1, max: 3, precision: 0.01 }),
        averageHoldingPeriod: faker.number.int({ min: 1, max: 30 }),
        preferredTimeframe: faker.helpers.arrayElement(['1h', '4h', '1d', '1w']),
        preferredAssets: faker.helpers.arrayElements(['BTC', 'ETH', 'SOL', 'DOT', 'AVAX'], { min: 2, max: 4 })
      },
      // 新增专业投资指标
      professionalMetrics: {
        alpha: faker.number.float({ min: -0.1, max: 0.3, precision: 0.01 }),
        beta: faker.number.float({ min: 0.5, max: 2, precision: 0.01 }),
        informationRatio: faker.number.float({ min: 0.5, max: 2, precision: 0.01 }),
        sortinoRatio: faker.number.float({ min: 1, max: 3, precision: 0.01 }),
        calmarRatio: faker.number.float({ min: 0.5, max: 2, precision: 0.01 }),
        omegaRatio: faker.number.float({ min: 1, max: 2, precision: 0.01 }),
        valueAtRisk: faker.number.float({ min: 0.01, max: 0.1, precision: 0.01 }),
        expectedShortfall: faker.number.float({ min: 0.02, max: 0.15, precision: 0.01 })
      },
      // 新增交易策略分析
      tradingStrategy: {
        strategyType: faker.helpers.arrayElement(['趋势跟踪', '均值回归', '动量交易', '套利交易']),
        entryTiming: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
        exitTiming: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
        positionSizing: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
        riskRewardRatio: faker.number.float({ min: 1.5, max: 3, precision: 0.01 }),
        marketRegime: faker.helpers.arrayElement(['牛市', '熊市', '震荡市']),
        preferredMarketCap: faker.helpers.arrayElement(['大市值', '中市值', '小市值'])
      },
      lastUpdated: new Date().toISOString()
    };
  }

  // 生成交易历史
  generateTransactionHistory(address: string, page: number, limit: number) {
    const transactions = Array.from({ length: limit }, () => ({
      id: nanoid(),
      timestamp: faker.date.recent().toISOString(),
      type: faker.helpers.arrayElement(['Buy', 'Sell']),
      asset: faker.helpers.arrayElement(['BTC', 'ETH', 'SOL', 'DOT', 'AVAX']),
      amount: faker.number.float({ min: 0.1, max: 10, precision: 0.01 }),
      price: faker.number.float({ min: 1000, max: 50000, precision: 0.01 }),
      value: faker.number.float({ min: 100, max: 10000, precision: 0.01 }),
      profitLoss: faker.number.float({ min: -1000, max: 1000, precision: 0.01 }),
      dex: faker.helpers.arrayElement(['Uniswap', 'SushiSwap', 'PancakeSwap']),
      chain: faker.helpers.arrayElement(['Ethereum', 'BSC', 'Polygon']),
      // 新增交易分析数据
      analysis: {
        marketCap: faker.number.float({ min: 100000000, max: 10000000000, precision: 0.01 }),
        volume24h: faker.number.float({ min: 1000000, max: 100000000, precision: 0.01 }),
        priceChange24h: faker.number.float({ min: -0.2, max: 0.2, precision: 0.01 }),
        liquidity: faker.number.float({ min: 100000, max: 1000000, precision: 0.01 }),
        slippage: faker.number.float({ min: 0.001, max: 0.05, precision: 0.001 }),
        gasPrice: faker.number.float({ min: 10, max: 100, precision: 0.01 }),
        executionTime: faker.number.int({ min: 1, max: 60 })
      }
    }));

    return {
      address,
      transactions,
      pagination: {
        page,
        limit,
        total: faker.number.int({ min: 100, max: 1000 })
      },
      // 新增交易统计
      statistics: {
        totalVolume: faker.number.float({ min: 100000, max: 1000000, precision: 0.01 }),
        averageTradeSize: faker.number.float({ min: 1000, max: 10000, precision: 0.01 }),
        largestTrade: faker.number.float({ min: 50000, max: 500000, precision: 0.01 }),
        averageExecutionTime: faker.number.int({ min: 1, max: 30 }),
        averageSlippage: faker.number.float({ min: 0.001, max: 0.05, precision: 0.001 }),
        preferredTradingTime: faker.helpers.arrayElement(['亚洲时段', '欧洲时段', '美洲时段']),
        preferredDex: faker.helpers.arrayElement(['Uniswap', 'SushiSwap', 'PancakeSwap'])
      }
    };
  }

  // 生成相似地址
  generateSimilarAddresses(address: string) {
    return {
      address,
      similarAddresses: Array.from({ length: 5 }, () => ({
        address: faker.string.alphanumeric({ length: 42, prefix: '0x' }),
        similarity: faker.number.float({ min: 0.5, max: 0.95, precision: 0.01 }),
        commonTraits: faker.helpers.arrayElements([
          '相似交易模式',
          '共同资产偏好',
          '相似持仓周期',
          '相似风险偏好',
          '共同DEX使用',
          '相似交易规模',
          '相似交易时间',
          '相似交易策略'
        ], { min: 2, max: 4 }),
        // 新增相似度分析
        similarityAnalysis: {
          tradingPattern: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
          assetPreference: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
          timingPattern: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
          strategyAlignment: faker.number.float({ min: 0.5, max: 1, precision: 0.01 }),
          riskProfile: faker.number.float({ min: 0.5, max: 1, precision: 0.01 })
        }
      })),
      lastUpdated: new Date().toISOString()
    };
  }
}

export const mockDataService = new MockDataService(); 