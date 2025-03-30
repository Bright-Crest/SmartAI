/**
 * 聪明钱分析示例数据
 * 用于在无网络连接的情况下进行演示
 */

// 示例地址1: 长期价值投资者
export const valueInvestorExample = {
  address: '0x28c6c06298d514db089934071355e5743bf21d60',
  isSmartMoney: true,
  smartMoneyInfo: {
    isSmartMoney: true,
    score: 0.88,
    confidence: 0.92,
    reason: '满足聪明钱指标：长期投资表现优异',
    investorType: 'value_investor',
    traits: {
      entryTiming: 0.85,
      exitTiming: 0.75,
      hodlStrength: 0.94,
      diversification: 0.65,
      contrarian: 0.8,
    },
    expertiseAreas: ['DeFi', '头部项目', 'Layer1'],
    performanceMetrics: {
      overallROI: 0.52,
      monthlyROI: [0.05, 0.08, 0.12, -0.03, 0.07, 0.11],
      winRate: 0.78,
      sharpeRatio: 2.1,
      volatility: 0.24,
      maxDrawdown: 0.18,
    },
    tags: ['长期持有者', '价值投资者', '蓝筹资产'],
    scoreComponents: {
      performance: 0.85,
      timing: 0.79,
      portfolioManagement: 0.75,
      riskManagement: 0.8,
      insight: 0.82,
    },
  },
  portfolio: [
    {
      token: 'ETH',
      symbol: 'ETH',
      amount: 156.3,
      valueUSD: 487000,
      allocation: 0.48,
      entryPrice: 1850,
    },
    {
      token: 'BTC',
      symbol: 'BTC',
      amount: 5.2,
      valueUSD: 312000,
      allocation: 0.31,
      entryPrice: 36000,
    },
    {
      token: 'AAVE',
      symbol: 'AAVE',
      amount: 420,
      valueUSD: 52000,
      allocation: 0.05,
      entryPrice: 95,
    },
    {
      token: 'MKR',
      symbol: 'MKR',
      amount: 85,
      valueUSD: 93500,
      allocation: 0.09,
      entryPrice: 820,
    },
    {
      token: 'UNI',
      symbol: 'UNI',
      amount: 7500,
      valueUSD: 45000,
      allocation: 0.04,
      entryPrice: 4.2,
    },
    {
      token: 'LINK',
      symbol: 'LINK',
      amount: 2100,
      valueUSD: 25200,
      allocation: 0.03,
      entryPrice: 7.8,
    },
  ],
  transactionPatterns: {
    overview: {
      transactionCount: 156,
      firstTransaction: '2019-08-12T08:23:45Z',
      lastTransaction: new Date().toISOString(),
      avgTransactionSize: 18500,
    },
    frequencyPatterns: {
      dailyAvg: 0.12,
      weeklyAvg: 0.85,
      monthlyAvg: 3.6,
      averageFrequency: 'low',
    },
    sizePatterns: {
      averageSize: 18500,
      maxSize: 120000,
      sizeDistribution: {
        small: 0.1,
        medium: 0.35,
        large: 0.55,
      },
    },
    strategies: ['价值投资', '长期持有', '逆势加仓', '分散投资'],
  },
  analysisTimestamp: new Date().toISOString(),
  score: 0.88,
};

// 示例地址2: DeFi策略师
export const defiStrategistExample = {
  address: '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0',
  isSmartMoney: true,
  smartMoneyInfo: {
    isSmartMoney: true,
    score: 0.82,
    confidence: 0.88,
    reason: '满足聪明钱指标：DeFi策略使用精湛',
    investorType: 'defi_strategist',
    traits: {
      entryTiming: 0.78,
      exitTiming: 0.81,
      hodlStrength: 0.65,
      diversification: 0.85,
      contrarian: 0.72,
    },
    expertiseAreas: ['DeFi', '收益耕种', '借贷协议'],
    performanceMetrics: {
      overallROI: 0.68,
      monthlyROI: [0.12, 0.15, -0.05, 0.22, 0.08, 0.09],
      winRate: 0.82,
      sharpeRatio: 1.9,
      volatility: 0.35,
      maxDrawdown: 0.22,
    },
    tags: ['DeFi专家', '收益农民', '杠杆使用者'],
    scoreComponents: {
      performance: 0.86,
      timing: 0.75,
      portfolioManagement: 0.8,
      riskManagement: 0.65,
      insight: 0.78,
    },
  },
  portfolio: [
    {
      token: 'ETH',
      symbol: 'ETH',
      amount: 42.8,
      valueUSD: 133336,
      allocation: 0.28,
      entryPrice: 1950,
    },
    {
      token: 'AAVE',
      symbol: 'AAVE',
      amount: 385,
      valueUSD: 47740,
      allocation: 0.1,
      entryPrice: 105,
    },
    {
      token: 'CRV',
      symbol: 'CRV',
      amount: 15800,
      valueUSD: 28440,
      allocation: 0.06,
      entryPrice: 1.35,
    },
    {
      token: 'COMP',
      symbol: 'COMP',
      amount: 250,
      valueUSD: 23750,
      allocation: 0.05,
      entryPrice: 75,
    },
    {
      token: 'SNX',
      symbol: 'SNX',
      amount: 3600,
      valueUSD: 18000,
      allocation: 0.04,
      entryPrice: 3.8,
    },
    {
      token: 'USDC',
      symbol: 'USDC',
      amount: 42500,
      valueUSD: 42500,
      allocation: 0.09,
      entryPrice: 1.0,
    },
    {
      token: 'DAI',
      symbol: 'DAI',
      amount: 38700,
      valueUSD: 38700,
      allocation: 0.08,
      entryPrice: 1.0,
    },
    {
      token: 'SUSHI',
      symbol: 'SUSHI',
      amount: 12400,
      valueUSD: 24800,
      allocation: 0.05,
      entryPrice: 1.2,
    },
    {
      token: 'UNI-V3-POS',
      symbol: 'LP-Tokens',
      amount: 5,
      valueUSD: 120000,
      allocation: 0.25,
      entryPrice: null,
    },
  ],
  transactionPatterns: {
    overview: {
      transactionCount: 485,
      firstTransaction: '2020-03-18T14:53:12Z',
      lastTransaction: new Date().toISOString(),
      avgTransactionSize: 8200,
    },
    frequencyPatterns: {
      dailyAvg: 0.45,
      weeklyAvg: 3.2,
      monthlyAvg: 13.8,
      averageFrequency: 'medium',
    },
    sizePatterns: {
      averageSize: 8200,
      maxSize: 65000,
      sizeDistribution: {
        small: 0.35,
        medium: 0.45,
        large: 0.2,
      },
    },
    strategies: ['收益耕种', '借贷套利', '流动性挖矿', '闪电贷'],
  },
  analysisTimestamp: new Date().toISOString(),
  score: 0.82,
};

// 示例地址3: 交易员/市场择时者
export const traderExample = {
  address: '0xdac17f958d2ee523a2206206994597c13d831ec7',
  isSmartMoney: true,
  smartMoneyInfo: {
    isSmartMoney: true,
    score: 0.76,
    confidence: 0.82,
    reason: '满足聪明钱指标：市场择时能力强',
    investorType: 'market_timer',
    traits: {
      entryTiming: 0.88,
      exitTiming: 0.9,
      hodlStrength: 0.35,
      diversification: 0.6,
      contrarian: 0.55,
    },
    expertiseAreas: ['高频交易', '市场波动', '趋势捕捉'],
    performanceMetrics: {
      overallROI: 0.92,
      monthlyROI: [0.15, -0.08, 0.32, 0.18, -0.12, 0.28],
      winRate: 0.68,
      sharpeRatio: 1.6,
      volatility: 0.42,
      maxDrawdown: 0.28,
    },
    tags: ['交易员', '波段交易', '高换手率'],
    scoreComponents: {
      performance: 0.82,
      timing: 0.89,
      portfolioManagement: 0.62,
      riskManagement: 0.58,
      insight: 0.72,
    },
  },
  portfolio: [
    {
      token: 'ETH',
      symbol: 'ETH',
      amount: 28.5,
      valueUSD: 88730,
      allocation: 0.24,
      entryPrice: 2120,
    },
    {
      token: 'USDT',
      symbol: 'USDT',
      amount: 150000,
      valueUSD: 150000,
      allocation: 0.41,
      entryPrice: 1.0,
    },
    {
      token: 'SOL',
      symbol: 'SOL',
      amount: 1200,
      valueUSD: 54000,
      allocation: 0.15,
      entryPrice: 32,
    },
    {
      token: 'AVAX',
      symbol: 'AVAX',
      amount: 850,
      valueUSD: 29750,
      allocation: 0.08,
      entryPrice: 25,
    },
    {
      token: 'MATIC',
      symbol: 'MATIC',
      amount: 18500,
      valueUSD: 22200,
      allocation: 0.06,
      entryPrice: 0.85,
    },
    {
      token: 'FTM',
      symbol: 'FTM',
      amount: 25000,
      valueUSD: 10000,
      allocation: 0.03,
      entryPrice: 0.28,
    },
    {
      token: 'DOGE',
      symbol: 'DOGE',
      amount: 150000,
      valueUSD: 12000,
      allocation: 0.03,
      entryPrice: 0.06,
    },
  ],
  transactionPatterns: {
    overview: {
      transactionCount: 1250,
      firstTransaction: '2021-01-24T19:12:38Z',
      lastTransaction: new Date().toISOString(),
      avgTransactionSize: 4800,
    },
    frequencyPatterns: {
      dailyAvg: 1.8,
      weeklyAvg: 12.5,
      monthlyAvg: 52.3,
      averageFrequency: 'high',
    },
    sizePatterns: {
      averageSize: 4800,
      maxSize: 48000,
      sizeDistribution: {
        small: 0.55,
        medium: 0.35,
        large: 0.1,
      },
    },
    strategies: ['短线交易', '趋势追踪', '资金管理', '技术分析'],
  },
  analysisTimestamp: new Date().toISOString(),
  score: 0.76,
};

// 导出示例地址列表
export const exampleAddresses = [
  '0x28c6c06298d514db089934071355e5743bf21d60', // 价值投资者
  '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0', // DeFi策略师
  '0xdac17f958d2ee523a2206206994597c13d831ec7', // 交易员
];

// 通过地址获取示例数据
export const getExampleByAddress = address => {
  switch (address.toLowerCase()) {
    case '0x28c6c06298d514db089934071355e5743bf21d60':
      return valueInvestorExample;
    case '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0':
      return defiStrategistExample;
    case '0xdac17f958d2ee523a2206206994597c13d831ec7':
      return traderExample;
    default:
      return null;
  }
};

// 检查地址是否为示例地址
export const isExampleAddress = address => {
  return exampleAddresses.includes(address);
};
