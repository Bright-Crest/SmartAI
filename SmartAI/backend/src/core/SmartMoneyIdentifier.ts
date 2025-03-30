import { loggerWinston as logger } from '../utils/logger';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { ISmartMoneyProfile } from '../database/models/SmartMoneyProfile';
import { MongooseError } from 'mongoose';

/**
 * 聪明钱识别与分类服务
 * 负责识别、分类和分析聪明资金账户的交易行为特征
 */
export class SmartMoneyIdentifier {
  /**
   * 智能投资者类型枚举
   */
  static readonly INVESTOR_TYPES = {
    WHALE: '鲸鱼', // 大额持仓者
    EARLY_INVESTOR: '早期投资者', // 能够识别早期优质项目的投资者
    SWING_TRADER: '波段交易者', // 善于捕捉市场波动的交易者
    VALUE_INVESTOR: '价值投资者', // 关注项目基本面的长线投资者
    CONTRARIAN: '逆势交易者', // 反向操作、逆市场情绪的交易者
    MOMENTUM_TRADER: '动量交易者', // 追涨杀跌的交易者
    ARBITRAGEUR: '套利交易者', // 寻找套利机会的交易者
    MARKET_MAKER: '做市商', // 为市场提供流动性的机构
    QUANTITATIVE: '量化交易者', // 使用算法和量化模型的交易者
    INSTITUTION: '机构', // 投资机构
  };

  /**
   * 根据交易历史和持仓情况识别聪明钱地址
   * @param address 区块链地址
   * @param transactions 交易历史
   * @param holdings 当前持仓
   */
  static async identifySmartMoney(
    address: string,
    transactions: any[],
    holdings: any[]
  ): Promise<boolean> {
    try {
      logger.info('开始识别聪明钱地址', { address });

      // 计算关键指标
      const metrics = this.calculateKeyMetrics(transactions, holdings);
      
      // 聪明钱判断标准
      // 1. ROI高于阈值
      // 2. 或者交易胜率高
      // 3. 或者交易次数多且平均收益为正
      const isSmartMoney = 
        metrics.overallROI > 100 || // ROI > 100%
        metrics.winRate > 0.7 || // 胜率 > 70%
        (metrics.totalTrades > 50 && metrics.averageROI > 0); // 频繁交易且平均收益为正
      
      logger.info('聪明钱识别结果', { 
        address, 
        isSmartMoney, 
        metrics: {
          overallROI: metrics.overallROI,
          winRate: metrics.winRate,
          totalTrades: metrics.totalTrades
        } 
      });
      
      return isSmartMoney;
    } catch (error) {
      logger.error('识别聪明钱地址失败', { error, address });
      return false;
    }
  }
  
  /**
   * 根据交易行为和持仓对聪明钱进行分类
   * @param address 区块链地址
   * @param transactions 交易历史
   * @param holdings 当前持仓
   */
  static async classifyInvestor(
    address: string,
    transactions: any[],
    holdings: any[]
  ): Promise<string[]> {
    try {
      logger.info('开始对投资者进行分类', { address });
      
      const investorTypes: string[] = [];
      const metrics = this.calculateKeyMetrics(transactions, holdings);
      
      // 根据当前持仓总价值判断是否为鲸鱼
      if (metrics.totalHoldingsValue > 1000000) { // 持仓价值 > 100万美元
        investorTypes.push(this.INVESTOR_TYPES.WHALE);
      }
      
      // 根据早期投资行为识别早期投资者
      if (metrics.earlyInvestmentsCount > 5) { // 5次以上早期投资
        investorTypes.push(this.INVESTOR_TYPES.EARLY_INVESTOR);
      }
      
      // 根据交易频率和胜率识别波段交易者
      if (metrics.tradingFrequency > 10 && metrics.shortTermWinRate > 0.6) { // 频繁交易且短期胜率高
        investorTypes.push(this.INVESTOR_TYPES.SWING_TRADER);
      }
      
      // 根据持仓时间和收益识别价值投资者
      if (metrics.averageHoldingPeriod > 180 && metrics.longTermROI > 50) { // 平均持仓>180天且长期ROI>50%
        investorTypes.push(this.INVESTOR_TYPES.VALUE_INVESTOR);
      }
      
      // 根据逆市场操作识别逆势交易者
      if (metrics.contrarianScore > 0.7) { // 逆势交易得分高
        investorTypes.push(this.INVESTOR_TYPES.CONTRARIAN);
      }
      
      // 根据追涨杀跌特征识别动量交易者
      if (metrics.momentumScore > 0.7) { // 动量交易得分高
        investorTypes.push(this.INVESTOR_TYPES.MOMENTUM_TRADER);
      }
      
      // 根据套利行为识别套利交易者
      if (metrics.arbitrageCount > 10) { // 套利次数多
        investorTypes.push(this.INVESTOR_TYPES.ARBITRAGEUR);
      }
      
      // 根据流动性提供行为识别做市商
      if (metrics.liquidityProvisionScore > 0.8) { // 流动性提供得分高
        investorTypes.push(this.INVESTOR_TYPES.MARKET_MAKER);
      }
      
      // 根据交易模式识别量化交易者
      if (metrics.algorithmicPatternScore > 0.8) { // 算法交易模式得分高
        investorTypes.push(this.INVESTOR_TYPES.QUANTITATIVE);
      }
      
      // 根据交易规模和多链活动识别机构
      if (metrics.averageTradeSize > 100000 && metrics.crossChainActivityCount > 3) { // 大额交易且多链活动
        investorTypes.push(this.INVESTOR_TYPES.INSTITUTION);
      }
      
      logger.info('投资者分类结果', { address, investorTypes });
      
      return investorTypes;
    } catch (error) {
      logger.error('投资者分类失败', { error, address });
      return [];
    }
  }
  
  /**
   * 分析投资者交易特征
   * @param address 区块链地址
   * @param transactions 交易历史
   */
  static async analyzeTraits(
    address: string,
    transactions: any[],
    holdings: any[]
  ): Promise<any> {
    try {
      logger.info('开始分析投资者特征', { address });
      
      // 计算关键指标
      const metrics = this.calculateKeyMetrics(transactions, holdings);
      
      // 分析入场时机把握能力 (0-1)
      const entryTiming = this.calculateEntryTimingScore(transactions);
      
      // 分析出场时机把握能力 (0-1)
      const exitTiming = this.calculateExitTimingScore(transactions);
      
      // 分析多样化程度 (0-1)
      const diversification = this.calculateDiversificationScore(holdings);
      
      // 分析持币韧性 (0-1)
      const hodlStrength = this.calculateHodlStrengthScore(transactions);
      
      // 分析逆势操作倾向 (0-1)
      const contrarian = this.calculateContrarianScore(transactions);
      
      // 构建特征结果
      const traits = {
        entryTiming,
        exitTiming,
        diversification,
        hodlStrength,
        contrarian
      };
      
      logger.info('投资者特征分析结果', { address, traits });
      
      return traits;
    } catch (error) {
      logger.error('投资者特征分析失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 分析投资表现指标
   * @param address 区块链地址
   * @param transactions 交易历史
   */
  static async analyzePerformance(
    address: string,
    transactions: any[]
  ): Promise<any> {
    try {
      logger.info('开始分析投资表现', { address });
      
      // 计算总体ROI
      const overallROI = this.calculateOverallROI(transactions);
      
      // 计算月度ROI
      const monthlyROI = this.calculateMonthlyROI(transactions);
      
      // 计算胜率
      const winRate = this.calculateWinRate(transactions);
      
      // 计算夏普比率
      const sharpeRatio = this.calculateSharpeRatio(transactions);
      
      // 计算波动率
      const volatility = this.calculateVolatility(monthlyROI);
      
      // 计算最大回撤
      const maxDrawdown = this.calculateMaxDrawdown(transactions);
      
      // 构建表现结果
      const performance = {
        overallROI,
        monthlyROI,
        winRate,
        sharpeRatio,
        volatility,
        maxDrawdown
      };
      
      logger.info('投资表现分析结果', { 
        address, 
        performance: {
          overallROI,
          winRate,
          sharpeRatio
        } 
      });
      
      return performance;
    } catch (error) {
      logger.error('投资表现分析失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 处理新地址，判断是否是聪明钱并创建画像
   * @param address 区块链地址
   * @param transactions 交易历史
   * @param holdings 当前持仓
   */
  static async processNewAddress(
    address: string,
    transactions: any[],
    holdings: any[]
  ): Promise<ISmartMoneyProfile | null> {
    try {
      logger.info('处理新地址', { address });
      
      // 判断是否为聪明钱
      const isSmartMoney = await this.identifySmartMoney(address, transactions, holdings);
      
      // 如果不是聪明钱，则不创建画像
      if (!isSmartMoney) {
        logger.info('该地址不符合聪明钱标准', { address });
        return null;
      }
      
      // 对聪明钱进行分类
      const investorTypes = await this.classifyInvestor(address, transactions, holdings);
      
      // 分析交易特征
      const traits = await this.analyzeTraits(address, transactions, holdings);
      
      // 分析投资表现
      const performance = await this.analyzePerformance(address, transactions);
      
      // 分析活动统计
      const activityStats = this.analyzeActivityStats(transactions);
      
      // 创建聪明钱画像
      const profileData: Partial<ISmartMoneyProfile> = {
        address,
        investorTypes,
        tags: this.generateTags(investorTypes, traits, performance),
        classification: this.determineClassification(transactions, holdings),
        expertiseAreas: this.identifyExpertiseAreas(transactions, holdings),
        traits,
        performance,
        activityStats,
        currentHoldings: this.formatHoldings(holdings),
        successCases: this.identifySuccessCases(transactions),
        influence: this.estimateInfluence(address, transactions),
        metadata: {
          createdAt: new Date(),
          updatedAt: new Date(),
          dataQuality: this.assessDataQuality(transactions, holdings),
          dataSources: ['链上数据', '交易所API', '外部数据聚合']
        }
      };
      
      try {
        // 创建新画像
        const profile = await SmartMoneyProfileDAO.create(profileData as ISmartMoneyProfile);
        logger.info('聪明钱画像创建成功', { address });
        
        return profile;
      } catch (error) {
        if (error instanceof MongooseError && error.message.includes('duplicate key error')) {
          logger.warn('聪明钱画像已存在', { address });
          
          // 尝试更新现有画像
          const updatedProfile = await SmartMoneyProfileDAO.update(address, profileData);
          return updatedProfile;
        }
        
        throw error;
      }
    } catch (error) {
      logger.error('处理新地址失败', { error, address });
      return null;
    }
  }
  
  /**
   * 计算关键指标
   * @private
   */
  private static calculateKeyMetrics(transactions: any[], holdings: any[]): any {
    // 这里应该根据实际交易数据格式计算各类指标
    // 下面是一个简化的示例
    
    // 计算总体ROI
    const overallROI = this.calculateOverallROI(transactions);
    
    // 计算胜率(盈利交易/总交易)
    const winRate = this.calculateWinRate(transactions);
    
    // 计算总交易次数
    const totalTrades = transactions.length;
    
    // 计算平均ROI
    const averageROI = overallROI / Math.max(1, totalTrades);
    
    // 计算持仓总价值
    const totalHoldingsValue = holdings.reduce((sum, h) => sum + (h.valueUSD || 0), 0);
    
    // 计算早期投资次数(假设有entryDate和projectAge字段)
    const earlyInvestmentsCount = transactions.filter(
      (tx: any) => tx.projectAge && tx.projectAge < 30 // 项目上线30天内
    ).length;
    
    // 计算交易频率(每周交易次数)
    const oldestTxTime = Math.min(...transactions.map((tx: any) => tx.timestamp || Date.now()));
    const timeSpanInWeeks = (Date.now() - oldestTxTime) / (7 * 24 * 60 * 60 * 1000);
    const tradingFrequency = totalTrades / Math.max(1, timeSpanInWeeks);
    
    // 短期交易胜率(持有时间<7天的交易)
    const shortTermTrades = transactions.filter((tx: any) => tx.holdingPeriod && tx.holdingPeriod < 7);
    const shortTermWins = shortTermTrades.filter((tx: any) => tx.roi && tx.roi > 0);
    const shortTermWinRate = shortTermWins.length / Math.max(1, shortTermTrades.length);
    
    // 长期持仓ROI(持有时间>180天的交易)
    const longTermTrades = transactions.filter((tx: any) => tx.holdingPeriod && tx.holdingPeriod > 180);
    const longTermROI = longTermTrades.reduce((sum, tx: any) => sum + (tx.roi || 0), 0) / Math.max(1, longTermTrades.length);
    
    // 平均持仓时间(天)
    const averageHoldingPeriod = transactions.reduce((sum, tx: any) => sum + (tx.holdingPeriod || 0), 0) / Math.max(1, totalTrades);
    
    // 其他指标(这些需要更复杂的计算，这里只是示例)
    const contrarianScore = 0.5; // 逆势交易得分
    const momentumScore = 0.5; // 动量交易得分
    const arbitrageCount = 0; // 套利交易次数
    const liquidityProvisionScore = 0.0; // 流动性提供得分
    const algorithmicPatternScore = 0.0; // 算法交易模式得分
    const averageTradeSize = transactions.reduce((sum, tx: any) => sum + (tx.valueUSD || 0), 0) / Math.max(1, totalTrades);
    const crossChainActivityCount = new Set(transactions.map((tx: any) => tx.chainId)).size;
    
    return {
      overallROI,
      winRate,
      totalTrades,
      averageROI,
      totalHoldingsValue,
      earlyInvestmentsCount,
      tradingFrequency,
      shortTermWinRate,
      longTermROI,
      averageHoldingPeriod,
      contrarianScore,
      momentumScore,
      arbitrageCount,
      liquidityProvisionScore,
      algorithmicPatternScore,
      averageTradeSize,
      crossChainActivityCount
    };
  }
  
  /**
   * 计算总体ROI
   * @private
   */
  private static calculateOverallROI(transactions: any[]): number {
    // 简化计算，实际情况下需要考虑入金出金，计算真实ROI
    const totalInvestment = transactions.reduce((sum, tx: any) => tx.type === 'buy' ? sum + (tx.valueUSD || 0) : sum, 0);
    const totalReturn = transactions.reduce((sum, tx: any) => tx.type === 'sell' ? sum + (tx.valueUSD || 0) : sum, 0);
    
    if (totalInvestment === 0) return 0;
    
    return ((totalReturn - totalInvestment) / totalInvestment) * 100;
  }
  
  /**
   * 计算月度ROI
   * @private
   */
  private static calculateMonthlyROI(transactions: any[]): number[] {
    // 这里简化处理，真实情况需要按月聚合交易并计算每月ROI
    return [5, 10, -3, 8, 12, 7, -2, 9, 15, 4, -5, 11];
  }
  
  /**
   * 计算胜率
   * @private
   */
  private static calculateWinRate(transactions: any[]): number {
    const tradesWithROI = transactions.filter((tx: any) => tx.roi !== undefined);
    
    if (tradesWithROI.length === 0) return 0;
    
    const winningTrades = tradesWithROI.filter((tx: any) => tx.roi > 0);
    return winningTrades.length / tradesWithROI.length;
  }
  
  /**
   * 计算夏普比率
   * @private
   */
  private static calculateSharpeRatio(transactions: any[]): number {
    // 简化计算，实际需要计算收益率的平均值和标准差
    const riskFreeRate = 2; // 假设无风险利率为2%
    
    // 获取所有交易ROI
    const rois = transactions
      .filter((tx: any) => tx.roi !== undefined)
      .map((tx: any) => tx.roi);
    
    if (rois.length === 0) return 0;
    
    // 计算平均ROI
    const avgROI = rois.reduce((sum, roi) => sum + roi, 0) / rois.length;
    
    // 计算ROI标准差
    const variance = rois.reduce((sum, roi) => sum + Math.pow(roi - avgROI, 2), 0) / rois.length;
    const stdDev = Math.sqrt(variance);
    
    if (stdDev === 0) return 0;
    
    // 计算夏普比率
    return (avgROI - riskFreeRate) / stdDev;
  }
  
  /**
   * 计算波动率
   * @private
   */
  private static calculateVolatility(monthlyROI: number[]): number {
    // 简化计算，使用标准差作为波动率指标
    if (monthlyROI.length === 0) return 0;
    
    const avgROI = monthlyROI.reduce((sum, roi) => sum + roi, 0) / monthlyROI.length;
    const variance = monthlyROI.reduce((sum, roi) => sum + Math.pow(roi - avgROI, 2), 0) / monthlyROI.length;
    
    return Math.sqrt(variance);
  }
  
  /**
   * 计算最大回撤
   * @private
   */
  private static calculateMaxDrawdown(transactions: any[]): number {
    // 简化计算，实际情况需要根据账户价值历史计算
    return 0.25; // 假设最大回撤为25%
  }
  
  /**
   * 计算入场时机把握分数
   * @private
   */
  private static calculateEntryTimingScore(transactions: any[]): number {
    // 简化计算，实际需要分析买入时机与市场底部的关系
    return 0.75; // 假设入场时机得分为75%
  }
  
  /**
   * 计算出场时机把握分数
   * @private
   */
  private static calculateExitTimingScore(transactions: any[]): number {
    // 简化计算，实际需要分析卖出时机与市场顶部的关系
    return 0.68; // 假设出场时机得分为68%
  }
  
  /**
   * 计算多样化程度分数
   * @private
   */
  private static calculateDiversificationScore(holdings: any[]): number {
    if (holdings.length === 0) return 0;
    
    // 简化计算，使用持仓数量与集中度的组合指标
    const diversity = Math.min(1, holdings.length / 15); // 假设15种资产为完全多样化
    
    // 计算集中度(HHI指数)
    const totalValue = holdings.reduce((sum, h) => sum + (h.valueUSD || 0), 0);
    
    if (totalValue === 0) return diversity;
    
    const hhi = holdings.reduce((sum, h) => {
      const marketShare = (h.valueUSD || 0) / totalValue;
      return sum + (marketShare * marketShare);
    }, 0);
    
    // 多样化得分(考虑资产数量和集中度)
    return diversity * (1 - Math.min(1, hhi));
  }
  
  /**
   * 计算持币韧性分数
   * @private
   */
  private static calculateHodlStrengthScore(transactions: any[]): number {
    // 简化计算，实际需要分析市场下跌期间的持币行为
    return 0.82; // 假设持币韧性得分为82%
  }
  
  /**
   * 计算逆势操作倾向分数
   * @private
   */
  private static calculateContrarianScore(transactions: any[]): number {
    // 简化计算，实际需要分析交易方向与市场趋势的关系
    return 0.63; // 假设逆势操作倾向得分为63%
  }
  
  /**
   * 分析活动统计
   * @private
   */
  private static analyzeActivityStats(transactions: any[]): any {
    // 总交易次数
    const totalTrades = transactions.length;
    
    // 计算平均交易金额
    const totalValue = transactions.reduce((sum, tx: any) => sum + (tx.valueUSD || 0), 0);
    const avgTradeValue = totalValue / Math.max(1, totalTrades);
    
    // 计算最高交易金额
    const highestTradeValue = Math.max(...transactions.map((tx: any) => tx.valueUSD || 0));
    
    // 计算交易频率(每周交易次数)
    const oldestTxTime = Math.min(...transactions.map((tx: any) => tx.timestamp || Date.now()));
    const timeSpanInWeeks = (Date.now() - oldestTxTime) / (7 * 24 * 60 * 60 * 1000);
    const tradingFrequency = totalTrades / Math.max(1, timeSpanInWeeks);
    
    // 最后活跃时间
    const lastActive = new Date(Math.max(...transactions.map((tx: any) => tx.timestamp || 0)));
    
    return {
      totalTrades,
      avgTradeValue,
      highestTradeValue,
      tradingFrequency,
      lastActive
    };
  }
  
  /**
   * 格式化持仓数据
   * @private
   */
  private static formatHoldings(holdings: any[]): any[] {
    return holdings.map(h => ({
      token: h.symbol || h.tokenAddress,
      amount: h.amount || 0,
      valueUSD: h.valueUSD || 0,
      allocation: h.allocation || (h.valueUSD / holdings.reduce((sum, h) => sum + (h.valueUSD || 0), 0)),
      entryPrice: h.entryPrice
    }));
  }
  
  /**
   * 识别成功案例
   * @private
   */
  private static identifySuccessCases(transactions: any[]): any[] {
    // 简化处理，实际需要分析完整的买卖交易对并计算ROI
    const successfulTrades = transactions
      .filter((tx: any) => tx.roi && tx.roi > 50) // ROI > 50%的交易视为成功案例
      .slice(0, 5); // 最多取5个
    
    return successfulTrades.map(tx => ({
      token: tx.symbol || tx.tokenAddress,
      roi: tx.roi,
      entryDate: new Date(tx.buyTimestamp || tx.timestamp),
      exitDate: tx.sellTimestamp ? new Date(tx.sellTimestamp) : undefined,
      description: `成功投资${tx.symbol || tx.tokenAddress}，获得${tx.roi}%回报`
    }));
  }
  
  /**
   * 评估影响力
   * @private
   */
  private static estimateInfluence(address: string, transactions: any[]): any {
    // 简化处理，实际需要分析社交媒体影响力和钱包跟随者
    return {
      followerCount: Math.floor(Math.random() * 50), // 模拟数据
      copyTradingValue: Math.floor(Math.random() * 1000000), // 模拟数据
      marketImpact: 0.01 + Math.random() * 0.2 // 模拟数据，1%-21%
    };
  }
  
  /**
   * 确定账户分类(个人/机构/项目)
   * @private
   */
  private static determineClassification(transactions: any[], holdings: any[]): string {
    // 简化处理，实际需要分析交易模式和规模
    const avgTradeSize = transactions.reduce((sum, tx: any) => sum + (tx.valueUSD || 0), 0) / Math.max(1, transactions.length);
    const totalHoldingsValue = holdings.reduce((sum, h) => sum + (h.valueUSD || 0), 0);
    
    if (avgTradeSize > 50000 || totalHoldingsValue > 1000000) {
      return '机构';
    }
    
    // 检查是否可能是项目方地址
    const isProjectAddress = holdings.some(h => h.allocation > 0.8); // 某个代币占比超过80%
    
    return isProjectAddress ? '项目' : '个人';
  }
  
  /**
   * 识别擅长领域
   * @private
   */
  private static identifyExpertiseAreas(transactions: any[], holdings: any[]): string[] {
    // 简化处理，实际需要分析交易代币的类别和领域
    const expertiseAreas = new Set<string>();
    
    // 根据交易代币类别添加领域
    transactions.forEach((tx: any) => {
      if (tx.category) {
        expertiseAreas.add(tx.category);
      }
    });
    
    // 根据持仓代币类别添加领域
    holdings.forEach((h: any) => {
      if (h.category) {
        expertiseAreas.add(h.category);
      }
    });
    
    // 如果没有识别到任何领域，添加一些默认领域
    if (expertiseAreas.size === 0) {
      // 分析DeFi相关交易
      const defiTokenCount = transactions.filter((tx: any) => 
        (tx.symbol || '').toLowerCase().includes('swap') || 
        (tx.symbol || '').toLowerCase().includes('lp') ||
        (tx.tokenAddress || '').toLowerCase().includes('0x')
      ).length;
      
      if (defiTokenCount > 5) {
        expertiseAreas.add('DeFi');
      }
      
      // 分析NFT相关交易
      const nftCount = transactions.filter((tx: any) => 
        tx.tokenType === 'ERC721' || 
        tx.tokenType === 'ERC1155'
      ).length;
      
      if (nftCount > 3) {
        expertiseAreas.add('NFT');
      }
    }
    
    return Array.from(expertiseAreas);
  }
  
  /**
   * 生成标签
   * @private
   */
  private static generateTags(
    investorTypes: string[],
    traits: any,
    performance: any
  ): string[] {
    const tags = [...investorTypes]; // 把投资者类型作为标签的一部分
    
    // 根据特征添加标签
    if (traits.entryTiming > 0.7) tags.push('买点准确');
    if (traits.exitTiming > 0.7) tags.push('卖点准确');
    if (traits.diversification > 0.7) tags.push('多元化投资');
    if (traits.hodlStrength > 0.8) tags.push('坚定持有者');
    if (traits.contrarian > 0.7) tags.push('逆势思考者');
    
    // 根据表现添加标签
    if (performance.overallROI > 200) tags.push('高收益');
    if (performance.winRate > 0.7) tags.push('高胜率');
    if (performance.sharpeRatio > 2) tags.push('高夏普比率');
    if (performance.volatility < 0.2) tags.push('低波动');
    if (performance.maxDrawdown < 0.15) tags.push('稳健投资');
    
    return tags;
  }
  
  /**
   * 评估数据质量
   * @private
   */
  private static assessDataQuality(transactions: any[], holdings: any[]): number {
    // 数据完整性检查
    const hasCompleteTransactions = transactions.every((tx: any) => 
      tx.timestamp && tx.valueUSD && (tx.symbol || tx.tokenAddress)
    );
    
    const hasCompleteHoldings = holdings.every((h: any) => 
      h.valueUSD && (h.symbol || h.tokenAddress) && h.amount
    );
    
    // 数据量检查
    const sufficientTransactions = transactions.length >= 10;
    const sufficientHoldings = holdings.length >= 2;
    
    // 数据时效性检查
    const latestTxTime = Math.max(...transactions.map((tx: any) => tx.timestamp || 0));
    const dataFreshness = Date.now() - latestTxTime < 30 * 24 * 60 * 60 * 1000; // 30天内
    
    // 计算总体数据质量分数
    let qualityScore = 0.5; // 基础分
    
    if (hasCompleteTransactions) qualityScore += 0.1;
    if (hasCompleteHoldings) qualityScore += 0.1;
    if (sufficientTransactions) qualityScore += 0.1;
    if (sufficientHoldings) qualityScore += 0.1;
    if (dataFreshness) qualityScore += 0.1;
    
    return Math.min(1, qualityScore);
  }
} 