import { loggerWinston as logger } from '../utils/logger';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { ISmartMoneyProfile } from '../database/models/SmartMoneyProfile';

/**
 * 聪明钱评分系统 - 负责计算和更新地址的聪明钱评分
 */
export class SmartMoneyScorer {
  /**
   * 计算地址的聪明钱总体评分
   * @param address 区块链地址
   */
  static async calculateScore(address: string): Promise<{ 
    overall: number;
    components: { 
      performance: number; 
      timing: number;
      portfolioManagement: number;
      riskManagement: number;
      insight: number;
    };
    confidence: number;
    trend: 'rising' | 'stable' | 'declining';
  }> {
    try {
      logger.info('开始计算聪明钱评分', { address });
      
      // 获取地址画像
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      if (!profile) {
        throw new Error(`未找到地址画像: ${address}`);
      }
      
      // 计算投资表现分数 (权重35%)
      const performanceScore = await this.calculatePerformanceScore(profile);
      
      // 计算时机把握分数 (权重25%)
      const timingScore = await this.calculateTimingScore(profile);
      
      // 计算组合管理分数 (权重20%)
      const portfolioManagementScore = await this.calculatePortfolioManagementScore(profile);
      
      // 计算风险管理分数 (权重10%)
      const riskManagementScore = await this.calculateRiskManagementScore(profile);
      
      // 计算洞察力分数 (权重10%)
      const insightScore = await this.calculateInsightScore(profile);
      
      // 加权计算总体评分
      const overallScore = Math.round(
        performanceScore * 0.35 +
        timingScore * 0.25 +
        portfolioManagementScore * 0.2 +
        riskManagementScore * 0.1 +
        insightScore * 0.1
      );
      
      // 计算置信度 (基于数据完整性和交易历史)
      const confidence = this.calculateConfidence(profile);
      
      // 判断趋势 (基于最近表现)
      const trend = this.determineTrend(profile);
      
      // 组装结果
      const result = {
        overall: overallScore,
        components: {
          performance: performanceScore,
          timing: timingScore,
          portfolioManagement: portfolioManagementScore,
          riskManagement: riskManagementScore,
          insight: insightScore
        },
        confidence,
        trend
      };
      
      logger.info('聪明钱评分计算完成', { address, score: result.overall });
      
      return result;
    } catch (error) {
      logger.error('计算聪明钱评分失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 计算投资表现评分 (0-100)
   * 考虑ROI、胜率和夏普比率
   */
  private static async calculatePerformanceScore(profile: ISmartMoneyProfile): Promise<number> {
    // 提取各项指标
    const { overallROI, winRate, sharpeRatio } = profile.performance;
    
    // ROI评分 (0-40分)
    // ROI越高，分数越高，但有上限
    const roiScore = Math.min(40, overallROI <= 0 ? 0 : (10 + Math.log10(overallROI) * 10));
    
    // 胜率评分 (0-40分)
    // 胜率越高，分数越高，呈现S型增长曲线
    const winRateScore = Math.min(40, winRate * 50);
    
    // 夏普比率评分 (0-20分)
    // 夏普比率越高，分数越高，但有上限
    const sharpeScore = Math.min(20, Math.max(0, (sharpeRatio - 0.5) * 10));
    
    // 总评分
    const totalScore = Math.round(roiScore + winRateScore + sharpeScore);
    
    return Math.min(100, Math.max(0, totalScore));
  }
  
  /**
   * 计算时机把握评分 (0-100)
   * 考虑入场时机、出场时机
   */
  private static async calculateTimingScore(profile: ISmartMoneyProfile): Promise<number> {
    // 提取各项指标
    const { entryTiming, exitTiming } = profile.traits;
    
    // 入场时机评分 (0-50分)
    const entryScore = entryTiming * 50;
    
    // 出场时机评分 (0-50分)
    const exitScore = exitTiming * 50;
    
    // 总评分
    const totalScore = Math.round(entryScore + exitScore);
    
    return Math.min(100, Math.max(0, totalScore));
  }
  
  /**
   * 计算投资组合管理评分 (0-100)
   * 考虑多样化程度、持仓调整、资产配置
   */
  private static async calculatePortfolioManagementScore(profile: ISmartMoneyProfile): Promise<number> {
    // 提取各项指标
    const { diversification, hodlStrength } = profile.traits;
    const { currentHoldings } = profile;
    
    // 多样化评分 (0-40分)
    const diversificationScore = diversification * 40;
    
    // 持仓韧性评分 (0-30分)
    const hodlScore = hodlStrength * 30;
    
    // 资产配置评分 (0-30分)
    // 基于持仓的规模分布
    let allocationScore = 15; // 默认基础分
    
    // 如果有足够的持仓数据，进行更详细的评估
    if (currentHoldings && currentHoldings.length > 0) {
      // 计算持仓价值分布的均衡性
      const totalValue = currentHoldings.reduce((sum, holding) => sum + holding.valueUSD, 0);
      
      // 计算集中度指数
      let concentrationIndex = 0;
      for (const holding of currentHoldings) {
        const percentage = holding.valueUSD / totalValue;
        concentrationIndex += percentage * percentage;
      }
      
      // 转换为均衡性分数 (越分散越高)
      const balanceScore = Math.min(30, Math.max(0, 30 - concentrationIndex * 100));
      
      allocationScore = balanceScore;
    }
    
    // 总评分
    const totalScore = Math.round(diversificationScore + hodlScore + allocationScore);
    
    return Math.min(100, Math.max(0, totalScore));
  }
  
  /**
   * 计算风险管理评分 (0-100)
   * 考虑波动率、最大回撤
   */
  private static async calculateRiskManagementScore(profile: ISmartMoneyProfile): Promise<number> {
    // 提取各项指标
    const { volatility, maxDrawdown } = profile.performance;
    
    // 波动率评分 (0-50分)
    // 波动率越低越好
    const volatilityScore = Math.min(50, Math.max(0, 50 - volatility * 25));
    
    // 最大回撤评分 (0-50分)
    // 最大回撤越小越好 (注意：maxDrawdown是正数，越小越好)
    const drawdownScore = Math.min(50, Math.max(0, 50 - maxDrawdown * 100));
    
    // 总评分
    const totalScore = Math.round(volatilityScore + drawdownScore);
    
    return Math.min(100, Math.max(0, totalScore));
  }
  
  /**
   * 计算洞察力评分 (0-100)
   * 考虑早期投资项目、创新投资
   */
  private static async calculateInsightScore(profile: ISmartMoneyProfile): Promise<number> {
    // 提取各项指标
    const { successCases, investorTypes } = profile;
    
    // 早期投资评分 (0-60分)
    let earlyInvestmentScore = 0;
    
    // 判断是否为早期投资者
    const isEarlyInvestor = investorTypes.includes('早期投资者');
    if (isEarlyInvestor) {
      earlyInvestmentScore += 20;
    }
    
    // 分析成功案例中的早期投资
    if (successCases && successCases.length > 0) {
      // 计算成功案例的ROI平均值
      const avgROI = successCases.reduce((sum, c) => sum + c.roi, 0) / successCases.length;
      
      // 根据平均ROI加分
      const roiBonus = Math.min(40, avgROI / 10);
      
      earlyInvestmentScore += roiBonus;
    }
    
    // 创新投资评分 (0-40分)
    let innovationScore = 0;
    
    // 分析投资类型
    const isContrarian = profile.traits.contrarian > 0.7;
    if (isContrarian) {
      innovationScore += 20;
    }
    
    // 分析擅长领域
    if (profile.expertiseAreas && profile.expertiseAreas.length > 0) {
      const innovativeAreas = ['DeFi', 'NFT', 'GameFi', 'AI', 'ZK', 'Layer2'];
      const hasInnovativeAreas = profile.expertiseAreas.some(area => 
        innovativeAreas.includes(area)
      );
      
      if (hasInnovativeAreas) {
        innovationScore += 20;
      }
    }
    
    // 总评分
    const totalScore = Math.round(earlyInvestmentScore + innovationScore);
    
    return Math.min(100, Math.max(0, totalScore));
  }
  
  /**
   * 计算评分置信度 (0-1)
   * 基于数据完整性和交易历史
   */
  private static calculateConfidence(profile: ISmartMoneyProfile): number {
    // 交易历史长度加分
    const historyConfidence = Math.min(0.5, profile.activityStats.totalTrades / 200);
    
    // 数据完整性加分
    let dataCompletenessScore = 0;
    
    // 检查各个关键字段是否有数据
    if (profile.performance.overallROI !== undefined) dataCompletenessScore += 0.1;
    if (profile.performance.winRate !== undefined) dataCompletenessScore += 0.1;
    if (profile.traits.entryTiming !== undefined) dataCompletenessScore += 0.05;
    if (profile.traits.exitTiming !== undefined) dataCompletenessScore += 0.05;
    if (profile.successCases && profile.successCases.length > 0) dataCompletenessScore += 0.1;
    if (profile.currentHoldings && profile.currentHoldings.length > 0) dataCompletenessScore += 0.1;
    
    // 总置信度
    const confidence = Math.min(1, Math.max(0, historyConfidence + dataCompletenessScore));
    
    return parseFloat(confidence.toFixed(2));
  }
  
  /**
   * 判断评分趋势
   * 基于最近表现
   */
  private static determineTrend(profile: ISmartMoneyProfile): 'rising' | 'stable' | 'declining' {
    // 提取月度ROI数据
    const { monthlyROI } = profile.performance;
    
    // 如果没有足够的历史数据，返回稳定
    if (!monthlyROI || monthlyROI.length < 3) {
      return 'stable';
    }
    
    // 计算近3个月和前3个月的平均ROI
    const recentMonths = monthlyROI.slice(-3);
    const previousMonths = monthlyROI.slice(-6, -3);
    
    // 如果前3个月数据不足，返回稳定
    if (previousMonths.length < 3) {
      return 'stable';
    }
    
    const recentAvg = recentMonths.reduce((sum, roi) => sum + roi, 0) / recentMonths.length;
    const previousAvg = previousMonths.reduce((sum, roi) => sum + roi, 0) / previousMonths.length;
    
    // 判断趋势
    const difference = recentAvg - previousAvg;
    
    if (difference > 10) {
      return 'rising';
    } else if (difference < -10) {
      return 'declining';
    } else {
      return 'stable';
    }
  }
  
  /**
   * 更新地址的聪明钱评分
   * @param address 区块链地址
   */
  static async updateScore(address: string): Promise<ISmartMoneyProfile | null> {
    try {
      // 计算最新评分
      const score = await this.calculateScore(address);
      
      // 更新到数据库
      const updatedProfile = await SmartMoneyProfileDAO.update(address, {
        score: {
          overall: score.overall,
          components: score.components,
          confidence: score.confidence,
          trend: score.trend,
          lastUpdated: new Date()
        }
      });
      
      return updatedProfile;
    } catch (error) {
      logger.error('更新聪明钱评分失败', { error, address });
      throw error;
    }
  }
} 