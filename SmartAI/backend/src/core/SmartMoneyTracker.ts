import { loggerWinston as logger } from '../utils/logger';
import { SmartMoneyIdentifier } from './SmartMoneyIdentifier';
import { PortfolioTracker } from './PortfolioTracker';
import { TransactionPatternAnalyzer } from './TransactionPatternAnalyzer';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { cache } from '../utils/cache';
import { ISmartMoneyProfile } from '../database/models/SmartMoneyProfile';
import { AddressAnalysisResult, SmartMoneyIdentificationResult } from '../../../shared/api/SmartMoneyAPI';

// 创建标准错误响应
interface ErrorResponse {
  error: boolean;
  code: string;
  message: string;
  details?: any;
}

/**
 * 聪明钱追踪服务
 * 整合聪明钱识别、投资组合追踪和交易模式分析
 */
export class SmartMoneyTracker {
  private static knex: any;

  /**
   * 初始化
   */
  static init(knexInstance: any) {
    this.knex = knexInstance;
    logger.info('聪明钱追踪服务初始化完成');
  }

  /**
   * 分析地址是否为聪明钱
   * @param address 区块链地址
   */
  static async analyzeAddress(address: string, options = { useCache: true }): Promise<AddressAnalysisResult> {
    try {
      logger.info('开始分析地址', { address });
      
      // 使用缓存机制
      const cacheKey = `smartmoney_analysis:${address}`;
      if (options.useCache) {
        const cachedAnalysis = cache.get<AddressAnalysisResult>(cacheKey);
        if (cachedAnalysis) {
          logger.debug('从缓存获取地址分析结果', { address });
          return cachedAnalysis;
        }
      }
      
      // 获取交易历史
      const transactions = await this.getTransactionHistory(address);
      
      if (!transactions || transactions.length === 0) {
        logger.warn('地址没有交易历史', { address });
        return {
          address,
          isSmartMoney: false,
          reason: '没有交易历史',
          score: 0.3,
          analysisTimestamp: new Date().toISOString()
        };
      }
      
      // 获取当前持仓
      const holdings = await PortfolioTracker.getCurrentHoldings(address);
      
      // 识别是否为聪明钱
      const isSmartMoney = await SmartMoneyIdentifier.identifySmartMoney(address, transactions, holdings);
      
      // 创建标准结果对象
      let smartMoneyInfo: SmartMoneyIdentificationResult = {
        isSmartMoney,
        score: isSmartMoney ? 0.7 : 0.3,
        confidence: 0.8,
        reason: isSmartMoney ? '满足聪明钱指标' : '不满足聪明钱标准'
      };
      
      // 如果不是聪明钱，返回简化结果
      if (!isSmartMoney) {
        const result: AddressAnalysisResult = {
          address,
          isSmartMoney: false,
          reason: smartMoneyInfo.reason,
          score: smartMoneyInfo.score,
          analysisTimestamp: new Date().toISOString()
        };
        
        // 缓存结果（较短时间）
        cache.set(cacheKey, result, 1800); // 30分钟
        
        return result;
      }
      
      // 为聪明钱获取更多详细信息
      try {
        // 分析投资特征
        const traits = await SmartMoneyIdentifier.analyzeTraits(address, transactions, holdings);
        
        // 分析性能指标
        const performance = await SmartMoneyIdentifier.analyzePerformance(address, transactions);
        
        // 获取投资者类型
        const investorTypes = await SmartMoneyIdentifier.classifyInvestor(address, transactions, holdings);
        
        // 丰富结果对象
        smartMoneyInfo = {
          ...smartMoneyInfo,
          investorType: investorTypes[0] || 'unknown',
          traits,
          expertiseAreas: investorTypes
            .filter(type => Object.prototype.hasOwnProperty.call(SmartMoneyIdentifier.INVESTOR_TYPES, type))
            .map(type => (SmartMoneyIdentifier.INVESTOR_TYPES as any)[type]?.expertiseAreas || [])
            .flat(),
          performanceMetrics: performance,
          tags: investorTypes,
          scoreComponents: {
            performance: performance?.overallROI > 0.2 ? 0.8 : 0.4,
            timing: traits?.entryTiming || 0.5,
            portfolioManagement: traits?.diversification || 0.5,
            riskManagement: 1 - (performance?.volatility || 0.5),
            insight: traits?.contrarian || 0.5
          }
        };
      } catch (error) {
        logger.error('获取聪明钱详细信息失败', { error, address });
        // 出错时仍然保留基本信息
      }
      
      // 分析投资组合
      const portfolioResult = await PortfolioTracker.getCurrentHoldings(address);
      
      // 分析交易模式
      const patternResult = await TransactionPatternAnalyzer.analyzeTransactionPatterns(address, transactions);
      
      // 构建完整分析结果
      const result: AddressAnalysisResult = {
        address,
        isSmartMoney: true,
        smartMoneyInfo,
        portfolio: portfolioResult,
        transactionPatterns: patternResult,
        analysisTimestamp: new Date().toISOString(),
        score: smartMoneyInfo.score
      };
      
      // 缓存结果
      cache.set(cacheKey, result, 3600); // 1小时
      
      // 创建或更新聪明钱档案
      await this.updateSmartMoneyProfile(address, result);
      
      logger.info('地址分析完成', { address, isSmartMoney: true });
      
      return result;
    } catch (error) {
      logger.error('分析地址失败', { error, address });
      const errorResponse: AddressAnalysisResult = {
        address,
        isSmartMoney: false,
        reason: '分析失败',
        score: 0,
        analysisTimestamp: new Date().toISOString(),
        error: true,
        code: 'ANALYSIS_FAILED',
        message: '分析地址失败',
        details: error instanceof Error ? error.message : String(error)
      } as unknown as AddressAnalysisResult;
      return errorResponse;
    }
  }

  /**
   * 批量分析地址
   * @param addresses 区块链地址列表
   */
  static async batchAnalyzeAddresses(addresses: string[]): Promise<AddressAnalysisResult[]> {
    try {
      logger.info('开始批量分析地址', { count: addresses.length });
      
      const results: AddressAnalysisResult[] = [];
      
      // 按批次处理，避免一次处理过多
      const batchSize = 10;
      
      for (let i = 0; i < addresses.length; i += batchSize) {
        const batch = addresses.slice(i, i + batchSize);
        const batchPromises = batch.map(address => this.analyzeAddress(address));
        
        const batchResults = await Promise.allSettled(batchPromises);
        
        batchResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            logger.error('批量分析地址失败', { address: batch[index], error: result.reason });
            results.push({
              address: batch[index],
              isSmartMoney: false,
              reason: '分析失败',
              score: 0.3,
              analysisTimestamp: new Date().toISOString()
            });
          }
        });
      }
      
      logger.info('批量分析地址完成', {
        total: addresses.length,
        smartMoneyCount: results.filter(r => r.isSmartMoney).length
      });
      
      return results;
    } catch (error) {
      logger.error('批量分析地址失败', { error });
      throw error;
    }
  }

  /**
   * 获取交易历史
   * @private
   */
  private static async getTransactionHistory(address: string): Promise<any[]> {
    try {
      // 从缓存获取
      const cacheKey = `tx_history:${address}`;
      const cachedHistory = cache.get<any[]>(cacheKey);
      
      if (cachedHistory) {
        return cachedHistory;
      }
      
      // 这里应该调用区块链数据服务获取交易历史
      // 示例实现，实际项目中应该替换为真实的数据源
      const transactions = await this.knex('transactions')
        .where('from', address)
        .orWhere('to', address)
        .orderBy('timestamp', 'desc')
        .limit(1000);
      
      // 缓存结果
      if (transactions && transactions.length > 0) {
        cache.set(cacheKey, transactions, 3600); // 1小时
      }
      
      return transactions || [];
    } catch (error) {
      logger.error('获取交易历史失败', { error, address });
      return [];
    }
  }

  /**
   * 创建或更新聪明钱档案
   * @private
   */
  private static async updateSmartMoneyProfile(address: string, analysisResult: AddressAnalysisResult): Promise<void> {
    try {
      // 使用 DAO 查询现有档案
      const existingProfile = await SmartMoneyProfileDAO.findByAddress(address);
      
      const investorType = analysisResult.smartMoneyInfo?.investorType || 'unknown';
      
      // 创建元数据对象
      const metaData = {
        createdAt: new Date(),
        updatedAt: new Date(),
        dataQuality: 0.8,
        dataSources: ['on-chain', 'transaction-analysis', 'portfolio-tracking']
      };
      
      const profileData: Partial<ISmartMoneyProfile> = {
        address,
        investorTypes: [investorType],
        tags: analysisResult.smartMoneyInfo?.tags || [],
        classification: '个人', // 默认分类
        expertiseAreas: analysisResult.smartMoneyInfo?.expertiseAreas || [],
        
        performance: {
          overallROI: analysisResult.smartMoneyInfo?.performanceMetrics?.overallROI || 0,
          monthlyROI: analysisResult.smartMoneyInfo?.performanceMetrics?.monthlyROI || [],
          winRate: analysisResult.smartMoneyInfo?.performanceMetrics?.winRate || 0,
          sharpeRatio: analysisResult.smartMoneyInfo?.performanceMetrics?.sharpeRatio || 0,
          volatility: analysisResult.smartMoneyInfo?.performanceMetrics?.volatility || 0,
          maxDrawdown: analysisResult.smartMoneyInfo?.performanceMetrics?.maxDrawdown || 0
        },
        
        traits: {
          entryTiming: analysisResult.smartMoneyInfo?.traits?.entryTiming || 0.5,
          exitTiming: analysisResult.smartMoneyInfo?.traits?.exitTiming || 0.5,
          hodlStrength: analysisResult.smartMoneyInfo?.traits?.hodlStrength || 0.5,
          diversification: analysisResult.smartMoneyInfo?.traits?.diversification || 0.5,
          contrarian: analysisResult.smartMoneyInfo?.traits?.contrarian || 0.5
        },
        
        currentHoldings: analysisResult.portfolio?.map((holding: any) => ({
          token: holding.symbol || holding.tokenAddress,
          amount: holding.amount || 0,
          valueUSD: holding.valueUSD || 0,
          allocation: holding.allocation || 0,
          entryPrice: holding.entryPrice
        })) || [],
        
        activityStats: {
          totalTrades: analysisResult.transactionPatterns?.overview.transactionCount || 0,
          avgTradeValue: analysisResult.transactionPatterns?.sizePatterns?.averageSize || 0,
          highestTradeValue: analysisResult.transactionPatterns?.sizePatterns?.maxSize || 0,
          tradingFrequency: analysisResult.transactionPatterns?.frequencyPatterns?.averageFrequency || 0,
          lastActive: analysisResult.transactionPatterns?.overview.lastTransaction || new Date()
        },
        
        score: {
          overall: analysisResult.smartMoneyInfo?.score || 0,
          components: {
            performance: analysisResult.smartMoneyInfo?.scoreComponents?.performance || 0,
            timing: analysisResult.smartMoneyInfo?.scoreComponents?.timing || 0,
            portfolioManagement: analysisResult.smartMoneyInfo?.scoreComponents?.portfolioManagement || 0,
            riskManagement: analysisResult.smartMoneyInfo?.scoreComponents?.riskManagement || 0,
            insight: analysisResult.smartMoneyInfo?.scoreComponents?.insight || 0
          },
          confidence: analysisResult.smartMoneyInfo?.confidence || 0.5,
          trend: 'stable',
          lastUpdated: new Date()
        },
        
        metadata: metaData
      };
      
      if (existingProfile) {
        // 更新现有档案
        await SmartMoneyProfileDAO.update(address, profileData);
        logger.debug('更新聪明钱档案', { address });
      } else {
        // 创建新档案
        await SmartMoneyProfileDAO.create({
          ...profileData,
          relatedAddresses: [],
          successCases: [],
          influence: {
            followerCount: 0,
            copyTradingValue: 0,
            marketImpact: 0
          }
        } as ISmartMoneyProfile);
        logger.debug('创建聪明钱档案', { address });
      }
    } catch (error) {
      logger.error('更新聪明钱档案失败', { error, address });
      // 档案更新失败不应阻止分析过程
    }
  }

  /**
   * 获取聪明钱排行榜
   * @param options 选项
   */
  static async getLeaderboard(options: any = {}): Promise<ISmartMoneyProfile[]> {
    try {
      // 从数据库获取排行榜
      return await SmartMoneyProfileDAO.getLeaderboard(options);
    } catch (error) {
      logger.error('获取排行榜失败', { error });
      throw error;
    }
  }

  /**
   * 获取投资者类型分布
   */
  static async getInvestorTypeDistribution(): Promise<Record<string, number>> {
    try {
      return await SmartMoneyProfileDAO.getInvestorTypeDistribution();
    } catch (error) {
      logger.error('获取投资者类型分布失败', { error });
      throw error;
    }
  }

  /**
   * 获取ROI分布
   */
  static async getROIDistribution(): Promise<any> {
    try {
      return await SmartMoneyProfileDAO.getROIDistribution();
    } catch (error) {
      logger.error('获取ROI分布失败', { error });
      throw error;
    }
  }

  /**
   * 获取最近活跃的聪明钱
   * @param days 最近几天
   * @param limit 限制数量
   */
  static async getRecentlyActive(days: number = 7, limit: number = 10): Promise<ISmartMoneyProfile[]> {
    try {
      return await SmartMoneyProfileDAO.getRecentlyActive(days, limit);
    } catch (error) {
      logger.error('获取最近活跃的聪明钱失败', { error, days, limit });
      throw error;
    }
  }

  /**
   * 预测聪明钱未来可能的交易
   * @param address 区块链地址
   */
  static async predictFutureActivity(address: string): Promise<any> {
    try {
      logger.info('预测聪明钱未来活动', { address });
      
      // 检查是否是聪明钱
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      
      if (!profile) {
        throw new Error('该地址不是聪明钱或尚未分析');
      }
      
      // 获取交易历史
      const transactions = await this.getTransactionHistory(address);
      
      // 使用交易模式分析器预测未来行为
      const prediction = await TransactionPatternAnalyzer.predictFutureTrading(address, transactions);
      
      return {
        address,
        profile: {
          investorTypes: profile.investorTypes,
          score: profile.score.overall
        },
        prediction
      };
    } catch (error) {
      logger.error('预测聪明钱未来活动失败', { error, address });
      throw error;
    }
  }

  /**
   * 跟踪聪明钱投资组合变化
   * @param address 区块链地址
   */
  static async trackPortfolioChanges(address: string): Promise<any> {
    try {
      logger.info('跟踪聪明钱投资组合变化', { address });
      
      // 获取当前持仓
      const currentHoldings = await PortfolioTracker.getCurrentHoldings(address);
      
      // 获取档案中的持仓
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      const previousHoldings = profile?.currentHoldings || [];
      
      // 分析变化
      const changes = await PortfolioTracker.analyzePortfolioChanges(address, previousHoldings, currentHoldings);
      
      // 更新档案中的持仓
      if (profile) {
        await SmartMoneyProfileDAO.updateHoldings(address, currentHoldings);
      }
      
      return {
        address,
        changes,
        currentHoldings,
        previousHoldings
      };
    } catch (error) {
      logger.error('跟踪聪明钱投资组合变化失败', { error, address });
      throw error;
    }
  }

  /**
   * 监控多个聪明钱的共同行为
   * @param addresses 地址列表
   */
  static async monitorCommonBehavior(addresses: string[]): Promise<any> {
    try {
      logger.info('监控聪明钱共同行为', { count: addresses.length });
      
      if (addresses.length < 2) {
        throw new Error('需要至少两个地址进行比较');
      }
      
      // 获取所有地址的持仓
      const holdingsPromises = addresses.map(address => 
        PortfolioTracker.getCurrentHoldings(address)
      );
      
      const allHoldings = await Promise.all(holdingsPromises);
      
      // 查找共同持有的代币
      const commonTokens = this.findCommonTokens(allHoldings);
      
      // 获取所有地址的交易模式
      const transactionsPromises = addresses.map(address => 
        this.getTransactionHistory(address)
      );
      
      const allTransactions = await Promise.all(transactionsPromises);
      
      // 分析交易模式
      const patternPromises = addresses.map((address, index) => 
        TransactionPatternAnalyzer.analyzeTransactionPatterns(address, allTransactions[index])
      );
      
      const allPatterns = await Promise.all(patternPromises);
      
      // 查找共同的交易策略
      const commonStrategies = this.findCommonStrategies(allPatterns);
      
      return {
        addresses,
        commonTokens,
        commonStrategies,
        analysisTimestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('监控聪明钱共同行为失败', { error, addresses });
      throw error;
    }
  }

  /**
   * 查找共同持有的代币
   * @private
   */
  private static findCommonTokens(allHoldings: any[]): any[] {
    try {
      if (allHoldings.length === 0) {
        return [];
      }
      
      // 获取第一个地址的代币列表
      const firstAddressTokens = new Set(
        allHoldings[0].map((holding: any) => holding.tokenAddress || holding.symbol)
      );
      
      // 查找所有地址共同持有的代币
      const commonTokens = [...firstAddressTokens].filter(token => {
        return allHoldings.every(holdings => 
          holdings.some((h: any) => (h.tokenAddress || h.symbol) === token)
        );
      });
      
      // 获取共同代币的详细信息
      return commonTokens.map(token => {
        const tokenDetails = allHoldings.map(holdings => 
          holdings.find((h: any) => (h.tokenAddress || h.symbol) === token)
        );
        
        return {
          token,
          symbol: tokenDetails[0]?.symbol,
          name: tokenDetails[0]?.name,
          holdingAddresses: allHoldings.length,
          averageHoldingValue: tokenDetails.reduce((sum, details) => sum + (details?.valueUSD || 0), 0) / tokenDetails.length
        };
      });
    } catch (error) {
      logger.error('查找共同持有的代币失败', { error });
      return [];
    }
  }

  /**
   * 查找共同的交易策略
   * @private
   */
  private static findCommonStrategies(allPatterns: any[]): string[] {
    try {
      if (allPatterns.length === 0) {
        return [];
      }
      
      // 获取第一个地址的策略列表
      const firstAddressStrategies = new Set<string>(allPatterns[0]?.strategies || []);
      
      // 查找所有地址共同使用的策略
      return [...firstAddressStrategies].filter(strategy => {
        return allPatterns.every(pattern => 
          (pattern?.strategies || []).includes(strategy)
        );
      });
    } catch (error) {
      logger.error('查找共同的交易策略失败', { error });
      return [];
    }
  }

  /**
   * 获取热门代币和新兴代币
   */
  static async getTrendingTokens(): Promise<any> {
    try {
      logger.info('获取热门代币');
      
      // 从缓存获取
      const cacheKey = 'trending_tokens';
      const cachedTokens = cache.get<any>(cacheKey);
      
      if (cachedTokens) {
        return cachedTokens;
      }
      
      // 获取最近活跃的聪明钱
      const recentActive = await this.getRecentlyActive(7, 50);
      
      if (!recentActive || recentActive.length === 0) {
        return {
          trending: [],
          emerging: []
        };
      }
      
      // 获取这些聪明钱持有的代币
      const holdings = recentActive
        .filter(profile => profile.currentHoldings && profile.currentHoldings.length > 0)
        .map(profile => profile.currentHoldings)
        .flat();
      
      // 计算每个代币被持有的数量和总价值
      const tokenStats: Record<string, any> = {};
      
      holdings.forEach(holding => {
        const token = holding.token;
        
        if (!tokenStats[token]) {
          tokenStats[token] = {
            token,
            symbol: holding.token,
            name: holding.token,
            holdingCount: 0,
            totalValue: 0,
            lastAdded: new Date()
          };
        }
        
        tokenStats[token].holdingCount++;
        tokenStats[token].totalValue += (holding.valueUSD || 0);
      });
      
      // 按持有数量排序获取热门代币
      const trending = Object.values(tokenStats)
        .sort((a, b) => b.holdingCount - a.holdingCount)
        .slice(0, 10);
      
      // 按最近添加时间排序获取新兴代币
      const emerging = Object.values(tokenStats)
        .sort((a, b) => new Date(b.lastAdded).getTime() - new Date(a.lastAdded).getTime())
        .filter(token => token.holdingCount <= 3) // 较少人持有的新代币
        .slice(0, 10);
      
      const result = {
        trending,
        emerging
      };
      
      // 缓存结果
      cache.set(cacheKey, result, 3600); // 1小时
      
      return result;
    } catch (error) {
      logger.error('获取热门代币失败', { error });
      throw error;
    }
  }
} 