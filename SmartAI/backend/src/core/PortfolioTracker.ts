import { loggerWinston as logger } from '../utils/logger';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { cache } from '../utils/cache';

/**
 * 投资组合追踪服务
 * 负责追踪和分析聪明钱地址的投资组合变化
 */
export class PortfolioTracker {
  /**
   * 获取地址当前持仓
   * @param address 区块链地址
   */
  static async getCurrentHoldings(address: string): Promise<any[]> {
    try {
      logger.info('获取地址当前持仓', { address });
      
      // 从缓存获取
      const cacheKey = `holdings:${address}`;
      const cachedHoldings = cache.get<any[]>(cacheKey);
      
      if (cachedHoldings) {
        logger.debug('从缓存获取持仓数据', { address });
        return cachedHoldings;
      }
      
      // 从数据库获取最新持仓
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      if (!profile || !profile.currentHoldings) {
        logger.warn('未找到地址持仓数据', { address });
        return [];
      }
      
      // 更新缓存
      cache.set(cacheKey, profile.currentHoldings, 300); // 缓存5分钟
      
      return profile.currentHoldings;
    } catch (error) {
      logger.error('获取当前持仓失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 更新地址持仓信息
   * @param address 区块链地址
   * @param holdings 新的持仓信息
   */
  static async updateHoldings(address: string, holdings: any[]): Promise<boolean> {
    try {
      logger.info('更新地址持仓信息', { address, holdingsCount: holdings.length });
      
      // 格式化持仓数据
      const formattedHoldings = holdings.map(h => ({
        token: h.symbol || h.tokenAddress || h.token,
        amount: h.amount || 0,
        valueUSD: h.valueUSD || 0,
        allocation: h.allocation || (h.valueUSD / holdings.reduce((sum, item) => sum + (item.valueUSD || 0), 0)),
        entryPrice: h.entryPrice
      }));
      
      // 更新数据库
      const updatedProfile = await SmartMoneyProfileDAO.updateHoldings(address, formattedHoldings);
      
      if (!updatedProfile) {
        logger.warn('更新持仓信息失败，未找到该地址画像', { address });
        return false;
      }
      
      // 更新缓存
      const cacheKey = `holdings:${address}`;
      cache.set(cacheKey, formattedHoldings, 300);
      
      logger.info('持仓信息更新成功', { address });
      return true;
    } catch (error) {
      logger.error('更新持仓信息失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 分析投资组合变化
   * @param address 区块链地址
   * @param previousHoldings 之前的持仓
   * @param currentHoldings 当前的持仓
   */
  static async analyzePortfolioChanges(
    address: string,
    previousHoldings: any[],
    currentHoldings: any[]
  ): Promise<any> {
    try {
      logger.info('分析投资组合变化', { address });
      
      // 创建持仓映射
      const prevMap = new Map();
      previousHoldings.forEach(h => {
        prevMap.set(h.token, h);
      });
      
      const currMap = new Map();
      currentHoldings.forEach(h => {
        currMap.set(h.token, h);
      });
      
      // 分析新增的代币
      const addedTokens = currentHoldings
        .filter(h => !prevMap.has(h.token))
        .map(h => ({
          token: h.token,
          amount: h.amount,
          valueUSD: h.valueUSD,
          allocation: h.allocation
        }));
      
      // 分析移除的代币
      const removedTokens = previousHoldings
        .filter(h => !currMap.has(h.token))
        .map(h => ({
          token: h.token,
          amount: h.amount,
          valueUSD: h.valueUSD,
          allocation: h.allocation
        }));
      
      // 分析持仓变化
      const changedTokens = currentHoldings
        .filter(h => prevMap.has(h.token) && h.amount !== prevMap.get(h.token).amount)
        .map(h => {
          const prev = prevMap.get(h.token);
          return {
            token: h.token,
            previousAmount: prev.amount,
            currentAmount: h.amount,
            amountChange: h.amount - prev.amount,
            previousValueUSD: prev.valueUSD,
            currentValueUSD: h.valueUSD,
            valueChange: h.valueUSD - prev.valueUSD,
            percentChange: prev.valueUSD ? ((h.valueUSD - prev.valueUSD) / prev.valueUSD) * 100 : 0
          };
        });
      
      // 计算总价值变化
      const previousTotalValue = previousHoldings.reduce((sum, h) => sum + (h.valueUSD || 0), 0);
      const currentTotalValue = currentHoldings.reduce((sum, h) => sum + (h.valueUSD || 0), 0);
      const totalValueChange = currentTotalValue - previousTotalValue;
      const totalPercentChange = previousTotalValue ? (totalValueChange / previousTotalValue) * 100 : 0;
      
      // 计算资产配置变化
      const allocationChanges = currentHoldings
        .filter(h => prevMap.has(h.token) && h.allocation !== prevMap.get(h.token).allocation)
        .map(h => {
          const prev = prevMap.get(h.token);
          return {
            token: h.token,
            previousAllocation: prev.allocation,
            currentAllocation: h.allocation,
            allocationChange: h.allocation - prev.allocation
          };
        });
      
      // 构建结果
      const result = {
        address,
        timeframe: {
          from: new Date(Date.now() - 24 * 60 * 60 * 1000), // 假设前一天
          to: new Date()
        },
        summary: {
          previousTotalValue,
          currentTotalValue,
          totalValueChange,
          totalPercentChange,
          addedTokensCount: addedTokens.length,
          removedTokensCount: removedTokens.length,
          changedTokensCount: changedTokens.length
        },
        details: {
          addedTokens,
          removedTokens,
          changedTokens,
          allocationChanges
        }
      };
      
      logger.info('投资组合变化分析完成', { 
        address, 
        totalValueChange, 
        totalPercentChange 
      });
      
      return result;
    } catch (error) {
      logger.error('分析投资组合变化失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 计算投资组合多样化指标
   * @param holdings 持仓信息
   */
  static calculateDiversificationMetrics(holdings: any[]): any {
    try {
      if (!holdings || holdings.length === 0) {
        return {
          diversificationScore: 0,
          herfindahlIndex: 1,
          effectiveCount: 1,
          largestAllocation: 0,
          tokenCount: 0
        };
      }
      
      // 计算赫芬达尔-赫希曼指数(HHI)
      // HHI = sum(marketShare^2)，越低表示越分散
      const herfindahlIndex = holdings.reduce((sum, h) => {
        const allocation = h.allocation || 0;
        return sum + (allocation * allocation);
      }, 0);
      
      // 计算有效代币数量(Effective N)
      // Effective N = 1/HHI，表示等效的均匀分布资产数量
      const effectiveCount = herfindahlIndex > 0 ? 1 / herfindahlIndex : 0;
      
      // 最大持仓比例
      const largestAllocation = Math.max(...holdings.map(h => h.allocation || 0));
      
      // 代币数量
      const tokenCount = holdings.length;
      
      // 综合多样化得分(0-1)
      // 考虑有效数量和实际代币数量的比值，以及最大持仓比例
      const diversificationRatio = effectiveCount / Math.max(1, tokenCount);
      const diversificationScore = Math.min(1, diversificationRatio * (1 - Math.min(1, largestAllocation)));
      
      return {
        diversificationScore,
        herfindahlIndex,
        effectiveCount,
        largestAllocation,
        tokenCount
      };
    } catch (error) {
      logger.error('计算多样化指标失败', { error });
      return {
        diversificationScore: 0,
        herfindahlIndex: 1,
        effectiveCount: 1,
        largestAllocation: 0,
        tokenCount: 0
      };
    }
  }
  
  /**
   * 计算持仓价值分布
   * @param holdings 持仓信息
   */
  static calculateValueDistribution(holdings: any[]): any {
    try {
      if (!holdings || holdings.length === 0) {
        return {
          totalValue: 0,
          topHoldings: [],
          categories: {}
        };
      }
      
      // 计算总价值
      const totalValue = holdings.reduce((sum, h) => sum + (h.valueUSD || 0), 0);
      
      // 获取前5大持仓
      const topHoldings = [...holdings]
        .sort((a, b) => (b.valueUSD || 0) - (a.valueUSD || 0))
        .slice(0, 5)
        .map(h => ({
          token: h.token,
          valueUSD: h.valueUSD || 0,
          allocation: h.allocation || 0,
          percentage: totalValue > 0 ? ((h.valueUSD || 0) / totalValue) * 100 : 0
        }));
      
      // 按类别分组
      const categories: Record<string, number> = {};
      
      holdings.forEach(h => {
        const category = h.category || '未分类';
        categories[category] = (categories[category] || 0) + (h.valueUSD || 0);
      });
      
      // 计算类别占比
      Object.keys(categories).forEach(key => {
        const categoryValue = categories[key];
        categories[key] = {
          valueUSD: categoryValue,
          percentage: totalValue > 0 ? (categoryValue / totalValue) * 100 : 0
        };
      });
      
      return {
        totalValue,
        topHoldings,
        categories
      };
    } catch (error) {
      logger.error('计算持仓价值分布失败', { error });
      return {
        totalValue: 0,
        topHoldings: [],
        categories: {}
      };
    }
  }
  
  /**
   * 分析投资组合风险
   * @param holdings 持仓信息
   * @param marketData 市场数据(代币相关性、波动率等)
   */
  static analyzePortfolioRisk(holdings: any[], marketData: any): any {
    try {
      if (!holdings || holdings.length === 0) {
        return {
          riskScore: 0,
          volatility: 0,
          correlationRisk: 0,
          concentrationRisk: 0,
          liquidityRisk: 0
        };
      }
      
      // 计算波动率风险 (基于持仓代币的历史波动率)
      const volatilityRisk = holdings.reduce((sum, h) => {
        const token = h.token;
        const allocation = h.allocation || 0;
        const tokenData = marketData?.tokens?.[token] || {};
        const tokenVolatility = tokenData.volatility || 0.5; // 默认波动率
        
        return sum + (allocation * tokenVolatility);
      }, 0);
      
      // 计算集中度风险 (基于赫芬达尔指数)
      const diversificationMetrics = this.calculateDiversificationMetrics(holdings);
      const concentrationRisk = diversificationMetrics.herfindahlIndex;
      
      // 计算流动性风险 (基于持仓代币的流动性)
      const liquidityRisk = holdings.reduce((sum, h) => {
        const token = h.token;
        const allocation = h.allocation || 0;
        const tokenData = marketData?.tokens?.[token] || {};
        const tokenLiquidity = tokenData.liquidity || 0.5; // 默认流动性
        
        // 流动性越低，风险越高
        return sum + (allocation * (1 - tokenLiquidity));
      }, 0);
      
      // 计算相关性风险 (基于持仓代币间的相关性)
      let correlationRisk = 0;
      
      // 简化计算，实际需要使用相关性矩阵
      if (holdings.length > 1 && marketData?.correlationMatrix) {
        correlationRisk = 0.3; // 假设数值
      }
      
      // 综合风险得分 (0-1)
      const riskScore = Math.min(1, 
        (volatilityRisk * 0.4) + 
        (concentrationRisk * 0.3) + 
        (liquidityRisk * 0.2) + 
        (correlationRisk * 0.1)
      );
      
      return {
        riskScore,
        volatility: volatilityRisk,
        correlationRisk,
        concentrationRisk,
        liquidityRisk
      };
    } catch (error) {
      logger.error('分析投资组合风险失败', { error });
      return {
        riskScore: 0,
        volatility: 0,
        correlationRisk: 0,
        concentrationRisk: 0,
        liquidityRisk: 0
      };
    }
  }
  
  /**
   * 分析投资组合表现
   * @param address 区块链地址
   * @param timeframe 时间范围(天)
   */
  static async analyzePerformance(address: string, timeframe: number = 30): Promise<any> {
    try {
      logger.info('分析投资组合表现', { address, timeframe });
      
      // 从画像中获取表现数据
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      if (!profile) {
        logger.warn('未找到地址画像', { address });
        return null;
      }
      
      // 提取表现指标
      const { performance } = profile;
      
      // 计算特定时间范围的ROI
      const periodROI = this.calculatePeriodROI(performance, timeframe);
      
      // 构建结果
      const result = {
        address,
        timeframe,
        overall: {
          roi: performance.overallROI,
          winRate: performance.winRate,
          sharpeRatio: performance.sharpeRatio
        },
        period: {
          roi: periodROI,
          volatility: performance.volatility,
          maxDrawdown: performance.maxDrawdown
        },
        comparison: {
          vsBTC: this.compareWithBenchmark(periodROI, 'BTC', timeframe),
          vsETH: this.compareWithBenchmark(periodROI, 'ETH', timeframe),
          vsMarket: this.compareWithBenchmark(periodROI, 'MARKET', timeframe)
        },
        riskAdjusted: {
          sharpeRatio: performance.sharpeRatio,
          sortinoRatio: performance.sortinoRatio || 0,
          calmarRatio: this.calculateCalmarRatio(periodROI, performance.maxDrawdown)
        }
      };
      
      logger.info('投资组合表现分析完成', { address, periodROI });
      
      return result;
    } catch (error) {
      logger.error('分析投资组合表现失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 计算特定时间段的ROI
   * @private
   */
  private static calculatePeriodROI(performance: any, timeframe: number): number {
    // 简化计算，实际需要根据月度ROI数据计算特定时间段的ROI
    const { monthlyROI, overallROI } = performance;
    
    if (!monthlyROI || monthlyROI.length === 0) {
      return overallROI || 0;
    }
    
    // 转换时间范围从天到月
    const months = Math.ceil(timeframe / 30);
    const relevantMonths = monthlyROI.slice(-Math.min(months, monthlyROI.length));
    
    // 计算累计收益率
    return relevantMonths.reduce((accumulator, roi) => {
      // 使用复利计算: (1+r1)*(1+r2)*(1+r3)...-1
      return accumulator * (1 + roi/100);
    }, 1) - 1;
  }
  
  /**
   * 与基准指标比较
   * @private
   */
  private static compareWithBenchmark(roi: number, benchmark: string, timeframe: number): number {
    // 简化计算，实际需要获取真实市场数据
    const benchmarkROI = {
      'BTC': 0.05, // 假设BTC在此期间上涨了5%
      'ETH': 0.08, // 假设ETH在此期间上涨了8%
      'MARKET': 0.03 // 假设整体市场在此期间上涨了3%
    };
    
    return roi - (benchmarkROI[benchmark] || 0);
  }
  
  /**
   * 计算卡尔马比率(Calmar Ratio)
   * @private
   */
  private static calculateCalmarRatio(roi: number, maxDrawdown: number): number {
    if (!maxDrawdown || maxDrawdown === 0) {
      return 0;
    }
    
    // 卡尔马比率 = 年化收益率 / 最大回撤
    // 简化计算，假设roi已经是年化的
    return roi / maxDrawdown;
  }
} 