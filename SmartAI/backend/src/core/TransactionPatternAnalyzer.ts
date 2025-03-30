import { loggerWinston as logger } from '../utils/logger';
import { cache } from '../utils/cache';

interface Transaction {
  hash: string;
  timestamp: number;
  type: string; // 'buy' | 'sell' | 'transfer' | 'swap' | 'other'
  tokenAddress?: string;
  tokenSymbol?: string;
  amount?: number;
  valueUSD?: number;
  price?: number;
  from?: string;
  to?: string;
  blockNumber?: number;
  gas?: number;
  gasPrice?: number;
  chainId?: number;
}

interface TimeSegment {
  start: number;
  end: number;
  transactions: Transaction[];
}

interface TimingPattern {
  timeOfDay: Record<string, number>;
  dayOfWeek: Record<string, number>;
  weeklyPattern: boolean;
  monthlyPattern: boolean;
  marketConditionCorrelation: number;
}

interface PricePattern {
  buyLowSellHigh: number; // 低买高卖倾向 (0-1)
  contrarian: number; // 逆势操作倾向 (0-1)
  momentumFollow: number; // 顺势操作倾向 (0-1)
  priceThresholds: {
    buyThresholds: number[]; // 价格区间购买倾向
    sellThresholds: number[]; // 价格区间卖出倾向
  };
}

interface TokenInteraction {
  token: string;
  buys: number;
  sells: number;
  totalValueUSD: number;
  avgBuyPrice: number;
  avgSellPrice: number;
  netValue: number;
  roi: number;
  averageHoldingPeriod: number;
  lastInteraction: number;
}

/**
 * 交易模式分析服务
 * 负责分析聪明钱地址的交易行为模式
 */
export class TransactionPatternAnalyzer {
  /**
   * 分析地址的交易模式
   * @param address 区块链地址
   * @param transactions 交易历史
   */
  static async analyzeTransactionPatterns(address: string, transactions: Transaction[]): Promise<any> {
    try {
      logger.info('开始分析交易模式', { address, transactionCount: transactions.length });
      
      // 从缓存获取
      const cacheKey = `tx_patterns:${address}`;
      const cachedPatterns = cache.get<any>(cacheKey);
      
      if (cachedPatterns) {
        logger.debug('从缓存获取交易模式分析', { address });
        return cachedPatterns;
      }
      
      // 分析时间模式
      const timingPatterns = this.analyzeTimingPatterns(transactions);
      
      // 分析价格模式
      const pricePatterns = this.analyzePricePatterns(transactions);
      
      // 分析代币交互模式
      const tokenInteractions = this.analyzeTokenInteractions(transactions);
      
      // 分析交易规模模式
      const sizePatterns = this.analyzeSizePatterns(transactions);
      
      // 分析交易频率模式
      const frequencyPatterns = this.analyzeFrequencyPatterns(transactions);
      
      // 分析交易策略
      const strategies = this.identifyStrategies(
        timingPatterns,
        pricePatterns,
        sizePatterns,
        frequencyPatterns
      );
      
      // 分析行为异常
      const anomalies = this.detectAnomalies(
        transactions,
        timingPatterns,
        pricePatterns,
        sizePatterns,
        frequencyPatterns
      );
      
      // 构建结果
      const result = {
        address,
        overview: {
          transactionCount: transactions.length,
          firstTransaction: new Date(Math.min(...transactions.map(tx => tx.timestamp))),
          lastTransaction: new Date(Math.max(...transactions.map(tx => tx.timestamp))),
          totalVolume: transactions.reduce((sum, tx) => sum + (tx.valueUSD || 0), 0),
          activeChains: [...new Set(transactions.map(tx => tx.chainId))],
          tokenCount: Object.keys(tokenInteractions).length
        },
        timingPatterns,
        pricePatterns,
        tokenInteractions: Object.values(tokenInteractions).sort((a, b) => b.totalValueUSD - a.totalValueUSD).slice(0, 10),
        sizePatterns,
        frequencyPatterns,
        strategies,
        anomalies
      };
      
      // 更新缓存
      cache.set(cacheKey, result, 3600); // 缓存1小时
      
      logger.info('交易模式分析完成', { address });
      
      return result;
    } catch (error) {
      logger.error('分析交易模式失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 分析时间模式
   * @private
   */
  private static analyzeTimingPatterns(transactions: Transaction[]): TimingPattern {
    try {
      // 按时间段分组
      const timeOfDay: Record<string, number> = {
        'morning': 0, // 6-12
        'afternoon': 0, // 12-18
        'evening': 0, // 18-24
        'night': 0 // 0-6
      };
      
      const dayOfWeek: Record<string, number> = {
        'monday': 0,
        'tuesday': 0,
        'wednesday': 0,
        'thursday': 0,
        'friday': 0,
        'saturday': 0,
        'sunday': 0
      };
      
      // 统计每个时间段的交易次数
      transactions.forEach(tx => {
        const date = new Date(tx.timestamp);
        const hour = date.getHours();
        const day = date.getDay();
        
        // 按小时分组
        if (hour >= 6 && hour < 12) {
          timeOfDay.morning++;
        } else if (hour >= 12 && hour < 18) {
          timeOfDay.afternoon++;
        } else if (hour >= 18 && hour < 24) {
          timeOfDay.evening++;
        } else {
          timeOfDay.night++;
        }
        
        // 按星期分组
        switch (day) {
          case 0: dayOfWeek.sunday++; break;
          case 1: dayOfWeek.monday++; break;
          case 2: dayOfWeek.tuesday++; break;
          case 3: dayOfWeek.wednesday++; break;
          case 4: dayOfWeek.thursday++; break;
          case 5: dayOfWeek.friday++; break;
          case 6: dayOfWeek.saturday++; break;
        }
      });
      
      // 判断是否存在每周模式
      const weeklyPattern = this.hasPeriodicPattern(transactions, 7 * 24 * 60 * 60 * 1000);
      
      // 判断是否存在每月模式
      const monthlyPattern = this.hasPeriodicPattern(transactions, 30 * 24 * 60 * 60 * 1000);
      
      // 计算与市场条件的相关性
      const marketConditionCorrelation = 0.3; // 简化，实际需要与市场数据比对
      
      return {
        timeOfDay,
        dayOfWeek,
        weeklyPattern,
        monthlyPattern,
        marketConditionCorrelation
      };
    } catch (error) {
      logger.error('分析时间模式失败', { error });
      return {
        timeOfDay: {},
        dayOfWeek: {},
        weeklyPattern: false,
        monthlyPattern: false,
        marketConditionCorrelation: 0
      };
    }
  }
  
  /**
   * 分析价格模式
   * @private
   */
  private static analyzePricePatterns(transactions: Transaction[]): PricePattern {
    try {
      // 对交易按代币和类型分组
      const tokenTransactions: Record<string, Transaction[]> = {};
      
      transactions.forEach(tx => {
        const token = tx.tokenSymbol || tx.tokenAddress || 'unknown';
        if (!tokenTransactions[token]) {
          tokenTransactions[token] = [];
        }
        tokenTransactions[token].push(tx);
      });
      
      // 计算低买高卖倾向
      let buyLowSellHighCount = 0;
      let totalPairs = 0;
      
      Object.keys(tokenTransactions).forEach(token => {
        const txs = tokenTransactions[token];
        const buys = txs.filter(tx => tx.type === 'buy').sort((a, b) => a.timestamp - b.timestamp);
        const sells = txs.filter(tx => tx.type === 'sell').sort((a, b) => a.timestamp - b.timestamp);
        
        // 使用简单配对方法 (实际应该使用更复杂的配对算法)
        let buyIndex = 0;
        let sellIndex = 0;
        
        while (buyIndex < buys.length && sellIndex < sells.length) {
          const buy = buys[buyIndex];
          const sell = sells[sellIndex];
          
          // 确保卖出在买入之后
          if (sell.timestamp > buy.timestamp) {
            // 检查是否低买高卖
            if (sell.price && buy.price && sell.price > buy.price) {
              buyLowSellHighCount++;
            }
            
            totalPairs++;
            buyIndex++;
            sellIndex++;
          } else {
            sellIndex++;
          }
        }
      });
      
      const buyLowSellHigh = totalPairs > 0 ? buyLowSellHighCount / totalPairs : 0;
      
      // 计算逆势操作倾向
      const contrarian = 0.4; // 简化，实际需要与市场趋势对比
      
      // 计算顺势操作倾向
      const momentumFollow = 0.6; // 简化，实际需要与市场趋势对比
      
      // 价格阈值分析
      const buyThresholds = [0.8, 0.9, 1.0, 1.1]; // 相对价格阈值
      const sellThresholds = [0.9, 1.0, 1.1, 1.2]; // 相对价格阈值
      
      return {
        buyLowSellHigh,
        contrarian,
        momentumFollow,
        priceThresholds: {
          buyThresholds,
          sellThresholds
        }
      };
    } catch (error) {
      logger.error('分析价格模式失败', { error });
      return {
        buyLowSellHigh: 0,
        contrarian: 0,
        momentumFollow: 0,
        priceThresholds: {
          buyThresholds: [],
          sellThresholds: []
        }
      };
    }
  }
  
  /**
   * 分析代币交互模式
   * @private
   */
  private static analyzeTokenInteractions(transactions: Transaction[]): Record<string, TokenInteraction> {
    try {
      const tokenInteractions: Record<string, TokenInteraction> = {};
      
      // 对交易按代币分组
      transactions.forEach(tx => {
        const token = tx.tokenSymbol || tx.tokenAddress || 'unknown';
        
        if (!tokenInteractions[token]) {
          tokenInteractions[token] = {
            token,
            buys: 0,
            sells: 0,
            totalValueUSD: 0,
            avgBuyPrice: 0,
            avgSellPrice: 0,
            netValue: 0,
            roi: 0,
            averageHoldingPeriod: 0,
            lastInteraction: 0
          };
        }
        
        const interaction = tokenInteractions[token];
        
        // 更新统计
        if (tx.type === 'buy') {
          interaction.buys++;
          interaction.netValue -= (tx.valueUSD || 0);
          interaction.avgBuyPrice = (interaction.avgBuyPrice * (interaction.buys - 1) + (tx.price || 0)) / interaction.buys;
        } else if (tx.type === 'sell') {
          interaction.sells++;
          interaction.netValue += (tx.valueUSD || 0);
          interaction.avgSellPrice = (interaction.avgSellPrice * (interaction.sells - 1) + (tx.price || 0)) / interaction.sells;
        }
        
        interaction.totalValueUSD += (tx.valueUSD || 0);
        interaction.lastInteraction = Math.max(interaction.lastInteraction, tx.timestamp);
      });
      
      // 计算ROI和平均持有时间
      Object.values(tokenInteractions).forEach(interaction => {
        // 简化ROI计算
        const totalBuys = interaction.buys * interaction.avgBuyPrice;
        const totalSells = interaction.sells * interaction.avgSellPrice;
        
        if (totalBuys > 0) {
          interaction.roi = ((totalSells - totalBuys) / totalBuys) * 100;
        }
        
        // 计算平均持有时间 (简化)
        interaction.averageHoldingPeriod = 30; // 假设30天
      });
      
      return tokenInteractions;
    } catch (error) {
      logger.error('分析代币交互模式失败', { error });
      return {};
    }
  }
  
  /**
   * 分析交易规模模式
   * @private
   */
  private static analyzeSizePatterns(transactions: Transaction[]): any {
    try {
      // 按交易金额分组
      const sizeBuckets = {
        'small': 0, // < $1000
        'medium': 0, // $1000 - $10000
        'large': 0, // $10000 - $100000
        'whale': 0 // > $100000
      };
      
      // 计算平均和最大交易规模
      let totalValue = 0;
      let maxValue = 0;
      
      transactions.forEach(tx => {
        const value = tx.valueUSD || 0;
        totalValue += value;
        maxValue = Math.max(maxValue, value);
        
        // 按规模分类
        if (value < 1000) {
          sizeBuckets.small++;
        } else if (value < 10000) {
          sizeBuckets.medium++;
        } else if (value < 100000) {
          sizeBuckets.large++;
        } else {
          sizeBuckets.whale++;
        }
      });
      
      const avgValue = transactions.length > 0 ? totalValue / transactions.length : 0;
      
      // 分析交易规模随时间的变化
      const sizeOverTime = this.analyzeSizeOverTime(transactions);
      
      // 计算交易规模分布
      const sizeDistribution = this.calculateSizeDistribution(transactions);
      
      return {
        sizeBuckets,
        averageSize: avgValue,
        maxSize: maxValue,
        sizeOverTime,
        sizeDistribution
      };
    } catch (error) {
      logger.error('分析交易规模模式失败', { error });
      return {
        sizeBuckets: {},
        averageSize: 0,
        maxSize: 0,
        sizeOverTime: [],
        sizeDistribution: []
      };
    }
  }
  
  /**
   * 分析交易频率模式
   * @private
   */
  private static analyzeFrequencyPatterns(transactions: Transaction[]): any {
    try {
      if (transactions.length < 2) {
        return {
          averageFrequency: 0,
          frequencyOverTime: [],
          burstiness: 0,
          periodicityScore: 0
        };
      }
      
      // 按时间排序
      const sortedTxs = [...transactions].sort((a, b) => a.timestamp - b.timestamp);
      
      // 计算交易间隔
      const intervals: number[] = [];
      for (let i = 1; i < sortedTxs.length; i++) {
        intervals.push(sortedTxs[i].timestamp - sortedTxs[i-1].timestamp);
      }
      
      // 计算平均交易频率 (每天交易次数)
      const timeSpanMs = sortedTxs[sortedTxs.length - 1].timestamp - sortedTxs[0].timestamp;
      const timeSpanDays = timeSpanMs / (24 * 60 * 60 * 1000);
      const averageFrequency = timeSpanDays > 0 ? transactions.length / timeSpanDays : 0;
      
      // 计算爆发性指标
      const avgInterval = intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length;
      const variance = intervals.reduce((sum, interval) => sum + Math.pow(interval - avgInterval, 2), 0) / intervals.length;
      const stdDev = Math.sqrt(variance);
      const burstiness = avgInterval > 0 ? stdDev / avgInterval : 0;
      
      // 分析交易频率随时间的变化
      const frequencyOverTime = this.analyzeFrequencyOverTime(sortedTxs);
      
      // 周期性评分
      const periodicityScore = this.calculatePeriodicityScore(intervals);
      
      return {
        averageFrequency,
        frequencyOverTime,
        burstiness,
        periodicityScore
      };
    } catch (error) {
      logger.error('分析交易频率模式失败', { error });
      return {
        averageFrequency: 0,
        frequencyOverTime: [],
        burstiness: 0,
        periodicityScore: 0
      };
    }
  }
  
  /**
   * 识别交易策略
   * @private
   */
  private static identifyStrategies(
    timingPatterns: TimingPattern,
    pricePatterns: PricePattern,
    sizePatterns: any,
    frequencyPatterns: any
  ): string[] {
    const strategies: string[] = [];
    
    // 识别价值投资策略
    if (pricePatterns.buyLowSellHigh > 0.7 && frequencyPatterns.averageFrequency < 1) {
      strategies.push('价值投资');
    }
    
    // 识别逆势交易策略
    if (pricePatterns.contrarian > 0.6) {
      strategies.push('逆势交易');
    }
    
    // 识别动量交易策略
    if (pricePatterns.momentumFollow > 0.6 && frequencyPatterns.averageFrequency > 3) {
      strategies.push('动量交易');
    }
    
    // 识别高频交易策略
    if (frequencyPatterns.averageFrequency > 10) {
      strategies.push('高频交易');
    }
    
    // 识别定投策略
    if (frequencyPatterns.periodicityScore > 0.7) {
      strategies.push('定期投资');
    }
    
    // 识别预设订单策略 (根据时间分布)
    if (timingPatterns.timeOfDay.night > (timingPatterns.timeOfDay.morning + timingPatterns.timeOfDay.afternoon + timingPatterns.timeOfDay.evening) * 0.5) {
      strategies.push('预设订单交易');
    }
    
    // 识别鲸鱼策略
    if (sizePatterns.sizeBuckets.whale > 3 || sizePatterns.maxSize > 1000000) {
      strategies.push('大额交易');
    }
    
    // 识别分散投资策略
    if (sizePatterns.sizeDistribution && sizePatterns.sizeDistribution.length > 5) {
      strategies.push('分散投资');
    }
    
    return strategies;
  }
  
  /**
   * 检测行为异常
   * @private
   */
  private static detectAnomalies(
    transactions: Transaction[],
    timingPatterns: TimingPattern,
    pricePatterns: PricePattern,
    sizePatterns: any,
    frequencyPatterns: any
  ): any[] {
    const anomalies: any[] = [];
    
    // 检测交易规模异常
    const avgSize = sizePatterns.averageSize;
    transactions.forEach(tx => {
      if ((tx.valueUSD || 0) > avgSize * 10) {
        anomalies.push({
          type: '交易规模异常',
          description: `交易${tx.hash}的规模是平均值的10倍以上`,
          transaction: tx.hash,
          timestamp: tx.timestamp,
          severity: 'medium'
        });
      }
    });
    
    // 检测交易频率异常
    if (frequencyPatterns.burstiness > 5) {
      anomalies.push({
        type: '交易频率异常',
        description: '交易频率波动性异常高',
        severity: 'low'
      });
    }
    
    // 检测交易时间异常
    const nightTxs = transactions.filter(tx => {
      const hour = new Date(tx.timestamp).getHours();
      return hour >= 0 && hour < 6;
    });
    
    if (nightTxs.length > transactions.length * 0.5) {
      anomalies.push({
        type: '交易时间异常',
        description: '超过50%的交易发生在深夜(0-6点)',
        severity: 'low'
      });
    }
    
    // 检测价格异常交易
    // 简化处理，实际需要根据市场数据判断
    
    return anomalies;
  }
  
  /**
   * 检测是否存在周期性模式
   * @private
   */
  private static hasPeriodicPattern(transactions: Transaction[], period: number): boolean {
    if (transactions.length < 5) {
      return false;
    }
    
    // 按时间排序
    const sortedTxs = [...transactions].sort((a, b) => a.timestamp - b.timestamp);
    
    // 将交易时间戳映射到周期内的位置
    const normalizedTimestamps = sortedTxs.map(tx => tx.timestamp % period);
    
    // 计算周期内的分布
    const buckets = 10;
    const distribution: number[] = new Array(buckets).fill(0);
    
    normalizedTimestamps.forEach(ts => {
      const bucketIndex = Math.floor((ts / period) * buckets);
      distribution[bucketIndex]++;
    });
    
    // 计算分布的标准差
    const mean = transactions.length / buckets;
    const variance = distribution.reduce((sum, count) => sum + Math.pow(count - mean, 2), 0) / buckets;
    const stdDev = Math.sqrt(variance);
    
    // 如果标准差足够大，认为存在周期性
    return stdDev / mean > 0.5;
  }
  
  /**
   * 分析交易规模随时间的变化
   * @private
   */
  private static analyzeSizeOverTime(transactions: Transaction[]): any[] {
    if (transactions.length < 2) {
      return [];
    }
    
    // 按时间排序
    const sortedTxs = [...transactions].sort((a, b) => a.timestamp - b.timestamp);
    
    // 将交易分成多个时间段
    const segments = this.segmentTransactions(sortedTxs, 10);
    
    // 计算每个时间段的平均交易规模
    return segments.map(segment => {
      const avgSize = segment.transactions.reduce((sum, tx) => sum + (tx.valueUSD || 0), 0) / segment.transactions.length;
      
      return {
        startTime: new Date(segment.start),
        endTime: new Date(segment.end),
        averageSize: avgSize,
        transactionCount: segment.transactions.length
      };
    });
  }
  
  /**
   * 计算交易规模分布
   * @private
   */
  private static calculateSizeDistribution(transactions: Transaction[]): any[] {
    if (transactions.length === 0) {
      return [];
    }
    
    // 对交易按规模进行分组
    const sizes = transactions.map(tx => tx.valueUSD || 0).sort((a, b) => a - b);
    
    // 计算百分位数
    const percentiles = [0, 10, 25, 50, 75, 90, 100];
    
    return percentiles.map(p => {
      const index = Math.min(Math.floor(sizes.length * (p / 100)), sizes.length - 1);
      return {
        percentile: p,
        size: sizes[index]
      };
    });
  }
  
  /**
   * 分析交易频率随时间的变化
   * @private
   */
  private static analyzeFrequencyOverTime(transactions: Transaction[]): any[] {
    if (transactions.length < 2) {
      return [];
    }
    
    // 将交易分成多个时间段
    const segments = this.segmentTransactions(transactions, 10);
    
    // 计算每个时间段的交易频率(每天)
    return segments.map(segment => {
      const durationDays = (segment.end - segment.start) / (24 * 60 * 60 * 1000);
      const frequency = durationDays > 0 ? segment.transactions.length / durationDays : 0;
      
      return {
        startTime: new Date(segment.start),
        endTime: new Date(segment.end),
        frequency,
        transactionCount: segment.transactions.length
      };
    });
  }
  
  /**
   * 计算周期性评分
   * @private
   */
  private static calculatePeriodicityScore(intervals: number[]): number {
    if (intervals.length < 5) {
      return 0;
    }
    
    // 计算间隔的平均值和标准差
    const mean = intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length;
    const variance = intervals.reduce((sum, interval) => sum + Math.pow(interval - mean, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);
    
    // 计算变异系数
    const cv = mean > 0 ? stdDev / mean : 0;
    
    // 周期性评分 (变异系数越小，周期性越强)
    return Math.max(0, 1 - Math.min(1, cv));
  }
  
  /**
   * 将交易分成固定数量的时间段
   * @private
   */
  private static segmentTransactions(transactions: Transaction[], segmentCount: number): TimeSegment[] {
    if (transactions.length === 0) {
      return [];
    }
    
    // 获取时间范围
    const minTime = Math.min(...transactions.map(tx => tx.timestamp));
    const maxTime = Math.max(...transactions.map(tx => tx.timestamp));
    const timeRange = maxTime - minTime;
    
    if (timeRange === 0) {
      return [{
        start: minTime,
        end: maxTime,
        transactions: [...transactions]
      }];
    }
    
    // 创建时间段
    const segments: TimeSegment[] = [];
    const segmentSize = timeRange / segmentCount;
    
    for (let i = 0; i < segmentCount; i++) {
      const start = minTime + (i * segmentSize);
      const end = start + segmentSize;
      
      segments.push({
        start,
        end,
        transactions: []
      });
    }
    
    // 将交易分配到时间段
    transactions.forEach(tx => {
      const segmentIndex = Math.min(
        Math.floor(((tx.timestamp - minTime) / timeRange) * segmentCount),
        segmentCount - 1
      );
      segments[segmentIndex].transactions.push(tx);
    });
    
    return segments;
  }
  
  /**
   * 预测未来可能的交易行为
   * @param address 区块链地址
   * @param transactions 交易历史
   */
  static async predictFutureTrading(address: string, transactions: Transaction[]): Promise<any> {
    try {
      logger.info('预测未来交易行为', { address });
      
      // 分析交易模式
      const patterns = await this.analyzeTransactionPatterns(address, transactions);
      
      // 根据历史模式预测未来行为
      
      // 1. 预测可能的交易时间
      const predictedTimes = this.predictTradingTimes(patterns.timingPatterns);
      
      // 2. 预测可能的交易规模
      const predictedSizes = this.predictTradingSizes(patterns.sizePatterns);
      
      // 3. 预测可能的交易代币
      const predictedTokens = this.predictTradingTokens(patterns.tokenInteractions);
      
      // 4. 预测可能的交易价格范围
      const predictedPrices = this.predictTradingPrices(patterns.pricePatterns);
      
      return {
        address,
        predictedTimes,
        predictedSizes,
        predictedTokens,
        predictedPrices,
        confidence: 0.7, // 置信度
        timeframe: '7d' // 预测时间范围
      };
    } catch (error) {
      logger.error('预测未来交易行为失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 预测可能的交易时间
   * @private
   */
  private static predictTradingTimes(timingPatterns: TimingPattern): any {
    // 根据历史交易时间预测未来可能的交易时间
    
    // 找出最活跃的时间段
    const timeOfDay = Object.entries(timingPatterns.timeOfDay)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([time]) => time);
    
    // 找出最活跃的星期几
    const dayOfWeek = Object.entries(timingPatterns.dayOfWeek)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([day]) => day);
    
    return {
      likelyTimeOfDay: timeOfDay,
      likelyDayOfWeek: dayOfWeek,
      isPeriodicWeekly: timingPatterns.weeklyPattern,
      isPeriodicMonthly: timingPatterns.monthlyPattern
    };
  }
  
  /**
   * 预测可能的交易规模
   * @private
   */
  private static predictTradingSizes(sizePatterns: any): any {
    // 根据历史交易规模预测未来可能的交易规模
    
    // 计算最可能的交易规模范围
    const sizeBuckets = Object.entries(sizePatterns.sizeBuckets)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([bucket]) => bucket);
    
    // 根据分布计算可能的规模范围
    const likelyMin = sizePatterns.averageSize * 0.5;
    const likelyMax = sizePatterns.averageSize * 2;
    
    return {
      likelySizeBuckets: sizeBuckets,
      likelyRange: {
        min: likelyMin,
        max: likelyMax
      },
      averageSize: sizePatterns.averageSize
    };
  }
  
  /**
   * 预测可能的交易代币
   * @private
   */
  private static predictTradingTokens(tokenInteractions: any[]): any {
    // 根据历史代币交互预测未来可能交易的代币
    
    // 找出交互最频繁的代币
    const frequentTokens = [...tokenInteractions]
      .sort((a, b) => (b.buys + b.sells) - (a.buys + a.sells))
      .slice(0, 3)
      .map(t => t.token);
    
    // 找出最近交互的代币
    const recentTokens = [...tokenInteractions]
      .sort((a, b) => b.lastInteraction - a.lastInteraction)
      .slice(0, 3)
      .map(t => t.token);
    
    // 找出ROI最高的代币
    const profitableTokens = [...tokenInteractions]
      .sort((a, b) => b.roi - a.roi)
      .slice(0, 3)
      .map(t => t.token);
    
    return {
      frequentTokens,
      recentTokens,
      profitableTokens
    };
  }
  
  /**
   * 预测可能的交易价格范围
   * @private
   */
  private static predictTradingPrices(pricePatterns: PricePattern): any {
    // 根据历史价格模式预测未来可能的交易价格
    
    // 根据买入/卖出的价格阈值预测
    const likelyBuyThresholds = pricePatterns.priceThresholds.buyThresholds.slice(0, 2);
    const likelySellThresholds = pricePatterns.priceThresholds.sellThresholds.slice(0, 2);
    
    return {
      likelyBuyThresholds,
      likelySellThresholds,
      buyLowSellHighProbability: pricePatterns.buyLowSellHigh
    };
  }
} 