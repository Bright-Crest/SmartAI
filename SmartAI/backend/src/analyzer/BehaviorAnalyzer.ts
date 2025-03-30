import { NormalizedEvent } from '../types/events';
import { AddressProfile } from '../types/profile';
import { BehaviorTag } from '../types/riskAnalysis';
import { EventDAO } from '../database/dao/EventDAO';
import { logger } from '../utils/logger';
import { config } from '../config/riskConfig';

/**
 * 行为分析器
 * 负责识别和分析地址的行为特征与模式
 */
export class BehaviorAnalyzer {
  /**
   * 分析地址行为并提取标签
   * @param event 当前交易事件
   * @param profile 地址画像
   * @returns 行为分析结果，包含标签和详细信息
   */
  static async analyze(
    event: NormalizedEvent,
    profile: AddressProfile
  ): Promise<{ tags: BehaviorTag[]; details: Record<string, any> }> {
    try {
      logger.info('开始行为特征分析', {
        traceId: event.traceId,
        address: event.from,
      });

      const tags: BehaviorTag[] = [];
      const details: Record<string, any> = {};
      
      // 获取地址历史交易
      const recentEvents = await this.getAddressHistory(event.from);
      
      // 分析基本行为特征
      await this.analyzeBasicFeatures(event, profile, recentEvents, tags, details);
      
      // 分析交易模式
      await this.analyzeTransactionPatterns(event, recentEvents, tags, details);
      
      // 分析交互行为
      await this.analyzeInteractionBehavior(event, recentEvents, tags, details);
      
      // 分析可疑行为标记
      await this.analyzeSuspiciousBehavior(event, profile, tags, details);

      logger.info('行为特征分析完成', {
        traceId: event.traceId,
        address: event.from,
        tagsCount: tags.length,
      });

      return { tags, details };
    } catch (error) {
      logger.error('行为特征分析失败', {
        traceId: event.traceId,
        error: error instanceof Error ? error.message : String(error),
      });
      
      return { 
        tags: [{
          name: 'analysis_error',
          confidence: 1.0,
          category: 'system',
          description: '行为分析过程发生错误'
        }], 
        details: { error: true } 
      };
    }
  }
  
  /**
   * 获取地址历史交易记录
   * @param address 地址
   * @returns 历史交易记录
   */
  private static async getAddressHistory(address: string): Promise<NormalizedEvent[]> {
    try {
      // 获取最近30天的交易记录，最多100条
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      
      const events = await EventDAO.findByAddressAndTimeRange(
        address, 
        Math.floor(thirtyDaysAgo.getTime() / 1000),
        Math.floor(Date.now() / 1000),
        100
      );
      
      return events.map(record => record.event);
    } catch (error) {
      logger.warn('获取地址历史交易失败', {
        address,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }
  
  /**
   * 分析基本行为特征
   * @param event 当前事件
   * @param profile 地址画像
   * @param history 历史交易
   * @param tags 标签数组
   * @param details 详细信息
   */
  private static async analyzeBasicFeatures(
    event: NormalizedEvent,
    profile: AddressProfile,
    history: NormalizedEvent[],
    tags: BehaviorTag[],
    details: Record<string, any>
  ): Promise<void> {
    // 分析账户年龄
    const accountAgeDays = profile.firstSeen 
      ? Math.floor((Date.now() - new Date(profile.firstSeen).getTime()) / (24 * 60 * 60 * 1000))
      : 0;
    
    details.accountAgeDays = accountAgeDays;
    
    // 新账户标记 (少于7天)
    if (accountAgeDays < 7) {
      tags.push({
        name: 'new_account',
        confidence: 0.9,
        category: 'historical',
        description: '新创建账户（不足7天）'
      });
    }
    
    // 分析交易频率
    if (history.length > 0) {
      const txPerDay = history.length / Math.max(1, accountAgeDays);
      details.txPerDay = txPerDay;
      
      // 高频交易账户 (平均每天超过10笔交易)
      if (txPerDay > 10) {
        tags.push({
          name: 'high_frequency_account',
          confidence: 0.8,
          category: 'behavior',
          description: '高频交易账户'
        });
      }
    }
    
    // 检查是否为休眠账户激活
    if (accountAgeDays > 30 && history.length < 5) {
      const lastTxTime = history.length > 0 
        ? Math.max(...history.map(e => e.timestamp))
        : 0;
        
      const dormantDays = lastTxTime 
        ? Math.floor((Date.now() / 1000 - lastTxTime) / (24 * 60 * 60))
        : 0;
      
      if (dormantDays > 30) {
        tags.push({
          name: 'dormant_activated',
          confidence: 0.7,
          category: 'historical',
          description: '长期休眠账户被激活'
        });
      }
    }
  }
  
  /**
   * 分析交易模式
   * @param event 当前事件
   * @param history 历史交易
   * @param tags 标签数组
   * @param details 详细信息
   */
  private static async analyzeTransactionPatterns(
    event: NormalizedEvent,
    history: NormalizedEvent[],
    tags: BehaviorTag[],
    details: Record<string, any>
  ): Promise<void> {
    if (history.length < 3) return;
    
    // 分析交易金额模式
    const amounts = history
      .filter(e => e.value)
      .map(e => BigInt(e.value || '0'));
      
    if (amounts.length > 3) {
      // 计算平均值和标准差
      const avg = this.calculateAverage(amounts);
      const stdDev = this.calculateStdDev(amounts, avg);
      
      details.valueStats = {
        average: avg.toString(),
        stdDev: stdDev.toString()
      };
      
      // 检查当前交易是否异常
      if (event.value) {
        const currentValue = BigInt(event.value);
        const zScore = Number((currentValue - avg)) / Number(stdDev);
        
        if (zScore > 3) {
          tags.push({
            name: 'unusual_large_tx',
            confidence: Math.min(0.9, 0.5 + zScore / 10),
            category: 'flow',
            description: '异常大额交易'
          });
        }
      }
    }
    
    // 检测分散转账模式 (从一个地址短时间内向多个地址转账)
    const recentTxs = history.filter(e => 
      e.timestamp > (Date.now() / 1000) - 24 * 60 * 60);
      
    if (recentTxs.length >= config.thresholds.dispersalThreshold) {
      const uniqueRecipients = new Set(recentTxs.map(e => e.to));
      
      if (uniqueRecipients.size >= config.thresholds.dispersalThreshold) {
        tags.push({
          name: 'fund_dispersal',
          confidence: 0.7,
          category: 'flow',
          description: '资金分散转出'
        });
      }
    }
    
    // 检测周期性模式
    if (history.length >= 10) {
      const isPeriodicPattern = this.detectPeriodicPattern(history);
      
      if (isPeriodicPattern) {
        tags.push({
          name: 'periodic_activity',
          confidence: 0.6,
          category: 'behavior',
          description: '周期性交易活动'
        });
      }
    }
  }
  
  /**
   * 分析交互行为
   * @param event 当前事件
   * @param history 历史交易
   * @param tags 标签数组
   * @param details 详细信息
   */
  private static async analyzeInteractionBehavior(
    event: NormalizedEvent,
    history: NormalizedEvent[],
    tags: BehaviorTag[],
    details: Record<string, any>
  ): Promise<void> {
    // 检查是否频繁与特定合约交互
    const contractInteractions = history.filter(e => 
      e.type === 'contract_call' && e.to);
      
    if (contractInteractions.length > 0) {
      const contractCounts: Record<string, number> = {};
      
      contractInteractions.forEach(e => {
        if (e.to) {
          contractCounts[e.to] = (contractCounts[e.to] || 0) + 1;
        }
      });
      
      // 寻找高频交互合约
      const frequentContracts = Object.entries(contractCounts)
        .filter(([_, count]) => count >= 5)
        .map(([address]) => address);
        
      if (frequentContracts.length > 0) {
        details.frequentContracts = frequentContracts;
        
        // 当前交易是否与高频合约交互
        if (event.to && frequentContracts.includes(event.to)) {
          tags.push({
            name: 'frequent_contract_user',
            confidence: 0.6,
            category: 'behavior',
            description: '频繁合约用户'
          });
        }
      }
      
      // 检查交互方法
      if (event.type === 'contract_call' && event.methodName) {
        // DEX交互检测
        if (this.isDexInteraction(event.methodName)) {
          tags.push({
            name: 'dex_user',
            confidence: 0.8,
            category: 'behavior',
            description: '去中心化交易所用户'
          });
        }
        
        // DeFi协议交互检测
        if (this.isDefiInteraction(event.methodName)) {
          tags.push({
            name: 'defi_user',
            confidence: 0.8,
            category: 'behavior',
            description: 'DeFi协议用户'
          });
        }
      }
    }
  }
  
  /**
   * 分析可疑行为
   * @param event 当前事件
   * @param profile 地址画像
   * @param tags 标签数组
   * @param details 详细信息
   */
  private static async analyzeSuspiciousBehavior(
    event: NormalizedEvent,
    profile: AddressProfile,
    tags: BehaviorTag[],
    details: Record<string, any>
  ): Promise<void> {
    // 检查是否与已知的混币器交互
    if (event.to && this.isMixerAddress(event.to)) {
      tags.push({
        name: 'mixer_interaction',
        confidence: 0.9,
        category: 'flow',
        description: '与混币器交互'
      });
    }
    
    // 检查是否使用隐私增强型转账
    if (event.type === 'contract_call' && 
        event.methodName && 
        this.isPrivacyEnhancedTransfer(event.methodName)) {
      tags.push({
        name: 'privacy_tx',
        confidence: 0.7,
        category: 'behavior',
        description: '使用隐私增强型转账'
      });
    }
    
    // 检查地址标签中的危险指标
    if (profile.tags && profile.tags.length > 0) {
      // 检查钓鱼相关标签
      if (profile.tags.some(tag => tag.includes('phish'))) {
        tags.push({
          name: 'phishing_related',
          confidence: 0.9,
          category: 'association',
          description: '关联钓鱼活动'
        });
      }
      
      // 检查欺诈相关标签
      if (profile.tags.some(tag => tag.includes('scam') || tag.includes('fraud'))) {
        tags.push({
          name: 'fraud_related',
          confidence: 0.9,
          category: 'association',
          description: '关联欺诈活动'
        });
      }
    }
  }
  
  /**
   * 计算BigInt数组平均值
   */
  private static calculateAverage(values: bigint[]): bigint {
    if (values.length === 0) return BigInt(0);
    
    const sum = values.reduce((acc, val) => acc + val, BigInt(0));
    return sum / BigInt(values.length);
  }
  
  /**
   * 计算BigInt数组标准差
   */
  private static calculateStdDev(values: bigint[], avg: bigint): bigint {
    if (values.length <= 1) return BigInt(0);
    
    const squaredDiffs = values.map(val => {
      const diff = val > avg ? val - avg : avg - val;
      return diff * diff;
    });
    
    const sumSquaredDiffs = squaredDiffs.reduce((acc, val) => acc + val, BigInt(0));
    return BigInt(Math.sqrt(Number(sumSquaredDiffs / BigInt(values.length))));
  }
  
  /**
   * 检测周期性交易模式
   */
  private static detectPeriodicPattern(events: NormalizedEvent[]): boolean {
    // 简单实现：检查时间间隔的一致性
    const timestamps = events.map(e => e.timestamp).sort((a, b) => a - b);
    const intervals: number[] = [];
    
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i - 1]);
    }
    
    // 计算间隔的平均值和标准差
    const avg = intervals.reduce((sum, val) => sum + val, 0) / intervals.length;
    const variance = intervals.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);
    
    // 计算变异系数 (CV)，低于0.5认为是周期性的
    const cv = stdDev / avg;
    
    return cv < 0.5;
  }
  
  /**
   * 检查是否为DEX交互
   */
  private static isDexInteraction(methodName: string): boolean {
    const dexMethods = [
      'swap', 'swapExactTokens', 'swapTokens', 'addLiquidity', 
      'removeLiquidity', 'trade', 'exchange'
    ];
    
    return dexMethods.some(method => methodName.toLowerCase().includes(method.toLowerCase()));
  }
  
  /**
   * 检查是否为DeFi交互
   */
  private static isDefiInteraction(methodName: string): boolean {
    const defiMethods = [
      'borrow', 'repay', 'deposit', 'withdraw', 'stake', 'unstake',
      'claim', 'farm', 'harvest', 'yield', 'supply', 'redeem'
    ];
    
    return defiMethods.some(method => methodName.toLowerCase().includes(method.toLowerCase()));
  }
  
  /**
   * 检查是否为混币器地址
   */
  private static isMixerAddress(address: string): boolean {
    // 这里应该查询数据库或配置文件中的混币器地址列表
    const knownMixers = config.mixerAddresses || [];
    return knownMixers.some(mixer => mixer.toLowerCase() === address.toLowerCase());
  }
  
  /**
   * 检查是否为隐私增强型转账
   */
  private static isPrivacyEnhancedTransfer(methodName: string): boolean {
    const privacyMethods = [
      'tornado', 'mix', 'shield', 'private', 'anonymize', 
      'zether', 'aztec', 'nightfall'
    ];
    
    return privacyMethods.some(method => methodName.toLowerCase().includes(method.toLowerCase()));
  }
} 