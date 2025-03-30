import { ScoreCalculator } from '../analyzer/ScoreCalculator';
import { 
  BehaviorTag, 
  ExtendedRiskLevel, 
  RiskLevel, 
  RiskScore, 
  TrackingConfig 
} from '../types/riskAnalysis';
import { RiskScoreResponse, DimensionMetrics, RiskFactor } from '../types/api';
import { config } from '../config/riskConfig';
import { AddressProfileDAO } from '../database/dao/AddressProfileDAO';
import { TransactionDAO } from '../database/dao/TransactionDAO';
import { EventDAO } from '../database/dao/EventDAO';
import { logger } from '../utils/logger';

/**
 * 风险评分服务
 * 提供风险评分计算和获取相关功能
 */
export class RiskScoringService {
  /**
   * 计算地址的SmartScore
   * @param address 区块链地址
   * @returns 风险评分响应
   */
  static async calculateSmartScore(address: string): Promise<RiskScoreResponse> {
    try {
      // 标准化地址格式
      const normalizedAddress = address.toLowerCase();
      
      // 查询地址档案
      const profile = await AddressProfileDAO.findByAddress(normalizedAddress);
      
      // 获取最近交易记录
      const recentTransactions = await TransactionDAO.findByAddress(normalizedAddress, 50);
      
      // 获取最近事件记录
      const recentEvents = await EventDAO.findByAddress(normalizedAddress, 20);
      
      // 收集所有行为标签
      const behaviorTags: BehaviorTag[] = [];
      
      // 如果有现有的档案和标签，则添加到行为标签中
      if (profile && profile.tags) {
        profile.tags.forEach(tag => {
          if (typeof tag === 'string') {
            // 将字符串标签转换为BehaviorTag对象
            behaviorTags.push({
              name: tag,
              confidence: 0.7,
              category: this.getCategoryForTag(tag),
              description: this.getDescriptionForTag(tag)
            });
          }
        });
      }
      
      // 分析交易和事件以生成更多标签
      // 实际项目中应该有更复杂的分析逻辑
      const generatedTags = this.generateBehaviorTags(
        normalizedAddress, 
        recentTransactions, 
        recentEvents
      );
      
      behaviorTags.push(...generatedTags);
      
      // 计算各维度分数
      const dimensionScores = ScoreCalculator.calculateAllDimensions(behaviorTags);
      
      // 计算复合风险分数
      const compositeScore = ScoreCalculator.calculateCompositeRiskScore(dimensionScores);
      
      // 标准化为0-100分数
      const normalizedScore = ScoreCalculator.normalizeScore(compositeScore);
      
      // 确定风险等级
      const riskLevel = ScoreCalculator.getRiskLevel(compositeScore);
      const extendedRiskLevel = ScoreCalculator.getExtendedRiskLevel(compositeScore);
      
      // 获取风险等级描述
      const riskDescription = ScoreCalculator.getRiskLevelDescription(extendedRiskLevel);
      
      // 准备维度分数数据
      const dimensions = {
        flow: dimensionScores[config.categories.FLOW]?.value || 0,
        behavior: dimensionScores[config.categories.BEHAVIOR]?.value || 0,
        association: dimensionScores[config.categories.ASSOCIATION]?.value || 0,
        historical: dimensionScores[config.categories.HISTORICAL]?.value || 0,
        technical: dimensionScores[config.categories.TECHNICAL]?.value || 0
      };
      
      // 合并相似的标签，减少冗余
      const mergedTags = ScoreCalculator.mergeSimilarTags(behaviorTags);
      
      // 准备风险标签数据
      const riskTags = mergedTags.map(tag => ({
        name: tag.name,
        category: tag.category,
        confidence: tag.confidence,
        description: tag.description
      }));
      
      // 准备风险因素数据 (兼容FastAPI格式)
      const riskFactors = this.generateRiskFactors(dimensions, mergedTags);
      
      // 准备需要关注的点 (兼容FastAPI格式)
      const attentionPoints = this.generateAttentionPoints(mergedTags, normalizedScore);
      
      // 更新档案中的风险分数
      if (normalizedScore > 0) {
        await AddressProfileDAO.updateRiskScore(normalizedAddress, normalizedScore / 100);
      }
      
      // 提取简化的特征用于FastAPI兼容
      const features = {
        transaction_count: recentTransactions.length,
        event_count: recentEvents.length,
        age_days: profile?.createdAt ? Math.floor((Date.now() - new Date(profile.createdAt).getTime()) / (1000 * 60 * 60 * 24)) : 0,
        tag_count: mergedTags.length,
        // 添加主要维度特征
        dimensions: {
          flow: dimensions.flow,
          behavior: dimensions.behavior,
          association: dimensions.association,
          historical: dimensions.historical,
          technical: dimensions.technical,
        }
      };
      
      // 返回标准化的响应 (兼容FastAPI格式)
      return {
        address: normalizedAddress,
        score: compositeScore,
        risk_score: normalizedScore, // FastAPI兼容字段
        risk_level: this.getRiskLevelString(riskLevel), // FastAPI兼容字段
        risk_description: riskDescription, // FastAPI兼容字段
        risk_explanation: `该地址风险评分为${normalizedScore.toFixed(2)}，风险等级为${this.getRiskLevelString(riskLevel)}。${riskDescription}`,
        risk_factors: riskFactors,
        attention_points: attentionPoints,
        dimensions,
        riskTags,
        features,
        updatedAt: Date.now()
      };
    } catch (error) {
      logger.error('计算SmartScore失败', {
        address,
        error: error instanceof Error ? error.message : String(error)
      });
      
      throw new Error(`计算风险评分失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  /**
   * 根据维度和标签生成风险因素
   * @param dimensions 维度分数
   * @param tags 行为标签
   * @returns 风险因素数组
   */
  private static generateRiskFactors(
    dimensions: Record<string, number>,
    tags: BehaviorTag[]
  ): RiskFactor[] {
    const factors: RiskFactor[] = [];
    
    // 从维度生成风险因素
    for (const [key, value] of Object.entries(dimensions)) {
      if (value > 0.4) { // 只添加较高影响的因素
        let category = '';
        let name = '';
        
        switch (key) {
          case 'flow':
            category = '资金流动';
            name = '异常资金流动模式';
            break;
          case 'behavior':
            category = '行为模式';
            name = '可疑行为模式';
            break;
          case 'association':
            category = '关联分析';
            name = '风险地址关联';
            break;
          case 'historical':
            category = '历史特征';
            name = '历史风险记录';
            break;
          case 'technical':
            category = '技术特征';
            name = '异常技术特征';
            break;
          default:
            category = '其他';
            name = '未知风险因素';
        }
        
        factors.push({
          name,
          description: `${this.getDimensionDescription(key)}异常，影响风险评分。`,
          impact: value,
          category
        });
      }
    }
    
    // 从高置信度标签生成额外的风险因素
    tags.forEach(tag => {
      if (tag.confidence > 0.6) {
        factors.push({
          name: this.getReadableTagName(tag.name),
          description: tag.description,
          impact: tag.confidence,
          category: this.getReadableCategoryName(tag.category)
        });
      }
    });
    
    // 按影响程度排序
    return factors.sort((a, b) => b.impact - a.impact);
  }
  
  /**
   * 生成需要关注的点
   * @param tags 行为标签
   * @param score 风险分数
   * @returns 关注点数组
   */
  private static generateAttentionPoints(tags: BehaviorTag[], score: number): string[] {
    const points: string[] = [];
    
    // 根据风险分数等级添加关注点
    if (score > 80) {
      points.push('该地址风险极高，建议立即停止任何与该地址的交互。');
    } else if (score > 60) {
      points.push('该地址存在明显风险，与该地址交互前请充分了解其背景。');
    } else if (score > 40) {
      points.push('该地址有一定风险，建议谨慎与该地址进行大额交易。');
    }
    
    // 基于特定标签添加关注点
    const hasMixerTag = tags.some(tag => tag.name.includes('mixer'));
    if (hasMixerTag) {
      points.push('该地址与混币服务有关联，可能涉及匿名交易，需特别关注。');
    }
    
    const hasBlacklistTag = tags.some(tag => tag.name.includes('blacklist'));
    if (hasBlacklistTag) {
      points.push('该地址与黑名单地址有直接或间接关联，建议避免交互。');
    }
    
    const hasNewAddressTag = tags.some(tag => tag.name === 'new_address');
    if (hasNewAddressTag) {
      points.push('该地址为新创建地址，历史数据有限，风险评估可能不充分。');
    }
    
    const hasHighFreqTag = tags.some(tag => tag.name === 'high_frequency_trading');
    if (hasHighFreqTag) {
      points.push('该地址存在高频交易行为，可能是自动化交易程序或机器人。');
    }
    
    return points;
  }
  
  /**
   * 获取维度描述
   * @param dimension 维度键名
   * @returns 维度描述
   */
  private static getDimensionDescription(dimension: string): string {
    switch (dimension) {
      case 'flow':
        return '资金流动模式';
      case 'behavior':
        return '交易行为模式';
      case 'association':
        return '地址关联关系';
      case 'historical':
        return '历史活动记录';
      case 'technical':
        return '技术操作特征';
      default:
        return '未知维度';
    }
  }
  
  /**
   * 获取可读的标签名称
   * @param tagName 原始标签名
   * @returns 可读标签名
   */
  private static getReadableTagName(tagName: string): string {
    const tagMap: Record<string, string> = {
      'mixer_interaction': '混币服务交互',
      'high_frequency_trading': '高频交易',
      'contract_deployer': '合约部署者',
      'blacklisted_address': '黑名单地址',
      'connected_to_blacklist': '关联黑名单',
      'new_address': '新地址',
      'contract_caller': '合约调用者',
      'unusual_large_tx': '异常大额交易',
      'suspicious_cluster': '可疑地址集群',
      'abnormal_code_patterns': '异常代码模式',
    };
    
    return tagMap[tagName] || tagName.replace(/_/g, ' ');
  }
  
  /**
   * 获取可读的类别名称
   * @param category 原始类别
   * @returns 可读类别名
   */
  private static getReadableCategoryName(category: string): string {
    const categoryMap: Record<string, string> = {
      'flow': '资金流动',
      'behavior': '行为模式',
      'association': '关联分析',
      'historical': '历史特征',
      'technical': '技术特征'
    };
    
    return categoryMap[category] || category;
  }
  
  /**
   * 获取地址的五维指标数据
   * @param address 区块链地址
   * @returns 五维指标数据数组
   */
  static async getDimensionMetrics(address: string): Promise<DimensionMetrics[]> {
    try {
      // 计算SmartScore，以获取维度分数
      const smartScore = await this.calculateSmartScore(address);
      
      // 准备结果数组
      const metrics: DimensionMetrics[] = [];
      
      // 添加各维度的指标数据
      config.scoreDimensions.forEach(dim => {
        // 根据维度键名获取维度分数
        const dimensionScore = (smartScore.dimensions as any)[dim.key] || 0;
        
        // 获取维度权重
        const weight = this.getDimensionWeight(dim.key);
        
        // 创建指标数据
        const indicators = this.generateIndicatorsForDimension(dim.key, dimensionScore, smartScore);
        
        // 添加到结果数组
        metrics.push({
          name: dim.name,
          key: dim.key,
          description: dim.description,
          score: dimensionScore * 100, // 转换为0-100分数
          weight,
          indicators
        });
      });
      
      return metrics;
    } catch (error) {
      logger.error('获取维度指标失败', {
        address,
        error: error instanceof Error ? error.message : String(error)
      });
      
      throw new Error(`获取维度指标失败: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  
  /**
   * 根据维度获取相应的指标
   * @param dimension 维度名称
   * @param score 维度分数
   * @param smartScore SmartScore数据
   * @returns 指标数组
   */
  private static generateIndicatorsForDimension(
    dimension: string,
    score: number,
    smartScore: RiskScoreResponse
  ): Array<{ name: string; value: number; description: string; isAnomaly: boolean; }> {
    // 根据不同维度生成不同的指标
    switch (dimension) {
      case config.categories.FLOW:
        return [
          {
            name: '资金流量异常度',
            value: score * 100,
            description: '资金流入流出模式的异常程度',
            isAnomaly: score > 0.7
          },
          {
            name: '大额交易比例',
            value: Math.min(score * 1.2 * 100, 100),
            description: '大额交易在总交易中的比例',
            isAnomaly: score > 0.65
          },
          {
            name: '混币器关联度',
            value: this.hasTag(smartScore.riskTags, 'mixer_interaction') ? 90 : score * 40,
            description: '与混币服务的关联程度',
            isAnomaly: this.hasTag(smartScore.riskTags, 'mixer_interaction')
          }
        ];
        
      case config.categories.BEHAVIOR:
        return [
          {
            name: '交易模式异常度',
            value: score * 100,
            description: '交易行为模式的异常程度',
            isAnomaly: score > 0.7
          },
          {
            name: '合约调用复杂度',
            value: this.hasTag(smartScore.riskTags, 'contract_deployer') ? 70 : score * 60,
            description: '智能合约调用的复杂程度',
            isAnomaly: score > 0.75
          },
          {
            name: '操作频率异常度',
            value: this.hasTag(smartScore.riskTags, 'high_frequency_trading') ? 85 : score * 50,
            description: '交易操作频率的异常程度',
            isAnomaly: this.hasTag(smartScore.riskTags, 'high_frequency_trading')
          }
        ];
        
      case config.categories.ASSOCIATION:
        return [
          {
            name: '高风险地址关联度',
            value: score * 100,
            description: '与已知高风险地址的关联程度',
            isAnomaly: score > 0.6
          },
          {
            name: '黑名单关联度',
            value: this.hasTag(smartScore.riskTags, 'connected_to_blacklist') ? 95 : score * 40,
            description: '与黑名单地址的关联程度',
            isAnomaly: this.hasTag(smartScore.riskTags, 'connected_to_blacklist')
          },
          {
            name: '风险聚类度',
            value: this.hasTag(smartScore.riskTags, 'suspicious_cluster') ? 80 : score * 70,
            description: '在风险地址聚类中的位置',
            isAnomaly: this.hasTag(smartScore.riskTags, 'suspicious_cluster')
          }
        ];
        
      case config.categories.HISTORICAL:
        return [
          {
            name: '历史活动异常度',
            value: score * 100,
            description: '历史活动的异常程度',
            isAnomaly: score > 0.65
          },
          {
            name: '账户年龄风险',
            value: this.hasTag(smartScore.riskTags, 'new_address') ? 75 : score * 30,
            description: '基于账户年龄的风险',
            isAnomaly: this.hasTag(smartScore.riskTags, 'new_address')
          },
          {
            name: '历史违规记录',
            value: this.hasTag(smartScore.riskTags, 'previous_violations') ? 95 : score * 20,
            description: '历史违规记录的严重程度',
            isAnomaly: this.hasTag(smartScore.riskTags, 'previous_violations')
          }
        ];
        
      case config.categories.TECHNICAL:
        return [
          {
            name: '合约代码风险',
            value: score * 100,
            description: '智能合约代码的风险程度',
            isAnomaly: score > 0.7
          },
          {
            name: '合约交互复杂度',
            value: this.hasTag(smartScore.riskTags, 'smart_contract_owner') ? 65 : score * 60,
            description: '与合约交互的复杂程度',
            isAnomaly: score > 0.8
          },
          {
            name: '异常代码模式',
            value: this.hasTag(smartScore.riskTags, 'abnormal_code_patterns') ? 90 : score * 40,
            description: '检测到的异常代码模式',
            isAnomaly: this.hasTag(smartScore.riskTags, 'abnormal_code_patterns')
          }
        ];
        
      default:
        return [
          {
            name: '默认指标',
            value: score * 100,
            description: '一般风险指标',
            isAnomaly: score > 0.7
          }
        ];
    }
  }
  
  /**
   * 检查标签是否存在
   * @param tags 标签数组
   * @param tagName 标签名称
   * @returns 是否存在
   */
  private static hasTag(
    tags: Array<{name: string; category: string; confidence: number; description: string;}>,
    tagName: string
  ): boolean {
    return tags.some(tag => tag.name === tagName);
  }
  
  /**
   * 生成行为标签
   * @param address 地址
   * @param transactions 交易记录
   * @param events 事件记录
   * @returns 行为标签数组
   */
  private static generateBehaviorTags(
    address: string,
    transactions: any[],
    events: any[]
  ): BehaviorTag[] {
    const tags: BehaviorTag[] = [];
    
    // 这里简化实现，实际项目中应该有更复杂的分析逻辑
    
    // 根据交易数量判断是否为高频交易
    if (transactions.length > 30) {
      tags.push({
        name: 'high_frequency_trading',
        confidence: 0.8,
        category: config.categories.BEHAVIOR,
        description: '高频交易行为'
      });
    }
    
    // 检查是否有大额交易
    const hasLargeTx = transactions.some(tx => {
      const value = parseFloat(tx.value);
      return value > 10; // 假设10个以太币为大额交易阈值
    });
    
    if (hasLargeTx) {
      tags.push({
        name: 'unusual_large_tx',
        confidence: 0.75,
        category: config.categories.FLOW,
        description: '异常大金额交易'
      });
    }
    
    // 根据地址是否为新地址
    if (transactions.length < 10) {
      tags.push({
        name: 'new_address',
        confidence: 0.6,
        category: config.categories.HISTORICAL,
        description: '新创建的地址'
      });
    }
    
    // 判断合约调用比例
    const contractCalls = transactions.filter(tx => tx.input && tx.input.length > 10);
    if (contractCalls.length > transactions.length * 0.7) {
      tags.push({
        name: 'contract_caller',
        confidence: 0.7,
        category: config.categories.TECHNICAL,
        description: '频繁调用合约'
      });
    }
    
    return tags;
  }
  
  /**
   * 获取标签的类别
   * @param tag 标签名称
   * @returns 标签类别
   */
  private static getCategoryForTag(tag: string): string {
    // 简单映射
    const tagCategories: Record<string, string> = {
      'mixer_interaction': config.categories.FLOW,
      'high_frequency_trading': config.categories.BEHAVIOR,
      'contract_deployer': config.categories.TECHNICAL,
      'blacklisted_address': config.categories.FLOW,
      'connected_to_blacklist': config.categories.ASSOCIATION,
      'new_address': config.categories.HISTORICAL
    };
    
    return tagCategories[tag] || config.categories.BEHAVIOR;
  }
  
  /**
   * 获取标签的描述
   * @param tag 标签名称
   * @returns 标签描述
   */
  private static getDescriptionForTag(tag: string): string {
    // 简单映射
    const tagDescriptions: Record<string, string> = {
      'mixer_interaction': '与混币服务进行交互',
      'high_frequency_trading': '高频交易行为',
      'contract_deployer': '智能合约部署者',
      'blacklisted_address': '地址在黑名单中',
      'connected_to_blacklist': '与黑名单地址有关联',
      'new_address': '新创建的地址'
    };
    
    return tagDescriptions[tag] || '未知标签';
  }
  
  /**
   * 获取维度权重
   * @param dimension 维度名称
   * @returns 权重值
   */
  private static getDimensionWeight(dimension: string): number {
    switch (dimension) {
      case config.categories.FLOW:
        return config.weights.FLOW;
      case config.categories.BEHAVIOR:
        return config.weights.BEHAVIOR;
      case config.categories.ASSOCIATION:
        return config.weights.ASSOCIATION;
      case config.categories.HISTORICAL:
        return config.weights.HISTORICAL;
      case config.categories.TECHNICAL:
        return config.weights.TECHNICAL;
      default:
        return 0.1;
    }
  }
  
  /**
   * 获取风险等级字符串表示
   * @param level 风险等级枚举
   * @returns 风险等级字符串
   */
  private static getRiskLevelString(level: RiskLevel): string {
    switch (level) {
      case RiskLevel.CRITICAL:
        return 'critical';
      case RiskLevel.HIGH:
        return 'high';
      case RiskLevel.MEDIUM:
        return 'medium';
      case RiskLevel.LOW:
        return 'low';
      default:
        return 'unknown';
    }
  }
} 