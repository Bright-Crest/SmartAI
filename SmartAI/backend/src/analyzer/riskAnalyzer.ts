import { RiskAnalysis, RiskLevel, NormalizedEvent } from '../types/events';
import { EnhancedRiskAnalysis, BehaviorTag } from '../types/riskAnalysis';
import { AddressProfile } from '../types/profile';
import { MLModel } from './MLModel';
import { RiskPatternAnalyzer } from './RiskPatternAnalyzer';
import { MEVDetector } from './MEVDetector';
import { GraphAnalyzer } from './GraphAnalyzer';
import { BehaviorAnalyzer } from './BehaviorAnalyzer';
import { config } from '../config/riskConfig';
import { logger } from '../utils/logger';

export const riskAnalyzer = {
  /**
   * 分析交易风险，结合多维度评分算法
   * @param event 规范化的交易事件
   * @param profile 地址画像
   * @returns 增强的风险分析结果
   */
  async analyze(event: NormalizedEvent, profile: AddressProfile): Promise<EnhancedRiskAnalysis> {
    try {
      logger.info('开始多维度风险分析', {
        traceId: event.traceId,
        transactionHash: event.transactionHash,
      });

      // 初始化风险因素和特征
      const factors: string[] = [];
      const features: Array<{ description: string; score: number; category: string }> = [];
      const behaviorTags: BehaviorTag[] = [];
      
      // 多维度评分初始化
      const dimensions: Record<string, { score: number; weight: number; tags: string[] }> = {
        flow: { score: 0, weight: config.weights.FLOW, tags: [] },
        behavior: { score: 0, weight: config.weights.BEHAVIOR, tags: [] },
        association: { score: 0, weight: config.weights.ASSOCIATION, tags: [] },
        historical: { score: 0, weight: config.weights.HISTORICAL, tags: [] },
        technical: { score: 0, weight: config.weights.TECHNICAL, tags: [] },
      };

      // 1. 基础风险评分 (来自地址画像)
      dimensions.historical.score += profile.riskScore;
      
      if (profile.riskScore > 0.5) {
        dimensions.historical.tags.push('high_risk_profile');
        behaviorTags.push({
          name: 'high_risk_profile',
          confidence: 0.8,
          category: 'historical',
          description: '地址历史风险评分较高'
        });
      }

      // 2. 分析交易金额
      const amountScore = this.analyzeTransactionAmount(event);
      if (amountScore > 0.3) {
        dimensions.flow.score += amountScore;
        dimensions.flow.tags.push('unusual_amount');
        
        features.push({
          description: '交易金额异常',
          score: amountScore,
          category: 'flow'
        });
        
        behaviorTags.push({
          name: 'unusual_amount',
          confidence: amountScore,
          category: 'flow',
          description: '交易金额明显偏离历史平均水平'
        });
      }

      // 3. 分析合约调用
      const contractScore = this.analyzeContractCall(event);
      if (contractScore > 0) {
        dimensions.technical.score += contractScore;
        dimensions.technical.tags.push('contract_interaction');
        
        features.push({
          description: '涉及合约调用',
          score: contractScore,
          category: 'technical'
        });
      }

      // 4. 检测 MEV 行为
      const mevResult = await MEVDetector.detect(event, []);
      if (mevResult.detected) {
        dimensions.behavior.score += 0.7;
        dimensions.behavior.tags.push('mev_activity');
        
        features.push({
          description: 'MEV行为检测',
          score: 0.7,
          category: 'behavior'
        });
        
        behaviorTags.push({
          name: 'mev_activity',
          confidence: mevResult.confidence,
          category: 'behavior',
          description: mevResult.description
        });
      }

      // 5. 获取行为分析结果
      const behaviorResult = await BehaviorAnalyzer.analyze(event, profile);
      behaviorResult.tags.forEach(tag => {
        behaviorTags.push(tag);
        dimensions[tag.category].tags.push(tag.name);
        dimensions[tag.category].score = Math.max(dimensions[tag.category].score, tag.confidence);
        
        features.push({
          description: tag.description,
          score: tag.confidence,
          category: tag.category
        });
      });

      // 6. 获取 AI 模型评分
      const aiScore = await MLModel.analyzeRisk(event);

      // 合并 AI 模型检测到的风险因素
      aiScore.factors.forEach((factor) => {
        if (!factors.includes(factor)) {
          factors.push(factor);
        }
      });

      // 7. 获取交易模式分析
      const patternScore = await RiskPatternAnalyzer.evaluate(event);

      // 合并交易模式分析检测到的风险因素
      patternScore.factors.forEach((factor) => {
        if (!factors.includes(factor)) {
          factors.push(factor);
        }
      });
      
      // 8. 获取图分析结果
      const graphAnalysis = await GraphAnalyzer.analyze(event.from, event.to);
      
      if (graphAnalysis.riskPaths.length > 0) {
        dimensions.association.score += 0.6;
        dimensions.association.tags.push('risk_path_detected');
        
        features.push({
          description: '检测到风险关联路径',
          score: 0.6,
          category: 'association'
        });
      }
      
      if (graphAnalysis.centrality > 0.7) {
        dimensions.association.score += 0.5;
        dimensions.association.tags.push('high_centrality');
        
        features.push({
          description: '地址中心度高',
          score: 0.5,
          category: 'association'
        });
      }

      // 9. 汇总各维度的标签
      Object.values(dimensions).forEach(dim => {
        dim.tags.forEach(tag => {
          if (!factors.includes(tag)) {
            factors.push(tag);
          }
        });
      });

      // 10. 计算多维度综合风险分数
      let totalWeight = 0;
      let weightedScore = 0;
      
      Object.entries(dimensions).forEach(([key, dim]) => {
        weightedScore += dim.score * dim.weight;
        totalWeight += dim.weight;
      });
      
      const score = totalWeight > 0 ? weightedScore / totalWeight : 0.1;

      // 11. 确定风险等级
      const level =
        score >= 0.9
          ? RiskLevel.CRITICAL
          : score >= 0.7
            ? RiskLevel.HIGH
            : score >= 0.4
              ? RiskLevel.MEDIUM
              : RiskLevel.LOW;

      // 12. 确定处理动作
      const action =
        score >= 0.9 ? 'block' : score >= 0.7 ? 'alert' : score >= 0.4 ? 'monitor' : 'none';

      // 13. 构建增强的风险分析结果
      const result: EnhancedRiskAnalysis = {
        score,
        level,
        factors,
        features,
        behaviorTags,
        dimensions: Object.entries(dimensions).map(([key, dim]) => ({
          name: key,
          score: dim.score,
          weight: dim.weight,
          tags: dim.tags
        })),
        aiAnalysis: {
          behaviorAnalysis: {
            pattern: factors.includes('high_frequency_trading')
              ? 'high_frequency'
              : factors.includes('mev_activity')
                ? 'mev'
                : 'normal',
            confidence: aiScore.confidence,
            details: behaviorResult.details,
          },
          graphAnalysis: {
            centrality: graphAnalysis.centrality,
            degree: graphAnalysis.degree,
            clustering: graphAnalysis.clustering,
            paths: graphAnalysis.riskPaths,
          },
          summary: `地址 ${event.from} 的风险评分为 ${score.toFixed(2)}，风险等级为 ${level}`,
        },
        combinations: [
          {
            factors: dimensions.historical.tags,
            score: dimensions.historical.score,
            description: `基于历史数据的风险分析`,
          },
          {
            factors: dimensions.behavior.tags,
            score: dimensions.behavior.score,
            description: `基于行为模式的风险分析`,
          },
          {
            factors: dimensions.flow.tags,
            score: dimensions.flow.score,
            description: `基于资金流动的风险分析`,
          },
          {
            factors: dimensions.association.tags,
            score: dimensions.association.score,
            description: `基于关联分析的风险评估`,
          },
          {
            factors: dimensions.technical.tags,
            score: dimensions.technical.score,
            description: `基于技术特征的风险评估`,
          },
        ],
        timestamp: Date.now(),
        action,
      };

      logger.info('多维度风险分析完成', {
        traceId: event.traceId,
        score,
        level,
        factorsCount: factors.length,
        dimensionsCount: Object.keys(dimensions).length,
      });

      return result;
    } catch (error) {
      logger.error('风险分析失败', {
        traceId: event.traceId,
        error: error instanceof Error ? error.message : String(error),
      });

      // 发生错误时返回默认低风险评分
      return {
        score: 0.2,
        level: RiskLevel.LOW,
        factors: ['analysis_failed'],
        features: [
          {
            description: '分析过程发生错误',
            score: 0.2,
            category: 'system'
          },
        ],
        behaviorTags: [],
        dimensions: [],
        aiAnalysis: {
          behaviorAnalysis: {
            pattern: 'unknown',
            confidence: 0.3,
            details: {},
          },
          graphAnalysis: {
            centrality: 0,
            degree: 0,
            clustering: 0,
            paths: [],
          },
          summary: `风险分析失败`,
        },
        combinations: [
          {
            factors: ['analysis_failed'],
            score: 0.2,
            description: `风险分析过程发生错误`,
          },
        ],
        timestamp: Date.now(),
        action: 'monitor',
      };
    }
  },

  /**
   * 分析交易金额
   * @param event 交易事件
   * @returns 金额风险评分 (0-1)
   */
  analyzeTransactionAmount(event: NormalizedEvent): number {
    if (!event.value) return 0;

    try {
      const value = BigInt(event.value);

      // 超大额交易 (>10000 ETH)
      if (value > BigInt('10000000000000000000000')) {
        return 0.9;
      }

      // 大额交易 (>1000 ETH)
      if (value > BigInt('1000000000000000000000')) {
        return 0.7;
      }

      // 中等金额 (>100 ETH)
      if (value > BigInt('100000000000000000000')) {
        return 0.5;
      }

      // 小额交易 (>10 ETH)
      if (value > BigInt('10000000000000000000')) {
        return 0.3;
      }

      // 微小交易
      return 0.1;
    } catch (error) {
      // 解析失败，返回默认分数
      return 0.2;
    }
  },

  /**
   * 分析合约调用
   * @param event 交易事件
   * @returns 合约调用风险评分 (0-1)
   */
  analyzeContractCall(event: NormalizedEvent): number {
    // 检查是否为合约调用
    if (event.type !== 'contract_call') {
      return 0;
    }

    // 检查是否有方法名
    if (!event.methodName) {
      return 0.3; // 未知方法的合约调用
    }

    // 高风险方法列表
    const highRiskMethods = [
      'transfer',
      'transferFrom',
      'approve',
      'swap',
      'exchange',
      'execute',
      'delegateCall',
      'flashLoan',
    ];

    // 检查是否为高风险方法
    const methodName = event.methodName.toLowerCase();
    if (highRiskMethods.some((method) => methodName.includes(method))) {
      return 0.6; // 高风险方法
    }

    return 0.3; // 普通合约调用
  },
};
