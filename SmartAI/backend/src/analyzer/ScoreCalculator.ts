import { BehaviorTag, ExtendedRiskLevel, RiskLevel } from '../types/riskAnalysis';
import { config } from '../config/riskConfig';
import { logger } from '../utils/logger';

/**
 * 评分因子接口
 */
interface ScoreFactor {
  name: string;
  value: number;
  weight: number;
  category: string;
  description: string;
}

/**
 * 复合评分接口
 */
interface CompositeScore {
  value: number;
  factors: ScoreFactor[];
  confidence: number;
}

/**
 * 评分计算器
 * 实现复杂的评分计算逻辑
 */
export class ScoreCalculator {
  /**
   * 使用行为标签计算维度分数
   * @param behaviorTags 行为标签数组
   * @param category 维度类别
   * @returns 维度评分
   */
  static calculateDimensionScore(
    behaviorTags: BehaviorTag[],
    category: string
  ): CompositeScore {
    // 过滤出指定类别的标签
    const categoryTags = behaviorTags.filter(tag => tag.category === category);
    
    if (categoryTags.length === 0) {
      return { value: 0, factors: [], confidence: 0 };
    }
    
    // 转换为评分因子
    const factors: ScoreFactor[] = categoryTags.map(tag => {
      // 从配置中获取标签权重，如果没有则使用默认值
      const tagWeight = config.tagRiskWeights[tag.name as keyof typeof config.tagRiskWeights] || tag.confidence;
      
      return {
        name: tag.name,
        value: tag.confidence,
        weight: tagWeight,
        category,
        description: tag.description
      };
    });
    
    // 计算评分和置信度
    const { score, confidence } = this.calculateWeightedScore(factors);
    
    return {
      value: score,
      factors,
      confidence
    };
  }
  
  /**
   * 根据行为标签计算所有维度的分数
   * @param behaviorTags 行为标签数组
   * @returns 所有维度的评分
   */
  static calculateAllDimensions(
    behaviorTags: BehaviorTag[]
  ): Record<string, CompositeScore> {
    const dimensions: Record<string, CompositeScore> = {};
    
    // 获取所有维度
    const categories = Object.values(config.categories);
    
    // 计算每个维度的分数
    categories.forEach(category => {
      dimensions[category] = this.calculateDimensionScore(behaviorTags, category);
    });
    
    return dimensions;
  }
  
  /**
   * 使用加权平均法计算综合评分
   * @param factors 评分因子数组
   * @returns 计算结果，包含分数和置信度
   */
  private static calculateWeightedScore(
    factors: ScoreFactor[]
  ): { score: number; confidence: number } {
    if (factors.length === 0) {
      return { score: 0, confidence: 0 };
    }
    
    let weightedSum = 0;
    let weightSum = 0;
    
    // 计算加权平均分
    factors.forEach(factor => {
      weightedSum += factor.value * factor.weight;
      weightSum += factor.weight;
    });
    
    // 计算最终分数
    const score = weightSum > 0 ? weightedSum / weightSum : 0;
    
    // 计算置信度 (基于样本数量和因子权重)
    const confidence = Math.min(
      0.95, 
      0.5 + (factors.length / 10) * 0.5 + (score / 2) * 0.3
    );
    
    return { score, confidence };
  }
  
  /**
   * 计算集合风险评分
   * 使用非线性组合方法，会给予高风险因素更高权重
   * @param dimensionScores 维度评分映射
   * @returns 集合风险评分
   */
  static calculateCompositeRiskScore(
    dimensionScores: Record<string, CompositeScore>
  ): number {
    // 获取各维度分数
    const dimensions = Object.entries(dimensionScores);
    if (dimensions.length === 0) return 0;
    
    // 非线性组合公式参数
    const baseScore = 0.1; // 基础分数
    const powerFactor = 1.5; // 非线性幂次因子
    
    let totalWeight = 0;
    let weightedSum = 0;
    
    // 对每个维度应用非线性变换后加权
    dimensions.forEach(([category, score]) => {
      const dimensionWeight = this.getDimensionWeight(category);
      
      // 应用非线性变换，突出高风险维度
      const nonlinearScore = Math.pow(score.value, powerFactor);
      
      weightedSum += nonlinearScore * dimensionWeight;
      totalWeight += dimensionWeight;
    });
    
    // 归一化到 0-1 范围并结合基础分数
    let finalScore = totalWeight > 0 ? (weightedSum / totalWeight) : 0;
    
    // 防止过低评分，添加基础分数
    finalScore = baseScore + (1 - baseScore) * finalScore;
    
    // 应用饱和函数，确保评分在 0-1 范围内并有适当的分布
    finalScore = this.applySaturationFunction(finalScore);
    
    // 最终限制在 0-1 范围内
    return Math.max(0, Math.min(1, finalScore));
  }
  
  /**
   * 获取维度的权重
   * @param category 维度类别
   * @returns 权重值
   */
  private static getDimensionWeight(category: string): number {
    switch (category) {
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
   * 应用饱和函数对评分进行调整
   * 使用 S 形曲线让中间分数更加分散，两端更加聚集
   * @param score 原始评分 (0-1)
   * @returns 调整后的评分 (0-1)
   */
  private static applySaturationFunction(score: number): number {
    // S形函数参数
    const k = 6; // 斜率参数，值越大中间变化越陡峭
    const midpoint = 0.5; // 中点
    
    // 应用S形函数: 1 / (1 + e^(-k * (x - midpoint)))
    return 1 / (1 + Math.exp(-k * (score - midpoint)));
  }
  
  /**
   * 将评分标准化到0-100范围，便于展示
   * @param score 原始评分 (0-1)
   * @returns 标准化评分 (0-100)
   */
  static normalizeScore(score: number): number {
    return Math.round(score * 100);
  }
  
  /**
   * 根据评分获取风险等级
   * @param score 风险评分 (0-1)
   * @returns 风险等级
   */
  static getRiskLevel(score: number): RiskLevel {
    const normalizedScore = this.normalizeScore(score);
    
    if (normalizedScore >= 90) {
      return RiskLevel.CRITICAL;
    } else if (normalizedScore >= 75) {
      return RiskLevel.HIGH;
    } else if (normalizedScore >= 50) {
      return RiskLevel.MEDIUM;
    } else {
      return RiskLevel.LOW;
    }
  }
  
  /**
   * 根据评分获取扩展风险等级
   * @param score 风险评分 (0-1)
   * @returns 扩展风险等级
   */
  static getExtendedRiskLevel(score: number): ExtendedRiskLevel {
    const normalizedScore = this.normalizeScore(score);
    
    if (normalizedScore >= 95) {
      return ExtendedRiskLevel.CRITICAL;
    } else if (normalizedScore >= 85) {
      return ExtendedRiskLevel.VERY_HIGH;
    } else if (normalizedScore >= 75) {
      return ExtendedRiskLevel.HIGH;
    } else if (normalizedScore >= 65) {
      return ExtendedRiskLevel.MEDIUM_HIGH;
    } else if (normalizedScore >= 50) {
      return ExtendedRiskLevel.MEDIUM;
    } else if (normalizedScore >= 35) {
      return ExtendedRiskLevel.MEDIUM_LOW;
    } else if (normalizedScore >= 20) {
      return ExtendedRiskLevel.LOW;
    } else if (normalizedScore >= 5) {
      return ExtendedRiskLevel.VERY_LOW;
    } else {
      return ExtendedRiskLevel.UNKNOWN;
    }
  }
  
  /**
   * 获取风险等级描述
   * @param level 风险等级
   * @returns 风险等级描述
   */
  static getRiskLevelDescription(level: RiskLevel | ExtendedRiskLevel): string {
    switch (level) {
      case RiskLevel.CRITICAL:
      case ExtendedRiskLevel.CRITICAL:
        return '极高风险';
      case ExtendedRiskLevel.VERY_HIGH:
        return '非常高风险';
      case RiskLevel.HIGH:
      case ExtendedRiskLevel.HIGH:
        return '高风险';
      case ExtendedRiskLevel.MEDIUM_HIGH:
        return '中高风险';
      case RiskLevel.MEDIUM:
      case ExtendedRiskLevel.MEDIUM:
        return '中等风险';
      case ExtendedRiskLevel.MEDIUM_LOW:
        return '中低风险';
      case RiskLevel.LOW:
      case ExtendedRiskLevel.LOW:
        return '低风险';
      case ExtendedRiskLevel.VERY_LOW:
        return '非常低风险';
      case ExtendedRiskLevel.UNKNOWN:
      default:
        return '未知风险';
    }
  }
  
  /**
   * 组合多个标签并合并相同类型的标签
   * @param tags 标签数组
   * @returns 合并后的标签数组
   */
  static mergeSimilarTags(tags: BehaviorTag[]): BehaviorTag[] {
    // 如果标签太少，不需要合并
    if (tags.length <= 3) return tags;
    
    const mergedTags: BehaviorTag[] = [];
    const tagsByCategory: Record<string, BehaviorTag[]> = {};
    
    // 按类别分组
    tags.forEach(tag => {
      if (!tagsByCategory[tag.category]) {
        tagsByCategory[tag.category] = [];
      }
      tagsByCategory[tag.category].push(tag);
    });
    
    // 处理每个类别
    Object.entries(tagsByCategory).forEach(([category, categoryTags]) => {
      // 如果类别标签较少，保留所有标签
      if (categoryTags.length <= 2) {
        mergedTags.push(...categoryTags);
        return;
      }
      
      // 按置信度排序
      const sortedTags = [...categoryTags].sort((a, b) => b.confidence - a.confidence);
      
      // 保留置信度最高的标签
      mergedTags.push(sortedTags[0]);
      
      // 如果有多个相似标签，合并为一个总结性标签
      if (sortedTags.length > 1) {
        const remainingCount = sortedTags.length - 1;
        
        if (remainingCount > 0) {
          // 创建一个总结标签
          mergedTags.push({
            name: `${category}_summary`,
            confidence: sortedTags[1].confidence,
            category,
            description: `还有${remainingCount}个相关${this.getCategoryName(category)}标签`
          });
        }
      }
    });
    
    return mergedTags;
  }
  
  /**
   * 获取类别的中文名称
   * @param category 类别代码
   * @returns 类别名称
   */
  private static getCategoryName(category: string): string {
    switch (category) {
      case config.categories.FLOW:
        return "资金流动";
      case config.categories.BEHAVIOR:
        return "行为模式";
      case config.categories.ASSOCIATION:
        return "关联分析";
      case config.categories.HISTORICAL:
        return "历史特征";
      case config.categories.TECHNICAL:
        return "技术特征";
      case config.categories.SYSTEM:
        return "系统";
      default:
        return "其他";
    }
  }
} 