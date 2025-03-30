/**
 * SmartScore计算工具
 * 提供智能评分计算逻辑
 */

/**
 * 权重配置（各指标在总分中的权重）
 */
const WEIGHTS = {
  activity: 0.25, // 活跃度
  protocol: 0.2, // 协议互动
  profitability: 0.15, // 收益能力
  fundFlow: 0.2, // 资金流向
  distillation: 0.2, // 蒸馏模型
};

/**
 * 根据五维数据计算SmartScore
 *
 * @param {Object} radarData - 五维雷达图数据
 * @param {number} radarData.activityScore - 活跃度评分(0-100)
 * @param {number} radarData.protocolInteractionScore - 协议互动评分(0-100)
 * @param {number} radarData.profitabilityScore - 收益能力评分(0-100)
 * @param {number} radarData.fundFlowScore - 资金流评分(0-100)
 * @param {number} radarData.distillationScore - 蒸馏模型评分(0-100)
 * @returns {number} 最终计算的SmartScore(0-100)
 */
export function calculateSmartScore(radarData) {
  if (!radarData) return 0;

  // 计算加权分数
  const weightedScore =
    (radarData.activityScore || 0) * WEIGHTS.activity +
    (radarData.protocolInteractionScore || 0) * WEIGHTS.protocol +
    (radarData.profitabilityScore || 0) * WEIGHTS.profitability +
    (radarData.fundFlowScore || 0) * WEIGHTS.fundFlow +
    (radarData.distillationScore || 0) * WEIGHTS.distillation;

  // 四舍五入到整数
  return Math.round(weightedScore);
}

/**
 * 根据SmartScore确定行为类型
 *
 * @param {number} score - SmartScore分数
 * @param {Object} radarData - 五维雷达图数据
 * @returns {string} 行为类型
 */
export function determineBehaviorType(score, radarData) {
  // 各维度最高分
  const highestDimension = Object.entries({
    activityScore: radarData?.activityScore || 0,
    protocolInteractionScore: radarData?.protocolInteractionScore || 0,
    profitabilityScore: radarData?.profitabilityScore || 0,
    fundFlowScore: radarData?.fundFlowScore || 0,
    distillationScore: radarData?.distillationScore || 0,
  }).reduce(
    (highest, [key, value]) => {
      if (value > highest.value) {
        return { key, value };
      }
      return highest;
    },
    { key: '', value: 0 }
  );

  // 基于最高分维度确定行为类型
  switch (highestDimension.key) {
    case 'activityScore':
      return score >= 75 ? '活跃交易者' : '普通交易者';
    case 'protocolInteractionScore':
      return score >= 75 ? 'DeFi专家' : 'DeFi用户';
    case 'profitabilityScore':
      return score >= 75 ? '套利达人' : '价值投资者';
    case 'fundFlowScore':
      return score >= 80 ? '鲸鱼' : '资金管理者';
    case 'distillationScore':
      return score >= 75 ? '开发者' : '高级用户';
    default:
      return score >= 70 ? '价值持有者' : '普通用户';
  }
}

/**
 * 生成行为标签
 *
 * @param {Object} radarData - 五维雷达图数据
 * @param {number} score - SmartScore分数
 * @returns {string[]} 行为标签数组
 */
export function generateBehaviorTags(radarData, score) {
  const tags = [];

  // 基于五维评分生成标签
  if (radarData?.activityScore >= 70) tags.push('频繁交易');
  if (radarData?.activityScore <= 30) tags.push('长期持有');
  if (radarData?.protocolInteractionScore >= 70) tags.push('多协议用户');
  if (radarData?.profitabilityScore >= 80) tags.push('高收益者');
  if (radarData?.fundFlowScore >= 75) tags.push('大额交易');
  if (radarData?.distillationScore >= 75) tags.push('智能合约部署者');

  // 基于综合评分添加标签
  if (score >= 85) tags.push('专业用户');
  if (score >= 90) tags.push('链上专家');
  if (score <= 40) tags.push('新手用户');

  return tags;
}

/**
 * 综合计算地址智能分析结果
 *
 * @param {Object} radarData - 五维雷达图数据
 * @returns {Object} 分析结果对象，包含分数、行为类型和标签
 */
export function analyzeAddress(radarData) {
  const score = calculateSmartScore(radarData);
  const behaviorType = determineBehaviorType(score, radarData);
  const tags = generateBehaviorTags(radarData, score);

  return {
    score,
    behaviorType,
    tags,
  };
}
