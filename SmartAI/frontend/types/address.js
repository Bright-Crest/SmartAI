/**
 * 地址相关类型定义文件
 */

/**
 * 地址类型枚举
 * @typedef {'eoa' | 'contract' | 'exchange' | 'mixer' | 'unknown'} AddressType
 */

/**
 * 风险等级
 * @typedef {'high' | 'medium' | 'low' | 'critical'} RiskLevel
 */

/**
 * 地址实体
 * @typedef {Object} Address
 * @property {string} address - 区块链地址
 * @property {AddressType} type - 地址类型
 * @property {RiskLevel} risk - 风险级别
 * @property {number} riskScore - 风险分数(0-100)
 * @property {string[]} [riskFactors] - 风险因素
 * @property {number} balance - 余额(ETH)
 * @property {Object} [tokens] - 代币余额
 * @property {number} transactionCount - 交易数量
 * @property {string} [label] - 地址标签
 * @property {string[]} [tags] - 相关标签
 * @property {Date} firstActivityDate - 首次活动时间
 * @property {Date} lastActivityDate - 最后活动时间
 * @property {Object} [metadata] - 元数据
 */

/**
 * 地址详情
 * @typedef {Object} AddressDetails
 * @property {string} address - 区块链地址
 * @property {AddressType} type - 地址类型
 * @property {RiskLevel} risk - 风险级别
 * @property {number} riskScore - 风险分数(0-100)
 * @property {string[]} riskFactors - 风险因素
 * @property {number} balance - 余额(ETH)
 * @property {Object} tokens - 代币余额
 * @property {number} transactionCount - 交易数量
 * @property {string} [label] - 地址标签
 * @property {string[]} tags - 相关标签
 * @property {Date} firstActivity - 首次活动时间
 * @property {Date} lastActivity - 最后活动时间
 * @property {Transaction[]} [recentTransactions] - 最近交易
 * @property {AddressActivity} activity - 活动统计
 * @property {Object} relatedAddresses - 相关地址
 * @property {Object} metadata - 元数据
 */

/**
 * 地址活动统计
 * @typedef {Object} AddressActivity
 * @property {Object} transactionsByTime - 按时间段统计的交易数量
 * @property {Object} transactionsByType - 按类型统计的交易数量
 * @property {Object} valueByTime - 按时间段统计的交易金额
 * @property {Object} riskDistribution - 风险交易分布
 */

/**
 * 地址搜索结果
 * @typedef {Object} AddressSearchResult
 * @property {string} address - 区块链地址
 * @property {AddressType} type - 地址类型
 * @property {RiskLevel} risk - 风险级别
 * @property {string} [label] - 地址标签
 * @property {number} balance - 余额(ETH)
 * @property {number} transactionCount - 交易数量
 * @property {Date} lastActivity - 最后活动时间
 */

/**
 * 高风险地址
 * @typedef {Object} HighRiskAddress
 * @property {string} address - 区块链地址
 * @property {AddressType} type - 地址类型
 * @property {number} riskScore - 风险分数(0-100)
 * @property {string[]} riskFactors - 风险因素
 * @property {string} [label] - 地址标签
 * @property {Date} lastActivity - 最后活动时间
 */

/**
 * 地址关系
 * @typedef {Object} AddressRelation
 * @property {string} address - 关联地址
 * @property {string} relationshipType - 关系类型(交易/合约调用等)
 * @property {number} strength - 关系强度(0-100)
 * @property {number} transactionCount - 交易数量
 * @property {number} totalValue - 总交易价值
 * @property {Date} firstInteractionDate - 首次交互时间
 * @property {Date} lastInteractionDate - 最后交互时间
 */

// 用于与TypeScript兼容的空导出
export default {};
