/**
 * 交易相关类型定义文件
 */

/**
 * 交易类型枚举
 * @typedef {'standard' | 'contract_call' | 'token_transfer' | 'contract_creation' | 'internal'} TransactionType
 */

/**
 * 交易状态枚举
 * @typedef {'pending' | 'confirmed' | 'failed'} TransactionStatus
 */

/**
 * 风险等级
 * @typedef {'high' | 'medium' | 'low' | 'critical'} RiskLevel
 */

/**
 * 交易实体
 * @typedef {Object} Transaction
 * @property {string} hash - 交易哈希
 * @property {number} blockNumber - 区块号
 * @property {Date} timestamp - 时间戳
 * @property {string} from - 发送地址
 * @property {string} to - 接收地址
 * @property {number} value - 交易金额(ETH)
 * @property {number} gasPrice - Gas价格
 * @property {number} gasUsed - 消耗的Gas
 * @property {TransactionType} type - 交易类型
 * @property {TransactionStatus} status - 交易状态
 * @property {RiskLevel} [risk] - 风险级别
 * @property {string[]} [riskFactors] - 风险因素
 * @property {string} [input] - 输入数据
 * @property {Object} [metadata] - 元数据
 */

/**
 * 交易详情
 * @typedef {Object} TransactionDetails
 * @property {string} hash - 交易哈希
 * @property {number} blockNumber - 区块号
 * @property {Date} timestamp - 时间戳
 * @property {string} from - 发送地址
 * @property {string} to - 接收地址
 * @property {string} [fromLabel] - 发送地址标签
 * @property {string} [toLabel] - 接收地址标签
 * @property {number} value - 交易金额(ETH)
 * @property {number} gasPrice - Gas价格
 * @property {number} gasUsed - 消耗的Gas
 * @property {number} gasLimit - Gas限制
 * @property {number} nonce - 交易序号
 * @property {TransactionType} type - 交易类型
 * @property {TransactionStatus} status - 交易状态
 * @property {RiskLevel} [risk] - 风险级别
 * @property {number} [riskScore] - 风险分数(0-100)
 * @property {string[]} [riskFactors] - 风险因素
 * @property {string} [input] - 输入数据
 * @property {Array} [events] - 事件日志
 * @property {Object} [decodedInput] - 解码后的输入数据
 * @property {Array} [relatedTransactions] - 相关交易
 * @property {Object} [metadata] - 元数据
 */

/**
 * 交易事件
 * @typedef {Object} TransactionEvent
 * @property {string} address - 合约地址
 * @property {Array} topics - 事件主题
 * @property {string} data - 事件数据
 * @property {string} [name] - 事件名称
 * @property {Array} [args] - 解码参数
 */

/**
 * 交易搜索结果
 * @typedef {Object} TransactionSearchResult
 * @property {string} hash - 交易哈希
 * @property {Date} timestamp - 时间戳
 * @property {string} from - 发送地址
 * @property {string} to - 接收地址
 * @property {number} value - 交易金额(ETH)
 * @property {TransactionType} type - 交易类型
 * @property {RiskLevel} [risk] - 风险级别
 */

/**
 * 交易统计信息
 * @typedef {Object} TransactionStats
 * @property {number} count - 总交易数
 * @property {number} totalValue - 总交易额(ETH)
 * @property {number} avgValue - 平均交易金额(ETH)
 * @property {number} avgGasPrice - 平均Gas价格
 * @property {Object} riskDistribution - 风险分布
 * @property {Object} timeDistribution - 时间分布
 * @property {Object} typeDistribution - 类型分布
 */

/**
 * 异常交易
 * @typedef {Object} AnomalousTransaction
 * @property {string} hash - 交易哈希
 * @property {Date} timestamp - 时间戳
 * @property {string} from - 发送地址
 * @property {string} to - 接收地址
 * @property {number} value - 交易金额(ETH)
 * @property {RiskLevel} risk - 风险级别
 * @property {number} riskScore - 风险分数(0-100)
 * @property {string[]} anomalyFactors - 异常因素
 * @property {number} confidence - 置信度(0-100)
 */

/**
 * 交易筛选条件
 * @typedef {Object} TransactionFilters
 * @property {Date} [fromDate] - 开始日期
 * @property {Date} [toDate] - 结束日期
 * @property {string} [fromAddress] - 发送地址
 * @property {string} [toAddress] - 接收地址
 * @property {number} [minValue] - 最小金额
 * @property {number} [maxValue] - 最大金额
 * @property {TransactionType} [type] - 交易类型
 * @property {RiskLevel} [risk] - 风险级别
 */

// 用于与TypeScript兼容的空导出
export default {};
