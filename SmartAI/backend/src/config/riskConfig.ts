/**
 * 风险分析配置
 * 定义风险分析所需的权重、阈值和其他参数
 */
export const config = {
  // 风险分析维度类别
  categories: {
    FLOW: 'flow',             // 资金流动
    BEHAVIOR: 'behavior',     // 行为模式
    ASSOCIATION: 'association', // 关联分析
    HISTORICAL: 'historical', // 历史特征
    TECHNICAL: 'technical',   // 技术特征
    SYSTEM: 'system'          // 系统标签
  },
  
  // 各维度权重配置
  weights: {
    FLOW: 0.25,        // 资金流动维度权重
    BEHAVIOR: 0.25,    // 行为模式维度权重
    ASSOCIATION: 0.20, // 关联分析维度权重
    HISTORICAL: 0.15,  // 历史特征维度权重
    TECHNICAL: 0.15,   // 技术特征维度权重
  },
  
  // 标签风险权重映射
  tagRiskWeights: {
    // 资金流动相关标签
    'mixer_interaction': 0.85,    // 混币器交互 
    'blacklisted_address': 0.95,  // 黑名单地址
    'phishing_related': 0.90,     // 钓鱼相关
    'fraud_related': 0.90,        // 欺诈相关
    'high_frequency_trading': 0.60, // 高频交易
    'unusual_large_tx': 0.65,     // 异常大金额交易
    'fund_dispersal': 0.70,       // 资金分散
    
    // 行为模式相关标签
    'contract_deployer': 0.30,    // 合约部署者
    'flash_loan_user': 0.60,      // 闪电贷使用者
    'multiple_token_swaps': 0.50, // 多重代币交换
    'gas_price_manipulation': 0.80, // Gas价格操纵
    'token_dumping': 0.75,        // 代币抛售
    
    // 关联分析相关标签
    'connected_to_blacklist': 0.80, // 关联黑名单
    'exchange_funding': 0.40,     // 交易所资金
    'suspicious_cluster': 0.70,   // 可疑聚类
    'dark_pool_connection': 0.75, // 暗池连接
    'risk_path_detected': 0.75,   // 风险路径
    
    // 历史特征相关标签
    'new_address': 0.50,          // 新地址
    'dormant_reactivated': 0.60,  // 休眠重激活
    'previous_violations': 0.85,  // 之前违规记录
    'abandoned_tokens': 0.55,     // 废弃代币
    
    // 技术特征相关标签
    'smart_contract_owner': 0.35, // 智能合约所有者
    'multisig_wallet': 0.30,      // 多签钱包
    'proxy_contract': 0.45,       // 代理合约
    'abnormal_code_patterns': 0.75, // 异常代码模式
    'custom_signatures': 0.60,    // 自定义签名
    
    // 系统标签
    'system_flagged': 0.85,       // 系统标记
    'under_investigation': 0.70,  // 调查中
    'confidence_low': 0.30,       // 低置信度
    'historical_data_limited': 0.40 // 历史数据有限
  },
  
  // 风险评分阈值
  riskThresholds: {
    LOW_RISK: 0.3,      // 低风险阈值
    MEDIUM_RISK: 0.6,   // 中风险阈值
    HIGH_RISK: 0.8,     // 高风险阈值
    DIRECT_CONNECTION_CUTOFF: 0.4, // 直接连接阈值
    SAFE_PATH_THRESHOLD: 0.3     // 安全路径阈值
  },
  
  // 风险评分信息设置
  riskLevels: [
    { level: '极高风险', min: 90, color: '#FF0000', description: '极度危险，建议立即中断交互' },
    { level: '高风险', min: 80, color: '#FF4500', description: '存在高度危险，不建议交互' },
    { level: '中高风险', min: 70, color: '#FFA500', description: '显著风险，谨慎交互' },
    { level: '中等风险', min: 60, color: '#FFD700', description: '中等风险，需要谨慎' },
    { level: '中低风险', min: 50, color: '#ADFF2F', description: '低至中等风险，相对安全' },
    { level: '低风险', min: 30, color: '#7CFC00', description: '风险较低，通常安全' },
    { level: '极低风险', min: 0, color: '#00FF00', description: '几乎无风险，非常安全' }
  ],
  
  // 评分分析维度配置
  scoreDimensions: [
    {
      name: '资金流动分析',
      key: 'flow',
      description: '分析地址资金流入流出、交易对象、转账模式等'
    },
    {
      name: '行为模式分析',
      key: 'behavior',
      description: '分析地址的交易行为、交互模式、操作习惯等'
    },
    {
      name: '关联关系分析',
      key: 'association',
      description: '分析地址与高风险地址的直接和间接关联'
    },
    {
      name: '历史特征分析',
      key: 'historical',
      description: '分析地址的历史活动、活跃度、存在时间等'
    },
    {
      name: '技术特征分析',
      key: 'technical',
      description: '分析合约代码特征、交易调用模式等技术指标'
    }
  ],
  
  // 高风险地址列表
  knownRiskAddresses: [
    '0x8589427373d6d84e98730d7795d8f6f8731fda16',  // Tornado Cash: 以太坊主网
    '0x722122df12d4e14e13ac3b6895a86e84145b6967',  // Tornado Cash: BSC
    '0xd90e2f925da726b50c4ed8d0fb90ad053324f31b',  // Tornado Cash: Polygon
    '0x169ad27a470d064dede56a2d3ff727986b15d52b'   // Tornado Cash: Optimism
  ],
    
  // AI模型配置
  aiModel: {
    maxContextLength: 4096,
    minConfidenceThreshold: 0.65,
    maxTagsPerAddress: 12,
    riskFactorBalancing: {
      behavioralWeight: 0.35,
      networkAnalysisWeight: 0.4,
      historicalWeight: 0.25
    }
  },
    
  // 风险处理动作映射
  riskActions: {
    HIGH: ['block', 'report', 'alert'],
    MEDIUM: ['warn', 'review', 'monitor'],
    LOW: ['monitor', 'log']
  },
    
  // 关系分析配置
  relationshipAnalysis: {
    maxDepth: 3,           // 最大深度
    maxNodes: 100,         // 最大节点数
    minFlowValue: 0.05,    // 最小流量阈值
    directInfluenceDecay: 0.5,  // 直接影响衰减系数
    indirectInfluenceDecay: 0.7 // 间接影响衰减系数
  }
}; 