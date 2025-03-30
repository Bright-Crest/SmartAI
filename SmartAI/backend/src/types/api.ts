/**
 * 统一API响应格式
 */
export interface ApiResponse<T = any> {
  /** 状态码 */
  code: number;
  
  /** 消息 */
  message: string;
  
  /** 数据 */
  data?: T;
  
  /** 请求ID */
  requestId?: string;
  
  /** 时间戳 */
  timestamp: number;
}

/**
 * 分页响应接口
 */
export interface PaginatedResponse<T = any> extends ApiResponse {
  data: {
    /** 数据列表 */
    items: T[];
    
    /** 总数 */
    total: number;
    
    /** 页码 */
    page: number;
    
    /** 每页大小 */
    pageSize: number;
    
    /** 总页数 */
    totalPages: number;
  };
}

/**
 * 风险因素接口
 */
export interface RiskFactor {
  /** 因素名称 */
  name: string;
  
  /** 因素描述 */
  description: string;
  
  /** 影响程度 (0-1) */
  impact: number;
  
  /** 因素类别 */
  category: string;
}

/**
 * 风险评分响应接口 (与FastAPI兼容)
 */
export interface RiskScoreResponse {
  /** 地址 */
  address: string;
  
  /** 原始风险分数 */
  score: number;
  
  /** 标准化风险分数(0-100) - Express.js中称为smartScore */
  risk_score: number; // 与FastAPI兼容
  
  /** 风险等级 */
  risk_level: string; // 与FastAPI兼容
  
  /** 风险等级描述 */
  risk_description: string; // 与FastAPI兼容
  
  /** 风险解释 */
  risk_explanation?: string;
  
  /** 风险因素列表 */
  risk_factors: RiskFactor[];
  
  /** 需要关注的点 */
  attention_points?: string[];
  
  /** 风险维度分数 - Express.js特有 */
  dimensions: {
    /** 资金流动维度分数 */
    flow: number;
    
    /** 行为模式维度分数 */
    behavior: number;
    
    /** 关联分析维度分数 */
    association: number;
    
    /** 历史特征维度分数 */
    historical: number;
    
    /** 技术特征维度分数 */
    technical: number;
  };
  
  /** 风险标签 - Express.js特有 */
  riskTags: Array<{
    name: string;
    category: string;
    confidence: number;
    description: string;
  }>;
  
  /** 提取的特征 - FastAPI特有 */
  features?: Record<string, any>;
  
  /** 上次更新时间 */
  updatedAt: number;
}

/**
 * 维度指标接口 - Express.js特有
 */
export interface DimensionMetrics {
  /** 维度名称 */
  name: string;
  
  /** 维度键名 */
  key: string;
  
  /** 维度描述 */
  description: string;
  
  /** 维度分数 */
  score: number;
  
  /** 维度权重 */
  weight: number;
  
  /** 维度指标列表 */
  indicators: Array<{
    /** 指标名称 */
    name: string;
    
    /** 指标值 */
    value: number;
    
    /** 指标描述 */
    description: string;
    
    /** 是否异常 */
    isAnomaly: boolean;
  }>;
} 