import { Request, Response } from 'express';
import { mockDataService } from '../services/mockDataService';
import { CacheService } from '../services/cacheService';
import { dataOptimizer } from '../services/dataOptimizer';
import { logger } from '../utils/logger';
import { SmartMoneyAnalyzer } from '../services/smartMoneyAnalyzer';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { SmartMoneyScorer } from '../core/SmartMoneyScorer';

interface SmartScoreData {
  address: string;
  score: number;
  traits: string[];
}

interface InvestmentMetrics {
  address: string;
  metrics: {
    roi: number;
    risk: number;
    volume: number;
  };
}

interface AddressDetails {
  address: string;
  balance: string;
  transactions: number;
}

interface TransactionHistory {
  address: string;
  transactions: Array<{
    hash: string;
    timestamp: number;
    value: string;
  }>;
}

interface SimilarAddresses {
  address: string;
  similar: Array<{
    address: string;
    similarity: number;
  }>;
}

export class SmartMoneyController {
  private analyzer: SmartMoneyAnalyzer;
  private cacheService: CacheService;

  constructor() {
    this.analyzer = new SmartMoneyAnalyzer();
    this.cacheService = new CacheService();
  }

  // 获取智能投资评分
  async getSmartScore(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `smart_score:${address}`;
      
      // 尝试从缓存获取
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData as SmartScoreData);
      }

      // 生成模拟数据
      const mockData = mockDataService.generateSmartScore(address);
      
      // 优化数据
      const optimizedData = dataOptimizer.optimizeSmartScore(mockData);
      
      // 缓存结果
      await this.cacheService.set(cacheKey, optimizedData, 300); // 缓存5分钟
      
      res.json(optimizedData as SmartScoreData);
    } catch (error) {
      logger.error('获取智能投资评分失败:', error as Error);
      res.status(500).json({ error: '获取智能投资评分失败' });
    }
  }

  // 获取投资行为指标
  async getInvestmentMetrics(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `investment_metrics:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData as InvestmentMetrics);
      }

      const mockData = mockDataService.generateInvestmentMetrics(address);
      const optimizedData = dataOptimizer.optimizeMetrics(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData, 300);
      
      res.json(optimizedData as InvestmentMetrics);
    } catch (error) {
      logger.error('获取投资行为指标失败:', error as Error);
      res.status(500).json({ error: '获取投资行为指标失败' });
    }
  }

  // 获取地址详情
  async getAddressDetails(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `address_details:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData as AddressDetails);
      }

      const mockData = mockDataService.generateAddressDetails(address);
      const optimizedData = dataOptimizer.optimizeDetails(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData, 300);
      
      res.json(optimizedData as AddressDetails);
    } catch (error) {
      logger.error('获取地址详情失败:', error as Error);
      res.status(500).json({ error: '获取地址详情失败' });
    }
  }

  // 获取交易历史
  async getTransactionHistory(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const { page = 1, limit = 10 } = req.query;
      const cacheKey = `tx_history:${address}:${page}:${limit}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData as TransactionHistory);
      }

      const mockData = mockDataService.generateTransactionHistory(address, Number(page), Number(limit));
      const optimizedData = dataOptimizer.optimizeTransactions(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData, 300);
      
      res.json(optimizedData as TransactionHistory);
    } catch (error) {
      logger.error('获取交易历史失败:', error as Error);
      res.status(500).json({ error: '获取交易历史失败' });
    }
  }

  // 获取相似地址
  async getSimilarAddresses(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `similar_addresses:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData as SimilarAddresses);
      }

      const mockData = mockDataService.generateSimilarAddresses(address);
      const optimizedData = dataOptimizer.optimizeSimilarAddresses(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData, 300);
      
      res.json(optimizedData as SimilarAddresses);
    } catch (error) {
      logger.error('获取相似地址失败:', error as Error);
      res.status(500).json({ error: '获取相似地址失败' });
    }
  }

  // 批量获取智能投资评分
  async getBatchSmartScores(req: Request, res: Response) {
    try {
      const { addresses } = req.body;
      if (!Array.isArray(addresses) || addresses.length === 0) {
        return res.status(400).json({ error: '无效的地址列表' });
      }

      if (addresses.length > 50) {
        return res.status(400).json({ error: '地址数量超过限制' });
      }

      const results = await Promise.all(
        addresses.map(async (address) => {
          const cacheKey = `smart_score:${address}`;
          const cachedData = await this.cacheService.get(cacheKey);
          
          if (cachedData) {
            return cachedData;
          }

          const mockData = mockDataService.generateSmartScore(address);
          const optimizedData = dataOptimizer.optimizeSmartScore(mockData);
          
          await this.cacheService.set(cacheKey, optimizedData, 300);
          
          return optimizedData;
        })
      );

      res.json({ results });
    } catch (error) {
      logger.error('批量获取智能投资评分失败:', error);
      res.status(500).json({ error: '批量获取智能投资评分失败' });
    }
  }

  // 获取聪明钱特征分析
  async getSmartMoneyTraits(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `smart_money_traits:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData);
      }

      const mockData = mockDataService.generateSmartScore(address);
      const optimizedData = dataOptimizer.optimizeSmartScore(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData.smartMoneyTraits, 300);
      
      res.json(optimizedData.smartMoneyTraits);
    } catch (error) {
      logger.error('获取聪明钱特征分析失败:', error);
      res.status(500).json({ error: '获取聪明钱特征分析失败' });
    }
  }

  // 获取专业投资指标
  async getProfessionalMetrics(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `professional_metrics:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData);
      }

      const mockData = mockDataService.generateInvestmentMetrics(address);
      const optimizedData = dataOptimizer.optimizeMetrics(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData.professionalMetrics, 300);
      
      res.json(optimizedData.professionalMetrics);
    } catch (error) {
      logger.error('获取专业投资指标失败:', error);
      res.status(500).json({ error: '获取专业投资指标失败' });
    }
  }

  // 获取交易策略分析
  async getTradingStrategy(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `trading_strategy:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData);
      }

      const mockData = mockDataService.generateInvestmentMetrics(address);
      const optimizedData = dataOptimizer.optimizeMetrics(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData.tradingStrategy, 300);
      
      res.json(optimizedData.tradingStrategy);
    } catch (error) {
      logger.error('获取交易策略分析失败:', error);
      res.status(500).json({ error: '获取交易策略分析失败' });
    }
  }

  // 获取交易统计分析
  async getTransactionStatistics(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `transaction_statistics:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData);
      }

      const mockData = mockDataService.generateTransactionHistory(address, 1, 10);
      const optimizedData = dataOptimizer.optimizeTransactions(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData.statistics, 300);
      
      res.json(optimizedData.statistics);
    } catch (error) {
      logger.error('获取交易统计分析失败:', error);
      res.status(500).json({ error: '获取交易统计分析失败' });
    }
  }

  // 获取相似地址分析
  async getSimilarAddressAnalysis(req: Request, res: Response) {
    try {
      const { address } = req.params;
      const cacheKey = `similar_address_analysis:${address}`;
      
      const cachedData = await this.cacheService.get(cacheKey);
      if (cachedData) {
        return res.json(cachedData);
      }

      const mockData = mockDataService.generateSimilarAddresses(address);
      const optimizedData = dataOptimizer.optimizeSimilarAddresses(mockData);
      
      await this.cacheService.set(cacheKey, optimizedData.similarAddresses, 300);
      
      res.json(optimizedData.similarAddresses);
    } catch (error) {
      logger.error('获取相似地址分析失败:', error);
      res.status(500).json({ error: '获取相似地址分析失败' });
    }
  }

  // 批量获取聪明钱特征
  async getBatchSmartMoneyTraits(req: Request, res: Response) {
    try {
      const { addresses } = req.body;
      if (!Array.isArray(addresses) || addresses.length === 0) {
        return res.status(400).json({ error: '无效的地址列表' });
      }

      if (addresses.length > 50) {
        return res.status(400).json({ error: '地址数量超过限制' });
      }

      const results = await Promise.all(
        addresses.map(async (address) => {
          const cacheKey = `smart_money_traits:${address}`;
          const cachedData = await this.cacheService.get(cacheKey);
          
          if (cachedData) {
            return cachedData;
          }

          const mockData = mockDataService.generateSmartScore(address);
          const optimizedData = dataOptimizer.optimizeSmartScore(mockData);
          
          await this.cacheService.set(cacheKey, optimizedData.smartMoneyTraits, 300);
          
          return optimizedData.smartMoneyTraits;
        })
      );

      res.json({ results });
    } catch (error) {
      logger.error('批量获取聪明钱特征失败:', error);
      res.status(500).json({ error: '批量获取聪明钱特征失败' });
    }
  }

  public async analyzeAddress(req: Request, res: Response) {
    try {
      const { address } = req.params;
      
      if (!address) {
        return res.status(400).json({
          error: '缺少地址参数'
        });
      }

      const analysis = await this.analyzer.analyzeAddress(address);
      
      res.json({
        success: true,
        data: analysis
      });
    } catch (error) {
      logger.error('地址分析失败:', error);
      res.status(500).json({
        success: false,
        error: '分析失败，请稍后重试'
      });
    }
  }

  public async getSmartMoneyDashboard(req: Request, res: Response) {
    try {
      // 获取热门聪明钱地址
      const hotAddresses = await this.getHotAddresses();
      
      // 获取市场趋势
      const marketTrends = await this.getMarketTrends();
      
      // 获取DeFi项目分析
      const defiAnalysis = await this.getDeFiAnalysis();

      res.json({
        success: true,
        data: {
          hotAddresses,
          marketTrends,
          defiAnalysis
        }
      });
    } catch (error) {
      logger.error('获取仪表板数据失败:', error);
      res.status(500).json({
        success: false,
        error: '获取数据失败，请稍后重试'
      });
    }
  }

  private async getHotAddresses() {
    // 实现获取热门地址的逻辑
    return [
      {
        address: '0x123...',
        smartScore: 0.85,
        lastActivity: '2024-03-20T10:00:00Z'
      }
    ];
  }

  private async getMarketTrends() {
    // 实现获取市场趋势的逻辑
    return {
      overallSentiment: '看涨',
      confidence: 0.75,
      keyFactors: ['机构资金流入', 'DeFi锁仓量增加']
    };
  }

  private async getDeFiAnalysis() {
    // 实现获取DeFi分析的逻辑
    return {
      topProjects: [
        {
          name: 'Uniswap',
          tvl: '1000000',
          volume24h: '500000'
        }
      ],
      trends: ['流动性增加', '收益率提升']
    };
  }

  /**
   * 获取地址的聪明钱画像
   */
  static async getProfile(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 查询画像
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      
      if (!profile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      res.json({ success: true, data: profile });
    } catch (error) {
      logger.error('获取聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 创建聪明钱画像
   */
  static async createProfile(req: Request, res: Response): Promise<void> {
    try {
      const profileData = req.body;
      
      // 参数验证
      if (!profileData || !profileData.address) {
        res.status(400).json({ success: false, message: '画像数据缺失或无效' });
        return;
      }
      
      // 检查是否已存在
      const existingProfile = await SmartMoneyProfileDAO.findByAddress(profileData.address);
      if (existingProfile) {
        res.status(409).json({ success: false, message: '该地址的聪明钱画像已存在' });
        return;
      }
      
      // 创建画像
      const newProfile = await SmartMoneyProfileDAO.create(profileData);
      
      // 计算初始评分
      await SmartMoneyScorer.updateScore(profileData.address);
      
      res.status(201).json({ success: true, data: newProfile });
    } catch (error) {
      logger.error('创建聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 更新聪明钱画像
   */
  static async updateProfile(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      const updates = req.body;
      
      // 参数验证
      if (!address || !updates) {
        res.status(400).json({ success: false, message: '参数缺失或无效' });
        return;
      }
      
      // 更新画像
      const updatedProfile = await SmartMoneyProfileDAO.update(address, updates);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      // 更新评分
      await SmartMoneyScorer.updateScore(address);
      
      res.json({ success: true, data: updatedProfile });
    } catch (error) {
      logger.error('更新聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 删除聪明钱画像
   */
  static async deleteProfile(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 删除画像
      const result = await SmartMoneyProfileDAO.delete(address);
      
      if (!result) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      res.json({ success: true, message: '聪明钱画像已删除' });
    } catch (error) {
      logger.error('删除聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取聪明钱排行榜
   */
  static async getLeaderboard(req: Request, res: Response): Promise<void> {
    try {
      const { 
        sortBy = 'score.overall',
        investorTypes,
        minScore,
        limit = 20,
        skip = 0
      } = req.query;
      
      // 处理查询参数
      const options: any = {
        sortBy: sortBy as string,
        minScore: minScore ? Number(minScore) : undefined,
        limit: Number(limit),
        skip: Number(skip)
      };
      
      // 处理投资者类型
      if (investorTypes) {
        options.investorTypes = Array.isArray(investorTypes) 
          ? investorTypes 
          : [investorTypes as string];
      }
      
      // 查询排行榜
      const leaderboard = await SmartMoneyProfileDAO.getLeaderboard(options);
      
      res.json({ 
        success: true, 
        data: leaderboard,
        meta: {
          total: leaderboard.length,
          limit: options.limit,
          skip: options.skip
        }
      });
    } catch (error) {
      logger.error('获取聪明钱排行榜失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取投资者类型分布
   */
  static async getInvestorTypeDistribution(req: Request, res: Response): Promise<void> {
    try {
      const distribution = await SmartMoneyProfileDAO.getInvestorTypeDistribution();
      
      res.json({ success: true, data: distribution });
    } catch (error) {
      logger.error('获取投资者类型分布失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取近期活跃的聪明钱
   */
  static async getRecentlyActive(req: Request, res: Response): Promise<void> {
    try {
      const { days = 7, limit = 10 } = req.query;
      
      const profiles = await SmartMoneyProfileDAO.getRecentlyActive(
        Number(days),
        Number(limit)
      );
      
      res.json({ success: true, data: profiles });
    } catch (error) {
      logger.error('获取近期活跃聪明钱失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取ROI分布
   */
  static async getROIDistribution(req: Request, res: Response): Promise<void> {
    try {
      const distribution = await SmartMoneyProfileDAO.getROIDistribution();
      
      res.json({ success: true, data: distribution });
    } catch (error) {
      logger.error('获取ROI分布失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 添加成功案例
   */
  static async addSuccessCase(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      const successCase = req.body;
      
      // 参数验证
      if (!address || !successCase) {
        res.status(400).json({ success: false, message: '参数缺失或无效' });
        return;
      }
      
      // 添加成功案例
      const updatedProfile = await SmartMoneyProfileDAO.addSuccessCase(address, successCase);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      // 更新评分
      await SmartMoneyScorer.updateScore(address);
      
      res.json({ success: true, data: updatedProfile });
    } catch (error) {
      logger.error('添加成功案例失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 更新持仓
   */
  static async updateHoldings(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      const holdings = req.body;
      
      // 参数验证
      if (!address || !Array.isArray(holdings)) {
        res.status(400).json({ success: false, message: '参数缺失或无效' });
        return;
      }
      
      // 更新持仓
      const updatedProfile = await SmartMoneyProfileDAO.updateHoldings(address, holdings);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      // 更新评分
      await SmartMoneyScorer.updateScore(address);
      
      res.json({ success: true, data: updatedProfile });
    } catch (error) {
      logger.error('更新持仓失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取评分
   */
  static async getScore(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 计算最新评分
      const score = await SmartMoneyScorer.calculateScore(address);
      
      res.json({ success: true, data: score });
    } catch (error) {
      logger.error('获取评分失败', { error });
      
      // 处理特定错误
      if (error instanceof Error && error.message.includes('未找到地址画像')) {
        res.status(404).json({ success: false, message: error.message });
        return;
      }
      
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 更新评分
   */
  static async updateScore(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 更新评分
      const updatedProfile = await SmartMoneyScorer.updateScore(address);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      res.json({ success: true, data: updatedProfile.score });
    } catch (error) {
      logger.error('更新评分失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
} 