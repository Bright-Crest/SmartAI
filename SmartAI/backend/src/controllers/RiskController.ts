import { Request, Response, NextFunction } from 'express';
import { MockDataService } from '../services/mockDataService';
import { CacheService } from '../services/cacheService';
import { DataOptimizer } from '../utils/dataOptimizer';
import { logger } from '../utils/logger';
import { isValidAddress, normalizeAddress } from '../utils/blockchainValidators';

/**
 * 风险控制器 - 处理风险分析相关API
 */
export class RiskController {
  private mockDataService: MockDataService;
  private cacheService: CacheService;
  
  constructor() {
    this.mockDataService = MockDataService.getInstance();
    this.cacheService = CacheService.getInstance();
  }
  
  /**
   * 获取地址风险评分
   */
  public getAddressRiskScore = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 尝试从缓存获取
      const cacheKey = `risk:score:${normalizedAddress}`;
      const cachedData = await this.cacheService.get(cacheKey);
      
      if (cachedData) {
        logger.debug('从缓存获取风险评分', { address: normalizedAddress });
        res.json({
          success: true,
          data: cachedData
        });
        return;
      }
      
      // 获取风险评分数据
      let riskData: any;
      
      // 判断是使用Mock数据还是实际API
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据
        riskData = this.mockDataService.generateRiskScore(normalizedAddress);
      } else {
        // TODO: 实际风险评分计算API调用
        // 暂时使用Mock数据
        riskData = this.mockDataService.generateRiskScore(normalizedAddress);
      }
      
      // 缓存结果
      await this.cacheService.set(cacheKey, riskData, {
        ttl: 3600, // 1小时缓存
        tags: ['risk', `address:${normalizedAddress}`]
      });
      
      res.json({
        success: true,
        data: riskData
      });
    } catch (error) {
      logger.error('获取风险评分失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '获取风险评分时发生错误'
        }
      });
    }
  }
  
  /**
   * 获取地址风险指标详情
   */
  public getAddressMetrics = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 尝试从缓存获取
      const cacheKey = `risk:metrics:${normalizedAddress}`;
      const cachedData = await this.cacheService.get(cacheKey);
      
      if (cachedData) {
        logger.debug('从缓存获取风险指标', { address: normalizedAddress });
        res.json({
          success: true,
          data: cachedData
        });
        return;
      }
      
      // 获取风险指标数据
      let metricsData: any;
      
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据
        metricsData = this.mockDataService.generateRiskMetrics(normalizedAddress);
      } else {
        // TODO: 实际风险指标API调用
        // 暂时使用Mock数据
        metricsData = this.mockDataService.generateRiskMetrics(normalizedAddress);
      }
      
      // 缓存结果
      await this.cacheService.set(cacheKey, metricsData, {
        ttl: 3600, // 1小时缓存
        tags: ['risk', `address:${normalizedAddress}`]
      });
      
      res.json({
        success: true,
        data: metricsData
      });
    } catch (error) {
      logger.error('获取风险指标失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '获取风险指标时发生错误'
        }
      });
    }
  }
  
  /**
   * 获取地址详细信息
   */
  public getAddressDetails = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 尝试从缓存获取
      const cacheKey = `address:details:${normalizedAddress}`;
      const cachedData = await this.cacheService.get(cacheKey);
      
      if (cachedData) {
        logger.debug('从缓存获取地址详情', { address: normalizedAddress });
        res.json({
          success: true,
          data: cachedData
        });
        return;
      }
      
      // 获取地址详情
      let addressData: any;
      
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据
        addressData = this.mockDataService.generateAddressDetails(normalizedAddress);
      } else {
        // TODO: 实际地址详情API调用
        // 暂时使用Mock数据
        addressData = this.mockDataService.generateAddressDetails(normalizedAddress);
      }
      
      // 缓存结果
      await this.cacheService.set(cacheKey, addressData, {
        ttl: 3600, // 1小时缓存
        tags: ['address', `address:${normalizedAddress}`]
      });
      
      res.json({
        success: true,
        data: addressData
      });
    } catch (error) {
      logger.error('获取地址详情失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '获取地址详情时发生错误'
        }
      });
    }
  }
  
  /**
   * 获取地址交易历史
   */
  public getAddressTransactions = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      const page = parseInt(req.query.page as string) || 1;
      const pageSize = parseInt(req.query.pageSize as string) || 20;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 尝试从缓存获取
      const cacheKey = `address:transactions:${normalizedAddress}:${page}:${pageSize}`;
      const cachedData = await this.cacheService.get(cacheKey);
      
      if (cachedData) {
        logger.debug('从缓存获取交易历史', { address: normalizedAddress, page, pageSize });
        res.json({
          success: true,
          data: cachedData
        });
        return;
      }
      
      // 获取交易历史
      let transactionsData: any[];
      
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据
        transactionsData = this.mockDataService.generateTransactionHistory(normalizedAddress, 100);
      } else {
        // TODO: 实际交易历史API调用
        // 暂时使用Mock数据
        transactionsData = this.mockDataService.generateTransactionHistory(normalizedAddress, 100);
      }
      
      // 分页处理
      const paginatedData = DataOptimizer.paginateData(transactionsData, page, pageSize);
      
      // 缓存结果
      await this.cacheService.set(cacheKey, paginatedData, {
        ttl: 1800, // 30分钟缓存
        tags: ['transactions', `address:${normalizedAddress}`]
      });
      
      res.json({
        success: true,
        data: paginatedData
      });
    } catch (error) {
      logger.error('获取交易历史失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '获取交易历史时发生错误'
        }
      });
    }
  }
  
  /**
   * 批量获取地址风险评分
   */
  public getBatchRiskScores = async (req: Request, res: Response): Promise<void> => {
    try {
      const { addresses } = req.body;
      
      // 验证请求
      if (!Array.isArray(addresses) || addresses.length === 0) {
        res.status(400).json({
          success: false,
          error: {
            message: '请提供有效的地址数组'
          }
        });
        return;
      }
      
      if (addresses.length > 50) {
        res.status(400).json({
          success: false,
          error: {
            message: '一次最多处理50个地址'
          }
        });
        return;
      }
      
      // 标准化地址
      const normalizedAddresses = addresses.map(addr => normalizeAddress(addr));
      
      // 获取批量风险评分
      let batchResults: any;
      
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据
        batchResults = this.mockDataService.generateBatchRiskScores(normalizedAddresses);
      } else {
        // TODO: 实际批量评分API调用
        // 暂时使用Mock数据
        batchResults = this.mockDataService.generateBatchRiskScores(normalizedAddresses);
      }
      
      res.json({
        success: true,
        data: batchResults
      });
    } catch (error) {
      logger.error('批量获取风险评分失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '批量获取风险评分时发生错误'
        }
      });
    }
  }
  
  /**
   * 获取相似地址列表
   */
  public getSimilarAddresses = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 尝试从缓存获取
      const cacheKey = `address:similar:${normalizedAddress}`;
      const cachedData = await this.cacheService.get(cacheKey);
      
      if (cachedData) {
        logger.debug('从缓存获取相似地址', { address: normalizedAddress });
        res.json({
          success: true,
          data: cachedData
        });
        return;
      }
      
      // 获取相似地址
      let similarAddresses: any[];
      
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据生成5-15个相似地址
        const count = Math.floor(Math.random() * 10) + 5;
        similarAddresses = this.mockDataService.generateSimilarAddresses(normalizedAddress, count);
      } else {
        // TODO: 实际相似地址API调用
        // 暂时使用Mock数据
        const count = Math.floor(Math.random() * 10) + 5;
        similarAddresses = this.mockDataService.generateSimilarAddresses(normalizedAddress, count);
      }
      
      // 缓存结果
      await this.cacheService.set(cacheKey, similarAddresses, {
        ttl: 3600 * 4, // 4小时缓存
        tags: ['similar', `address:${normalizedAddress}`]
      });
      
      res.json({
        success: true,
        data: similarAddresses
      });
    } catch (error) {
      logger.error('获取相似地址失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '获取相似地址时发生错误'
        }
      });
    }
  }
  
  /**
   * 获取地址标签信息
   */
  public getAddressTags = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 尝试从缓存获取
      const cacheKey = `address:tags:${normalizedAddress}`;
      const cachedData = await this.cacheService.get(cacheKey);
      
      if (cachedData) {
        logger.debug('从缓存获取地址标签', { address: normalizedAddress });
        res.json({
          success: true,
          data: cachedData
        });
        return;
      }
      
      // 获取地址标签
      let tags: any[];
      
      if (this.mockDataService.isMockEnabled()) {
        // 使用Mock数据
        tags = this.mockDataService.generateAddressTags(normalizedAddress);
      } else {
        // TODO: 实际标签API调用
        // 暂时使用Mock数据
        tags = this.mockDataService.generateAddressTags(normalizedAddress);
      }
      
      // 缓存结果
      await this.cacheService.set(cacheKey, tags, {
        ttl: 3600 * 24, // 24小时缓存
        tags: ['tags', `address:${normalizedAddress}`]
      });
      
      res.json({
        success: true,
        data: tags
      });
    } catch (error) {
      logger.error('获取地址标签失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '获取地址标签时发生错误'
        }
      });
    }
  }
  
  /**
   * 清除地址相关缓存
   */
  public clearAddressCache = async (req: Request, res: Response): Promise<void> => {
    try {
      const { address } = req.params;
      
      // 验证地址
      if (!isValidAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            message: '无效的区块链地址'
          }
        });
        return;
      }
      
      // 标准化地址格式
      const normalizedAddress = normalizeAddress(address);
      
      // 清除该地址的所有缓存
      await this.cacheService.deleteByTag(`address:${normalizedAddress}`);
      
      logger.info('已清除地址缓存', { address: normalizedAddress });
      
      res.json({
        success: true,
        message: '地址缓存已清除'
      });
    } catch (error) {
      logger.error('清除地址缓存失败', {
        error: error instanceof Error ? error.message : String(error),
        path: req.path
      });
      
      res.status(500).json({
        success: false,
        error: {
          message: '清除地址缓存时发生错误'
        }
      });
    }
  }
} 