import { Router } from 'express';
import { SmartMoneyProfileController } from '../controllers/SmartMoneyProfileController';
import { SmartMoneyTracker } from '../core/SmartMoneyTracker';
import { loggerWinston as logger } from '../utils/logger';

const router = Router();

// 画像管理路由
router.get('/profiles/:address', SmartMoneyProfileController.getProfile);
router.post('/profiles', SmartMoneyProfileController.createProfile);
router.put('/profiles/:address', SmartMoneyProfileController.updateProfile);
router.delete('/profiles/:address', SmartMoneyProfileController.deleteProfile);

// 排行榜和统计路由
router.get('/leaderboard', SmartMoneyProfileController.getLeaderboard);
router.get('/stats/investor-types', SmartMoneyProfileController.getInvestorTypeDistribution);
router.get('/stats/roi-distribution', SmartMoneyProfileController.getROIDistribution);
router.get('/stats/recently-active', SmartMoneyProfileController.getRecentlyActive);

// 评分路由
router.get('/scores/:address', SmartMoneyProfileController.getScore);
router.post('/scores/:address/update', SmartMoneyProfileController.updateScore);

// 操作路由
router.post('/profiles/:address/success-cases', SmartMoneyProfileController.addSuccessCase);
router.put('/profiles/:address/holdings', SmartMoneyProfileController.updateHoldings);

/**
 * @api {get} /api/smart-money/analyze/:address 分析地址是否为聪明钱
 * @apiName AnalyzeAddress
 * @apiGroup SmartMoney
 * 
 * @apiParam {String} address 要分析的区块链地址
 * 
 * @apiSuccess {Boolean} isSmartMoney 是否是聪明钱
 * @apiSuccess {Object} data 分析结果详情
 */
router.get('/analyze/:address', async (req, res) => {
  try {
    const { address } = req.params;
    
    if (!address || typeof address !== 'string') {
      return res.status(400).json({ success: false, message: '无效的地址' });
    }

    // 在开发环境下，使用模拟数据
    if (process.env.NODE_ENV === 'development') {
      logger.info('使用模拟数据分析地址', { address });
      
      // 根据地址生成一致的模拟数据
      const isSmartMoney = address.toLowerCase().includes('1') || 
                          address.toLowerCase().includes('a') || 
                          address.toLowerCase().includes('e') || 
                          address.toLowerCase().includes('f');
      
      if (!isSmartMoney) {
        return res.json({
          success: true,
          data: {
            address,
            isSmartMoney: false,
            reason: '不满足聪明钱标准',
            score: 0.3,
            analysisTimestamp: new Date().toISOString()
          }
        });
      }
      
      // 模拟聪明钱数据
      const mockSmartMoneyData = {
        address,
        isSmartMoney: true,
        smartMoneyInfo: {
          isSmartMoney: true,
          score: 0.85,
          confidence: 0.9,
          reason: '满足聪明钱指标',
          investorType: 'long_term_holder',
          traits: {
            entryTiming: 0.8,
            exitTiming: 0.75,
            hodlStrength: 0.9,
            diversification: 0.6,
            contrarian: 0.7
          },
          expertiseAreas: ['DeFi', '稳定币', '头部项目'],
          performanceMetrics: {
            overallROI: 0.45,
            monthlyROI: [0.05, 0.08, 0.12, -0.03, 0.07],
            winRate: 0.72,
            sharpeRatio: 1.8,
            volatility: 0.28,
            maxDrawdown: 0.15
          },
          tags: ['长期持有者', '价值投资者', 'DeFi玩家'],
          scoreComponents: {
            performance: 0.8,
            timing: 0.7,
            portfolioManagement: 0.6,
            riskManagement: 0.75,
            insight: 0.7
          }
        },
        portfolio: [
          {
            token: 'ETH',
            symbol: 'ETH',
            amount: 12.5,
            valueUSD: 35000,
            allocation: 0.4,
            entryPrice: 2100
          },
          {
            token: 'BTC',
            symbol: 'BTC',
            amount: 0.65,
            valueUSD: 28000,
            allocation: 0.32,
            entryPrice: 39000
          },
          {
            token: 'LINK',
            symbol: 'LINK',
            amount: 1200,
            valueUSD: 12000,
            allocation: 0.14,
            entryPrice: 8.5
          },
          {
            token: 'UNI',
            symbol: 'UNI',
            amount: 850,
            valueUSD: 6000,
            allocation: 0.07,
            entryPrice: 5.8
          },
          {
            token: 'AAVE',
            symbol: 'AAVE',
            amount: 45,
            valueUSD: 5500,
            allocation: 0.06,
            entryPrice: 110
          }
        ],
        transactionPatterns: {
          overview: {
            transactionCount: 87,
            firstTransaction: '2021-05-12T08:23:45Z',
            lastTransaction: new Date().toISOString(),
            avgTransactionSize: 5200
          },
          frequencyPatterns: {
            dailyAvg: 0.3,
            weeklyAvg: 2.1,
            monthlyAvg: 8.5,
            averageFrequency: 'low'
          },
          sizePatterns: {
            averageSize: 5200,
            maxSize: 28000,
            sizeDistribution: {
              small: 0.2,
              medium: 0.5,
              large: 0.3
            }
          },
          strategies: ['价值投资', '长期持有', '逢低买入']
        },
        analysisTimestamp: new Date().toISOString(),
        score: 0.85
      };
      
      return res.json({
        success: true,
        data: mockSmartMoneyData
      });
    }
    
    // 使用SmartMoneyTracker服务分析地址
    const result = await SmartMoneyTracker.analyzeAddress(address);
    
    return res.json({
      success: true,
      data: result
    });
  } catch (error: any) {
    logger.error('分析地址失败', { error });
    return res.status(500).json({ success: false, message: '分析地址失败', error: error.message });
  }
});

/**
 * @api {post} /api/smart-money/batch-analyze 批量分析地址
 * @apiName BatchAnalyzeAddresses
 * @apiGroup SmartMoney
 * 
 * @apiParam {Array} addresses 要分析的区块链地址列表
 * 
 * @apiSuccess {Array} results 批量分析结果
 */
router.post('/batch-analyze', async (req, res) => {
  try {
    const { addresses } = req.body;
    
    if (!addresses || !Array.isArray(addresses)) {
      return res.status(400).json({ success: false, message: '无效的地址列表' });
    }
    
    if (addresses.length > 50) {
      return res.status(400).json({ success: false, message: '一次最多分析50个地址' });
    }
    
    const results = await SmartMoneyTracker.batchAnalyzeAddresses(addresses);
    
    return res.json({
      success: true,
      data: results
    });
  } catch (error: any) {
    logger.error('批量分析地址失败', { error });
    return res.status(500).json({ success: false, message: '批量分析地址失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/leaderboard 获取聪明钱排行榜
 * @apiName GetLeaderboard
 * @apiGroup SmartMoney
 * 
 * @apiParam {String} [sortBy=score.overall] 排序依据
 * @apiParam {Array} [investorTypes] 投资者类型筛选
 * @apiParam {Number} [minScore] 最低分数筛选
 * @apiParam {Number} [limit=20] 返回结果数量限制
 * @apiParam {Number} [skip=0] 跳过结果数量
 * 
 * @apiSuccess {Array} profiles 聪明钱档案列表
 */
router.get('/leaderboard', async (req, res) => {
  try {
    const { sortBy, investorTypes, minScore, limit, skip } = req.query;
    
    // 准备查询选项
    const options: any = {};
    
    if (sortBy) options.sortBy = sortBy as string;
    if (investorTypes) options.investorTypes = (investorTypes as string).split(',');
    if (minScore) options.minScore = Number(minScore);
    if (limit) options.limit = Number(limit);
    if (skip) options.skip = Number(skip);
    
    const profiles = await SmartMoneyTracker.getLeaderboard(options);
    
    return res.json({
      success: true,
      data: profiles
    });
  } catch (error: any) {
    logger.error('获取排行榜失败', { error });
    return res.status(500).json({ success: false, message: '获取排行榜失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/investor-types 获取投资者类型分布
 * @apiName GetInvestorTypeDistribution
 * @apiGroup SmartMoney
 * 
 * @apiSuccess {Object} distribution 投资者类型分布
 */
router.get('/investor-types', async (req, res) => {
  try {
    const distribution = await SmartMoneyTracker.getInvestorTypeDistribution();
    
    return res.json({
      success: true,
      data: distribution
    });
  } catch (error: any) {
    logger.error('获取投资者类型分布失败', { error });
    return res.status(500).json({ success: false, message: '获取投资者类型分布失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/roi-distribution 获取ROI分布
 * @apiName GetROIDistribution
 * @apiGroup SmartMoney
 * 
 * @apiSuccess {Array} distribution ROI分布数据
 */
router.get('/roi-distribution', async (req, res) => {
  try {
    const distribution = await SmartMoneyTracker.getROIDistribution();
    
    return res.json({
      success: true,
      data: distribution
    });
  } catch (error: any) {
    logger.error('获取ROI分布失败', { error });
    return res.status(500).json({ success: false, message: '获取ROI分布失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/recent 获取最近活跃的聪明钱
 * @apiName GetRecentlyActive
 * @apiGroup SmartMoney
 * 
 * @apiParam {Number} [days=7] 最近几天
 * @apiParam {Number} [limit=10] 返回结果数量限制
 * 
 * @apiSuccess {Array} profiles 聪明钱档案列表
 */
router.get('/recent', async (req, res) => {
  try {
    const days = req.query.days ? Number(req.query.days) : 7;
    const limit = req.query.limit ? Number(req.query.limit) : 10;
    
    const profiles = await SmartMoneyTracker.getRecentlyActive(days, limit);
    
    return res.json({
      success: true,
      data: profiles
    });
  } catch (error: any) {
    logger.error('获取最近活跃的聪明钱失败', { error });
    return res.status(500).json({ success: false, message: '获取最近活跃的聪明钱失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/predict/:address 预测聪明钱未来可能的交易
 * @apiName PredictFutureActivity
 * @apiGroup SmartMoney
 * 
 * @apiParam {String} address 区块链地址
 * 
 * @apiSuccess {Object} prediction 预测结果
 */
router.get('/predict/:address', async (req, res) => {
  try {
    const { address } = req.params;
    
    if (!address || typeof address !== 'string') {
      return res.status(400).json({ success: false, message: '无效的地址' });
    }
    
    const prediction = await SmartMoneyTracker.predictFutureActivity(address);
    
    return res.json({
      success: true,
      data: prediction
    });
  } catch (error: any) {
    logger.error('预测聪明钱未来活动失败', { error });
    return res.status(500).json({ success: false, message: '预测聪明钱未来活动失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/portfolio/:address 跟踪聪明钱投资组合变化
 * @apiName TrackPortfolioChanges
 * @apiGroup SmartMoney
 * 
 * @apiParam {String} address 区块链地址
 * 
 * @apiSuccess {Object} changes 投资组合变化
 */
router.get('/portfolio/:address', async (req, res) => {
  try {
    const { address } = req.params;
    
    if (!address || typeof address !== 'string') {
      return res.status(400).json({ success: false, message: '无效的地址' });
    }
    
    const changes = await SmartMoneyTracker.trackPortfolioChanges(address);
    
    return res.json({
      success: true,
      data: changes
    });
  } catch (error: any) {
    logger.error('跟踪聪明钱投资组合变化失败', { error });
    return res.status(500).json({ success: false, message: '跟踪聪明钱投资组合变化失败', error: error.message });
  }
});

/**
 * @api {post} /api/smart-money/common-behavior 监控多个聪明钱的共同行为
 * @apiName MonitorCommonBehavior
 * @apiGroup SmartMoney
 * 
 * @apiParam {Array} addresses 区块链地址列表
 * 
 * @apiSuccess {Object} commonBehavior 共同行为分析结果
 */
router.post('/common-behavior', async (req, res) => {
  try {
    const { addresses } = req.body;
    
    if (!addresses || !Array.isArray(addresses) || addresses.length < 2) {
      return res.status(400).json({ success: false, message: '需要至少两个有效地址' });
    }
    
    if (addresses.length > 10) {
      return res.status(400).json({ success: false, message: '一次最多比较10个地址' });
    }
    
    const commonBehavior = await SmartMoneyTracker.monitorCommonBehavior(addresses);
    
    return res.json({
      success: true,
      data: commonBehavior
    });
  } catch (error: any) {
    logger.error('监控聪明钱共同行为失败', { error });
    return res.status(500).json({ success: false, message: '监控聪明钱共同行为失败', error: error.message });
  }
});

/**
 * @api {get} /api/smart-money/trending-tokens 获取热门代币和新兴代币
 * @apiName GetTrendingTokens
 * @apiGroup SmartMoney
 * 
 * @apiSuccess {Array} trending 热门代币列表
 * @apiSuccess {Array} emerging 新兴代币列表
 */
router.get('/trending-tokens', async (req, res) => {
  try {
    const tokens = await SmartMoneyTracker.getTrendingTokens();
    
    return res.json({
      success: true,
      data: tokens
    });
  } catch (error: any) {
    logger.error('获取热门代币失败', { error });
    return res.status(500).json({ success: false, message: '获取热门代币失败', error: error.message });
  }
});

export default router; 