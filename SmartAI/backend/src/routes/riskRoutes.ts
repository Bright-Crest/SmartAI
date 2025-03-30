import { Router } from 'express';
import { RiskController } from '../controllers/riskController';
import { cacheMiddleware } from '../middleware/cacheMiddleware';
import { validateAddress, validateBatchAddresses, validatePagination } from '../middleware/validator';
import { DataOptimizer } from '../utils/dataOptimizer';
import { authMiddleware } from '../middleware/auth';

const router = Router();
const riskController = new RiskController();

// 应用数据优化中间件
router.use(DataOptimizer.fieldSelector());

// 缓存配置
const SHORT_CACHE = { ttl: 5 * 60 }; // 5分钟
const MEDIUM_CACHE = { ttl: 30 * 60 }; // 30分钟
const LONG_CACHE = { ttl: 60 * 60 }; // 1小时

/**
 * @swagger
 * /api/risk/score/{address}:
 *   get:
 *     summary: 获取地址的SmartScore风险评分
 *     description: 计算指定区块链地址的风险评分和相关分析数据
 *     tags: [风险分析]
 *     parameters:
 *       - in: path
 *         name: address
 *         required: true
 *         schema:
 *           type: string
 *         description: 区块链地址
 *     responses:
 *       200:
 *         description: 风险评分计算成功
 *       400:
 *         description: 无效的请求参数
 *       500:
 *         description: 服务器错误
 */
router.get(
  '/score/:address',
  validateAddress,
  cacheMiddleware(SHORT_CACHE),
  riskController.getAddressRiskScore
);

/**
 * @swagger
 * /api/risk/metrics/{address}:
 *   get:
 *     summary: 获取地址的五维指标数据
 *     description: 获取指定区块链地址的五维风险指标详细数据
 *     tags: [风险分析]
 *     parameters:
 *       - in: path
 *         name: address
 *         required: true
 *         schema:
 *           type: string
 *         description: 区块链地址
 *     responses:
 *       200:
 *         description: 五维指标数据获取成功
 *       400:
 *         description: 无效的请求参数
 *       500:
 *         description: 服务器错误
 */
router.get(
  '/metrics/:address',
  validateAddress,
  cacheMiddleware(SHORT_CACHE),
  riskController.getAddressMetrics
);

/**
 * @swagger
 * /api/risk/address/{address}:
 *   get:
 *     summary: 获取地址的详细信息
 *     description: 获取指定区块链地址的详细信息
 *     tags: [风险分析]
 *     parameters:
 *       - in: path
 *         name: address
 *         required: true
 *         schema:
 *           type: string
 *         description: 区块链地址
 *     responses:
 *       200:
 *         description: 地址详细信息获取成功
 *       400:
 *         description: 无效的请求参数
 *       500:
 *         description: 服务器错误
 */
router.get(
  '/address/:address',
  validateAddress,
  cacheMiddleware(MEDIUM_CACHE),
  riskController.getAddressDetails
);

/**
 * @swagger
 * /api/risk/transactions/{address}:
 *   get:
 *     summary: 获取地址的交易历史
 *     description: 获取指定区块链地址的交易历史
 *     tags: [风险分析]
 *     parameters:
 *       - in: path
 *         name: address
 *         required: true
 *         schema:
 *           type: string
 *         description: 区块链地址
 *     responses:
 *       200:
 *         description: 交易历史获取成功
 *       400:
 *         description: 无效的请求参数
 *       500:
 *         description: 服务器错误
 */
router.get(
  '/transactions/:address',
  validateAddress,
  validatePagination,
  cacheMiddleware(SHORT_CACHE),
  riskController.getAddressTransactions
);

/**
 * @swagger
 * /api/risk/batch-score:
 *   post:
 *     summary: 批量计算多个地址的SmartScore
 *     description: 同时计算多个区块链地址的风险评分
 *     tags: [风险分析]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               addresses:
 *                 type: array
 *                 items:
 *                   type: string
 *     responses:
 *       200:
 *         description: 批量风险评分计算完成
 *       400:
 *         description: 无效的请求参数
 *       500:
 *         description: 服务器错误
 */
router.post(
  '/batch',
  validateBatchAddresses,
  cacheMiddleware(SHORT_CACHE),
  riskController.getBatchRiskScores
);

// 获取相似地址
router.get(
  '/similar/:address',
  validateAddress,
  cacheMiddleware(MEDIUM_CACHE),
  riskController.getSimilarAddresses
);

// 地址标签查询
router.get(
  '/tags/:address',
  validateAddress,
  cacheMiddleware(LONG_CACHE),
  riskController.getAddressTags
);

// 清除地址缓存
router.delete(
  '/cache/:address',
  validateAddress,
  riskController.clearAddressCache
);

// 管理员路由，需要认证
router.use('/admin', authMiddleware);

// 清除所有风险数据缓存
router.delete(
  '/admin/cache/all',
  (req: Request, res: Response) => riskController.clearAllRiskCache(req, res)
);

export default router; 