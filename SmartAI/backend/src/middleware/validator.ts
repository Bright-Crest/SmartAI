import { Request, Response, NextFunction } from 'express';
import { body, param, query, validationResult } from 'express-validator';
import { isValidAddress } from '../utils/blockchainValidators';
import { logger } from '../utils/logger';

/**
 * 验证请求中间件
 * 处理验证结果并返回适当的错误信息
 */
export const validateRequest = (req: Request, res: Response, next: NextFunction) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.warn(`验证失败: ${JSON.stringify(errors.array())}`);
    return res.status(400).json({
      success: false,
      error: {
        code: 'VALIDATION_ERROR',
        message: '请求参数验证失败',
        details: errors.array()
      }
    });
  }
  next();
};

/**
 * 验证单个地址参数
 * 用于 /api/risk/score/:address 等接口
 */
export const validateAddress = [
  param('address')
    .notEmpty().withMessage('地址不能为空')
    .custom((value) => {
      if (!isValidAddress(value)) {
        throw new Error('无效的区块链地址格式');
      }
      return true;
    }),
  validateRequest
];

/**
 * 验证批量地址参数
 * 用于 /api/risk/batch 接口
 */
export const validateBatchAddresses = [
  body('addresses')
    .isArray().withMessage('addresses必须是数组')
    .notEmpty().withMessage('addresses不能为空')
    .custom((addresses) => {
      if (!Array.isArray(addresses)) {
        throw new Error('addresses必须是数组');
      }
      
      if (addresses.length > 50) {
        throw new Error('批量请求最多支持50个地址');
      }
      
      // 验证每个地址格式
      const invalidAddresses = addresses
        .filter(addr => !isValidAddress(addr))
        .map((addr, index) => ({ index, address: addr }));
      
      if (invalidAddresses.length > 0) {
        throw new Error(`包含${invalidAddresses.length}个无效地址: ${JSON.stringify(invalidAddresses)}`);
      }
      
      return true;
    }),
  validateRequest
];

/**
 * 验证分页参数
 * 用于带分页的列表接口
 */
export const validatePagination = [
  query('page')
    .optional()
    .isInt({ min: 1 }).withMessage('页码必须是大于0的整数'),
  query('pageSize')
    .optional()
    .isInt({ min: 1, max: 100 }).withMessage('每页条数必须是1-100之间的整数'),
  validateRequest
];

/**
 * 验证日期范围参数
 * 用于带时间筛选的接口
 */
export const validateDateRange = [
  query('startDate')
    .optional()
    .isISO8601().withMessage('开始日期必须是有效的ISO日期格式(YYYY-MM-DD)'),
  query('endDate')
    .optional()
    .isISO8601().withMessage('结束日期必须是有效的ISO日期格式(YYYY-MM-DD)')
    .custom((endDate, { req }) => {
      const startDate = req.query && req.query.startDate ? req.query.startDate as string : null;
      if (endDate && startDate && new Date(endDate) < new Date(startDate)) {
        throw new Error('结束日期不能早于开始日期');
      }
      return true;
    }),
  validateRequest
];

/**
 * 验证API密钥
 * 用于需要API认证的接口
 */
export const validateApiKey = [
  query('apiKey')
    .notEmpty().withMessage('API密钥不能为空')
    .isLength({ min: 32, max: 64 }).withMessage('API密钥长度必须在32-64位之间'),
  validateRequest
]; 