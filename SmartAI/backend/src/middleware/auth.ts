import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { logger } from '../utils/logger';

// 接口扩展，添加用户信息
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
        permissions?: string[];
      };
    }
  }
}

/**
 * JWT身份验证中间件
 */
export const authenticate = (req: Request, res: Response, next: NextFunction) => {
  try {
    // 从请求头获取Authorization
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        code: 401,
        message: '未提供身份验证凭据',
        timestamp: Date.now()
      });
    }
    
    // 提取Token
    const token = authHeader.split(' ')[1];
    
    // 验证Token
    const secret = process.env.JWT_SECRET || 'smartai-api-secret-key';
    const decoded = jwt.verify(token, secret);
    
    // 将用户信息添加到请求对象
    req.user = decoded as {
      id: string;
      email: string;
      role: string;
      permissions?: string[];
    };
    
    next();
  } catch (error) {
    logger.warn('身份验证失败', {
      error: error instanceof Error ? error.message : String(error),
      path: req.path
    });
    
    return res.status(401).json({
      code: 401,
      message: '身份验证失败',
      timestamp: Date.now()
    });
  }
};

/**
 * 角色授权中间件
 * @param roles 允许的角色列表
 */
export const authorize = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({
        code: 401,
        message: '未进行身份验证',
        timestamp: Date.now()
      });
    }
    
    if (roles.includes(req.user.role)) {
      return next();
    }
    
    logger.warn('未授权访问', {
      user: req.user.id,
      role: req.user.role,
      requiredRoles: roles,
      path: req.path
    });
    
    return res.status(403).json({
      code: 403,
      message: '无权访问此资源',
      timestamp: Date.now()
    });
  };
};

/**
 * API密钥验证中间件
 */
export const apiKeyAuth = (req: Request, res: Response, next: NextFunction) => {
  try {
    const apiKey = req.headers['x-api-key'] as string;
    
    if (!apiKey) {
      return res.status(401).json({
        code: 401,
        message: 'API密钥未提供',
        timestamp: Date.now()
      });
    }
    
    // API密钥验证逻辑
    const validApiKeys = process.env.API_KEYS?.split(',') || [];
    
    if (!validApiKeys.includes(apiKey)) {
      logger.warn('无效的API密钥', {
        key: apiKey.substring(0, 4) + '...',
        path: req.path,
        ip: req.ip
      });
      
      return res.status(401).json({
        code: 401,
        message: 'API密钥无效',
        timestamp: Date.now()
      });
    }
    
    next();
  } catch (error) {
    logger.error('API密钥验证错误', {
      error: error instanceof Error ? error.message : String(error),
      path: req.path
    });
    
    return res.status(500).json({
      code: 500,
      message: 'API密钥验证失败',
      timestamp: Date.now()
    });
  }
};

/**
 * 统一的身份验证中间件
 * 支持JWT和API密钥两种方式
 */
export const authMiddleware = (req: Request, res: Response, next: NextFunction) => {
  // 检查是否使用API密钥
  if (req.headers['x-api-key']) {
    return apiKeyAuth(req, res, next);
  }
  
  // 默认使用JWT验证
  return authenticate(req, res, next);
}; 