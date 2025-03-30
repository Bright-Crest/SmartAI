import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import cookieParser from 'cookie-parser';
import rateLimit from 'express-rate-limit';
import { logger } from '../utils/logger';

/**
 * 配置Express应用
 * @param app Express应用实例
 */
export const configureExpress = (app: express.Application): void => {
  // 安全头
  app.use(helmet());
  
  // CORS配置
  const corsOptions = {
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
    credentials: true,
    maxAge: 86400 // 24小时
  };
  app.use(cors(corsOptions));
  
  // 请求体解析
  app.use(express.json({ limit: '2mb' }));
  app.use(express.urlencoded({ extended: true, limit: '2mb' }));
  
  // Cookie解析
  app.use(cookieParser());
  
  // 响应压缩
  app.use(compression());
  
  // 日志
  app.use(morgan('combined', { 
    stream: { 
      write: (message: string) => logger.info(message.trim()) 
    } 
  }));
  
  // 全局请求限流
  const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15分钟
    limit: 300, // 每个IP最多300个请求
    standardHeaders: true,
    legacyHeaders: false,
    message: {
      code: 429,
      message: '请求过于频繁，请稍后再试',
      timestamp: Date.now()
    }
  });
  app.use(limiter);
  
  // 信任代理
  app.set('trust proxy', 1);
  
  // 设置安全的HTTP头
  app.disable('x-powered-by');
  
  // JSON格式化
  app.set('json spaces', 2);
  
  // 错误处理中间件
  app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
    logger.error('应用错误', {
      error: err.message,
      stack: err.stack,
      path: req.path,
      method: req.method,
      ip: req.ip
    });
    
    res.status(500).json({
      code: 500,
      message: '服务器内部错误',
      requestId: req.headers['x-request-id'] || '',
      timestamp: Date.now()
    });
  });
};

/**
 * 配置API响应格式
 */
export const configureResponseFormat = (app: express.Application): void => {
  // 添加统一的响应方法
  app.use((req: express.Request, res: any, next: express.NextFunction) => {
    // 成功响应
    res.success = (data: any = null, message: string = '操作成功') => {
      return res.json({
        code: 200,
        message,
        data,
        requestId: req.headers['x-request-id'] || '',
        timestamp: Date.now()
      });
    };
    
    // 失败响应
    res.fail = (code: number = 400, message: string = '操作失败', data: any = null) => {
      return res.status(code).json({
        code,
        message,
        data,
        requestId: req.headers['x-request-id'] || '',
        timestamp: Date.now()
      });
    };
    
    next();
  });
}; 