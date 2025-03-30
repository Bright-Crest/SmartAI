/**
 * @file 日志工具
 * @description 定义日志记录器接口和实现
 */

import winston from 'winston';
import { format } from 'winston';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import fs from 'fs';

export interface LoggerContext {
  traceId?: string;
  [key: string]: any;
}

export interface LoggerConfig {
  level?: string;
  format?: 'json' | 'text';
  timestampFormat?: string;
}

export interface ILogger {
  info(message: string, context?: Record<string, unknown>): void;
  error(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  debug(message: string, context?: Record<string, unknown>): void;
  getTraceId(): string;
  setTraceId(traceId: string): void;
  setDefaultContext(context: Record<string, unknown>): void;
}

export class Logger implements ILogger {
  protected traceId: string;
  private defaultContext: Record<string, unknown>;

  constructor() {
    this.traceId = '';
    this.defaultContext = {};
  }

  info(message: string, context?: Record<string, unknown>): void {
    console.log(`[INFO] ${this.formatMessage(message)}`, { ...this.defaultContext, ...context });
  }

  error(message: string, context?: Record<string, unknown>): void {
    console.error(`[ERROR] ${this.formatMessage(message)}`, { ...this.defaultContext, ...context });
  }

  warn(message: string, context?: Record<string, unknown>): void {
    console.warn(`[WARN] ${this.formatMessage(message)}`, { ...this.defaultContext, ...context });
  }

  debug(message: string, context?: Record<string, unknown>): void {
    console.debug(`[DEBUG] ${this.formatMessage(message)}`, { ...this.defaultContext, ...context });
  }

  getTraceId(): string {
    return this.traceId;
  }

  setTraceId(traceId: string): void {
    this.traceId = traceId;
  }

  setDefaultContext(context: Record<string, unknown>): void {
    this.defaultContext = context;
  }

  private formatMessage(message: string): string {
    return this.traceId ? `[${this.traceId}] ${message}` : message;
  }
}

export const logger = new Logger();

export class WinstonLogger implements ILogger {
  private logger: any;
  private traceId: string;
  private defaultContext: Record<string, unknown>;

  constructor() {
    this.logger = console;
    this.traceId = '';
    this.defaultContext = {};
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.logger.info(this.formatMessage(message), { ...this.defaultContext, ...context });
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.logger.error(this.formatMessage(message), { ...this.defaultContext, ...context });
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.logger.warn(this.formatMessage(message), { ...this.defaultContext, ...context });
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.logger.debug(this.formatMessage(message), { ...this.defaultContext, ...context });
  }

  getTraceId(): string {
    return this.traceId;
  }

  setTraceId(traceId: string): void {
    this.traceId = traceId;
  }

  setDefaultContext(context: Record<string, unknown>): void {
    this.defaultContext = context;
  }

  private formatMessage(message: string): string {
    return this.traceId ? `[${this.traceId}] ${message}` : message;
  }
}

export const createLogger = (config: LoggerConfig): ILogger => {
  return new Logger();
};

interface LogEntry {
  level: string;
  message: string;
  timestamp: string;
  [key: string]: unknown;
}

// 确保日志目录存在
const logDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir);
}

// 定义日志格式
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.splat(),
  winston.format.json()
);

// 创建自定义格式用于控制台输出
const consoleFormat = winston.format.printf(({ level, message, timestamp, ...meta }) => {
  const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
  return `${timestamp} [${level.toUpperCase()}]: ${message}${metaStr}`;
});

// 创建日志器
export const loggerWinston = winston.createLogger({
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  format: logFormat,
  defaultMeta: { service: 'smart-money-api' },
  transports: [
    // 写入所有日志到 combined.log
    new winston.transports.File({ 
      filename: path.join(logDir, 'combined.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    // 写入所有错误日志到 error.log
    new winston.transports.File({ 
      filename: path.join(logDir, 'error.log'),
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    // 控制台输出
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        consoleFormat
      )
    })
  ],
  // 退出时不结束所有事件循环以允许日志记录完成
  exitOnError: false
});

// 捕获未处理的异常并记录日志
process.on('uncaughtException', (error) => {
  loggerWinston.error('未捕获的异常', { error });
  // 给日志一些时间写入后退出
  setTimeout(() => {
    process.exit(1);
  }, 1000);
});

// 捕获未处理的promise拒绝并记录日志
process.on('unhandledRejection', (reason, promise) => {
  loggerWinston.error('未处理的Promise拒绝', { reason, promise });
});

// 创建HTTP请求记录中间件
export const httpLogger = {
  // Express中间件
  expressMiddleware: (req: any, res: any, next: any) => {
    const start = Date.now();
    
    // 请求完成时记录
    res.on('finish', () => {
      const duration = Date.now() - start;
      const logLevel = res.statusCode >= 400 ? 'error' : 'info';
      
      loggerWinston.log(logLevel, 'HTTP请求', {
        method: req.method,
        url: req.originalUrl || req.url,
        status: res.statusCode,
        duration: `${duration}ms`,
        ip: req.ip,
        userAgent: req.get('user-agent')
      });
    });
    
    next();
  }
};
