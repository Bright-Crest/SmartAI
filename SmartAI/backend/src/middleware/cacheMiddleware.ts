import { Request, Response, NextFunction } from 'express';
import { CacheService, CacheOptions } from '../services/cacheService';
import { logger } from '../utils/logger';

/**
 * 缓存中间件配置选项
 */
export interface CacheMiddlewareOptions extends CacheOptions {
  enabled?: boolean; // 是否启用缓存
  condition?: (req: Request) => boolean; // 缓存条件
}

/**
 * 生成缓存键
 * @param req 请求对象
 * @returns 缓存键
 */
const generateCacheKey = (req: Request): string => {
  // 基础键：方法 + 路径
  let key = `${req.method}:${req.originalUrl}`;
  
  // 对于GET请求，加入查询参数
  if (req.method === 'GET') {
    const queryParams = new URLSearchParams(req.query as Record<string, string>);
    queryParams.sort(); // 排序参数，确保相同参数不同顺序生成相同的键
    const queryString = queryParams.toString();
    if (queryString) {
      key += `?${queryString}`;
    }
  } 
  // 对于POST请求，可以考虑加入请求体的哈希值
  else if (req.method === 'POST' && req.body) {
    // 为了避免缓存键过长，这里只使用一些关键字段
    const bodyHash = JSON.stringify(req.body)
      .split('')
      .reduce((hash, char) => ((hash << 5) - hash) + char.charCodeAt(0), 0)
      .toString(36);
    key += `:${bodyHash}`;
  }
  
  // 添加授权信息的哈希（如果有），确保不同用户有不同的缓存
  if (req.headers.authorization) {
    const authHash = req.headers.authorization
      .split('')
      .reduce((hash, char) => ((hash << 5) - hash) + char.charCodeAt(0), 0)
      .toString(36);
    key += `:auth=${authHash}`;
  }
  
  return key;
};

/**
 * API响应缓存中间件
 * @param options 缓存选项
 */
export const cacheMiddleware = (options: CacheMiddlewareOptions = {}) => {
  const cacheService = CacheService.getInstance();
  
  // 默认选项
  const defaultOptions: CacheMiddlewareOptions = {
    enabled: process.env.API_CACHE_ENABLED === 'true',
    ttl: parseInt(process.env.API_CACHE_TTL || '300'), // 默认5分钟
    condition: (req) => req.method === 'GET' // 默认只缓存GET请求
  };
  
  // 合并选项
  const mergedOptions = { ...defaultOptions, ...options };
  
  return async (req: Request, res: Response, next: NextFunction) => {
    // 跳过缓存的条件
    if (!mergedOptions.enabled || 
        (mergedOptions.condition && !mergedOptions.condition(req)) ||
        req.headers['cache-control'] === 'no-cache') {
      return next();
    }
    
    // 生成缓存键
    const cacheKey = options.key || generateCacheKey(req);
    
    try {
      // 尝试从缓存获取
      const cachedResponse = await cacheService.get<any>(cacheKey);
      
      if (cachedResponse) {
        // 添加缓存标记头
        res.setHeader('X-Cache', 'HIT');
        
        // 返回缓存的响应
        return res.status(cachedResponse.status)
          .set(cachedResponse.headers)
          .send(cachedResponse.data);
      }
      
      // 缓存未命中，标记头
      res.setHeader('X-Cache', 'MISS');
      
      // 拦截响应
      const originalSend = res.send;
      res.send = function(body: any): Response {
        // 恢复原始方法
        res.send = originalSend;
        
        // 只缓存成功的响应
        if (res.statusCode >= 200 && res.statusCode < 400) {
          const responseToCache = {
            status: res.statusCode,
            headers: res.getHeaders(),
            data: body
          };
          
          // 异步缓存，不阻塞响应
          cacheService.set(cacheKey, responseToCache, {
            ttl: mergedOptions.ttl,
            tags: mergedOptions.tags
          }).catch(err => {
            logger.error('响应缓存写入失败', { 
              key: cacheKey, 
              error: err instanceof Error ? err.message : String(err) 
            });
          });
        }
        
        // 调用原始方法发送响应
        return originalSend.call(res, body);
      };
      
      next();
    } catch (error) {
      logger.error('缓存中间件错误', { 
        error: error instanceof Error ? error.message : String(error) 
      });
      next();
    }
  };
};

/**
 * 清除指定API路由的缓存
 * @param pattern 路由模式（例如：'GET:/api/risk/*'）
 */
export const clearRouteCache = async (pattern: string): Promise<void> => {
  try {
    const cacheService = CacheService.getInstance();
    
    if (cacheService.getMode() === 'redis') {
      const redisClient = cacheService.getRedisClient();
      if (redisClient) {
        const keys = await redisClient.keys(`smartai:${pattern}*`);
        if (keys.length > 0) {
          await redisClient.del(...keys);
          logger.info(`已清除路由缓存: ${pattern}`, { count: keys.length });
        }
      }
    } else {
      // 内存缓存清除逻辑相对简单，标签系统提供了足够的功能
      // 所以这里不实现额外的模式匹配逻辑
      logger.warn('内存缓存模式不支持按路由模式清除缓存，请使用标签代替');
    }
  } catch (error) {
    logger.error('清除路由缓存失败', { 
      pattern, 
      error: error instanceof Error ? error.message : String(error) 
    });
  }
}; 