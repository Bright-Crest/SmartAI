import Redis from 'ioredis';
import NodeCache from 'node-cache';
import { logger } from '../utils/logger';

/**
 * 缓存模式
 */
export enum CacheMode {
  REDIS = 'redis',
  MEMORY = 'memory',
  NONE = 'none'
}

/**
 * 缓存选项
 */
export interface CacheOptions {
  ttl?: number; // 过期时间（秒）
  key?: string; // 自定义缓存键
  tags?: string[]; // 缓存标签，用于批量清除
}

/**
 * 统一的缓存服务
 * 支持Redis和内存缓存两种模式
 */
export class CacheService {
  private static instance: CacheService;
  private redisClient: Redis | null = null;
  private memoryCache: NodeCache;
  private mode: CacheMode = CacheMode.MEMORY;
  private defaultTTL: number = 300; // 默认5分钟
  private keyPrefix: string = 'smartai:';

  private constructor() {
    // 内存缓存初始化
    this.memoryCache = new NodeCache({
      stdTTL: this.defaultTTL,
      checkperiod: 120, // 每2分钟检查过期项
      useClones: false // 不使用深拷贝以提高性能
    });

    // 根据环境变量初始化缓存模式
    const cacheMode = process.env.CACHE_MODE || CacheMode.MEMORY;
    this.setMode(cacheMode as CacheMode);
  }

  /**
   * 获取缓存服务实例
   */
  public static getInstance(): CacheService {
    if (!CacheService.instance) {
      CacheService.instance = new CacheService();
    }
    return CacheService.instance;
  }

  /**
   * 设置缓存模式
   * @param mode 缓存模式
   */
  public setMode(mode: CacheMode): void {
    this.mode = mode;
    
    // 如果设置为Redis模式，初始化Redis连接
    if (mode === CacheMode.REDIS && !this.redisClient) {
      try {
        const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
        this.redisClient = new Redis(redisUrl);
        
        this.redisClient.on('connect', () => {
          logger.info('Redis连接成功');
        });
        
        this.redisClient.on('error', (err) => {
          logger.error('Redis连接错误', { error: err.message });
          // 回退到内存缓存
          this.mode = CacheMode.MEMORY;
        });
      } catch (error) {
        logger.error('Redis初始化失败', { 
          error: error instanceof Error ? error.message : String(error) 
        });
        // 回退到内存缓存
        this.mode = CacheMode.MEMORY;
      }
    }
  }

  /**
   * 获取缓存键
   * @param key 原始键
   * @returns 带前缀的缓存键
   */
  private getKey(key: string): string {
    return `${this.keyPrefix}${key}`;
  }

  /**
   * 设置缓存
   * @param key 缓存键
   * @param value 缓存值
   * @param options 缓存选项
   */
  public async set(key: string, value: any, options: CacheOptions = {}): Promise<void> {
    const cacheKey = this.getKey(key);
    const ttl = options.ttl || this.defaultTTL;
    
    try {
      if (this.mode === CacheMode.REDIS && this.redisClient) {
        // Redis缓存
        const valueStr = JSON.stringify(value);
        await this.redisClient.set(cacheKey, valueStr, 'EX', ttl);
        
        // 如果有标签，存储键与标签的关系
        if (options.tags && options.tags.length > 0) {
          for (const tag of options.tags) {
            const tagKey = this.getKey(`tag:${tag}`);
            await this.redisClient.sadd(tagKey, cacheKey);
          }
        }
      } else if (this.mode === CacheMode.MEMORY) {
        // 内存缓存
        this.memoryCache.set(cacheKey, value, ttl);
        
        // 如果有标签，存储键与标签的关系
        if (options.tags && options.tags.length > 0) {
          for (const tag of options.tags) {
            const tagKey = this.getKey(`tag:${tag}`);
            let tagSet = this.memoryCache.get<Set<string>>(tagKey);
            if (!tagSet) {
              tagSet = new Set<string>();
              this.memoryCache.set(tagKey, tagSet);
            }
            tagSet.add(cacheKey);
          }
        }
      }
    } catch (error) {
      logger.error('缓存写入失败', { 
        key: cacheKey, 
        error: error instanceof Error ? error.message : String(error) 
      });
    }
  }

  /**
   * 获取缓存
   * @param key 缓存键
   * @returns 缓存值，不存在返回null
   */
  public async get<T>(key: string): Promise<T | null> {
    if (this.mode === CacheMode.NONE) {
      return null;
    }
    
    const cacheKey = this.getKey(key);
    
    try {
      if (this.mode === CacheMode.REDIS && this.redisClient) {
        // Redis缓存
        const valueStr = await this.redisClient.get(cacheKey);
        if (!valueStr) {
          return null;
        }
        return JSON.parse(valueStr) as T;
      } else if (this.mode === CacheMode.MEMORY) {
        // 内存缓存
        const value = this.memoryCache.get<T>(cacheKey);
        return value !== undefined ? value : null;
      }
    } catch (error) {
      logger.error('缓存读取失败', { 
        key: cacheKey, 
        error: error instanceof Error ? error.message : String(error) 
      });
    }
    
    return null;
  }

  /**
   * 删除缓存
   * @param key 缓存键
   */
  public async delete(key: string): Promise<void> {
    const cacheKey = this.getKey(key);
    
    try {
      if (this.mode === CacheMode.REDIS && this.redisClient) {
        await this.redisClient.del(cacheKey);
      } else if (this.mode === CacheMode.MEMORY) {
        this.memoryCache.del(cacheKey);
      }
    } catch (error) {
      logger.error('缓存删除失败', { 
        key: cacheKey, 
        error: error instanceof Error ? error.message : String(error) 
      });
    }
  }

  /**
   * 按标签删除缓存
   * @param tag 缓存标签
   */
  public async deleteByTag(tag: string): Promise<void> {
    const tagKey = this.getKey(`tag:${tag}`);
    
    try {
      if (this.mode === CacheMode.REDIS && this.redisClient) {
        // 获取标签对应的所有缓存键
        const keys = await this.redisClient.smembers(tagKey);
        
        if (keys.length > 0) {
          // 删除所有缓存键
          await this.redisClient.del(...keys);
          // 删除标签集合
          await this.redisClient.del(tagKey);
        }
      } else if (this.mode === CacheMode.MEMORY) {
        const tagSet = this.memoryCache.get<Set<string>>(tagKey);
        if (tagSet) {
          // 删除所有缓存键
          tagSet.forEach(key => {
            this.memoryCache.del(key);
          });
          // 删除标签集合
          this.memoryCache.del(tagKey);
        }
      }
    } catch (error) {
      logger.error('按标签删除缓存失败', { 
        tag, 
        error: error instanceof Error ? error.message : String(error) 
      });
    }
  }

  /**
   * 清空所有缓存
   */
  public async flushAll(): Promise<void> {
    try {
      if (this.mode === CacheMode.REDIS && this.redisClient) {
        // 只清除指定前缀的键
        const keys = await this.redisClient.keys(`${this.keyPrefix}*`);
        if (keys.length > 0) {
          await this.redisClient.del(...keys);
        }
      } else if (this.mode === CacheMode.MEMORY) {
        this.memoryCache.flushAll();
      }
    } catch (error) {
      logger.error('清空缓存失败', { 
        error: error instanceof Error ? error.message : String(error) 
      });
    }
  }

  /**
   * 获取Redis客户端
   * @returns Redis客户端实例
   */
  public getRedisClient(): Redis | null {
    return this.redisClient;
  }

  /**
   * 获取内存缓存实例
   * @returns NodeCache实例
   */
  public getMemoryCache(): NodeCache {
    return this.memoryCache;
  }

  /**
   * 获取当前缓存模式
   * @returns 缓存模式
   */
  public getMode(): CacheMode {
    return this.mode;
  }
} 