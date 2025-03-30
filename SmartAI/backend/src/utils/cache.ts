import NodeCache from 'node-cache';
import { logger } from './logger';

/**
 * 全局应用缓存
 * 用于优化数据访问性能
 */
class AppCache {
  private cache: NodeCache;
  
  constructor() {
    // 设置默认TTL为1小时
    this.cache = new NodeCache({
      stdTTL: 3600,
      checkperiod: 120
    });
    
    // 监听删除事件，用于日志记录
    this.cache.on('del', (key, value) => {
      logger.debug('缓存条目已过期', { key });
    });
    
    logger.info('缓存系统初始化完成');
  }
  
  /**
   * 设置缓存
   * @param key 缓存键名
   * @param value 缓存值
   * @param ttl 过期时间（秒），默认为1小时
   */
  set<T>(key: string, value: T, ttl: number = 3600): boolean {
    try {
      const result = this.cache.set(key, value, ttl);
      logger.debug('缓存已设置', { key, ttl });
      return result;
    } catch (error) {
      logger.error('设置缓存失败', { error, key });
      return false;
    }
  }
  
  /**
   * 获取缓存
   * @param key 缓存键名
   */
  get<T>(key: string): T | undefined {
    try {
      const value = this.cache.get<T>(key);
      if (value !== undefined) {
        logger.debug('缓存命中', { key });
      } else {
        logger.debug('缓存未命中', { key });
      }
      return value;
    } catch (error) {
      logger.error('获取缓存失败', { error, key });
      return undefined;
    }
  }
  
  /**
   * 删除缓存
   * @param key 缓存键名
   */
  del(key: string): number {
    try {
      const result = this.cache.del(key);
      if (result > 0) {
        logger.debug('缓存已删除', { key });
      }
      return result;
    } catch (error) {
      logger.error('删除缓存失败', { error, key });
      return 0;
    }
  }
  
  /**
   * 批量删除匹配模式的键
   * @param pattern 键名匹配模式
   */
  delStartWith(pattern: string): void {
    try {
      const keys = this.cache.keys().filter(key => key.startsWith(pattern));
      if (keys.length > 0) {
        this.cache.del(keys);
        logger.debug('批量删除缓存成功', { pattern, count: keys.length });
      }
    } catch (error) {
      logger.error('批量删除缓存失败', { error, pattern });
    }
  }
  
  /**
   * 检查缓存是否存在
   * @param key 缓存键名
   */
  has(key: string): boolean {
    return this.cache.has(key);
  }
  
  /**
   * 清空所有缓存
   */
  flush(): void {
    try {
      this.cache.flushAll();
      logger.info('缓存已清空');
    } catch (error) {
      logger.error('清空缓存失败', { error });
    }
  }
  
  /**
   * 获取所有键名
   */
  keys(): string[] {
    return this.cache.keys();
  }
  
  /**
   * 获取缓存统计信息
   */
  getStats(): NodeCache.Stats {
    return this.cache.getStats();
  }
  
  /**
   * 通过缓存获取数据，如果缓存不存在则通过工厂函数获取并缓存
   * @param key 缓存键名
   * @param factory 工厂函数，用于在缓存不存在时生成数据
   * @param ttl 缓存过期时间（秒）
   */
  async getOrSet<T>(key: string, factory: () => Promise<T>, ttl: number = 3600): Promise<T> {
    // 尝试从缓存获取
    const cached = this.get<T>(key);
    if (cached !== undefined) {
      return cached;
    }
    
    // 缓存不存在，通过工厂函数获取
    try {
      const value = await factory();
      // 缓存结果
      this.set(key, value, ttl);
      return value;
    } catch (error) {
      logger.error('通过工厂函数获取数据失败', { error, key });
      throw error;
    }
  }
}

// 导出单例
export const cache = new AppCache(); 