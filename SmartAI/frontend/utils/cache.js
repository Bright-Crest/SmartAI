/**
 * 缓存工具
 * 提供内存缓存和本地存储缓存的实现
 */

// 内存缓存映射
export const memCache = new Map();

/**
 * 内存缓存类
 */
class MemoryCache {
  /**
   * 设置缓存
   *
   * @param {string} key - 缓存键
   * @param {any} value - 缓存值
   * @param {number} ttl - 过期时间（毫秒），默认为1小时
   */
  set(key, value, ttl = 60 * 60 * 1000) {
    const now = Date.now();
    const item = {
      value,
      expiry: now + ttl,
    };
    memCache.set(key, item);
  }

  /**
   * 获取缓存
   *
   * @param {string} key - 缓存键
   * @returns {any} 缓存值，如果缓存不存在或已过期则返回null
   */
  get(key) {
    const item = memCache.get(key);

    if (!item) {
      return null;
    }

    const now = Date.now();

    if (now > item.expiry) {
      this.delete(key);
      return null;
    }

    return item.value;
  }

  /**
   * 删除缓存
   *
   * @param {string} key - 缓存键
   * @returns {boolean} 是否成功删除
   */
  delete(key) {
    return memCache.delete(key);
  }

  /**
   * 清空所有缓存
   */
  clear() {
    memCache.clear();
  }

  /**
   * 获取缓存内容的键列表
   *
   * @returns {Array<string>} 缓存键数组
   */
  keys() {
    return Array.from(memCache.keys());
  }
}

/**
 * 浏览器存储缓存类
 */
class StorageCache {
  /**
   * 构造函数
   *
   * @param {Storage} storage - 存储对象，默认为localStorage
   * @param {string} prefix - 缓存键前缀
   */
  constructor(storage = null, prefix = 'cache_') {
    this.storage = typeof window !== 'undefined' ? storage || window.localStorage : null;
    this.prefix = prefix;

    // 清理过期的缓存项
    if (this.storage) {
      this.clearExpired();
    }
  }

  /**
   * 设置缓存
   *
   * @param {string} key - 缓存键
   * @param {any} value - 缓存值
   * @param {number} ttl - 过期时间（毫秒），默认为1天
   */
  set(key, value, ttl = 24 * 60 * 60 * 1000) {
    if (!this.storage) {
      return;
    }

    const now = Date.now();
    const prefixedKey = this.prefix + key;
    const item = {
      value,
      expiry: now + ttl,
    };

    try {
      this.storage.setItem(prefixedKey, JSON.stringify(item));
    } catch (error) {
      console.error('存储缓存失败:', error);

      // 存储失败时尝试清理过期数据
      if (error.name === 'QuotaExceededError') {
        this.clearExpired();
        try {
          this.storage.setItem(prefixedKey, JSON.stringify(item));
        } catch (e) {
          // 如果还是失败，可能需要更激进的清理策略
          console.error('存储空间已满，无法缓存数据:', e);
        }
      }
    }
  }

  /**
   * 获取缓存
   *
   * @param {string} key - 缓存键
   * @returns {any} 缓存值，如果缓存不存在或已过期则返回null
   */
  get(key) {
    if (!this.storage) {
      return null;
    }

    const prefixedKey = this.prefix + key;
    const itemStr = this.storage.getItem(prefixedKey);

    if (!itemStr) {
      return null;
    }

    try {
      const item = JSON.parse(itemStr);
      const now = Date.now();

      if (now > item.expiry) {
        this.delete(key);
        return null;
      }

      return item.value;
    } catch (error) {
      console.error('解析缓存数据失败:', error);
      return null;
    }
  }

  /**
   * 删除缓存
   *
   * @param {string} key - 缓存键
   */
  delete(key) {
    if (!this.storage) {
      return;
    }

    const prefixedKey = this.prefix + key;
    this.storage.removeItem(prefixedKey);
  }

  /**
   * 清空所有缓存
   */
  clear() {
    if (!this.storage) {
      return;
    }

    for (let i = this.storage.length - 1; i >= 0; i--) {
      const key = this.storage.key(i);
      if (key && key.startsWith(this.prefix)) {
        this.storage.removeItem(key);
      }
    }
  }

  /**
   * 清理所有过期的缓存
   */
  clearExpired() {
    if (!this.storage) {
      return;
    }

    const now = Date.now();

    for (let i = this.storage.length - 1; i >= 0; i--) {
      const key = this.storage.key(i);

      if (key && key.startsWith(this.prefix)) {
        try {
          const itemStr = this.storage.getItem(key);
          const item = JSON.parse(itemStr);

          if (now > item.expiry) {
            this.storage.removeItem(key);
          }
        } catch (error) {
          // 如果解析失败，最好删除这个可能损坏的项
          this.storage.removeItem(key);
        }
      }
    }
  }
}

// 导出实例
export const memoryCache = new MemoryCache();
export const storageCache = typeof window !== 'undefined' ? new StorageCache() : null;

/**
 * 使用缓存包装异步函数
 *
 * @param {Function} fn - 要包装的异步函数
 * @param {Object} options - 选项
 * @param {string} options.cacheKeyPrefix - 缓存键前缀
 * @param {number} options.ttl - 缓存有效期（毫秒）
 * @param {boolean} options.useStorage - 是否使用持久化存储
 * @returns {Function} 包装后的函数
 */
export function withCache(fn, options = {}) {
  const {
    cacheKeyPrefix = '',
    ttl = 5 * 60 * 1000, // 默认5分钟
    useStorage = false,
  } = options;

  return async function cachedFn(...args) {
    // 根据函数名和参数生成缓存键
    const key = `${cacheKeyPrefix}:${JSON.stringify(args)}`;

    // 选择缓存实现
    const cache = useStorage ? storageCache : memoryCache;

    // 尝试从缓存获取
    const cachedResult = cache.get(key);
    if (cachedResult !== null) {
      return cachedResult;
    }

    // 执行原始函数
    const startTime = Date.now();
    const result = await fn(...args);
    const endTime = Date.now();

    // 记录执行时间
    if (process.env.NODE_ENV === 'development') {
      console.log(`函数 ${fn.name || '匿名函数'} 执行耗时: ${endTime - startTime}ms`);
    }

    // 缓存结果
    cache.set(key, result, ttl);

    return result;
  };
}

export default {
  memCache: memoryCache,
  storageCache,
  withCache,
};
