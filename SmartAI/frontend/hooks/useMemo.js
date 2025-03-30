import { useMemo, useRef, useEffect } from 'react';

/**
 * 增强版的useMemo，带有缓存功能
 *
 * 与React内置的useMemo类似，但增加了缓存命中统计和调试信息
 *
 * @template T
 * @param {() => T} factory - 创建值的工厂函数
 * @param {any[]} deps - 依赖数组，当这些值改变时，工厂函数将重新执行
 * @param {Object} [options] - 配置选项
 * @param {string} [options.key] - 缓存键，用于调试
 * @param {boolean} [options.debug] - 是否启用调试日志
 * @returns {T} 缓存的计算值
 */
export function useEnhancedMemo(factory, deps, options = {}) {
  const { key = 'cache', debug = false } = options;

  // 使用React的useMemo进行基本的缓存
  const value = useMemo(factory, deps);

  // 用于跟踪缓存命中和计算次数的引用
  const statsRef = useRef({
    computeCount: 0,
    hitCount: 0,
    lastUpdateTime: Date.now(),
  });

  // 用于判断是否为缓存命中
  const prevDepsRef = useRef([]);

  useEffect(() => {
    // 增加计算次数
    statsRef.current.computeCount++;

    // 检查是否为缓存命中
    const isHit =
      prevDepsRef.current.length > 0 &&
      deps.length === prevDepsRef.current.length &&
      deps.every((dep, i) => Object.is(dep, prevDepsRef.current[i]));

    if (isHit) {
      statsRef.current.hitCount++;
    }

    // 更新上次更新时间
    statsRef.current.lastUpdateTime = Date.now();

    // 保存当前依赖数组以便下次比较
    prevDepsRef.current = [...deps];

    // 输出调试信息
    if (debug) {
      console.log(`[EnhancedMemo:${key}] ${isHit ? '缓存命中' : '重新计算'}`);
      console.log(`[EnhancedMemo:${key}] 统计:`, {
        计算次数: statsRef.current.computeCount,
        命中次数: statsRef.current.hitCount,
        命中率: `${Math.round((statsRef.current.hitCount / statsRef.current.computeCount) * 100)}%`,
        最后更新: new Date(statsRef.current.lastUpdateTime).toLocaleTimeString(),
      });
    }
  }, [key, debug, ...deps]);

  return value;
}

/**
 * 使用存储在localStorage中的持久缓存的值
 *
 * @template T
 * @param {string} key - 缓存的键
 * @param {T} initialValue - 初始值
 * @param {Object} [options] - 配置选项
 * @param {boolean} [options.debug] - 是否启用调试日志
 * @param {number} [options.ttl] - 缓存生存时间(毫秒)，默认为1小时
 * @returns {[T, (value: T) => void]} 缓存的值和更新函数
 */
export function usePersistentCache(key, initialValue, options = {}) {
  const { debug = false, ttl = 60 * 60 * 1000 } = options;

  // 检查localStorage中是否有缓存，如果有且未过期则使用它
  const getInitialValue = () => {
    try {
      const item = window.localStorage.getItem(key);
      if (item) {
        const parsed = JSON.parse(item);

        // 检查缓存是否过期
        if (parsed.timestamp && Date.now() - parsed.timestamp < ttl) {
          if (debug) {
            console.log(
              `[PersistentCache:${key}] 使用缓存数据，缓存时间: ${new Date(parsed.timestamp).toLocaleTimeString()}`
            );
          }
          return parsed.value;
        } else {
          if (debug) {
            console.log(`[PersistentCache:${key}] 缓存已过期，使用初始值`);
          }
          // 缓存已过期，删除它
          window.localStorage.removeItem(key);
        }
      }
    } catch (error) {
      console.error(`[PersistentCache:${key}] 读取缓存出错:`, error);
    }

    return initialValue;
  };

  // 使用ref存储值，避免不必要的重渲染
  const valueRef = useRef(typeof window !== 'undefined' ? getInitialValue() : initialValue);

  // 更新缓存和当前值
  const setValue = value => {
    try {
      // 创建包含值和时间戳的对象
      const item = {
        value,
        timestamp: Date.now(),
      };

      // 保存到localStorage
      window.localStorage.setItem(key, JSON.stringify(item));

      // 更新ref
      valueRef.current = value;

      if (debug) {
        console.log(`[PersistentCache:${key}] 更新缓存数据`);
      }
    } catch (error) {
      console.error(`[PersistentCache:${key}] 保存缓存出错:`, error);
    }
  };

  // 初始化客户端缓存
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // 在客户端侧重新检查缓存，避免SSR问题
      valueRef.current = getInitialValue();
    }
  }, [key]);

  return [valueRef.current, setValue];
}

/**
 * 使用防抖计算值的hook
 *
 * @template T
 * @param {() => T} factory - 创建值的工厂函数
 * @param {any[]} deps - 依赖数组
 * @param {Object} [options] - 配置选项
 * @param {number} [options.delay=300] - 防抖延迟(毫秒)
 * @param {boolean} [options.debug] - 是否启用调试日志
 * @returns {T} 防抖后的计算值
 */
export function useDebouncedMemo(factory, deps, options = {}) {
  const { delay = 300, debug = false } = options;

  // 使用ref存储当前值
  const valueRef = useRef(factory());

  useEffect(() => {
    if (debug) {
      console.log('[DebouncedMemo] 依赖变化，启动防抖计算');
    }

    // 设置防抖定时器
    const timer = setTimeout(() => {
      if (debug) {
        console.log('[DebouncedMemo] 执行防抖计算');
      }

      valueRef.current = factory();
    }, delay);

    // 清理定时器
    return () => clearTimeout(timer);
  }, [...deps, delay]);

  return valueRef.current;
}

export default {
  useEnhancedMemo,
  usePersistentCache,
  useDebouncedMemo,
};
