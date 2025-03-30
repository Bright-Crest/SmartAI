/**
 * 图片加载相关工具函数
 * 实现图片懒加载、预加载和优化等功能
 */

// 跟踪待加载和已加载的图片
const imageCache = new Map<string, boolean>();
const loadingPromises = new Map<string, Promise<HTMLImageElement>>();

/**
 * 预加载单张图片
 * @param src 图片URL
 * @returns Promise，解析为已加载的图片元素
 */
export function preloadImage(src: string): Promise<HTMLImageElement> {
  // 如果图片已经加载过，直接返回
  if (imageCache.get(src)) {
    return Promise.resolve(new Image());
  }
  
  // 如果正在加载，返回已有Promise
  if (loadingPromises.has(src)) {
    return loadingPromises.get(src)!;
  }
  
  // 创建新的加载Promise
  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    
    img.onload = () => {
      imageCache.set(src, true);
      loadingPromises.delete(src);
      resolve(img);
    };
    
    img.onerror = () => {
      loadingPromises.delete(src);
      reject(new Error(`Failed to load image: ${src}`));
    };
    
    img.src = src;
  });
  
  loadingPromises.set(src, promise);
  return promise;
}

/**
 * 批量预加载图片
 * @param srcs 图片URL数组
 * @param concurrencyLimit 并发加载数量限制
 * @returns Promise，在所有图片加载完成后解析
 */
export async function preloadImages(srcs: string[], concurrencyLimit = 4): Promise<void> {
  // 过滤掉已加载的图片
  const filteredSrcs = srcs.filter(src => !imageCache.get(src));
  
  if (filteredSrcs.length === 0) return;
  
  // 使用并发限制加载图片
  const batches = Math.ceil(filteredSrcs.length / concurrencyLimit);
  
  for (let i = 0; i < batches; i++) {
    const startIdx = i * concurrencyLimit;
    const endIdx = Math.min(startIdx + concurrencyLimit, filteredSrcs.length);
    const batch = filteredSrcs.slice(startIdx, endIdx);
    
    // 并发加载当前批次
    await Promise.all(batch.map(src => preloadImage(src).catch(err => {
      console.warn('图片预加载失败:', err);
    })));
  }
}

/**
 * 优先级图片加载
 * 先加载低分辨率版本，然后加载高分辨率版本
 * @param lowResSrc 低分辨率图片URL
 * @param highResSrc 高分辨率图片URL
 * @param onLowResLoaded 低分辨率图片加载完成的回调
 * @param onHighResLoaded 高分辨率图片加载完成的回调
 */
export function loadWithPriority(
  lowResSrc: string,
  highResSrc: string,
  onLowResLoaded?: (img: HTMLImageElement) => void,
  onHighResLoaded?: (img: HTMLImageElement) => void
): void {
  // 先加载低分辨率版本
  preloadImage(lowResSrc)
    .then(img => {
      onLowResLoaded?.(img);
      
      // 然后加载高分辨率版本
      return preloadImage(highResSrc);
    })
    .then(img => {
      onHighResLoaded?.(img);
    })
    .catch(err => {
      console.error('加载优先级图片失败:', err);
    });
}

/**
 * 当网络空闲时进行资源预加载
 * @param srcs 需要预加载的资源URL数组
 */
export function preloadOnNetworkIdle(srcs: string[]): void {
  if (typeof window === 'undefined') return;
  
  // 检测网络是否空闲
  if ('connection' in navigator && (navigator as any).connection.saveData) {
    // 如果用户启用了省流量模式，跳过预加载
    return;
  }
  
  // 使用requestIdleCallback等待浏览器空闲
  const requestIdleCallback = 
    (window as any).requestIdleCallback || 
    ((cb: Function) => setTimeout(cb, 1000));
  
  requestIdleCallback(() => {
    preloadImages(srcs).catch(err => {
      console.warn('网络空闲时预加载失败:', err);
    });
  });
}

/**
 * 生成适合当前设备分辨率的响应式图片URL
 * @param src 原始图片URL
 * @param width 所需宽度
 * @returns 适合当前设备的图片URL
 */
export function getResponsiveImageUrl(src: string, width?: number): string {
  if (!src) return '';
  
  // 如果图片URL已经包含尺寸参数，或者是外部URL，直接返回
  if (src.includes('?w=') || src.includes('http') || src.includes('data:')) {
    return src;
  }
  
  // 计算设备DPR
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  
  // 根据DPR和请求宽度生成合适尺寸的图片
  const calculatedWidth = width ? Math.round(width * dpr) : undefined;
  const query = calculatedWidth ? `?w=${calculatedWidth}` : '';
  
  return `${src}${query}`;
}

/**
 * 判断浏览器是否支持WebP格式
 * @returns 是否支持WebP的Promise
 */
export async function supportsWebP(): Promise<boolean> {
  if (typeof window === 'undefined') return false;
  
  // 使用缓存结果
  if ((window as any)._supportsWebP !== undefined) {
    return (window as any)._supportsWebP;
  }
  
  try {
    // 检查是否原生支持
    if (typeof self !== 'undefined' && 'createImageBitmap' in self) {
      const webpData = 'data:image/webp;base64,UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAfQ//73v/+BiOh/AAA=';
      const blob = await fetch(webpData).then(r => r.blob());
      const result = await createImageBitmap(blob).then(() => true, () => false);
      (window as any)._supportsWebP = result;
      return result;
    }
    
    // 回退检测方法
    const elem = document.createElement('canvas');
    if (elem.getContext && elem.getContext('2d')) {
      const result = elem.toDataURL('image/webp').indexOf('data:image/webp') === 0;
      (window as any)._supportsWebP = result;
      return result;
    }
    
    return false;
  } catch (e) {
    return false;
  }
}

/**
 * 清除图片缓存
 * @param src 如果提供则清除特定图片的缓存，否则清除所有缓存
 */
export function clearImageCache(src?: string): void {
  if (src) {
    imageCache.delete(src);
    loadingPromises.delete(src);
  } else {
    imageCache.clear();
    loadingPromises.clear();
  }
} 