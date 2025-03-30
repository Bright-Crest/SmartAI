/**
 * 前端性能分析工具
 * 用于监控前端性能指标并发送到分析服务
 */

// 是否启用分析功能
const isAnalyticsEnabled = process.env.NEXT_PUBLIC_ENABLE_ANALYTICS === 'true';

// 性能指标缓存
let perfMetrics = {};

/**
 * 获取网页性能指标
 * @returns {Object} 性能指标对象
 */
const getPerformanceMetrics = () => {
  if (typeof window === 'undefined' || !window.performance) {
    return null;
  }

  try {
    const navigationTiming = window.performance.timing;

    // 基本页面加载指标
    const metrics = {
      // 总页面加载时间
      pageLoadTime: navigationTiming.loadEventEnd - navigationTiming.navigationStart,

      // DNS解析时间
      dnsTime: navigationTiming.domainLookupEnd - navigationTiming.domainLookupStart,

      // TCP连接时间
      tcpTime: navigationTiming.connectEnd - navigationTiming.connectStart,

      // 服务器响应时间 (TTFB - Time To First Byte)
      serverResponseTime: navigationTiming.responseStart - navigationTiming.requestStart,

      // 页面下载时间
      downloadTime: navigationTiming.responseEnd - navigationTiming.responseStart,

      // DOM解析和渲染时间
      domProcessingTime: navigationTiming.domComplete - navigationTiming.domLoading,

      // 首次内容绘制时间 (如果浏览器支持)
      firstPaint: 0,

      // 用户可交互时间
      timeToInteractive: navigationTiming.domInteractive - navigationTiming.navigationStart,

      // 页面资源总数
      resourceCount: window.performance.getEntriesByType('resource').length,

      // 来源
      referrer: document.referrer || '',

      // 页面URL
      url: window.location.href,

      // 用户代理
      userAgent: window.navigator.userAgent,

      // 设备类型
      deviceType: getDeviceType(),

      // 屏幕分辨率
      screenResolution: `${window.screen.width}x${window.screen.height}`,

      // 时间戳
      timestamp: new Date().toISOString(),
    };

    // 尝试获取更高级的性能指标
    if (window.performance.getEntriesByType) {
      const paintMetrics = window.performance.getEntriesByType('paint');

      if (paintMetrics && paintMetrics.length) {
        // 查找首次内容绘制指标
        const firstPaint = paintMetrics.find(entry => entry.name === 'first-paint');
        const firstContentfulPaint = paintMetrics.find(
          entry => entry.name === 'first-contentful-paint'
        );

        if (firstPaint) {
          metrics.firstPaint = Math.round(firstPaint.startTime);
        }

        if (firstContentfulPaint) {
          metrics.firstContentfulPaint = Math.round(firstContentfulPaint.startTime);
        }
      }
    }

    return metrics;
  } catch (error) {
    console.error('获取性能指标失败:', error);
    return null;
  }
};

/**
 * 获取设备类型
 * @returns {string} 设备类型
 */
const getDeviceType = () => {
  const ua = navigator.userAgent;
  if (/Mobi|Android/i.test(ua)) {
    return 'mobile';
  } else if (/iPad|Tablet/i.test(ua)) {
    return 'tablet';
  }
  return 'desktop';
};

/**
 * 发送性能数据到分析服务
 * @param {Object} metrics - 性能指标数据
 */
const sendMetricsToAnalyticsService = metrics => {
  if (!isAnalyticsEnabled || !metrics) {
    return;
  }

  try {
    // 尝试发送给实际的分析服务
    // 这里可以使用fetch或者beacon API发送数据
    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/analytics/performance', JSON.stringify(metrics));
    } else {
      fetch('/api/analytics/performance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(metrics),
        // 使用keepalive确保数据在页面卸载时也能发送
        keepalive: true,
      }).catch(err => console.error('发送性能指标失败:', err));
    }

    // 仅开发环境下输出
    if (process.env.NODE_ENV === 'development') {
      console.log('性能指标:', metrics);
    }
  } catch (error) {
    console.error('发送性能指标失败:', error);
  }
};

/**
 * 记录组件渲染时间
 * @param {string} componentName - 组件名称
 * @param {number} startTime - 开始时间
 * @param {number} endTime - 结束时间
 */
export const recordComponentRender = (componentName, startTime, endTime) => {
  if (!isAnalyticsEnabled) {
    return;
  }

  if (!perfMetrics.components) {
    perfMetrics.components = {};
  }

  perfMetrics.components[componentName] = {
    renderTime: endTime - startTime,
    timestamp: new Date().toISOString(),
  };
};

/**
 * 记录API调用性能
 * @param {string} endpoint - API端点
 * @param {number} duration - 持续时间(毫秒)
 * @param {boolean} isSuccess - 是否成功
 * @param {boolean} wasCached - 是否使用了缓存
 */
export const recordApiCall = (endpoint, duration, isSuccess, wasCached) => {
  if (!isAnalyticsEnabled) {
    return;
  }

  if (!perfMetrics.apiCalls) {
    perfMetrics.apiCalls = [];
  }

  perfMetrics.apiCalls.push({
    endpoint,
    duration,
    isSuccess,
    wasCached,
    timestamp: new Date().toISOString(),
  });
};

/**
 * 初始化性能监控
 */
export const initPerformanceMonitoring = () => {
  if (typeof window === 'undefined' || !isAnalyticsEnabled) {
    return;
  }

  // 页面加载完成后收集性能指标
  window.addEventListener('load', () => {
    // 等待所有内容完全加载后再收集指标
    setTimeout(() => {
      const metrics = getPerformanceMetrics();
      if (metrics) {
        perfMetrics.pageLoad = metrics;
        sendMetricsToAnalyticsService(perfMetrics);
      }
    }, 1000);
  });

  // 页面卸载前发送最终指标
  window.addEventListener('beforeunload', () => {
    sendMetricsToAnalyticsService(perfMetrics);
  });

  // 跟踪当前路由
  let currentPath = window.location.pathname;

  // 在单页应用中监听路由变化
  if (typeof window.history !== 'undefined') {
    const originalPushState = window.history.pushState;
    window.history.pushState = function (...args) {
      // 发送当前页面的性能数据
      sendMetricsToAnalyticsService({
        ...perfMetrics,
        navigationType: 'pushState',
        fromPath: currentPath,
        toPath: args[2],
      });

      // 重置指标收集
      perfMetrics = {
        routeChange: {
          fromPath: currentPath,
          toPath: args[2],
          timestamp: new Date().toISOString(),
        },
      };

      currentPath = args[2];
      return originalPushState.apply(this, args);
    };
  }
};

/**
 * 用于包装组件以测量其渲染性能的高阶组件
 * @param {React.Component} Component - 要包装的组件
 * @param {string} componentName - 组件名称
 * @returns {React.Component} 包装后的组件
 */
export const withPerformanceTracking = (Component, componentName) => {
  if (!isAnalyticsEnabled) {
    return Component;
  }

  return function PerformanceTrackedComponent(props) {
    if (typeof window === 'undefined') {
      return <Component {...props} />;
    }

    const startTime = performance.now();
    const result = <Component {...props} />;
    const endTime = performance.now();

    // 在下一个微任务中记录性能，以避免影响渲染
    setTimeout(() => {
      recordComponentRender(componentName, startTime, endTime);
    }, 0);

    return result;
  };
};

export default {
  initPerformanceMonitoring,
  recordApiCall,
  recordComponentRender,
  withPerformanceTracking,
};
