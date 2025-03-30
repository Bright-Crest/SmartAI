/**
 * 全局类型声明
 */

// 声明 worker-loader 模块
declare module '*.worker.ts' {
  // 您需要为您的 worker 文件添加适当的类型
  class WebpackWorker extends Worker {
    constructor();
  }

  // 导出默认 worker 构造函数
  export default WebpackWorker;
}

// 声明未提供类型的模块
declare module 'd3' {
  export * from '@types/d3';
} 