import { NetworkNode, NetworkLink } from '../components/BlockchainNetworkTypes';

/**
 * 网络布局Worker服务
 * 封装WebWorker相关操作，提供高级接口
 */
export class NetworkLayoutWorker {
  private worker: Worker | null = null;
  private activePromises: Map<string, { 
    resolve: Function, 
    reject: Function,
    timeout: NodeJS.Timeout | null
  }> = new Map();

  /**
   * 初始化WebWorker
   */
  constructor() {
    this.createWorker();
  }

  /**
   * 创建Worker实例
   */
  private createWorker() {
    if (typeof window === 'undefined') return;

    try {
      // 使用new Worker()创建worker
      this.worker = new Worker(new URL('../workers/networkLayoutWorker.ts', import.meta.url), { type: 'module' });

      // 设置消息处理
      this.worker.onmessage = this.handleMessage.bind(this);
      this.worker.onerror = this.handleError.bind(this);
    } catch (err) {
      console.error('创建WebWorker失败:', err);
    }
  }

  /**
   * 处理Worker消息
   */
  private handleMessage(event: MessageEvent) {
    const { type, data, requestId } = event.data;

    // 如果有对应的请求ID，表示是对特定请求的响应
    if (requestId && this.activePromises.has(requestId)) {
      const { resolve, timeout } = this.activePromises.get(requestId)!;
      
      // 对于终结性的消息，清理资源
      if (type.includes('result') || type === 'error') {
        this.clearRequest(requestId);
      }
      
      // 解析Promise
      resolve({ type, data });
      return;
    }

    // 处理进度更新等非请求特定的消息
    if (type === 'layout-progress') {
      // 可以在这里添加全局进度处理逻辑
      return;
    }
  }

  /**
   * 处理Worker错误
   */
  private handleError(error: ErrorEvent) {
    console.error('Worker错误:', error);
    
    // 拒绝所有待处理的请求
    this.activePromises.forEach(({ reject }, requestId) => {
      reject(new Error(`Worker错误: ${error.message}`));
      this.clearRequest(requestId);
    });
    
    // 重新创建Worker
    this.terminateWorker();
    this.createWorker();
  }

  /**
   * 清理请求资源
   */
  private clearRequest(requestId: string) {
    if (this.activePromises.has(requestId)) {
      const { timeout } = this.activePromises.get(requestId)!;
      if (timeout) clearTimeout(timeout);
      this.activePromises.delete(requestId);
    }
  }

  /**
   * 终止Worker
   */
  public terminateWorker() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }

  /**
   * 发送消息到Worker并等待响应
   */
  private async sendMessage(type: string, data: any, options: {
    timeout?: number,
    responseType?: string
  } = {}): Promise<any> {
    if (!this.worker) {
      this.createWorker();
      if (!this.worker) {
        throw new Error('WebWorker不可用');
      }
    }

    const { timeout = 30000, responseType = `${type.split('-')[0]}-result` } = options;
    
    return new Promise((resolve, reject) => {
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      
      // 设置超时
      const timeoutId = setTimeout(() => {
        this.clearRequest(requestId);
        reject(new Error(`请求超时: ${type}`));
      }, timeout);
      
      // 存储Promise解析函数
      this.activePromises.set(requestId, { 
        resolve, 
        reject,
        timeout: timeoutId
      });
      
      // 发送消息，确保worker存在
      if (this.worker) {
        this.worker.postMessage({
          type,
          data,
          requestId
        });
      } else {
        // 如果worker仍然不可用，立即拒绝Promise
        this.clearRequest(requestId);
        reject(new Error('WebWorker不可用'));
      }
    });
  }

  /**
   * 计算网络布局
   */
  public async calculateLayout(
    nodes: NetworkNode[],
    links: NetworkLink[],
    width: number,
    height: number
  ): Promise<NetworkNode[]> {
    try {
      const result = await this.sendMessage('calculate-layout', {
        nodes,
        links,
        width,
        height
      });
      
      return result.data.nodes;
    } catch (error) {
      console.error('计算布局失败:', error);
      throw error;
    }
  }

  /**
   * 聚合节点
   */
  public async clusterNodes(
    nodes: NetworkNode[],
    links: NetworkLink[],
    distance: number
  ): Promise<{ nodes: NetworkNode[], links: NetworkLink[], clusterMap: Map<string, string> }> {
    try {
      const result = await this.sendMessage('cluster-nodes', {
        nodes,
        links,
        distance
      });
      
      return result.data;
    } catch (error) {
      console.error('节点聚合失败:', error);
      throw error;
    }
  }

  /**
   * 优化网络数据
   */
  public async optimizeNetwork(
    nodes: NetworkNode[],
    links: NetworkLink[],
    options: {
      isMobile?: boolean,
      maxNodes?: number,
      preserveGroups?: boolean,
      clustering?: boolean,
      clusterDistance?: number,
      simplifyAttributes?: boolean
    } = {}
  ): Promise<{ nodes: NetworkNode[], links: NetworkLink[] }> {
    try {
      const result = await this.sendMessage('optimize-network', {
        nodes,
        links,
        options
      });
      
      return result.data;
    } catch (error) {
      console.error('网络数据优化失败:', error);
      
      // 如果Worker处理失败，在主线程简单处理
      console.warn('使用主线程备用优化策略');
      
      // 简单的备用优化策略
      let optimizedNodes = [...nodes];
      let optimizedLinks = [...links];
      
      // 如果节点数量太多，只保留前N个
      const maxNodes = options.maxNodes || 300;
      if (nodes.length > maxNodes) {
        optimizedNodes = nodes.slice(0, maxNodes);
        const nodeIds = new Set(optimizedNodes.map(n => n.id));
        optimizedLinks = links.filter(link => {
          const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
          const targetId = typeof link.target === 'string' ? link.target : link.target.id;
          return nodeIds.has(sourceId) && nodeIds.has(targetId);
        });
      }
      
      return { nodes: optimizedNodes, links: optimizedLinks };
    }
  }
}

// 单例实例
let layoutWorkerInstance: NetworkLayoutWorker | null = null;

/**
 * 获取NetworkLayoutWorker单例
 */
export function getNetworkLayoutWorker(): NetworkLayoutWorker {
  if (!layoutWorkerInstance) {
    layoutWorkerInstance = new NetworkLayoutWorker();
  }
  return layoutWorkerInstance;
}

/**
 * 清理NetworkLayoutWorker实例
 */
export function cleanupNetworkLayoutWorker() {
  if (layoutWorkerInstance) {
    layoutWorkerInstance.terminateWorker();
    layoutWorkerInstance = null;
  }
} 