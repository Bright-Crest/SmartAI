/**
 * Web Worker服务
 * 封装网络布局计算的Web Worker操作
 */

import { useState, useEffect, useCallback } from 'react';
import { NetworkNode, NetworkLink, LayoutResult, ClusteringResult } from '../components/BlockchainNetworkTypes';
// 导入Worker (webpack worker-loader会处理这个导入)
import NetworkLayoutWorker from './networkLayoutWorker';

// Worker响应类型
type WorkerResponse = {
  type: 'layout-result';
  data: LayoutResult;
} | {
  type: 'clustering-result';
  data: ClusteringResult;
} | {
  type: 'error';
  error: string;
};

// 布局选项
interface LayoutOptions {
  width: number;
  height: number;
  forceStrength?: number;
  linkDistance?: number;
  iterations?: number;
}

// 聚类选项
interface ClusterOptions {
  distance: number;
  minNodesForCluster?: number;
}

/**
 * Worker服务钩子
 * 提供对网络布局计算Worker的访问
 */
export function useWorkerService() {
  // 存储Worker实例
  const [worker, setWorker] = useState<Worker | null>(null);
  
  // 创建Worker
  useEffect(() => {
    // 检查浏览器是否支持Web Workers
    if (typeof window !== 'undefined' && 'Worker' in window) {
      try {
        // 创建新Worker
        const layoutWorker = new NetworkLayoutWorker();
        setWorker(layoutWorker);
        
        // 组件卸载时清理
        return () => {
          layoutWorker.terminate();
        };
      } catch (error) {
        console.error('创建Web Worker失败:', error);
      }
    } else {
      console.warn('当前浏览器不支持Web Workers，将在主线程执行计算');
    }
  }, []);
  
  /**
   * 计算网络布局
   * 使用力导向算法计算节点位置
   */
  const calculateLayout = useCallback(
    async (nodes: NetworkNode[], links: NetworkLink[], options: LayoutOptions): Promise<LayoutResult> => {
      if (!worker) {
        // 如果Worker不可用，返回原始节点
        console.warn('Worker不可用，跳过布局计算');
        return { nodes, finished: true, progress: 1 };
      }
      
      return new Promise<LayoutResult>((resolve, reject) => {
        // 创建一次性消息处理器
        const handleMessage = (event: MessageEvent) => {
          const response = event.data as WorkerResponse;
          
          if (response.type === 'error') {
            reject(new Error(response.error));
            worker.removeEventListener('message', handleMessage);
            return;
          }
          
          if (response.type === 'layout-result') {
            resolve(response.data);
            worker.removeEventListener('message', handleMessage);
          }
        };
        
        // 添加消息监听器
        worker.addEventListener('message', handleMessage);
        
        // 发送布局计算请求
        worker.postMessage({
          type: 'calculate-layout',
          nodes,
          links,
          options
        });
      });
    },
    [worker]
  );
  
  /**
   * 聚类节点
   * 将相近节点分组为聚合节点
   */
  const clusterNodes = useCallback(
    async (nodes: NetworkNode[], links: NetworkLink[], options: ClusterOptions): Promise<ClusteringResult> => {
      if (!worker) {
        // 如果Worker不可用，返回原始节点
        console.warn('Worker不可用，跳过节点聚类');
        return { nodes, links };
      }
      
      return new Promise<ClusteringResult>((resolve, reject) => {
        // 创建一次性消息处理器
        const handleMessage = (event: MessageEvent) => {
          const response = event.data as WorkerResponse;
          
          if (response.type === 'error') {
            reject(new Error(response.error));
            worker.removeEventListener('message', handleMessage);
            return;
          }
          
          if (response.type === 'clustering-result') {
            resolve(response.data);
            worker.removeEventListener('message', handleMessage);
          }
        };
        
        // 添加消息监听器
        worker.addEventListener('message', handleMessage);
        
        // 发送聚类请求
        worker.postMessage({
          type: 'cluster-nodes',
          nodes,
          links,
          options
        });
      });
    },
    [worker]
  );
  
  /**
   * 优化网络数据
   * 处理大型网络，减少渲染压力
   */
  const optimizeNetworkData = useCallback(
    (nodes: NetworkNode[], links: NetworkLink[], maxNodes = 1000): { nodes: NetworkNode[], links: NetworkLink[] } => {
      // 如果节点数量在合理范围内，不需要优化
      if (nodes.length <= maxNodes) {
        return { nodes, links };
      }
      
      // 按权重/重要性排序节点
      const sortedNodes = [...nodes].sort((a, b) => {
        // 优先保留有更多连接的节点
        const aWeight = a.weight || 1;
        const bWeight = b.weight || 1;
        return bWeight - aWeight;
      });
      
      // 只保留最重要的节点
      const keptNodes = sortedNodes.slice(0, maxNodes);
      const keptNodeIds = new Set(keptNodes.map(n => n.id));
      
      // 只保留连接保留节点的链接
      const filteredLinks = links.filter(link => {
        const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
        const targetId = typeof link.target === 'string' ? link.target : link.target.id;
        return keptNodeIds.has(sourceId) && keptNodeIds.has(targetId);
      });
      
      return { nodes: keptNodes, links: filteredLinks };
    },
    []
  );
  
  /**
   * 终止Worker
   */
  const terminate = useCallback(() => {
    if (worker) {
      worker.terminate();
      setWorker(null);
    }
  }, [worker]);
  
  // 返回Worker服务API
  return {
    calculateLayout,
    clusterNodes,
    optimizeNetworkData,
    terminate,
    isWorkerAvailable: !!worker
  };
}
