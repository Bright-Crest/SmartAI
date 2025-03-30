/**
 * 网络布局计算WebWorker
 * 处理计算密集型任务，如力导向布局计算和节点聚类
 */

import { NetworkNode, NetworkLink, LayoutResult, ClusteringResult } from '../components/BlockchainNetworkTypes';

// 布局计算选项
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

// 消息类型
type WorkerMessage = {
  type: 'calculate-layout';
  nodes: NetworkNode[];
  links: NetworkLink[];
  options: LayoutOptions;
} | {
  type: 'cluster-nodes';
  nodes: NetworkNode[];
  links: NetworkLink[];
  options: ClusterOptions;
};

// 响应类型
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

/**
 * 力导向布局算法实现
 * 基于弹簧-电荷模型
 */
function calculateForceDirectedLayout(
  nodes: NetworkNode[],
  links: NetworkLink[],
  options: LayoutOptions
): LayoutResult {
  const { width, height, forceStrength = 0.3, linkDistance = 50, iterations = 100 } = options;
  
  // 如果没有节点或节点已有坐标就直接返回
  if (nodes.length === 0) {
    return { nodes: [], finished: true, progress: 1 };
  }
  
  // 复制节点以避免修改原始数据
  const workingNodes = nodes.map(node => ({
    ...node,
    // 如果节点没有初始位置，随机分配一个
    x: node.x !== undefined ? node.x : Math.random() * width,
    y: node.y !== undefined ? node.y : Math.random() * height,
    // 为布局计算添加速度向量
    vx: 0,
    vy: 0
  }));
  
  // 处理链接，确保source和target是对象引用
  const workingLinks = links.map(link => {
    // 如果链接端点是字符串ID，转换为节点引用
    const sourceNode = typeof link.source === 'string' 
      ? workingNodes.find(n => n.id === link.source) 
      : link.source;
      
    const targetNode = typeof link.target === 'string' 
      ? workingNodes.find(n => n.id === link.target) 
      : link.target;
    
    if (!sourceNode || !targetNode) {
      console.warn(`链接 ${link.id || '未知'} 包含无效的节点引用`);
      return null;
    }
    
    return {
      ...link,
      source: sourceNode,
      target: targetNode,
      // 链接强度/距离
      distance: link.value ? linkDistance / link.value : linkDistance
    };
  }).filter(Boolean) as NetworkLink[];
  
  // 布局迭代
  for (let i = 0; i < iterations; i++) {
    // 计算斥力 (节点间)
    for (let a = 0; a < workingNodes.length; a++) {
      for (let b = a + 1; b < workingNodes.length; b++) {
        const nodeA = workingNodes[a];
        const nodeB = workingNodes[b];
        
        if (nodeA.x === undefined || nodeA.y === undefined || nodeB.x === undefined || nodeB.y === undefined) {
          continue;
        }
        
        // 计算节点间距离和方向
        const dx = nodeB.x - nodeA.x;
        const dy = nodeB.y - nodeA.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        
        // 避免距离太小导致的数值不稳定
        if (distance < 1) continue;
        
        // 计算斥力（反比于距离平方）
        const repulsiveForce = forceStrength * 1000 / (distance * distance);
        
        // 考虑节点大小权重
        const weightA = nodeA.weight || 1;
        const weightB = nodeB.weight || 1;
        
        // 应用到速度向量
        const fx = (dx / distance) * repulsiveForce;
        const fy = (dy / distance) * repulsiveForce;
        
        (nodeA as any).vx -= fx * weightB;
        (nodeA as any).vy -= fy * weightB;
        (nodeB as any).vx += fx * weightA;
        (nodeB as any).vy += fy * weightA;
      }
    }
    
    // 计算引力 (链接)
    for (const link of workingLinks) {
      const sourceNode = link.source as NetworkNode;
      const targetNode = link.target as NetworkNode;
      
      if (sourceNode.x === undefined || sourceNode.y === undefined || 
          targetNode.x === undefined || targetNode.y === undefined) {
        continue;
      }
      
      // 计算链接两端节点的距离和方向
      const dx = targetNode.x - sourceNode.x;
      const dy = targetNode.y - sourceNode.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1;
      
      // 计算与目标距离的差距
      const targetDistance = (link as any).distance || linkDistance;
      const displacement = (distance - targetDistance) / distance;
      
      // 链接权重
      const linkStrength = link.value || 1;
      
      // 应用弹簧力
      const fx = dx * displacement * 0.1 * linkStrength;
      const fy = dy * displacement * 0.1 * linkStrength;
      
      (sourceNode as any).vx += fx;
      (sourceNode as any).vy += fy;
      (targetNode as any).vx -= fx;
      (targetNode as any).vy -= fy;
    }
    
    // 应用速度并加入阻尼
    for (const node of workingNodes) {
      if (node.x === undefined || node.y === undefined) continue;
      
      // 如果节点位置被固定，跳过
      if (node.fx !== undefined && node.fy !== undefined) continue;
      
      // 应用速度，加入阻尼
      const damping = 0.8;
      const vx = (node as any).vx * damping;
      const vy = (node as any).vy * damping;
      
      node.x += vx;
      node.y += vy;
      
      // 限制在边界内
      node.x = Math.max(10, Math.min(width - 10, node.x));
      node.y = Math.max(10, Math.min(height - 10, node.y));
      
      // 重置速度
      (node as any).vx = 0;
      (node as any).vy = 0;
    }
    
    // 发送进度更新（每10次迭代）
    if (i % 10 === 0 && i < iterations - 1) {
      // 注意：这只是为了展示进度，实际实现时我们在一次消息中完成所有迭代
    }
  }
  
  // 清理临时速度属性，只返回必要的位置信息
  return {
    nodes: workingNodes.map(node => ({
      ...node,
      vx: undefined,
      vy: undefined
    })),
    finished: true,
    progress: 1
  };
}

/**
 * 基于距离的节点聚类算法
 * 将距离接近的节点聚合成单个节点
 */
function clusterNodesByDistance(
  nodes: NetworkNode[],
  links: NetworkLink[],
  options: ClusterOptions
): ClusteringResult {
  const { distance, minNodesForCluster = 3 } = options;
  
  if (nodes.length === 0) {
    return { nodes: [], links: [] };
  }
  
  // 复制节点和链接
  let workingNodes = [...nodes];
  let workingLinks = [...links];
  
  // 确保所有节点都有坐标
  const nodesWithCoords = workingNodes.filter(
    node => node.x !== undefined && node.y !== undefined
  );
  
  if (nodesWithCoords.length < 2) {
    return { nodes: workingNodes, links: workingLinks };
  }
  
  // 获取每个节点的邻居，基于连接
  const neighbors = new Map<string, Set<string>>();
  
  // 初始化邻居集合
  workingNodes.forEach(node => {
    neighbors.set(node.id, new Set<string>());
  });
  
  // 填充邻居信息
  workingLinks.forEach(link => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;
    
    neighbors.get(sourceId)?.add(targetId);
    neighbors.get(targetId)?.add(sourceId);
  });
  
  // 计算节点间距离并构建潜在聚类
  const clusters: NetworkNode[][] = [];
  const processedNodes = new Set<string>();
  
  // 对每个节点寻找潜在聚类
  for (const node of nodesWithCoords) {
    if (processedNodes.has(node.id)) continue;
    
    const cluster: NetworkNode[] = [node];
    processedNodes.add(node.id);
    
    // 寻找聚类中的其他节点
    for (const otherNode of nodesWithCoords) {
      if (processedNodes.has(otherNode.id) || node.id === otherNode.id) continue;
      
      // 计算欧几里得距离
      const dx = (node.x || 0) - (otherNode.x || 0);
      const dy = (node.y || 0) - (otherNode.y || 0);
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // 如果节点距离足够近且同类型，加入聚类
      if (dist <= distance && node.group === otherNode.group) {
        cluster.push(otherNode);
        processedNodes.add(otherNode.id);
      }
    }
    
    // 只保存有足够节点的聚类
    if (cluster.length >= minNodesForCluster) {
      clusters.push(cluster);
    } else {
      // 单个节点或太小的聚类，标记为未处理以便后续单独处理
      processedNodes.delete(node.id);
    }
  }
  
  // 创建聚类节点和更新链接
  const clusterMap = new Map<string, string>(); // 原始节点ID到聚类ID
  const resultNodes: NetworkNode[] = [];
  
  // 先添加聚类节点
  clusters.forEach((cluster, idx) => {
    // 计算聚类中心点
    let centerX = 0;
    let centerY = 0;
    
    cluster.forEach(node => {
      centerX += node.x || 0;
      centerY += node.y || 0;
    });
    
    centerX /= cluster.length;
    centerY /= cluster.length;
    
    // 创建聚类节点
    const clusterId = `cluster-${idx}`;
    const clusterNode: NetworkNode = {
      id: clusterId,
      x: centerX,
      y: centerY,
      isCluster: true,
      clusterSize: cluster.length,
      clusteredNodes: cluster,
      // 继承第一个节点的组和风险等级
      group: cluster[0].group,
      riskLevel: cluster[0].riskLevel,
      // 标签显示聚类大小
      label: `聚合: ${cluster.length}个节点`
    };
    
    resultNodes.push(clusterNode);
    
    // 更新映射关系
    cluster.forEach(node => {
      clusterMap.set(node.id, clusterId);
    });
  });
  
  // 添加未聚类的节点
  workingNodes.forEach(node => {
    if (!processedNodes.has(node.id)) {
      resultNodes.push({...node});
    }
  });
  
  // 更新链接
  const resultLinks: NetworkLink[] = [];
  const processedLinks = new Set<string>();
  
  workingLinks.forEach(link => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;
    
    // 如果源节点和目标节点都在同一聚类中，跳过此链接
    if (clusterMap.has(sourceId) && clusterMap.has(targetId) && 
        clusterMap.get(sourceId) === clusterMap.get(targetId)) {
      return;
    }
    
    // 确定新的源和目标
    const newSourceId = clusterMap.get(sourceId) || sourceId;
    const newTargetId = clusterMap.get(targetId) || targetId;
    
    // 创建链接的唯一标识符
    const linkId = newSourceId < newTargetId 
      ? `${newSourceId}-${newTargetId}` 
      : `${newTargetId}-${newSourceId}`;
      
    // 跳过重复链接
    if (processedLinks.has(linkId)) return;
    processedLinks.add(linkId);
    
    // 添加新链接
    resultLinks.push({
      ...link,
      id: linkId,
      source: newSourceId,
      target: newTargetId
    });
  });
  
  return {
    nodes: resultNodes,
    links: resultLinks,
    clusterMap
  };
}

/**
 * 处理主线程发来的消息
 */
self.addEventListener('message', (event: MessageEvent<WorkerMessage>) => {
  try {
    const { type } = event.data;
    
    switch (type) {
      case 'calculate-layout': {
        const { nodes, links, options } = event.data;
        const result = calculateForceDirectedLayout(nodes, links, options);
        
        self.postMessage({
          type: 'layout-result',
          data: result
        } as WorkerResponse);
        break;
      }
      
      case 'cluster-nodes': {
        const { nodes, links, options } = event.data;
        const result = clusterNodesByDistance(nodes, links, options);
        
        self.postMessage({
          type: 'clustering-result',
          data: result
        } as WorkerResponse);
        break;
      }
      
      default:
        throw new Error(`未知消息类型: ${type}`);
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error instanceof Error ? error.message : String(error)
    } as WorkerResponse);
  }
});

// 导出为默认模块供worker-loader使用
export default {} as typeof Worker & { new(): Worker }; 