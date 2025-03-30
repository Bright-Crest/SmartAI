/**
 * 网络布局计算Web Worker
 * 用于处理计算密集型的图布局任务，避免阻塞主线程
 */

// 节点和链接的缓存
let nodesCache: any[] = [];
let linksCache: any[] = [];
let width = 800;
let height = 600;
let simulation: any = null;

// 处理来自主线程的消息
self.onmessage = function (event) {
  const { type, data } = event.data;

  try {
    switch (type) {
      case 'init':
        // 初始化缓存数据
        nodesCache = data.nodes;
        linksCache = data.links;
        width = data.width || 800;
        height = data.height || 600;
        break;

      case 'calculate-layout':
        // 计算力导向布局
        calculateForceLayout(data.nodes, data.links, data.width, data.height);
        break;

      case 'cluster-nodes':
        // 计算节点聚合
        const result = clusterNodes(data.nodes, data.links, data.distance);
        self.postMessage({
          type: 'clustering-result',
          data: result,
        });
        break;

      case 'optimize-network':
        // 根据设备类型和性能需求优化网络数据
        const optimizedData = optimizeNetworkData(
          data.nodes, 
          data.links, 
          data.options
        );
        self.postMessage({
          type: 'optimization-result',
          data: optimizedData
        });
        break;

      default:
        console.warn('未知的消息类型:', type);
    }
  } catch (error: any) {
    self.postMessage({
      type: 'error',
      data: {
        message: error.message,
        stack: error.stack,
      },
    });
  }
};

/**
 * 计算力导向布局
 * @param {Array} nodes 节点数组
 * @param {Array} links 链接数组
 * @param {number} width 画布宽度
 * @param {number} height 画布高度
 */
function calculateForceLayout(nodes: any[], links: any[], width: number, height: number) {
  // 在Worker环境中无法直接使用d3，这里实现简化版的力导向布局
  // 实际项目中可以使用importScripts引入d3库，或使用其他轻量级布局算法
  
  // 简化版力导向布局实现
  const nodeMap = new Map();
  nodes.forEach(node => {
    // 初始化位置（如果没有的话）
    if (typeof node.x !== 'number') node.x = Math.random() * width;
    if (typeof node.y !== 'number') node.y = Math.random() * height;
    
    nodeMap.set(node.id, node);
  });
  
  // 处理链接，确保source和target是对象引用
  const processedLinks = links.map(link => {
    const source = typeof link.source === 'string' ? nodeMap.get(link.source) : link.source;
    const target = typeof link.target === 'string' ? nodeMap.get(link.target) : link.target;
    return { ...link, source, target };
  });
  
  // 发送初始布局
  self.postMessage({
    type: 'layout-progress',
    data: {
      nodes: [...nodes],
      progress: 0
    }
  });
  
  // 简单的力导向布局迭代
  const iterations = 50;
  for (let i = 0; i < iterations; i++) {
    // 计算斥力（节点之间）
    nodes.forEach(node1 => {
      node1.fx = node1.fx || null; // 固定位置
      node1.fy = node1.fy || null;
      
      // 跳过固定位置的节点
      if (node1.fx !== null && node1.fy !== null) return;
      
      let forceX = 0;
      let forceY = 0;
      
      // 节点之间的斥力
      nodes.forEach(node2 => {
        if (node1.id === node2.id) return;
        
        const dx = node1.x - node2.x;
        const dy = node1.y - node2.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        
        // 斥力计算
        const repulsiveForce = 100 / distance;
        forceX += dx / distance * repulsiveForce;
        forceY += dy / distance * repulsiveForce;
      });
      
      // 应用力
      node1.x += forceX * 0.05;
      node1.y += forceY * 0.05;
      
      // 确保节点在画布内
      node1.x = Math.max(10, Math.min(width - 10, node1.x));
      node1.y = Math.max(10, Math.min(height - 10, node1.y));
    });
    
    // 计算引力（链接）
    processedLinks.forEach(link => {
      const source = link.source;
      const target = link.target;
      
      // 跳过固定位置的节点
      if ((source.fx !== null && source.fy !== null) && 
          (target.fx !== null && target.fy !== null)) return;
      
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1;
      
      // 引力计算
      const attractiveForce = 0.1 * distance;
      const fx = dx / distance * attractiveForce;
      const fy = dy / distance * attractiveForce;
      
      // 应用引力
      if (source.fx === null) {
        source.x += fx * 0.5;
        source.y += fy * 0.5;
      }
      
      if (target.fx === null) {
        target.x -= fx * 0.5;
        target.y -= fy * 0.5;
      }
    });
    
    // 定期发送进度更新
    if (i % 10 === 0 || i === iterations - 1) {
      self.postMessage({
        type: 'layout-progress',
        data: {
          nodes: [...nodes], // 克隆节点数组
          progress: (i + 1) / iterations
        }
      });
    }
  }
  
  // 发送最终结果
  self.postMessage({
    type: 'layout-result',
    data: {
      nodes: nodes,
      finished: true
    }
  });
}

/**
 * 聚合节点算法
 * @param {Array} originalNodes 原始节点数组
 * @param {Array} originalLinks 原始链接数组
 * @param {number} distance 聚合距离
 * @returns {Object} 聚合后的节点和链接
 */
function clusterNodes(originalNodes: any[], originalLinks: any[], distance: number) {
  const clusteredNodes: any[] = [];
  const nodeClusters = new Map();
  const clusters = new Map();

  // 按照Group分组，然后在每个组内进行距离聚合
  const nodesByGroup = new Map();
  
  originalNodes.forEach(node => {
    const group = node.group || 'default';
    if (!nodesByGroup.has(group)) {
      nodesByGroup.set(group, []);
    }
    nodesByGroup.get(group).push(node);
  });
  
  // 为每个组创建聚合
  nodesByGroup.forEach((groupNodes, group) => {
    // 如果组内节点少于阈值，就不聚合
    if (groupNodes.length < 3) {
      groupNodes.forEach(node => {
        clusteredNodes.push({...node});
        nodeClusters.set(node.id, node.id);
      });
      return;
    }
    
    // 基于距离聚合节点
    let clusterIndex = 0;
    const processedNodes = new Set();
    
    groupNodes.forEach(node => {
      if (processedNodes.has(node.id)) return;
      
      const nearbyNodes = groupNodes.filter(other => {
        if (other.id === node.id || processedNodes.has(other.id)) return false;
        
        const dist = Math.sqrt(
          Math.pow((node.x || 0) - (other.x || 0), 2) +
          Math.pow((node.y || 0) - (other.y || 0), 2)
        );
        
        return dist < distance;
      });
      
      // 如果有足够多的邻近节点，创建聚合
      if (nearbyNodes.length > 0) {
        const clusterId = `cluster-${group}-${clusterIndex++}`;
        const allClusterNodes = [node, ...nearbyNodes];
        
        // 计算聚合位置（重心）
        const centerX = allClusterNodes.reduce((sum, n) => sum + (n.x || 0), 0) / allClusterNodes.length;
        const centerY = allClusterNodes.reduce((sum, n) => sum + (n.y || 0), 0) / allClusterNodes.length;
        
        // 创建聚合节点
        const clusterNode = {
          id: clusterId,
          x: centerX,
          y: centerY,
          isCluster: true,
          group: group,
          clusterSize: allClusterNodes.length,
          clusteredNodes: allClusterNodes,
          // 使用最高风险级别作为聚合风险级别
          riskLevel: getHighestRiskLevel(allClusterNodes)
        };
        
        clusteredNodes.push(clusterNode);
        clusters.set(clusterId, clusterNode);
        
        // 记录每个节点所属的聚合
        allClusterNodes.forEach(n => {
          nodeClusters.set(n.id, clusterId);
          processedNodes.add(n.id);
        });
      }
    });
    
    // 添加未处理的节点（未被聚合）
    groupNodes.forEach(node => {
      if (!processedNodes.has(node.id)) {
        clusteredNodes.push({...node});
        nodeClusters.set(node.id, node.id);
      }
    });
  });

  // 处理链接
  const clusteredLinks: any[] = [];
  const linkMap = new Map();

  originalLinks.forEach(link => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;

    const sourceClusterId = nodeClusters.get(sourceId);
    const targetClusterId = nodeClusters.get(targetId);

    if (sourceClusterId && targetClusterId) {
      // 如果源和目标在同一个聚合内，跳过
      if (sourceClusterId === targetClusterId) return;

      // 创建聚合间连接
      const linkId = `${sourceClusterId}-${targetClusterId}`;

      if (linkMap.has(linkId)) {
        // 增加现有链接的权重
        const existingLink = linkMap.get(linkId);
        existingLink.value = (existingLink.value || 1) + (link.value || 1);
        existingLink.width = Math.min(10, 1 + Math.log(existingLink.value));
      } else {
        // 创建新的聚合间链接
        const clusterLink = {
          id: linkId,
          source: sourceClusterId,
          target: targetClusterId,
          value: link.value || 1,
          width: link.width || 1
        };

        clusteredLinks.push(clusterLink);
        linkMap.set(linkId, clusterLink);
      }
    }
  });

  return { 
    nodes: clusteredNodes, 
    links: clusteredLinks,
    clusterMap: nodeClusters
  };
}

/**
 * 优化网络数据
 * @param {Array} nodes 节点数组
 * @param {Array} links 链接数组
 * @param {Object} options 优化选项
 * @returns {Object} 优化后的数据
 */
function optimizeNetworkData(nodes: any[], links: any[], options: any = {}) {
  const {
    isMobile = false,
    maxNodes = isMobile ? 100 : 300,
    preserveGroups = true,
    clustering = true,
    clusterDistance = isMobile ? 50 : 100,
    simplifyAttributes = true
  } = options;
  
  // 如果节点数量小于阈值，只做简单优化
  if (nodes.length <= maxNodes) {
    return simplifyData(nodes, links, simplifyAttributes);
  }
  
  // 重要节点识别 - 根据连接数和权重计算节点重要性
  const nodeImportance = new Map();
  nodes.forEach(node => {
    let importance = node.weight || 0;
    
    // 添加连接数量作为重要性指标
    links.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      
      if (sourceId === node.id || targetId === node.id) {
        importance += 1;
      }
    });
    
    // 风险级别会增加重要性
    if (node.riskLevel === 'critical') importance += 10;
    else if (node.riskLevel === 'high') importance += 5;
    else if (node.riskLevel === 'medium') importance += 2;
    
    nodeImportance.set(node.id, importance);
  });
  
  // 按组保留节点
  let optimizedNodes: any[] = [];
  let includedNodeIds: Set<string>;
  
  if (preserveGroups) {
    // 按组分类
    const nodesByGroup = new Map();
    nodes.forEach(node => {
      const group = node.group || 'default';
      if (!nodesByGroup.has(group)) {
        nodesByGroup.set(group, []);
      }
      nodesByGroup.get(group).push(node);
    });
    
    // 从每个组中选取最重要的节点
    includedNodeIds = new Set();
    
    nodesByGroup.forEach((groupNodes, group) => {
      // 按重要性排序
      const sortedNodes = [...groupNodes].sort((a, b) => 
        (nodeImportance.get(b.id) || 0) - (nodeImportance.get(a.id) || 0)
      );
      
      // 计算每个组应保留的节点数量比例
      const groupRatio = groupNodes.length / nodes.length;
      const nodesToKeep = Math.max(5, Math.ceil(maxNodes * groupRatio));
      
      // 保留重要节点
      sortedNodes.slice(0, nodesToKeep).forEach(node => {
        optimizedNodes.push(node);
        includedNodeIds.add(node.id);
      });
    });
  } else {
    // 不按组，直接选择最重要的节点
    const sortedNodes = [...nodes].sort((a, b) => 
      (nodeImportance.get(b.id) || 0) - (nodeImportance.get(a.id) || 0)
    );
    
    optimizedNodes = sortedNodes.slice(0, maxNodes);
    includedNodeIds = new Set(optimizedNodes.map(node => node.id));
  }
  
  // 保留连接重要节点的边
  const optimizedLinks = links.filter(link => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;
    
    return includedNodeIds.has(sourceId) && includedNodeIds.has(targetId);
  });
  
  // 对结果应用聚合
  if (clustering && optimizedNodes.length > 20) {
    const { nodes: clusteredNodes, links: clusteredLinks } = clusterNodes(
      optimizedNodes, 
      optimizedLinks, 
      clusterDistance
    );
    
    return simplifyData(clusteredNodes, clusteredLinks, simplifyAttributes);
  }
  
  return simplifyData(optimizedNodes, optimizedLinks, simplifyAttributes);
}

/**
 * 简化数据 - 移除不必要的属性
 */
function simplifyData(nodes: any[], links: any[], simplifyAttributes = true) {
  if (!simplifyAttributes) {
    return { nodes, links };
  }
  
  // 简化节点属性
  const simplifiedNodes = nodes.map(node => {
    const essential = {
      id: node.id,
      x: node.x,
      y: node.y,
      group: node.group,
      label: node.label,
      color: node.color,
      value: node.value,
      weight: node.weight,
      riskLevel: node.riskLevel,
    };
    
    // 保留聚合相关属性
    if (node.isCluster) {
      return {
        ...essential,
        isCluster: true,
        clusterSize: node.clusterSize,
      };
    }
    
    return essential;
  });
  
  // 简化链接属性
  const simplifiedLinks = links.map(link => ({
    id: link.id,
    source: link.source,
    target: link.target,
    value: link.value,
    width: link.width,
    color: link.color,
  }));
  
  return { nodes: simplifiedNodes, links: simplifiedLinks };
}

/**
 * 获取节点集合中的最高风险级别
 */
function getHighestRiskLevel(nodes: any[]): string {
  const riskLevels = ['low', 'medium', 'high', 'critical'];
  let highestIndex = -1;
  
  nodes.forEach(node => {
    if (node.riskLevel) {
      const index = riskLevels.indexOf(node.riskLevel);
      if (index > highestIndex) {
        highestIndex = index;
      }
    }
  });
  
  return highestIndex >= 0 ? riskLevels[highestIndex] : 'low';
}

export {}; 