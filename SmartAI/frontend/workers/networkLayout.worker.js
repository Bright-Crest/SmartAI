/**
 * 网络布局计算Web Worker
 * 用于处理计算密集型的图布局任务，避免阻塞主线程
 */

// 导入d3-force布局算法 (Worker环境中使用importScripts)
// 注意：在实际部署中，需要确保d3.js在public目录下可访问
importScripts('/d3.min.js');

// 节点和链接的缓存
let nodesCache = [];
let linksCache = [];
let width = 800;
let height = 600;
let simulation = null;

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

      default:
        console.warn('未知的消息类型:', type);
    }
  } catch (error) {
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
function calculateForceLayout(nodes, links, width, height) {
  // 停止之前的模拟
  if (simulation) {
    simulation.stop();
  }

  // 创建新的模拟
  simulation = d3
    .forceSimulation(nodes)
    .force(
      'link',
      d3
        .forceLink(links)
        .id(d => d.id)
        .distance(100)
    )
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('x', d3.forceX(width / 2).strength(0.1))
    .force('y', d3.forceY(height / 2).strength(0.1))
    .force(
      'collision',
      d3.forceCollide().radius(d => 10 + (d.value || 0) / 10)
    )
    .alphaDecay(0.01); // 减缓模拟冷却速度，提升布局质量

  // 发送初始布局
  self.postMessage({
    type: 'layout-result',
    data: {
      nodes: simulation.nodes(),
    },
  });

  // 每次tick后发送更新
  simulation.on('tick', () => {
    // 减少通信频率，只在特定迭代发送
    if (simulation.alpha() < 0.4 && Math.random() < 0.1) {
      self.postMessage({
        type: 'layout-result',
        data: {
          nodes: simulation.nodes(),
        },
      });
    }
  });

  // 完成后发送最终布局
  simulation.on('end', () => {
    self.postMessage({
      type: 'layout-result',
      data: {
        nodes: simulation.nodes(),
        finished: true,
      },
    });
  });

  // 运行多次迭代
  simulation.tick(300);
}

/**
 * 聚合节点算法
 * @param {Array} originalNodes 原始节点数组
 * @param {Array} originalLinks 原始链接数组
 * @param {number} distance 聚合距离
 * @returns {Object} 聚合后的节点和链接
 */
function clusterNodes(originalNodes, originalLinks, distance) {
  const clusteredNodes = [];
  const nodeClusters = new Map();
  const clusters = new Map();

  // 第一步：根据距离聚合节点
  originalNodes.forEach(node => {
    // 查找最近的现有簇
    let nearestCluster = null;
    let minDistance = Infinity;

    clusteredNodes.forEach(cluster => {
      if (cluster.isCluster) {
        const dist = Math.sqrt(
          Math.pow((node.x || 0) - (cluster.x || 0), 2) +
            Math.pow((node.y || 0) - (cluster.y || 0), 2)
        );

        if (dist < minDistance && dist < distance) {
          minDistance = dist;
          nearestCluster = cluster.id;
        }
      }
    });

    // 如果找到近的簇，添加到该簇
    if (nearestCluster) {
      nodeClusters.set(node.id, nearestCluster);
      const cluster = clusters.get(nearestCluster);
      cluster.clusterSize = (cluster.clusterSize || 0) + 1;
    } else {
      // 否则创建新簇
      const clusterId = `cluster-${clusteredNodes.length}`;
      const clusterNode = {
        ...node,
        id: clusterId,
        isCluster: true,
        clusterSize: 1,
        clusteredNodes: [node],
      };

      clusteredNodes.push(clusterNode);
      clusters.set(clusterId, clusterNode);
      nodeClusters.set(node.id, clusterId);
    }
  });

  // 第二步：处理链接
  const clusteredLinks = [];
  const linkMap = new Map();

  originalLinks.forEach(link => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;

    const sourceClusterId = nodeClusters.get(sourceId);
    const targetClusterId = nodeClusters.get(targetId);

    if (sourceClusterId && targetClusterId) {
      // 如果源和目标在同一个簇内，跳过
      if (sourceClusterId === targetClusterId) return;

      // 创建簇间连接
      const linkId = `${sourceClusterId}-${targetClusterId}`;

      if (linkMap.has(linkId)) {
        // 增加现有链接的权重
        const existingLink = linkMap.get(linkId);
        existingLink.value += link.value;
      } else {
        // 创建新的簇间链接
        const clusterLink = {
          source: sourceClusterId,
          target: targetClusterId,
          value: link.value,
        };

        clusteredLinks.push(clusterLink);
        linkMap.set(linkId, clusterLink);
      }
    }
  });

  return { clusteredNodes, clusteredLinks };
}

/**
 * 分层聚类算法，支持多级聚合
 * @param {Array} nodes 节点数组
 * @param {Array} links 链接数组
 * @param {number} levels 聚合的层数
 * @returns {Object} 分层聚合的结果
 */
function hierarchicalClustering(nodes, links, levels = 3) {
  // 实现分层聚类算法
  // 这里是一个简化的实现，实际项目中可能需要更复杂的算法
  let result = { nodes, links };

  for (let i = 0; i < levels; i++) {
    // 每一层使用不同的聚合距离
    const distance = 100 * (i + 1);
    result = clusterNodes(result.clusteredNodes || nodes, result.clusteredLinks || links, distance);
  }

  return result;
}
