import { logger } from '../utils/logger';
import { AddressProfileDAO } from '../database/dao/AddressProfileDAO';
import { TransactionDAO } from '../database/dao/TransactionDAO';

/**
 * 图分析结果接口
 */
export interface GraphAnalysisResult {
  centrality: number;
  degree: number;
  clustering: number;
  riskPaths: Array<{
    from: string;
    to: string;
    weight: number;
  }>;
  riskNodes: Array<{
    address: string;
    riskScore: number;
    distance: number;
  }>;
}

/**
 * 图分析器
 * 负责分析地址关系网络和风险传播路径
 */
export class GraphAnalyzer {
  /**
   * 分析地址之间的关系和风险传播
   * @param sourceAddress 源地址
   * @param targetAddress 目标地址（可选）
   * @param depth 分析深度（默认为2）
   * @returns 图分析结果
   */
  static async analyze(
    sourceAddress: string,
    targetAddress?: string,
    depth: number = 2
  ): Promise<GraphAnalysisResult> {
    try {
      logger.info('开始关系图分析', {
        sourceAddress,
        targetAddress: targetAddress || 'N/A',
        depth,
      });

      // 获取地址关系网络
      const network = await this.buildAddressNetwork(sourceAddress, depth);
      
      // 计算中心度
      const centrality = this.calculateCentrality(sourceAddress, network);
      
      // 计算度数 (直接连接数)
      const degree = this.calculateDegree(sourceAddress, network);
      
      // 计算聚类系数
      const clustering = this.calculateClustering(sourceAddress, network);
      
      // 查找风险路径
      const riskPaths = await this.findRiskPaths(sourceAddress, targetAddress, network);
      
      // 识别风险节点
      const riskNodes = await this.identifyRiskNodes(network);

      logger.info('关系图分析完成', {
        sourceAddress,
        networkSize: Object.keys(network).length,
        riskPathsCount: riskPaths.length,
        riskNodesCount: riskNodes.length,
      });

      return {
        centrality,
        degree,
        clustering,
        riskPaths,
        riskNodes,
      };
    } catch (error) {
      logger.error('关系图分析失败', {
        sourceAddress,
        error: error instanceof Error ? error.message : String(error),
      });
      
      return {
        centrality: 0,
        degree: 0,
        clustering: 0,
        riskPaths: [],
        riskNodes: [],
      };
    }
  }
  
  /**
   * 构建地址关系网络
   * @param sourceAddress 源地址
   * @param depth 分析深度
   * @returns 地址关系网络
   */
  private static async buildAddressNetwork(
    sourceAddress: string,
    depth: number
  ): Promise<Record<string, Array<{ address: string; weight: number }>>> {
    // 初始化网络
    const network: Record<string, Array<{ address: string; weight: number }>> = {};
    const visited = new Set<string>();
    const queue: Array<{ address: string; level: number }> = [];
    
    // 添加源地址
    queue.push({ address: sourceAddress, level: 0 });
    visited.add(sourceAddress);
    
    // BFS遍历
    while (queue.length > 0) {
      const { address, level } = queue.shift()!;
      
      // 初始化当前地址的连接
      if (!network[address]) {
        network[address] = [];
      }
      
      // 达到最大深度则停止扩展
      if (level >= depth) continue;
      
      try {
        // 获取与当前地址有交易的地址
        const transactions = await TransactionDAO.findByAddress(address, 50);
        
        // 处理每个交易
        for (const tx of transactions) {
          const otherAddress = tx.from === address ? tx.to : tx.from;
          
          // 添加到网络
          const txWeight = this.calculateTransactionWeight(tx);
          
          // 更新连接权重 (如果已存在，取最大权重)
          const existingConnection = network[address].find(conn => conn.address === otherAddress);
          if (existingConnection) {
            existingConnection.weight = Math.max(existingConnection.weight, txWeight);
          } else {
            network[address].push({ address: otherAddress, weight: txWeight });
          }
          
          // 双向连接
          if (!network[otherAddress]) {
            network[otherAddress] = [];
          }
          
          const reverseConnection = network[otherAddress].find(conn => conn.address === address);
          if (reverseConnection) {
            reverseConnection.weight = Math.max(reverseConnection.weight, txWeight);
          } else {
            network[otherAddress].push({ address, weight: txWeight });
          }
          
          // 添加到队列（如果未访问过）
          if (!visited.has(otherAddress)) {
            visited.add(otherAddress);
            queue.push({ address: otherAddress, level: level + 1 });
          }
        }
      } catch (error) {
        logger.warn('获取交易失败', {
          address,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    
    return network;
  }
  
  /**
   * 计算交易权重
   * 根据交易金额、时间等因素计算交易的重要性权重
   */
  private static calculateTransactionWeight(tx: any): number {
    // 简单实现：基于交易金额和时间计算权重
    const baseWeight = 0.5;
    
    // 根据交易金额增加权重 (假设tx.value是字符串形式的wei值)
    let amountWeight = 0;
    if (tx.value) {
      try {
        const value = BigInt(tx.value);
        // 大额交易 (>10 ETH)
        if (value > BigInt('10000000000000000000')) {
          amountWeight = 0.3;
        } 
        // 中等金额 (>1 ETH)
        else if (value > BigInt('1000000000000000000')) {
          amountWeight = 0.2;
        }
        // 小额交易
        else {
          amountWeight = 0.1;
        }
      } catch (e) {
        amountWeight = 0.1;
      }
    }
    
    // 根据交易时间增加权重 (近期交易更重要)
    let timeWeight = 0;
    if (tx.timestamp) {
      const txTime = new Date(tx.timestamp * 1000);
      const now = new Date();
      const daysDiff = (now.getTime() - txTime.getTime()) / (1000 * 60 * 60 * 24);
      
      if (daysDiff < 1) {
        timeWeight = 0.3; // 24小时内
      } else if (daysDiff < 7) {
        timeWeight = 0.2; // 一周内
      } else if (daysDiff < 30) {
        timeWeight = 0.1; // 一个月内
      } else {
        timeWeight = 0.05; // 更早
      }
    }
    
    return baseWeight + amountWeight + timeWeight;
  }
  
  /**
   * 计算地址中心度
   * @param address 目标地址
   * @param network 地址关系网络
   * @returns 中心度得分 (0-1)
   */
  private static calculateCentrality(
    address: string,
    network: Record<string, Array<{ address: string; weight: number }>>
  ): number {
    const connections = network[address] || [];
    if (connections.length === 0) return 0;
    
    // 计算一阶连接总权重
    const directWeight = connections.reduce((sum, conn) => sum + conn.weight, 0);
    
    // 计算二阶连接总数
    let secondOrderConnections = 0;
    for (const conn of connections) {
      const secondConnections = network[conn.address] || [];
      secondOrderConnections += secondConnections.length;
    }
    
    // 网络规模
    const networkSize = Object.keys(network).length;
    if (networkSize <= 1) return 0;
    
    // 综合计算中心度 (考虑直接连接权重和二阶连接数量)
    const directScore = directWeight / connections.length;
    const secondOrderScore = secondOrderConnections / (networkSize * 2);
    
    // 加权平均
    return directScore * 0.7 + secondOrderScore * 0.3;
  }
  
  /**
   * 计算地址度数 (直接连接数)
   * @param address 目标地址
   * @param network 地址关系网络
   * @returns 度数
   */
  private static calculateDegree(
    address: string,
    network: Record<string, Array<{ address: string; weight: number }>>
  ): number {
    return (network[address] || []).length;
  }
  
  /**
   * 计算聚类系数
   * 衡量地址连接的紧密程度
   * @param address 目标地址
   * @param network 地址关系网络
   * @returns 聚类系数 (0-1)
   */
  private static calculateClustering(
    address: string,
    network: Record<string, Array<{ address: string; weight: number }>>
  ): number {
    const neighbors = network[address] || [];
    const neighborCount = neighbors.length;
    
    if (neighborCount <= 1) return 0;
    
    // 计算邻居之间的连接数
    let connectionCount = 0;
    
    for (let i = 0; i < neighborCount; i++) {
      const neighbor1 = neighbors[i].address;
      
      for (let j = i + 1; j < neighborCount; j++) {
        const neighbor2 = neighbors[j].address;
        
        // 检查两个邻居之间是否有连接
        if (network[neighbor1]?.some(conn => conn.address === neighbor2)) {
          connectionCount++;
        }
      }
    }
    
    // 可能的最大连接数
    const maxPossibleConnections = (neighborCount * (neighborCount - 1)) / 2;
    
    return maxPossibleConnections > 0 ? connectionCount / maxPossibleConnections : 0;
  }
  
  /**
   * 查找风险传播路径
   * @param sourceAddress 源地址
   * @param targetAddress 目标地址
   * @param network 地址关系网络
   * @returns 风险路径数组
   */
  private static async findRiskPaths(
    sourceAddress: string,
    targetAddress: string | undefined,
    network: Record<string, Array<{ address: string; weight: number }>>
  ): Promise<Array<{ from: string; to: string; weight: number }>> {
    const paths: Array<{ from: string; to: string; weight: number }> = [];
    
    // 如果指定了目标地址，寻找从源到目标的路径
    if (targetAddress) {
      const path = await this.findShortestPath(sourceAddress, targetAddress, network);
      if (path.length > 0) {
        // 转换路径格式
        for (let i = 0; i < path.length - 1; i++) {
          const from = path[i];
          const to = path[i + 1];
          const connection = network[from]?.find(conn => conn.address === to);
          
          if (connection) {
            paths.push({
              from,
              to,
              weight: connection.weight
            });
          }
        }
      }
    } 
    // 否则，寻找通向高风险地址的路径
    else {
      // 获取网络中所有地址的风险评分
      const addressScores = await this.getAddressRiskScores(Object.keys(network));
      
      // 找出高风险地址 (风险分数 >= 0.7)
      const highRiskAddresses = Object.entries(addressScores)
        .filter(([_, score]) => score >= 0.7)
        .map(([address]) => address);
      
      // 为每个高风险地址寻找路径
      for (const riskAddress of highRiskAddresses) {
        if (riskAddress === sourceAddress) continue;
        
        const path = await this.findShortestPath(sourceAddress, riskAddress, network);
        if (path.length > 0) {
          // 转换路径格式
          for (let i = 0; i < path.length - 1; i++) {
            const from = path[i];
            const to = path[i + 1];
            const connection = network[from]?.find(conn => conn.address === to);
            
            if (connection) {
              paths.push({
                from,
                to,
                weight: connection.weight
              });
            }
          }
        }
      }
    }
    
    return paths;
  }
  
  /**
   * 识别网络中的风险节点
   * @param network 地址关系网络
   * @returns 风险节点数组
   */
  private static async identifyRiskNodes(
    network: Record<string, Array<{ address: string; weight: number }>>
  ): Promise<Array<{ address: string; riskScore: number; distance: number }>> {
    // 获取网络中所有地址的风险评分
    const addresses = Object.keys(network);
    const addressScores = await this.getAddressRiskScores(addresses);
    
    // 构建风险节点列表
    const riskNodes: Array<{ address: string; riskScore: number; distance: number }> = [];
    
    for (const address of addresses) {
      const riskScore = addressScores[address] || 0;
      
      // 只包含中高风险节点 (风险分数 >= 0.4)
      if (riskScore >= 0.4) {
        // 计算节点到源节点的距离
        // 简化实现：使用连接深度作为距离
        // 实际应用中应该使用网络距离算法
        const distance = 1; // 简化处理
        
        riskNodes.push({
          address,
          riskScore,
          distance
        });
      }
    }
    
    // 按风险分数降序排序
    return riskNodes.sort((a, b) => b.riskScore - a.riskScore);
  }
  
  /**
   * 查找两个地址之间的最短路径 (Dijkstra算法简化版)
   * @param from 起始地址
   * @param to 目标地址
   * @param network 地址关系网络
   * @returns 最短路径
   */
  private static async findShortestPath(
    from: string,
    to: string,
    network: Record<string, Array<{ address: string; weight: number }>>
  ): Promise<string[]> {
    // 如果起始或目标地址不在网络中，返回空路径
    if (!network[from] || !network[to]) return [];
    
    // 如果起始和目标是同一地址，返回单一节点路径
    if (from === to) return [from];
    
    // 初始化距离和前驱节点
    const distances: Record<string, number> = {};
    const previous: Record<string, string | undefined> = {};
    const unvisited = new Set<string>();
    
    // 初始化所有节点的距离为无穷大
    for (const address in network) {
      distances[address] = Infinity;
      previous[address] = undefined;
      unvisited.add(address);
    }
    
    // 起始节点距离为0
    distances[from] = 0;
    
    // Dijkstra算法
    while (unvisited.size > 0) {
      // 找出未访问节点中距离最小的
      let current: string | null = null;
      let minDistance = Infinity;
      
      for (const address of unvisited) {
        if (distances[address] < minDistance) {
          minDistance = distances[address];
          current = address;
        }
      }
      
      // 如果没有可达节点或已找到目标，结束
      if (current === null || current === to || minDistance === Infinity) break;
      
      // 标记为已访问
      unvisited.delete(current);
      
      // 更新邻居距离
      for (const { address: neighbor, weight } of network[current]) {
        if (unvisited.has(neighbor)) {
          // 使用连接权重的倒数作为距离（权重越高，距离越近）
          const edgeDistance = 1 / (weight > 0 ? weight : 0.1);
          const altDistance = distances[current] + edgeDistance;
          
          if (altDistance < distances[neighbor]) {
            distances[neighbor] = altDistance;
            previous[neighbor] = current;
          }
        }
      }
    }
    
    // 重建路径
    const path: string[] = [];
    let current: string | undefined = to;
    
    // 如果目标不可达
    if (previous[to] === undefined && from !== to) {
      return [];
    }
    
    while (current !== undefined) {
      path.unshift(current);
      current = previous[current];
    }
    
    return path;
  }
  
  /**
   * 获取地址风险评分
   * @param addresses 地址数组
   * @returns 地址风险分数映射
   */
  private static async getAddressRiskScores(addresses: string[]): Promise<Record<string, number>> {
    const scores: Record<string, number> = {};
    
    try {
      // 批量获取地址画像
      const profiles = await AddressProfileDAO.findByAddresses(addresses);
      
      // 映射风险分数
      for (const profile of profiles) {
        scores[profile.address] = profile.riskScore;
      }
      
      // 为未找到的地址设置默认低风险
      for (const address of addresses) {
        if (scores[address] === undefined) {
          scores[address] = 0.1;
        }
      }
    } catch (error) {
      logger.warn('获取地址风险评分失败', {
        addressCount: addresses.length,
        error: error instanceof Error ? error.message : String(error),
      });
      
      // 发生错误时，为所有地址设置默认低风险
      for (const address of addresses) {
        scores[address] = 0.1;
      }
    }
    
    return scores;
  }
} 