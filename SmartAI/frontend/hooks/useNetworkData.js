/**
 * 网络数据钩子
 *
 * 用于获取和管理网络图数据的React钩子
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import api from '../api/api';
import mockData from '../utils/mockDataService';

/**
 * 网络数据钩子
 *
 * @param {Object} options - 配置选项
 * @param {boolean} [options.useMockData=false] - 是否使用本地模拟数据(不发API请求)
 * @param {string} [options.initialSearchTerm=''] - 初始搜索关键词
 * @param {string} [options.initialRiskLevel=''] - 初始风险级别
 * @returns {Object} 网络数据和操作函数
 */
export default function useNetworkData(options = {}) {
  const { useMockData = false, initialSearchTerm = '', initialRiskLevel = '' } = options;

  // 状态
  const [networkData, setNetworkData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeLoading, setNodeLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState(initialSearchTerm);
  const [riskLevel, setRiskLevel] = useState(initialRiskLevel);
  const [networkStats, setNetworkStats] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // 保存组件挂载状态
  const isMountedRef = useRef(true);

  // 组件卸载时更新挂载状态
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // 加载网络数据
  const loadNetworkData = useCallback(async () => {
    if (!isMountedRef.current) return;

    setLoading(true);
    setError(null);

    try {
      let data;

      if (useMockData) {
        // 使用本地模拟数据(不发API请求)
        console.log('生成模拟网络数据');
        await mockData.simulateApiDelay(300, 800);
        data = mockData.generateNetworkData({ nodeCount: 30 });
        console.log('模拟数据生成完成', {
          nodes: data.nodes.length,
          links: data.links.length,
        });
      } else {
        // 从API获取数据
        const params = {};
        if (searchTerm) params.address = searchTerm;
        if (riskLevel) params.riskLevel = riskLevel;

        console.log('从API获取网络数据', params);
        data = await api.network.getNetworkData(params);
      }

      if (!isMountedRef.current) return;

      if (!data || !data.nodes || !data.links) {
        console.error('无效的网络数据格式', data);
        throw new Error('返回的网络数据格式无效');
      }

      // 格式化节点数据，确保字段一致性
      if (data && data.nodes) {
        data.nodes = data.nodes.map(node => ({
          ...node,
          id: String(node.id), // 确保id是字符串
          value: typeof node.value === 'number' ? node.value : 1,
        }));
      }

      // 格式化链接数据，确保字段一致性
      if (data && data.links) {
        data.links = data.links.map(link => ({
          ...link,
          source: String(link.source), // 确保source是字符串
          target: String(link.target), // 确保target是字符串
        }));
      }

      console.log('网络数据处理完成', {
        nodes: data.nodes.length,
        links: data.links.length,
      });

      setNetworkData(data);
      setLastUpdated(new Date().toISOString());
      setLoading(false);

      // 重置选中的节点
      if (selectedNodeId && data && data.nodes) {
        const node = data.nodes.find(n => n.id === selectedNodeId);
        if (!node) {
          setSelectedNodeId(null);
          setSelectedNode(null);
        }
      }
    } catch (err) {
      if (!isMountedRef.current) return;

      console.error('加载网络数据失败:', err);
      setError(err.message || '加载网络数据失败');
      setLoading(false);
    }
  }, [useMockData, searchTerm, riskLevel, selectedNodeId]);

  // 加载网络统计数据
  const loadNetworkStats = useCallback(async () => {
    if (!isMountedRef.current || !networkData) return;

    try {
      let stats;

      if (useMockData) {
        // 使用本地模拟数据
        await mockData.simulateApiDelay(300, 700);

        // 生成一些统计数据
        stats = {
          totalNodes: networkData?.nodes?.length || 0,
          totalLinks: networkData?.links?.length || 0,
          riskDistribution: {
            high: 0,
            medium: 0,
            low: 0,
            unknown: 0,
          },
        };

        // 计算风险分布
        if (networkData && networkData.nodes) {
          networkData.nodes.forEach(node => {
            if (node.risk === 'high') stats.riskDistribution.high++;
            else if (node.risk === 'medium') stats.riskDistribution.medium++;
            else if (node.risk === 'low') stats.riskDistribution.low++;
            else stats.riskDistribution.unknown++;
          });
        }
      } else {
        // 从API获取数据
        stats = await api.network.getNetworkStats({
          searchTerm,
          riskLevel,
        });
      }

      if (!isMountedRef.current) return;
      setNetworkStats(stats);
    } catch (err) {
      console.error('加载网络统计数据失败:', err);
      // 不设置错误状态，因为这是次要数据
    }
  }, [useMockData, networkData, searchTerm, riskLevel]);

  // 选择节点
  const selectNode = useCallback(
    async nodeId => {
      if (!isMountedRef.current) return;

      if (selectedNodeId === nodeId) {
        // 再次点击同一节点，取消选择
        setSelectedNodeId(null);
        setSelectedNode(null);
        return;
      }

      setSelectedNodeId(nodeId);
      setNodeLoading(true);

      try {
        let nodeDetails;

        if (useMockData) {
          // 使用本地模拟数据
          await mockData.simulateApiDelay(300, 800);

          // 查找节点
          const node = networkData.nodes.find(n => n.id === nodeId);
          if (!node) throw new Error('节点未找到');

          // 生成一些节点详情
          const address = node.address || mockData.generateRandomAddress();
          nodeDetails = mockData.generateAddressData(address);
        } else {
          // 从API获取数据
          nodeDetails = await api.network.getNodeDetails(nodeId, true);
        }

        if (!isMountedRef.current) return;
        setSelectedNode(nodeDetails);
        setNodeLoading(false);
      } catch (err) {
        if (!isMountedRef.current) return;

        console.error(`加载节点详情失败 (${nodeId}):`, err);
        setSelectedNode(null);
        setNodeLoading(false);

        // 不设置错误状态，因为这是选择操作
      }
    },
    [useMockData, selectedNodeId, networkData]
  );

  // 重置过滤器
  const resetFilters = useCallback(() => {
    setSearchTerm('');
    setRiskLevel('');
  }, []);

  // 初始加载
  useEffect(() => {
    loadNetworkData();
  }, [loadNetworkData]);

  // 当网络数据变化时加载统计信息
  useEffect(() => {
    if (networkData) {
      loadNetworkStats();
    }
  }, [networkData, loadNetworkStats]);

  return {
    // 数据
    networkData,
    loading,
    error,
    selectedNodeId,
    selectedNode,
    nodeLoading,
    searchTerm,
    riskLevel,
    networkStats,
    lastUpdated,

    // 操作函数
    loadNetworkData,
    selectNode,
    setSearchTerm,
    setRiskLevel,
    resetFilters,
    setNetworkData,
  };
}
