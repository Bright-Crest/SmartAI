import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import {
  FaRegCopy,
  FaExternalLinkAlt,
  FaChartPie,
  FaNetworkWired,
  FaHistory,
  FaShieldAlt,
  FaExchangeAlt,
  FaUserClock,
  FaTag,
  FaArrowLeft,
  FaInfoCircle,
  FaSearchDollar,
  FaRoute,
  FaFilter,
} from 'react-icons/fa';
import { getAddressAnalysis } from '../../api/addressAnalysis';
import SmartScore from '../../components/SmartScore';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { analyzeAddress } from '../../utils/smartScoreCalculator';

// 动态导入RadarChart组件，禁用SSR
const DynamicRadarChart = dynamic(() => import('../../components/RadarChart'), {
  ssr: false,
  loading: () => (
    <div className="h-[300px] flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  ),
});

// 动态导入网络图组件，禁用SSR
const DynamicNetworkGraph = dynamic(() => import('../../components/network/D3NetworkGraph'), {
  ssr: false,
  loading: () => (
    <div className="h-[500px] flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  ),
});

/**
 * 地址详情页面
 * 展示区块链地址的详细信息、风险分析和相关数据
 */
export default function AddressDetailPage() {
  const router = useRouter();
  const { address } = router.query;
  const [addressData, setAddressData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('分析报告');
  const [smartScoreData, setSmartScoreData] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [selectedDimension, setSelectedDimension] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const [networkLoading, setNetworkLoading] = useState(false);
  const [trackingDepth, setTrackingDepth] = useState(1);
  const [selectedNodeType, setSelectedNodeType] = useState('all');
  const [selectedNode, setSelectedNode] = useState(null);
  const [trackingStats, setTrackingStats] = useState(null);

  // 获取地址数据
  useEffect(() => {
    if (!address) return;

    const fetchAddressData = async () => {
      try {
        setIsLoading(true);
        const data = await getAddressAnalysis(address);
        setAddressData(data);

        // 计算SmartScore
        const calculatedScoreData = analyzeAddress(data.radarData);
        setSmartScoreData({
          ...calculatedScoreData,
          // 如果服务端已提供智能分数数据，优先使用服务端数据
          ...data.smartScore,
        });

        setIsLoading(false);
      } catch (error) {
        console.error('获取地址数据失败', error);
        setIsLoading(false);
      }
    };

    fetchAddressData();
  }, [address]);

  // 获取网络图数据
  useEffect(() => {
    if (!address || activeTab !== '关系图谱') return;

    // 只在首次加载关系图谱选项卡时获取数据
    if (!networkData && !networkLoading) {
      fetchNetworkData();
    }
  }, [address, activeTab]);

  // 获取网络图数据的函数
  const fetchNetworkData = async (depth = trackingDepth) => {
    if (!address) return;

    try {
      setNetworkLoading(true);
      setTrackingStats(null); // 重置追踪统计数据

      // 这里替换为实际的API调用
      // const data = await getAddressNetwork(address, depth);

      // 模拟数据（实际项目中替换为API调用）
      const mockData = generateMockNetworkData(address, depth);
      setNetworkData(mockData);

      // 生成追踪统计数据
      generateTrackingStats(mockData);

      setNetworkLoading(false);
    } catch (error) {
      console.error('获取关系图谱数据失败', error);
      setNetworkLoading(false);
    }
  };

  // 生成模拟网络数据（实际项目中应该从API获取真实数据）
  const generateMockNetworkData = (centerAddress, depth = 1) => {
    const nodes = [
      {
        id: centerAddress,
        label: '当前地址',
        type: 'address',
        value: 10,
        riskScore: 25,
      },
    ];

    const links = [];

    // 为了演示，生成一些模拟节点
    const nodeTypes = ['address', 'contract', 'exchange', 'mixer'];
    const riskLevels = [10, 30, 50, 75, 90];

    // 为主节点添加一些连接
    for (let i = 0; i < 5 + depth * 3; i++) {
      const nodeId = `0x${Math.random().toString(16).substr(2, 40)}`;
      const nodeType = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
      const risk = riskLevels[Math.floor(Math.random() * riskLevels.length)];

      nodes.push({
        id: nodeId,
        label: `${nodeType.charAt(0).toUpperCase() + nodeType.slice(1)} ${i + 1}`,
        type: nodeType,
        value: 5 + Math.random() * 5,
        riskScore: risk,
      });

      // 创建与中心节点的连接
      links.push({
        id: `link-${i}`,
        source: i % 2 === 0 ? centerAddress : nodeId,
        target: i % 2 === 0 ? nodeId : centerAddress,
        value: 1 + Math.random() * 3,
      });

      // 为一些节点添加相互连接
      if (i > 0 && i % 3 === 0) {
        links.push({
          id: `link-between-${i}-${i - 1}`,
          source: nodes[i].id,
          target: nodes[i - 1].id,
          value: 1 + Math.random() * 2,
        });
      }
    }

    // 添加二级连接（如果深度大于1）
    if (depth > 1) {
      const secondaryNodes = [...nodes];
      for (let i = 1; i < 5; i++) {
        const parentNode = nodes[i].id;

        for (let j = 0; j < 2 + Math.floor(Math.random() * 3); j++) {
          const nodeId = `0x${Math.random().toString(16).substr(2, 40)}`;
          const nodeType = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
          const risk = riskLevels[Math.floor(Math.random() * riskLevels.length)];

          secondaryNodes.push({
            id: nodeId,
            label: `${nodeType.charAt(0).toUpperCase() + nodeType.slice(1)} ${i}-${j}`,
            type: nodeType,
            value: 3 + Math.random() * 4,
            riskScore: risk,
          });

          links.push({
            id: `link-secondary-${i}-${j}`,
            source: parentNode,
            target: nodeId,
            value: 1 + Math.random() * 2,
          });
        }
      }

      // 使用扩展的节点集
      return { nodes: secondaryNodes, links };
    }

    return { nodes, links };
  };

  // 生成追踪统计数据
  const generateTrackingStats = data => {
    if (!data || !data.nodes || !data.links) return;

    // 计算高风险节点数量
    const highRiskNodes = data.nodes.filter(node => node.riskScore >= 70).length;

    // 计算中风险节点数量
    const mediumRiskNodes = data.nodes.filter(
      node => node.riskScore >= 40 && node.riskScore < 70
    ).length;

    // 按类型统计节点
    const nodeTypes = {};
    data.nodes.forEach(node => {
      if (node.type) {
        nodeTypes[node.type] = (nodeTypes[node.type] || 0) + 1;
      }
    });

    // 计算直接连接
    const directConnections = data.links.filter(
      link =>
        link.source === address ||
        link.target === address ||
        (typeof link.source === 'object' && link.source.id === address) ||
        (typeof link.target === 'object' && link.target.id === address)
    ).length;

    // 设置追踪统计
    setTrackingStats({
      totalNodes: data.nodes.length,
      totalLinks: data.links.length,
      highRiskNodes,
      mediumRiskNodes,
      nodeTypes,
      directConnections,
      trackingDepth,
    });
  };

  // 处理节点点击
  const handleNodeClick = node => {
    setSelectedNode(node);
  };

  // 处理节点双击 - 可以用于扩展某个节点的关系
  const handleNodeDoubleClick = node => {
    // 实际应用中可以通过API获取该节点的更多关联数据
    alert(`将展开节点 ${node.id} 的更多关系`);
  };

  // 处理追踪深度变化
  const handleDepthChange = newDepth => {
    setTrackingDepth(newDepth);
    fetchNetworkData(newDepth);
  };

  // 处理节点类型筛选
  const handleNodeTypeFilter = type => {
    setSelectedNodeType(type);
  };

  // 获取筛选后的网络数据
  const getFilteredNetworkData = () => {
    if (!networkData) return { nodes: [], links: [] };

    // 如果选择全部，返回完整数据
    if (selectedNodeType === 'all') {
      return networkData;
    }

    // 筛选符合类型的节点
    const filteredNodes = networkData.nodes.filter(
      node => node.type === selectedNodeType || node.id === address // 总是包含当前地址节点
    );

    // 节点ID列表，用于筛选连接
    const nodeIds = filteredNodes.map(node => node.id);

    // 筛选连接，两端节点都在筛选后的节点列表中
    const filteredLinks = networkData.links.filter(
      link =>
        nodeIds.includes(typeof link.source === 'object' ? link.source.id : link.source) &&
        nodeIds.includes(typeof link.target === 'object' ? link.target.id : link.target)
    );

    return { nodes: filteredNodes, links: filteredLinks };
  };

  // 复制地址到剪贴板
  const copyToClipboard = async () => {
    if (!address) return;

    try {
      await navigator.clipboard.writeText(address);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('复制失败:', err);
    }
  };

  // 前往区块浏览器
  const openInExplorer = () => {
    if (!address) return;
    window.open(`https://etherscan.io/address/${address}`, '_blank');
  };

  // 处理雷达图维度点击
  const handleDimensionClick = dimension => {
    setSelectedDimension(dimension);

    // 滚动到详情部分
    document.getElementById('dimension-details')?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  };

  // 获取所选维度的详细信息
  const getDimensionDetails = () => {
    if (!selectedDimension || !addressData) return null;

    const dimensionLabels = {
      activityScore: '活跃度',
      protocolInteractionScore: '协议互动',
      profitabilityScore: '收益能力',
      fundFlowScore: '资金流',
      distillationScore: '蒸馏模型',
    };

    const dimensionDescriptions = {
      activityScore:
        '活跃度反映了钱包地址在区块链上的活动频率和规律性。高活跃度表示该地址频繁参与交易和互动。',
      protocolInteractionScore:
        '协议互动评估与不同DeFi协议、智能合约的交互广度和深度，反映了用户对区块链生态系统的参与程度。',
      profitabilityScore: '收益能力分析投资和交易的盈利表现，评估用户在市场中的表现和策略有效性。',
      fundFlowScore:
        '资金流追踪和分析资金的流入流出模式、规模和网络特征，识别资金活动的性质和影响力。',
      distillationScore:
        '蒸馏模型是基于机器学习的综合行为评估，从复杂的交易数据中提取特征，识别行为的专业性和复杂度。',
    };

    return {
      name: dimensionLabels[selectedDimension] || selectedDimension,
      score: addressData.radarData[selectedDimension],
      description: dimensionDescriptions[selectedDimension] || '',
    };
  };

  // 渲染加载状态
  if (isLoading || !addressData) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <Head>
          <title>加载地址数据... | SmartAI</title>
        </Head>
        <div className="container mx-auto px-4 py-8">
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
          </div>
        </div>
      </div>
    );
  }

  const dimensionDetails = getDimensionDetails();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Head>
        <title>{formatAddress(address)} | SmartAI 分析</title>
      </Head>

      {/* 主内容 */}
      <main className="container mx-auto px-4 py-6">
        {/* 顶部导航 */}
        <div className="mb-6">
          <Button variant="ghost" className="mb-4" onClick={() => router.push('/')}>
            <FaArrowLeft className="mr-2" />
            返回首页
          </Button>
        </div>

        {/* 地址信息栏 */}
        <Card className="mb-6 p-6 border-t-4 border-primary-500">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <h1 className="text-2xl font-bold text-gray-800 dark:text-white">地址分析</h1>
                <span className="px-2 py-0.5 text-xs bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200 rounded-full">
                  以太坊
                </span>
                {smartScoreData?.tags && smartScoreData.tags.length > 0 && (
                  <span
                    className={`px-2 py-0.5 text-xs rounded-full 
                    ${
                      smartScoreData.score >= 80
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : smartScoreData.score >= 60
                          ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                          : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}
                  >
                    {smartScoreData?.behaviorType || '未分类'}
                  </span>
                )}
              </div>
              <div className="mt-1 flex items-center">
                <p className="text-sm font-mono text-gray-600 dark:text-gray-300">{address}</p>
                <button
                  onClick={copyToClipboard}
                  className="ml-2 text-gray-400 hover:text-primary-500 dark:text-gray-500 dark:hover:text-primary-400 transition-colors"
                  title="复制地址"
                >
                  <FaRegCopy />
                  {copySuccess && <span className="ml-1 text-xs text-green-500">已复制</span>}
                </button>
                <button
                  onClick={openInExplorer}
                  className="ml-2 text-gray-400 hover:text-primary-500 dark:text-gray-500 dark:hover:text-primary-400 transition-colors"
                  title="在区块浏览器中查看"
                >
                  <FaExternalLinkAlt />
                </button>
              </div>
            </div>
            <div className="flex gap-2">
              <Button className="text-sm" onClick={() => router.push('/')}>
                分析新地址
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 dark:bg-blue-900 mr-3">
                <FaExchangeAlt className="text-blue-500 dark:text-blue-300" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">交易总数</p>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {addressData.transactions || 0}
                </p>
              </div>
            </div>

            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 dark:bg-green-900 mr-3">
                <FaShieldAlt className="text-green-500 dark:text-green-300" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">余额</p>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {addressData.balance?.toFixed(4) || 0} ETH
                </p>
              </div>
            </div>

            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100 dark:bg-purple-900 mr-3">
                <FaUserClock className="text-purple-500 dark:text-purple-300" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">首次交易</p>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {addressData.firstSeen
                    ? new Date(addressData.firstSeen).toLocaleDateString('zh-CN')
                    : '未知'}
                </p>
              </div>
            </div>

            <div className="flex items-center">
              <div className="p-3 rounded-full bg-orange-100 dark:bg-orange-900 mr-3">
                <FaTag className="text-orange-500 dark:text-orange-300" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">最近活动</p>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {addressData.lastSeen
                    ? new Date(addressData.lastSeen).toLocaleDateString('zh-CN')
                    : '未知'}
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* SmartScore 和 分析选项卡 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧 - SmartScore */}
          <div>
            <Card className="p-0 overflow-hidden">
              <div className="bg-gradient-to-br from-primary-500 to-primary-700 p-4 text-white">
                <h3 className="text-lg font-medium mb-1">智能风险评分</h3>
                <p className="text-sm opacity-90">基于多维度链上行为分析</p>
              </div>
              <div className="p-6">
                <SmartScore
                  score={smartScoreData?.score || 0}
                  tags={smartScoreData?.tags || []}
                  behaviorType={smartScoreData?.behaviorType}
                  animate={true}
                  onViewDetails={() => setActiveTab('分析报告')}
                />
              </div>
            </Card>
          </div>

          {/* 中间和右侧 - 主要内容区 */}
          <div className="lg:col-span-2">
            {/* 标签页切换 */}
            <Card className="mb-6 p-0 overflow-hidden">
              {/* 标签页导航 */}
              <div className="px-4 border-b border-gray-200 dark:border-gray-700">
                <nav className="flex">
                  {['分析报告', '关系图谱', '交易历史'].map(tab => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className={`py-4 px-4 text-center font-medium border-b-2 transition-colors ${
                        activeTab === tab
                          ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                      }`}
                    >
                      <span className="flex items-center">
                        {tab === '分析报告' && <FaChartPie className="mr-2" />}
                        {tab === '关系图谱' && <FaNetworkWired className="mr-2" />}
                        {tab === '交易历史' && <FaHistory className="mr-2" />}
                        {tab}
                      </span>
                    </button>
                  ))}
                </nav>
              </div>

              {/* 标签页内容 */}
              <div className="p-6">
                {activeTab === '分析报告' && (
                  <div>
                    {/* 五维雷达图 */}
                    <div className="mb-8">
                      <div className="flex justify-between items-center mb-4">
                        <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300">
                          五维评分
                        </h4>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          点击雷达图查看详情
                        </div>
                      </div>
                      <div className="flex justify-center">
                        <div className="w-full max-w-md">
                          <DynamicRadarChart
                            data={addressData.radarData}
                            interactive={true}
                            showAverageUser={true}
                            onDimensionClick={handleDimensionClick}
                          />
                        </div>
                      </div>
                    </div>

                    {/* 维度详情 */}
                    {dimensionDetails && (
                      <div
                        id="dimension-details"
                        className="mb-8 bg-gray-50 dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700"
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h4 className="text-lg font-medium text-gray-800 dark:text-white flex items-center">
                            {dimensionDetails.name}
                            <span className="ml-2 px-2 py-1 text-xs rounded-full bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-300">
                              {dimensionDetails.score}/100
                            </span>
                          </h4>
                          <button
                            onClick={() => setSelectedDimension(null)}
                            className="text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
                          >
                            <span className="sr-only">关闭</span>
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-5 w-5"
                              viewBox="0 0 20 20"
                              fill="currentColor"
                            >
                              <path
                                fillRule="evenodd"
                                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                clipRule="evenodd"
                              />
                            </svg>
                          </button>
                        </div>
                        <p className="text-gray-600 dark:text-gray-400 mb-4">
                          {dimensionDetails.description}
                        </p>
                        <div className="flex items-center">
                          <div className="flex-1">
                            <div className="flex justify-between mb-1">
                              <span className="text-xs text-gray-500 dark:text-gray-400">一般</span>
                              <span className="text-xs text-gray-500 dark:text-gray-400">专业</span>
                            </div>
                            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                              <div
                                className="h-2 bg-primary-500 rounded-full"
                                style={{ width: `${dimensionDetails.score}%` }}
                              ></div>
                            </div>
                          </div>
                          <div className="ml-4">
                            <button
                              className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 text-sm flex items-center"
                              onClick={() => router.push('/radar-dimensions')}
                            >
                              <FaInfoCircle className="mr-1" /> 了解更多
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 行为总结 */}
                    <div className="mb-8">
                      <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4">
                        行为总结
                      </h4>
                      <div className="bg-white dark:bg-gray-800 p-5 rounded-lg border border-gray-200 dark:border-gray-700">
                        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                          {addressData.behaviorData?.summary || '暂无行为总结数据'}
                        </p>
                      </div>
                    </div>

                    {/* 行为洞察 */}
                    <div>
                      <h4 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-4">
                        行为洞察
                      </h4>
                      {addressData.behaviorData?.insights?.length > 0 ? (
                        <div className="grid grid-cols-1 gap-4">
                          {addressData.behaviorData.insights.map((insight, index) => (
                            <Card key={index} className="p-4 hover:shadow-md transition-shadow">
                              <h5 className="font-medium text-gray-800 dark:text-white mb-2 flex items-center">
                                <div className="w-2 h-2 rounded-full bg-primary-500 mr-2"></div>
                                {insight.title}
                              </h5>
                              <p className="text-gray-600 dark:text-gray-400 pl-4">
                                {insight.description}
                              </p>
                            </Card>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-10 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                          <FaInfoCircle className="mx-auto text-gray-400 text-2xl mb-2" />
                          <p className="text-gray-500 dark:text-gray-400">暂无洞察数据</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {activeTab === '关系图谱' && (
                  <div>
                    <h3 className="text-xl font-medium text-gray-800 dark:text-white mb-4 flex items-center">
                      <FaNetworkWired className="mr-2" />
                      地址关系图谱
                    </h3>

                    {/* 工具栏 */}
                    <div className="mb-6 bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 flex flex-wrap items-center gap-4">
                      {/* 追踪深度选择 */}
                      <div className="flex items-center">
                        <span className="text-sm text-gray-600 dark:text-gray-300 mr-2">
                          追踪深度:
                        </span>
                        <div className="flex border border-gray-200 dark:border-gray-700 rounded-md overflow-hidden">
                          {[1, 2, 3].map(depth => (
                            <button
                              key={depth}
                              onClick={() => handleDepthChange(depth)}
                              className={`px-3 py-1 text-sm ${
                                trackingDepth === depth
                                  ? 'bg-primary-500 text-white'
                                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                              }`}
                            >
                              {depth}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* 一键追踪按钮 */}
                      <Button
                        onClick={() => fetchNetworkData(trackingDepth)}
                        className="flex items-center text-sm"
                        disabled={networkLoading}
                      >
                        <FaRoute className="mr-1" />
                        {networkLoading ? '加载中...' : '一键追踪'}
                      </Button>

                      {/* 节点类型筛选 */}
                      <div className="flex items-center ml-auto">
                        <span className="text-sm text-gray-600 dark:text-gray-300 mr-2">
                          <FaFilter className="inline mr-1" />
                          筛选:
                        </span>
                        <div className="flex border border-gray-200 dark:border-gray-700 rounded-md overflow-hidden">
                          {['all', 'address', 'contract', 'exchange', 'mixer'].map(type => (
                            <button
                              key={type}
                              onClick={() => handleNodeTypeFilter(type)}
                              className={`px-3 py-1 text-sm ${
                                selectedNodeType === type
                                  ? 'bg-primary-500 text-white'
                                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                              }`}
                            >
                              {type === 'all'
                                ? '全部'
                                : type === 'address'
                                  ? '地址'
                                  : type === 'contract'
                                    ? '合约'
                                    : type === 'exchange'
                                      ? '交易所'
                                      : '混币器'}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* 网络图区域 */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {/* 主网络图区域 */}
                      <div className="md:col-span-2 bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700 min-h-[500px]">
                        {networkLoading ? (
                          <div className="h-[500px] flex items-center justify-center">
                            <div className="text-center">
                              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500 mx-auto mb-4"></div>
                              <p className="text-gray-500 dark:text-gray-400">加载关系图谱中...</p>
                            </div>
                          </div>
                        ) : networkData ? (
                          <div>
                            <DynamicNetworkGraph
                              data={getFilteredNetworkData()}
                              onNodeClick={handleNodeClick}
                              onNodeDoubleClick={handleNodeDoubleClick}
                              initialZoom={1.2}
                            />
                          </div>
                        ) : (
                          <div className="h-[500px] flex items-center justify-center">
                            <div className="text-center max-w-md">
                              <FaNetworkWired className="text-4xl text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                              <h4 className="text-xl font-medium text-gray-700 dark:text-gray-300 mb-2">
                                发现关系网络
                              </h4>
                              <p className="text-gray-500 dark:text-gray-400 mb-4">
                                点击"一键追踪"按钮，探索此地址与其他地址之间的交易关系和资金流动。
                              </p>
                              <Button onClick={() => fetchNetworkData(trackingDepth)}>
                                <FaSearchDollar className="mr-2" />
                                开始追踪
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* 节点详情/统计面板 */}
                      <div className="md:col-span-1">
                        <div className="space-y-4">
                          {/* 节点详情卡片 */}
                          <Card className="h-full p-0 overflow-hidden">
                            <div className="bg-gradient-to-br from-primary-600 to-primary-800 p-4 text-white">
                              <h4 className="text-lg font-medium mb-1">节点详情</h4>
                              <p className="text-sm opacity-90">查看选中节点的详细信息</p>
                            </div>
                            <div className="p-4">
                              {selectedNode ? (
                                <div>
                                  <div className="flex justify-between items-start mb-4">
                                    <div>
                                      <h5 className="font-medium text-lg text-gray-800 dark:text-white">
                                        {selectedNode.label || '未命名节点'}
                                      </h5>
                                      <p className="text-sm font-mono text-gray-500 dark:text-gray-400 break-all">
                                        {selectedNode.id}
                                      </p>
                                    </div>
                                    <div
                                      className={`px-2 py-1 text-xs font-medium rounded-full ${
                                        selectedNode.riskScore >= 70
                                          ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
                                          : selectedNode.riskScore >= 40
                                            ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300'
                                            : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                                      }`}
                                    >
                                      {selectedNode.riskScore >= 70
                                        ? '高风险'
                                        : selectedNode.riskScore >= 40
                                          ? '中风险'
                                          : '低风险'}
                                    </div>
                                  </div>

                                  <div className="space-y-3">
                                    <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
                                      <span className="text-gray-500 dark:text-gray-400">类型</span>
                                      <span className="font-medium text-gray-800 dark:text-gray-200 capitalize">
                                        {selectedNode.type === 'address'
                                          ? '钱包地址'
                                          : selectedNode.type === 'contract'
                                            ? '智能合约'
                                            : selectedNode.type === 'exchange'
                                              ? '交易所'
                                              : selectedNode.type === 'mixer'
                                                ? '混币器'
                                                : selectedNode.type}
                                      </span>
                                    </div>
                                    <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
                                      <span className="text-gray-500 dark:text-gray-400">
                                        风险评分
                                      </span>
                                      <span className="font-medium text-gray-800 dark:text-gray-200">
                                        {selectedNode.riskScore}/100
                                      </span>
                                    </div>
                                    <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
                                      <span className="text-gray-500 dark:text-gray-400">
                                        连接数
                                      </span>
                                      <span className="font-medium text-gray-800 dark:text-gray-200">
                                        {/* 计算选定节点的连接数 */}
                                        {
                                          networkData.links.filter(
                                            link =>
                                              (typeof link.source === 'object'
                                                ? link.source.id
                                                : link.source) === selectedNode.id ||
                                              (typeof link.target === 'object'
                                                ? link.target.id
                                                : link.target) === selectedNode.id
                                          ).length
                                        }
                                      </span>
                                    </div>
                                  </div>

                                  <div className="mt-6 flex gap-2">
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      onClick={() =>
                                        window.open(
                                          `https://etherscan.io/address/${selectedNode.id}`,
                                          '_blank'
                                        )
                                      }
                                      className="flex-1"
                                    >
                                      <FaExternalLinkAlt className="mr-1" />
                                      区块浏览器
                                    </Button>
                                    {selectedNode.id !== address && (
                                      <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={() => router.push(`/address/${selectedNode.id}`)}
                                        className="flex-1"
                                      >
                                        <FaChartPie className="mr-1" />
                                        分析此地址
                                      </Button>
                                    )}
                                  </div>
                                </div>
                              ) : networkData ? (
                                <div>
                                  <h5 className="font-medium text-lg text-gray-800 dark:text-white mb-3">
                                    高风险地址
                                  </h5>
                                  {networkData.nodes.filter(
                                    node => node.riskScore >= 70 && node.id !== address
                                  ).length > 0 ? (
                                    <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
                                      {networkData.nodes
                                        .filter(node => node.riskScore >= 70 && node.id !== address)
                                        .slice(0, 5)
                                        .map(node => (
                                          <div
                                            key={node.id}
                                            className="bg-red-50 dark:bg-red-900/20 p-3 rounded-lg cursor-pointer hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors"
                                            onClick={() => handleNodeClick(node)}
                                          >
                                            <div className="flex justify-between">
                                              <p className="font-medium text-gray-800 dark:text-white">
                                                {node.label || '高风险地址'}
                                              </p>
                                              <span className="text-sm text-red-600 dark:text-red-400 font-medium">
                                                {node.riskScore}/100
                                              </span>
                                            </div>
                                            <p className="text-xs font-mono text-gray-500 dark:text-gray-400 truncate">
                                              {node.id}
                                            </p>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                              类型:{' '}
                                              {node.type === 'address'
                                                ? '钱包地址'
                                                : node.type === 'contract'
                                                  ? '智能合约'
                                                  : node.type === 'exchange'
                                                    ? '交易所'
                                                    : node.type === 'mixer'
                                                      ? '混币器'
                                                      : node.type}
                                            </p>
                                          </div>
                                        ))}
                                    </div>
                                  ) : (
                                    <div className="text-center py-6 bg-gray-50 dark:bg-gray-800 rounded-lg">
                                      <p className="text-gray-500 dark:text-gray-400">
                                        未发现高风险地址
                                      </p>
                                    </div>
                                  )}

                                  <div className="mt-6 text-center">
                                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                                      点击图中的节点查看详细信息
                                    </p>
                                  </div>
                                </div>
                              ) : (
                                <div className="h-full min-h-[400px] flex items-center justify-center">
                                  <div className="text-center text-gray-500 dark:text-gray-400">
                                    <FaInfoCircle className="text-3xl mb-2 mx-auto" />
                                    <p>点击"开始追踪"按钮生成关系图谱</p>
                                  </div>
                                </div>
                              )}
                            </div>
                          </Card>

                          {/* 追踪路径分析（仅当有数据且有选中节点时显示） */}
                          {selectedNode && selectedNode.id !== address && networkData && (
                            <Card className="p-4">
                              <h5 className="font-medium text-gray-800 dark:text-white mb-3">
                                路径分析
                              </h5>
                              <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg">
                                <div className="flex items-center">
                                  <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center text-primary-600 dark:text-primary-400 mr-2">
                                    1
                                  </div>
                                  <div className="truncate flex-1">
                                    <p className="font-medium text-sm text-gray-800 dark:text-white">
                                      当前地址
                                    </p>
                                    <p className="text-xs font-mono text-gray-500 dark:text-gray-400 truncate">
                                      {address}
                                    </p>
                                  </div>
                                </div>
                                <div className="h-6 border-l-2 border-dashed border-primary-300 dark:border-primary-700 ml-4"></div>
                                <div className="flex items-center">
                                  <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center text-primary-600 dark:text-primary-400 mr-2">
                                    2
                                  </div>
                                  <div className="truncate flex-1">
                                    <p className="font-medium text-sm text-gray-800 dark:text-white">
                                      {selectedNode.label || '目标地址'}
                                    </p>
                                    <p className="text-xs font-mono text-gray-500 dark:text-gray-400 truncate">
                                      {selectedNode.id}
                                    </p>
                                  </div>
                                </div>
                              </div>
                              <div className="mt-3 text-sm text-gray-500 dark:text-gray-400">
                                <p>这是一个直接连接。资金在这两个地址之间直接流转。</p>
                              </div>
                              <div className="mt-3 flex justify-end">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() =>
                                    window.open(
                                      `https://etherscan.io/address/${selectedNode.id}`,
                                      '_blank'
                                    )
                                  }
                                >
                                  查看交易历史
                                </Button>
                              </div>
                            </Card>
                          )}
                        </div>
                      </div>
                    </div>

                    {networkData && !networkLoading && (
                      <div className="mb-6 bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <h4 className="text-lg font-medium text-gray-800 dark:text-white mb-3">
                          追踪统计
                        </h4>

                        {trackingStats && (
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                总节点数
                              </div>
                              <div className="text-xl font-semibold text-gray-800 dark:text-white">
                                {trackingStats.totalNodes}
                              </div>
                            </div>

                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                高风险地址
                              </div>
                              <div className="text-xl font-semibold text-red-600 dark:text-red-400">
                                {trackingStats.highRiskNodes}
                              </div>
                            </div>

                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                直接连接
                              </div>
                              <div className="text-xl font-semibold text-primary-600 dark:text-primary-400">
                                {trackingStats.directConnections}
                              </div>
                            </div>

                            <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                              <div className="text-sm text-gray-500 dark:text-gray-400">
                                追踪深度
                              </div>
                              <div className="text-xl font-semibold text-gray-800 dark:text-white">
                                {trackingStats.trackingDepth}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {activeTab === '交易历史' && (
                  <div>
                    <h3 className="text-xl font-medium text-gray-800 dark:text-white mb-6">
                      交易历史
                    </h3>
                    <div className="text-center py-20 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                      <FaHistory className="mx-auto text-gray-400 text-4xl mb-4" />
                      <p className="text-gray-500 dark:text-gray-400 mb-2">交易历史功能开发中...</p>
                      <p className="text-sm text-gray-400 dark:text-gray-500">
                        该功能将展示此地址的历史交易记录
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div>
      </main>

      {/* 页脚 */}
      <footer className="bg-white dark:bg-gray-900 mt-12 py-4 border-t border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          SmartAI © {new Date().getFullYear()} - 区块链智能分析平台
        </div>
      </footer>
    </div>
  );
}

// 格式化地址显示
const formatAddress = (address, length = 6) => {
  if (!address) return '';
  return `${address.substring(0, length)}...${address.substring(address.length - 4)}`;
};
