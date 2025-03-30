import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import {
  FaArrowLeft,
  FaChartPie,
  FaSearch,
  FaCopy,
  FaInfoCircle,
  FaExchangeAlt,
} from 'react-icons/fa';
// 正确导入UI组件
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';

// 动态导入RadarChart组件，禁用SSR
const DynamicRadarChart = dynamic(() => import('../components/RadarChart'), {
  ssr: false,
  loading: () => (
    <div className="h-80 flex items-center justify-center">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  ),
});

// 示例数据集
const DEMO_ADDRESSES = [
  {
    address: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
    type: '高频交易者',
    data: {
      activityScore: 92,
      protocolInteractionScore: 85,
      profitabilityScore: 76,
      fundFlowScore: 68,
      distillationScore: 90,
    },
    tags: ['高频', 'DeFi用户', '套利者'],
    insights: [
      '该地址活跃度极高，每日平均交易12次',
      '与超过15个DeFi协议有深入交互',
      '在市场波动期间展现出优秀的套利能力',
      '资金流向多样化，主要流向DEX和借贷协议',
    ],
  },
  {
    address: '0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD',
    type: '长期持有者',
    data: {
      activityScore: 28,
      protocolInteractionScore: 45,
      profitabilityScore: 82,
      fundFlowScore: 30,
      distillationScore: 65,
    },
    tags: ['长期持有', '低频交易', '稳健投资'],
    insights: [
      '该地址活跃度低，平均每月交易1-2次',
      '主要与质押和储蓄协议交互',
      '长期持有收益率高于市场平均',
      '资金流入后较少流出，典型的长期投资者模式',
    ],
  },
  {
    address: '0x6B175474E89094C44Da98b954EedeAC495271d0F',
    type: '机构账户',
    data: {
      activityScore: 60,
      protocolInteractionScore: 70,
      profitabilityScore: 75,
      fundFlowScore: 95,
      distillationScore: 85,
    },
    tags: ['大额交易', '机构投资', '多链布局'],
    insights: [
      '该地址交易频率中等，但单笔交易额度大',
      '在多个链上有活动，展现出机构级别的规模',
      '与高级DeFi产品和衍生品交互频繁',
      '资金流向多个顶级项目，显示风险分散投资策略',
    ],
  },
];

/**
 * SmartScore演示页面
 * 展示区块链钱包地址的五维分析和智能评分
 */
export default function SmartScoreDemo() {
  const [address, setAddress] = useState('');
  const [demoData, setDemoData] = useState(DEMO_ADDRESSES[0]);
  const [selectedDimension, setSelectedDimension] = useState(null);
  const [showCopiedTooltip, setShowCopiedTooltip] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // 处理地址搜索
  const handleSearch = () => {
    if (!address) return;

    setIsLoading(true);

    // 模拟API调用延迟
    setTimeout(() => {
      // 在演示地址中找匹配的，或者使用第一个演示地址
      const foundDemo =
        DEMO_ADDRESSES.find(demo => demo.address.toLowerCase() === address.toLowerCase()) ||
        DEMO_ADDRESSES[0];

      setDemoData(foundDemo);
      setIsLoading(false);
    }, 1500);
  };

  // 处理维度点击
  const handleDimensionClick = dimension => {
    setSelectedDimension(dimension);
    // 滚动到详情部分
    document.getElementById('dimension-details')?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  };

  // 处理地址复制
  const handleCopyAddress = () => {
    navigator.clipboard.writeText(demoData.address);
    setShowCopiedTooltip(true);
    setTimeout(() => setShowCopiedTooltip(false), 2000);
  };

  // 切换演示数据
  const switchDemoData = index => {
    setDemoData(DEMO_ADDRESSES[index]);
    setSelectedDimension(null);
  };

  // 获取所选维度的信息
  const getDimensionDetails = () => {
    if (!selectedDimension) return null;

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

    const dimensionTips = {
      activityScore: [
        '交易频率高表示用户积极参与市场',
        '规律性活动通常表明是个人日常使用',
        '高度规律、高频率通常表明可能是自动化交易',
        '长期持有者通常活跃度较低',
      ],
      protocolInteractionScore: [
        '高分表示与多种协议有复杂互动',
        '探索新协议和功能表明用户是早期采用者',
        '使用高级功能表明用户熟悉DeFi',
        '多链交互表明更广泛的生态系统参与',
      ],
      profitabilityScore: [
        '长期稳定收益通常表明策略成熟',
        '波动性大的收益可能代表高风险策略',
        '与市场相关性高表明跟随整体趋势',
        '与市场相关性低表明独立的交易策略',
      ],
      fundFlowScore: [
        '大额资金流动通常表明机构级别操作',
        '资金长期停留表明长期投资意图',
        '频繁的资金进出表明活跃交易者',
        '资金流向多样性表明投资组合多元化',
      ],
      distillationScore: [
        '高复杂度交易模式表明专业用户',
        '独特的行为特征区分专业与普通用户',
        '行为一致性反映用户的交易纪律',
        '复杂与简单操作的混合表明多策略投资者',
      ],
    };

    return {
      name: dimensionLabels[selectedDimension] || selectedDimension,
      score: demoData.data[selectedDimension],
      description: dimensionDescriptions[selectedDimension] || '',
      tips: dimensionTips[selectedDimension] || [],
    };
  };

  // 计算总体智能评分
  const calculateSmartScore = () => {
    const weights = {
      activityScore: 0.2,
      protocolInteractionScore: 0.25,
      profitabilityScore: 0.2,
      fundFlowScore: 0.15,
      distillationScore: 0.2,
    };

    let totalScore = 0;
    let totalWeight = 0;

    Object.keys(weights).forEach(key => {
      if (demoData.data[key] !== undefined) {
        totalScore += demoData.data[key] * weights[key];
        totalWeight += weights[key];
      }
    });

    return Math.round(totalWeight > 0 ? totalScore / totalWeight : 0);
  };

  const dimensionDetails = getDimensionDetails();
  const smartScore = calculateSmartScore();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Head>
        <title>SmartScore演示 | 区块链行为分析</title>
        <meta
          name="description"
          content="SmartScore区块链地址行为分析工具演示页面，通过五维雷达图展示钱包地址的行为特征"
        />
      </Head>

      <main className="container mx-auto px-4 py-6">
        <div className="mb-6">
          <Link href="/">
            <Button variant="ghost" className="mb-2">
              <FaArrowLeft className="mr-2" />
              返回首页
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">SmartScore演示</h1>
          <p className="text-gray-600 dark:text-gray-300 mt-1">
            通过区块链数据分析，智能评估钱包地址的行为特征、风险评估和投资风格。
          </p>
        </div>

        {/* 搜索栏 */}
        <Card className="mb-8 p-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <Input
                placeholder="输入以太坊地址查看分析结果..."
                value={address}
                onChange={e => setAddress(e.target.value)}
                className="w-full"
                icon={<FaSearch className="text-gray-400" />}
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading} className="whitespace-nowrap">
              {isLoading ? '分析中...' : '分析地址'}
            </Button>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <p className="text-sm text-gray-600 dark:text-gray-400 mr-2">演示地址:</p>
            {DEMO_ADDRESSES.map((demo, index) => (
              <Button
                key={index}
                variant="subtle"
                size="sm"
                onClick={() => switchDemoData(index)}
                className={`text-xs ${demoData.address === demo.address ? 'bg-primary-100 dark:bg-primary-900' : ''}`}
              >
                {demo.type}
              </Button>
            ))}
          </div>
        </Card>

        {/* 分析结果 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* 雷达图 */}
          <div className="lg:col-span-2">
            <Card className="p-4 h-full">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold flex items-center text-gray-900 dark:text-white">
                  <FaChartPie className="mr-2 text-primary-500" />
                  五维雷达分析
                </h2>
                <Link href="/radar-dimensions">
                  <Button
                    variant="text"
                    size="sm"
                    className="text-primary-600 dark:text-primary-400"
                  >
                    <FaInfoCircle className="mr-1" />
                    了解维度含义
                  </Button>
                </Link>
              </div>

              {isLoading ? (
                <div className="h-80 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
                </div>
              ) : (
                <DynamicRadarChart
                  data={demoData.data}
                  interactive={true}
                  showAverageUser={true}
                  onDimensionClick={handleDimensionClick}
                />
              )}

              <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                点击雷达图上的任意维度查看详细分析。比较当前地址与普通用户的行为差异。
              </div>
            </Card>
          </div>

          {/* 地址信息和评分 */}
          <div>
            <Card className="p-4 mb-6">
              <h2 className="text-lg font-bold mb-4 text-gray-900 dark:text-white">地址信息</h2>

              <div className="mb-4">
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">以太坊地址</div>
                <div className="flex items-center">
                  <div className="text-sm font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded flex-1 truncate">
                    {demoData.address}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleCopyAddress}
                    className="ml-2 relative"
                  >
                    <FaCopy />
                    {showCopiedTooltip && (
                      <span className="absolute -top-8 -left-6 bg-gray-900 text-white text-xs px-2 py-1 rounded">
                        已复制
                      </span>
                    )}
                  </Button>
                </div>
              </div>

              <div className="mb-4">
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">行为类型</div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {demoData.type}
                </div>
              </div>

              <div className="mb-4">
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">特征标签</div>
                <div className="flex flex-wrap gap-2">
                  {demoData.tags.map((tag, index) => (
                    <span
                      key={index}
                      className="text-xs bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-2 py-1 rounded-full"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold text-gray-900 dark:text-white">智能评分</h2>
                <div className="text-3xl font-bold text-primary-600 dark:text-primary-400">
                  {smartScore}
                </div>
              </div>

              <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full mb-4">
                <div
                  className="h-2 bg-primary-500 rounded-full"
                  style={{ width: `${smartScore}%` }}
                ></div>
              </div>

              <div className="mb-4">
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-1">主要洞察</div>
                <ul className="space-y-2">
                  {demoData.insights.map((insight, index) => (
                    <li key={index} className="flex items-start">
                      <FaExchangeAlt className="text-primary-500 mt-1 mr-2 flex-shrink-0" />
                      <span className="text-sm text-gray-700 dark:text-gray-300">{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </Card>
          </div>
        </div>

        {/* 维度详情 */}
        {dimensionDetails && (
          <div id="dimension-details" className="mb-8">
            <Card className="p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center text-gray-900 dark:text-white">
                {dimensionDetails.name} 详细分析
                <span className="ml-3 text-sm bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-300 px-2 py-0.5 rounded-full">
                  {dimensionDetails.score}/100
                </span>
              </h2>

              <p className="text-gray-700 dark:text-gray-300 mb-6">
                {dimensionDetails.description}
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-md font-medium mb-3 text-gray-900 dark:text-white">
                    分析洞察
                  </h3>
                  <ul className="space-y-2">
                    {dimensionDetails.tips.map((tip, index) => (
                      <li key={index} className="flex items-start">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary-500 mt-1.5 mr-2"></div>
                        <span className="text-sm text-gray-700 dark:text-gray-300">{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h3 className="text-md font-medium mb-3 text-gray-900 dark:text-white">
                    与平均用户比较
                  </h3>
                  <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-gray-600 dark:text-gray-400">平均用户</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        50/100
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full mb-4">
                      <div className="h-2 bg-gray-400 rounded-full" style={{ width: '50%' }}></div>
                    </div>

                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-gray-600 dark:text-gray-400">当前地址</span>
                      <span className="text-sm font-medium text-gray-900 dark:text-white">
                        {dimensionDetails.score}/100
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                      <div
                        className="h-2 bg-primary-500 rounded-full"
                        style={{ width: `${dimensionDetails.score}%` }}
                      ></div>
                    </div>

                    <div className="mt-4 text-sm text-gray-700 dark:text-gray-300">
                      {dimensionDetails.score > 50
                        ? `该地址在${dimensionDetails.name}维度上显著高于平均水平，表现出专业用户特征。`
                        : `该地址在${dimensionDetails.name}维度上低于平均水平，可能是普通用户或新手。`}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 text-center">
                <Link href="/radar-dimensions">
                  <Button variant="outline" size="sm">
                    <FaInfoCircle className="mr-2" />
                    了解更多关于{dimensionDetails.name}的信息
                  </Button>
                </Link>
              </div>
            </Card>
          </div>
        )}

        {/* 调用行动 */}
        <div className="bg-primary-50 dark:bg-primary-900/20 p-6 rounded-lg text-center">
          <h2 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
            开始分析您的区块链地址
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4 max-w-2xl mx-auto">
            SmartAI的区块链行为分析系统可以帮助您深入了解任何以太坊地址的行为特征、风险评估和投资风格。
            立即开始分析，获取专业的区块链智能分析。
          </p>
          <Link href="/">
            <Button size="lg">返回分析页面</Button>
          </Link>
        </div>
      </main>

      {/* 页脚 */}
      <footer className="bg-white dark:bg-gray-900 mt-12 py-6 border-t border-gray-200 dark:border-gray-800">
        <div className="container mx-auto px-4 text-center">
          <div className="text-sm text-gray-500 dark:text-gray-400">
            SmartAI © {new Date().getFullYear()} - 专业的区块链行为分析平台
          </div>
          <div className="mt-2 text-xs text-gray-400 dark:text-gray-500">
            注：本页面仅作演示用途，数据为模拟数据，非实际区块链分析结果
          </div>
        </div>
      </footer>
    </div>
  );
}
