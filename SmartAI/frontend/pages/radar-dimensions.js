import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import {
  FaChartPie,
  FaArrowLeft,
  FaExchangeAlt,
  FaCode,
  FaNetworkWired,
  FaChartLine,
  FaServer,
} from 'react-icons/fa';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

// 动态导入RadarChart组件，禁用SSR
const DynamicRadarChart = dynamic(() => import('../components/RadarChart'), {
  ssr: false,
  loading: () => (
    <div className="h-[400px] flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  ),
});

// 五维数据示例
const exampleData = {
  activityScore: 85,
  protocolInteractionScore: 72,
  profitabilityScore: 64,
  fundFlowScore: 90,
  distillationScore: 78,
};

// 维度详细信息
const dimensionDetails = [
  {
    key: 'activityScore',
    title: '活跃度',
    icon: <FaExchangeAlt className="text-blue-500" />,
    description:
      '活跃度维度评估钱包地址在区块链上的活动频率、规律性和连续性，反映用户的参与程度和使用习惯。',
    calculation: '基于交易频率、交易时间分布、活动周期和持续时间等因素计算。',
    highScore: '高分（80-100）表示该地址频繁交易，活动规律，可能是日常活跃用户或交易机器人。',
    mediumScore: '中分（50-79）表示该地址有规律但不频繁的交易行为，可能是普通活跃用户。',
    lowScore: '低分（0-49）表示该地址交易不频繁，可能是长期持有者或非活跃账户。',
    examples: [
      { name: '交易频率', description: '每日/每周/每月交易次数' },
      { name: '时间分布', description: '交易时间的分布模式' },
      { name: '连续性', description: '活动的持续时间和间隔' },
    ],
  },
  {
    key: 'protocolInteractionScore',
    title: '协议互动',
    icon: <FaServer className="text-purple-500" />,
    description:
      '协议互动维度衡量地址与不同DeFi协议和智能合约的交互广度和深度，反映用户对区块链生态系统的参与度。',
    calculation: '通过分析与不同协议的交互次数、交互类型复杂度和交互多样性计算。',
    highScore: '高分（80-100）表示该地址与多种协议有深入交互，使用复杂功能，可能是DeFi高级用户。',
    mediumScore: '中分（50-79）表示该地址使用多个常见协议，具有一定的交互深度。',
    lowScore: '低分（0-49）表示该地址只与少数简单协议交互，或主要进行基础转账。',
    examples: [
      { name: '协议多样性', description: '交互过的不同DeFi协议数量' },
      { name: '功能复杂度', description: '使用协议高级功能的频率' },
      { name: '交互深度', description: '与每个协议的交互深度' },
    ],
  },
  {
    key: 'profitabilityScore',
    title: '收益能力',
    icon: <FaChartLine className="text-green-500" />,
    description: '收益能力维度分析地址的投资和交易盈利能力，评估用户在市场中的表现和策略有效性。',
    calculation: '基于历史交易盈亏率、投资回报率、交易时机选择和套利效率等因素计算。',
    highScore: '高分（80-100）表示该地址具有优秀的盈利记录，可能是成功的交易者或投资者。',
    mediumScore: '中分（50-79）表示该地址有稳定的盈利能力，但不一定出众。',
    lowScore: '低分（0-49）表示该地址盈利能力有限或有亏损记录。',
    examples: [
      { name: '交易盈亏', description: '历史交易的盈亏比率' },
      { name: '市场时机', description: '交易时机选择的准确度' },
      { name: '套利效率', description: '套利交易的速度和效率' },
    ],
  },
  {
    key: 'fundFlowScore',
    title: '资金流',
    icon: <FaNetworkWired className="text-orange-500" />,
    description:
      '资金流维度追踪和分析地址资金的流入流出模式、规模和网络特征，反映资金活动的性质和影响力。',
    calculation: '通过分析资金流量大小、流向模式、交易对手多样性和资金集中度计算。',
    highScore: '高分（80-100）表示该地址有大量资金流动，可能是交易所、大型机构或鲸鱼账户。',
    mediumScore: '中分（50-79）表示该地址有适中的资金流动，交易对手较多样。',
    lowScore: '低分（0-49）表示该地址资金流动较小或简单，交易对手有限。',
    examples: [
      { name: '资金规模', description: '交易资金的总体规模' },
      { name: '流向模式', description: '资金流入流出的平衡性' },
      { name: '网络集中度', description: '资金流动的集中或分散程度' },
    ],
  },
  {
    key: 'distillationScore',
    title: '蒸馏模型',
    icon: <FaCode className="text-red-500" />,
    description:
      '蒸馏模型维度是基于机器学习的综合行为评估，通过分析复杂模式来识别行为的专业性和复杂度。',
    calculation: '使用神经网络和机器学习算法，从用户交易历史中提取特征，综合评估行为模式。',
    highScore: '高分（80-100）表示该地址行为模式复杂且专业，可能是专业交易者或开发者。',
    mediumScore: '中分（50-79）表示该地址有一定的行为复杂度，属于有经验的用户。',
    lowScore: '低分（0-49）表示该地址行为模式简单或常见，可能是普通或新手用户。',
    examples: [
      { name: '行为复杂度', description: '交易行为的复杂程度' },
      { name: '模式独特性', description: '行为模式的独特程度' },
      { name: '智能合约使用', description: '与智能合约交互的复杂度' },
    ],
  },
];

/**
 * 五维雷达图维度说明页面
 * 详细解释各个维度的含义和计算方式
 */
export default function RadarDimensionsPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Head>
        <title>五维雷达图说明 | SmartAI</title>
        <meta
          name="description"
          content="了解SmartAI区块链行为分析的五个维度：活跃度、协议互动、收益能力、资金流和蒸馏模型"
        />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <Link
              href="/"
              className="inline-flex items-center px-4 py-2 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
            >
              <FaArrowLeft className="mr-2" />
              返回首页
            </Link>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">五维雷达图维度说明</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-300 max-w-3xl">
              SmartAI通过五个关键维度分析区块链地址的行为特征，全面评估钱包的使用模式和风险特征。
              了解每个维度的含义，帮助您更好地理解智能分析结果。
            </p>
          </div>
        </div>

        {/* 示例雷达图 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="md:col-span-1">
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <FaChartPie className="mr-2 text-primary-500" />
                示例雷达图
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                右侧是一个示例雷达图，展示五个维度的评分。每个维度的分数范围从0到100，
                分数越高表示该维度的特征越显著。您可以点击各个维度查看详细信息。
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                在实际分析中，这些维度的组合构成了地址的整体行为特征，并用于计算SmartScore。
              </p>
            </Card>
          </div>

          <div className="md:col-span-2">
            <DynamicRadarChart
              data={exampleData}
              interactive={true}
              showAverageUser={true}
              showLegend={true}
            />
          </div>
        </div>

        {/* 维度详细说明 */}
        <div className="space-y-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">维度详细说明</h2>

          {dimensionDetails.map((dimension, index) => (
            <Card key={dimension.key} className="p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center mr-4">
                  {dimension.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {dimension.title}
                  <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">
                    (示例评分: {exampleData[dimension.key]}/100)
                  </span>
                </h3>
              </div>

              <p className="text-gray-700 dark:text-gray-300 mb-4">{dimension.description}</p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
                <div>
                  <h4 className="text-md font-medium text-gray-900 dark:text-white mb-2">
                    计算方式
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {dimension.calculation}
                  </p>
                </div>

                <div className="space-y-2">
                  <h4 className="text-md font-medium text-gray-900 dark:text-white mb-2">
                    分数解释
                  </h4>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {dimension.highScore}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {dimension.mediumScore}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{dimension.lowScore}</p>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">
                  评估指标示例
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {dimension.examples.map((example, i) => (
                    <div key={i} className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                      <h5 className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                        {example.name}
                      </h5>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {example.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          ))}
        </div>

        <div className="mt-10 bg-primary-50 dark:bg-primary-900/20 p-6 rounded-lg">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3">了解更多</h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            五维雷达图是SmartScore系统的重要组成部分，通过综合评估这五个维度，系统能够计算出地址的智能评分，
            并确定其行为类型和特征标签。要探索更多功能，请尝试我们的智能分析工具。
          </p>
          <div className="flex gap-4">
            <Link
              href="/smartscore-demo"
              className="inline-flex items-center px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              <FaChartPie className="mr-2" />
              试用SmartScore演示
            </Link>
            <Link
              href="/"
              className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              分析新地址
            </Link>
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
