import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import {
  FaSearch,
  FaEthereum,
  FaChartLine,
  FaNetworkWired,
  FaUserAlt,
  FaBrain,
} from 'react-icons/fa';
import { sampleAddresses } from '../utils/mockData';
import { isValidEthereumAddress } from '../api/addressAnalysis';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import { cn } from '../utils/cn';

// 特性卡片组件
const FeatureCard = ({ icon, title, description }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
      <div className="flex justify-center mb-4">{icon}</div>
      <h3 className="text-lg font-semibold mb-2 text-center text-gray-800 dark:text-white">
        {title}
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-300 text-center">{description}</p>
    </div>
  );
};

export default function HomePage() {
  const [address, setAddress] = useState('');
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const router = useRouter();

  // 处理地址分析
  const handleAnalyze = e => {
    e.preventDefault();

    // 验证地址
    if (!address.trim()) {
      setIsError(true);
      setErrorMessage('请输入以太坊地址');
      return;
    }

    if (!isValidEthereumAddress(address.trim())) {
      setIsError(true);
      setErrorMessage('无效的以太坊地址格式');
      return;
    }

    // 导航到聪明钱分析页面
    router.push(`/smart-money/address/${address.trim()}`);
  };

  // 选择示例地址
  const selectSampleAddress = sampleAddress => {
    setAddress(sampleAddress);
    setIsError(false);
  };

  return (
    <>
      <Head>
        <title>SmartAI - 区块链智能分析平台</title>
        <meta name="description" content="输入任意以太坊钱包地址，一键分析获得聪明钱分析报告" />
      </Head>

      <div className="min-h-screen flex flex-col">
        {/* 主内容区 */}
        <main className="flex-grow flex flex-col items-center justify-center p-4 sm:p-8 bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-indigo-950">
          <div className="w-full max-w-3xl mx-auto text-center">
            <div className="mb-8">
              <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
                <span className="text-primary">Smart</span>AI
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300">
                输入任意以太坊钱包地址，一键分析聪明钱投资行为
              </p>
            </div>

            {/* 搜索框 */}
            <form onSubmit={handleAnalyze} className="mb-8">
              <div className="relative">
                <div className="flex">
                  <div className="relative flex-grow">
                    <Input
                      type="text"
                      value={address}
                      onChange={e => {
                        setAddress(e.target.value);
                        setIsError(false);
                      }}
                      placeholder="输入以太坊地址 (0x...)"
                      className={cn(
                        'pr-12 text-base py-6 rounded-r-none',
                        isError ? 'border-red-500 focus-visible:ring-red-500' : ''
                      )}
                    />
                    <FaEthereum className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  </div>
                  <Button type="submit" className="rounded-l-none py-6 px-8 text-base font-medium">
                    <FaSearch className="mr-2" />
                    聪明钱分析
                  </Button>
                </div>
                {isError && <p className="absolute text-sm text-red-500 mt-1">{errorMessage}</p>}
              </div>
            </form>

            {/* 示例地址 */}
            <div className="mb-12">
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">示例地址:</p>
              <div className="flex flex-wrap justify-center gap-2">
                {sampleAddresses.map((sampleAddress, index) => (
                  <button
                    key={index}
                    onClick={() => selectSampleAddress(sampleAddress)}
                    className="px-3 py-1 text-xs rounded-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  >
                    {`${sampleAddress.substring(0, 6)}...${sampleAddress.substring(sampleAddress.length - 4)}`}
                  </button>
                ))}
              </div>
            </div>

            {/* 特性介绍 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
                <div className="text-primary text-3xl mb-3 flex justify-center">
                  <FaChartLine />
                </div>
                <h3 className="text-lg font-semibold mb-2">聪明钱识别</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  智能评估钱包是否为"聪明钱"，提供综合评分和特征分析
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
                <div className="text-primary text-3xl mb-3 flex justify-center">
                  <FaUserAlt />
                </div>
                <h3 className="text-lg font-semibold mb-2">投资组合分析</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  深入分析投资组合分布，追踪资产配置和风险管理策略
                </p>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md">
                <div className="text-primary text-3xl mb-3 flex justify-center">
                  <FaNetworkWired />
                </div>
                <h3 className="text-lg font-semibold mb-2">交易模式分析</h3>
                <p className="text-gray-600 dark:text-gray-300">
                  分析交易频率、规模和时机把握，洞察成功交易策略
                </p>
              </div>
            </div>

            <div className="mt-10 text-center">
              <Button
                onClick={() => router.push('/smart-money')}
                className="py-4 px-8 text-base font-medium"
              >
                <FaBrain className="mr-2" />
                进入聪明钱分析平台
              </Button>
            </div>
          </div>
        </main>

        {/* 页脚 */}
        <footer className="bg-white dark:bg-gray-900 py-4 border-t border-gray-200 dark:border-gray-800">
          <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
            SmartAI © {new Date().getFullYear()} - 区块链智能分析平台
          </div>
        </footer>
      </div>
    </>
  );
}
