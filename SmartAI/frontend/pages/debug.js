import React from 'react';
import Layout from '../components/Layout';
import ApiTester from '../components/debug/ApiTester';

/**
 * 调试页面
 * 用于测试前后端集成
 */
export default function DebugPage() {
  return (
    <Layout>
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6 text-gray-800 dark:text-white">API调试工具</h1>
        <p className="mb-6 text-gray-600 dark:text-gray-300">
          使用此工具测试前后端集成，可以切换不同服务（Express/FastAPI）来测试API响应。
        </p>
        
        <div className="mb-8">
          <ApiTester />
        </div>
        
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 dark:bg-blue-900 dark:border-blue-600">
          <h3 className="text-lg font-medium text-blue-800 dark:text-blue-200 mb-2">注意事项</h3>
          <ul className="list-disc pl-5 text-blue-700 dark:text-blue-300">
            <li className="mb-1">确保相应的后端服务已经启动（Express:3000/FastAPI:8002）</li>
            <li className="mb-1">使用不同的测试地址可能会返回不同的模拟数据</li>
            <li className="mb-1">在真实部署环境中，Nginx将处理所有的路由到相应的后端</li>
          </ul>
        </div>
      </div>
    </Layout>
  );
} 