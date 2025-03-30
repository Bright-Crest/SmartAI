import React from 'react';
import MainLayout from '../components/MainLayout';
import Link from 'next/link';

/**
 * 500 页面 - 处理服务器端错误
 */
export default function Custom500() {
  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-12">
        <div className="bg-white shadow rounded-lg p-6 max-w-lg mx-auto text-center">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">500 - 服务器错误</h1>

          <p className="text-gray-600 mb-6">抱歉，服务器遇到了问题。请稍后再试或联系管理员。</p>

          <div className="flex justify-center space-x-4">
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              刷新页面
            </button>

            <Link href="/" className="text-blue-500 hover:text-blue-700">
              返回首页
            </Link>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}
