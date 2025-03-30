import React from 'react';
import MainLayout from '../components/MainLayout';
import Link from 'next/link';

/**
 * 404 页面 - 处理未找到的路由
 */
export default function Custom404() {
  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-12">
        <div className="bg-white shadow rounded-lg p-6 max-w-lg mx-auto text-center">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">404 - 页面未找到</h1>

          <p className="text-gray-600 mb-6">抱歉，您请求的页面不存在。请检查URL或返回首页。</p>

          <div className="flex justify-center">
            <Link
              href="/"
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 inline-block"
            >
              返回首页
            </Link>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}
