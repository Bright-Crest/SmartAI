import React from 'react';
import MainLayout from '../components/MainLayout';
import Link from 'next/link';

/**
 * Next.js 默认错误处理组件
 * 处理应用运行时的错误
 */
function Error({ statusCode, errorMessage }) {
  return (
    <MainLayout>
      <div className="container mx-auto px-4 py-12">
        <div className="bg-white shadow rounded-lg p-6 max-w-lg mx-auto">
          <h1 className="text-2xl font-bold text-red-600 mb-4">
            {statusCode ? `错误 ${statusCode}` : '应用错误'}
          </h1>

          <p className="text-gray-700 mb-6">
            {errorMessage || '抱歉，应用遇到了一些问题。请尝试刷新页面或返回首页。'}
          </p>

          <div className="flex space-x-4">
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              刷新页面
            </button>

            <Link
              href="/"
              className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 inline-block"
            >
              返回首页
            </Link>
          </div>
        </div>
      </div>
    </MainLayout>
  );
}

Error.getInitialProps = ({ res, err }) => {
  const statusCode = res ? res.statusCode : err ? err.statusCode : 404;
  const errorMessage = err ? err.message : null;
  return { statusCode, errorMessage };
};

export default Error;
