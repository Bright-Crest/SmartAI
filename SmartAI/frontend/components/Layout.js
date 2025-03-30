import React from 'react';
import Head from 'next/head';
import Navbar from './Navbar';
import Footer from './Footer';

/**
 * 页面布局组件
 *
 * @param {Object} props 组件属性
 * @param {React.ReactNode} props.children 子组件
 * @returns {JSX.Element} 布局组件
 */
export default function Layout({ children }) {
  return (
    <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      {/* 导航栏 */}
      <Navbar />

      {/* 主要内容 */}
      <main className="flex-grow">{children}</main>

      {/* 页脚 */}
      <Footer />
    </div>
  );
}
