import React from 'react';
import Head from 'next/head';
import Link from 'next/link';

/**
 * 主布局组件 - 提供所有页面通用的布局结构
 *
 * @param {Object} props - 组件属性
 * @param {React.ReactNode} props.children - 子组件内容
 * @param {string} props.title - 页面标题
 * @param {string} props.description - 页面描述（用于SEO）
 */
export default function MainLayout({
  children,
  title = 'ChainIntelAI',
  description = 'ChainIntelAI区块链情报分析平台',
}) {
  return (
    <div
      style={{
        padding: '20px',
        fontFamily: 'Arial, sans-serif',
        maxWidth: '1200px',
        margin: '0 auto',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header
        style={{
          marginBottom: '30px',
          borderBottom: '1px solid #eaeaea',
          paddingBottom: '20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div>
          <h1 style={{ color: '#2c3e50', fontSize: '2rem', margin: 0 }}>
            <Link href="/" legacyBehavior>
              <a style={{ color: 'inherit', textDecoration: 'none' }}>ChainIntelAI</a>
            </Link>
          </h1>
          <p style={{ color: '#7f8c8d', margin: '5px 0 0 0' }}>区块链情报分析平台</p>
        </div>

        <nav>
          <ul
            style={{
              display: 'flex',
              listStyle: 'none',
              gap: '20px',
              margin: 0,
              padding: 0,
            }}
          >
            <li>
              <Link href="/" legacyBehavior>
                <a
                  style={{
                    color: '#3498db',
                    textDecoration: 'none',
                    fontWeight: 'bold',
                  }}
                >
                  首页
                </a>
              </Link>
            </li>
            <li>
              <Link href="/network" legacyBehavior>
                <a
                  style={{
                    color: '#3498db',
                    textDecoration: 'none',
                    fontWeight: 'bold',
                  }}
                >
                  网络分析
                </a>
              </Link>
            </li>
            <li>
              <Link href="/verify" legacyBehavior>
                <a
                  style={{
                    color: '#3498db',
                    textDecoration: 'none',
                    fontWeight: 'bold',
                  }}
                >
                  系统验证
                </a>
              </Link>
            </li>
          </ul>
        </nav>
      </header>

      <main style={{ flex: 1 }}>{children}</main>

      <footer
        style={{
          marginTop: '50px',
          borderTop: '1px solid #eaeaea',
          paddingTop: '20px',
          color: '#7f8c8d',
          textAlign: 'center',
        }}
      >
        <p>© {new Date().getFullYear()} ChainIntelAI - 区块链情报分析平台</p>
      </footer>
    </div>
  );
}
