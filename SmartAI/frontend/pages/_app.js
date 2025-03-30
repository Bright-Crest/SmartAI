import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Script from 'next/script';
import { useRouter } from 'next/router';
import dynamic from 'next/dynamic';
import '../styles/globals.css';
import ErrorBoundary from '../components/ErrorBoundary';
import { ToastContainer, useToast } from '../components/ui/Toast';
import { setGlobalErrorHandler } from '../utils/errorHandler';
import { initPerformanceMonitoring } from '../utils/analytics';
// 由于已经在i18n/index.js中进行了安全初始化，这里可以直接导入
import '../i18n';
import { ChakraProvider } from '@chakra-ui/react';
// 导入外部主题文件
import theme from '../theme';

// 创建React上下文用于Toast
export const ToastContext = React.createContext(null);

// 动态导入TranslatedErrorBoundary组件，禁用SSR
const DynamicTranslatedErrorBoundary = dynamic(
  () => import('../components/ErrorBoundary').then(mod => mod.TranslatedErrorBoundary),
  {
    ssr: false,
    loading: () => <ErrorBoundary />,
  }
);

function AppContent({ Component, pageProps }) {
  const router = useRouter();
  const toast = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // 客户端挂载检测
  useEffect(() => {
    setIsMounted(true);

    // 仅在客户端执行的初始化工作
    setGlobalErrorHandler(appError => {
      toast.error(appError.getFriendlyMessage());
      console.error('[全局错误]', appError.getLogInfo());
    });

    // 初始化性能监控
    initPerformanceMonitoring();

    // 路由变化监听
    const handleStart = () => setIsLoading(true);
    const handleComplete = () => setIsLoading(false);

    router.events.on('routeChangeStart', handleStart);
    router.events.on('routeChangeComplete', handleComplete);
    router.events.on('routeChangeError', handleComplete);

    // 预加载D3库
    if (typeof window !== 'undefined' && !window.d3) {
      console.log('预加载D3库到全局window对象...');

      // 动态加载D3并添加到全局对象
      import('d3')
        .then(d3Module => {
          // 将所有d3导出的方法直接挂载到window.d3上
          window.d3 = {};
          Object.keys(d3Module).forEach(key => {
            window.d3[key] = d3Module[key];
          });
          console.log('D3库加载成功，版本:', window.d3?.version);

          // 在控制台中显示可用的D3方法，以便调试
          if (process.env.NODE_ENV === 'development') {
            console.log('D3 methods:', Object.keys(window.d3).slice(0, 20).join(', ') + '...');
          }
        })
        .catch(err => {
          console.error('D3库加载失败:', err);
        });
    }

    return () => {
      router.events.off('routeChangeStart', handleStart);
      router.events.off('routeChangeComplete', handleComplete);
      router.events.off('routeChangeError', handleComplete);
    };
  }, [router, toast]);

  // 处理错误边界捕获的错误
  const handleError = (error, errorInfo) => {
    console.error('React Error Boundary 捕获错误:', error, errorInfo);
    // 这里可以添加错误上报逻辑
  };

  return (
    <ToastContext.Provider value={toast}>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>SmartAI - 区块链风险智能分析平台</title>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔗</text></svg>"
        />
      </Head>

      {/* 使用next/script正确引入D3.js */}
      <Script
        src="https://d3js.org/d3.v7.min.js"
        strategy="beforeInteractive"
        onLoad={() => {
          console.log('D3.js库加载成功，版本:', window.d3?.version);
        }}
      />

      {/* 页面切换进度条 */}
      {isLoading && (
        <div className="fixed top-0 left-0 right-0 h-1 bg-blue-600 dark:bg-blue-500 z-50 animate-pulse" />
      )}

      {/* 使用ErrorBoundary，仅在客户端渲染时使用国际化版本 */}
      <ErrorBoundary onError={handleError}>
        {isMounted ? <Component {...pageProps} /> : <Component {...pageProps} />}
      </ErrorBoundary>

      {/* Toast通知容器 */}
      <ToastContainer toasts={toast.toasts} removeToast={toast.removeToast} />
    </ToastContext.Provider>
  );
}

function MyApp({ Component, pageProps }) {
  return (
    <ChakraProvider theme={theme}>
      <AppContent Component={Component} pageProps={pageProps} />
    </ChakraProvider>
  );
}

export default MyApp;
