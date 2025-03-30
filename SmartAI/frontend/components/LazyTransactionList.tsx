import React, { Suspense, lazy } from 'react';

// 使用React.lazy懒加载TransactionList组件
const TransactionList = lazy(() => 
  import('./TransactionList')
    .then(module => ({
      default: module.default
    }))
);

// 加载状态组件
const LoadingFallback = () => (
  <div className="animate-pulse space-y-4">
    <div className="h-10 bg-gray-200 rounded w-full max-w-md"></div>
    <div className="h-8 bg-gray-100 rounded-full w-2/3"></div>
    <div className="space-y-2">
      {Array(5).fill(0).map((_, index) => (
        <div key={index} className="h-16 bg-gray-100 rounded"></div>
      ))}
    </div>
  </div>
);

// 错误处理组件
const ErrorBoundary = ({ children }: { children: React.ReactNode }) => {
  const [hasError, setHasError] = React.useState(false);

  React.useEffect(() => {
    const handleError = () => setHasError(true);
    window.addEventListener('error', handleError);
    return () => window.removeEventListener('error', handleError);
  }, []);

  if (hasError) {
    return (
      <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded">
        <h2 className="font-bold text-lg">加载组件失败</h2>
        <p>请刷新页面重试</p>
        <button 
          onClick={() => window.location.reload()}
          className="mt-2 bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 rounded transition-colors"
        >
          刷新页面
        </button>
      </div>
    );
  }

  return <>{children}</>;
};

/**
 * 懒加载的交易列表组件
 * 使用React.lazy和Suspense实现代码分割，减少初始加载时间
 */
const LazyTransactionList: React.FC = () => {
  return (
    <ErrorBoundary>
      <Suspense fallback={<LoadingFallback />}>
        <TransactionList />
      </Suspense>
    </ErrorBoundary>
  );
};

export default LazyTransactionList; 