import React, { useState, useEffect, useRef } from 'react';
import D3NetworkGraph from './D3NetworkGraph';
import styles from './NetworkGraph.module.css';

/**
 * 网络图表容器组件 - 封装D3网络图并提供控制界面
 */
const NetworkGraph = ({
  data,
  onNodeClick,
  selectedNode,
  width = '100%',
  height = '600px',
  loading = false,
  error = null,
}) => {
  const [enablePhysics, setEnablePhysics] = useState(true);
  const [showStats, setShowStats] = useState(false);
  const [performanceData, setPerformanceData] = useState({
    nodesCount: 0,
    linksCount: 0,
    renderTime: 0,
  });
  const [internalError, setInternalError] = useState(null);
  const [shouldRender, setShouldRender] = useState(true);

  const graphRef = useRef(null);

  // 添加日志以便调试
  console.log('NetworkGraph组件接收数据:', {
    hasData: !!data,
    dataNodes: data?.nodes?.length || 0,
    dataLinks: data?.links?.length || 0,
    selectedNode,
  });

  // 调试输出数据状态
  useEffect(() => {
    console.log('NetworkGraph 接收到数据:', {
      nodesCount: data?.nodes?.length || 0,
      linksCount: data?.links?.length || 0,
      selectedNode,
    });

    // 重置错误状态
    setInternalError(null);
  }, [data, selectedNode]);

  // 更新性能数据
  useEffect(() => {
    if (data && data.nodes && data.links) {
      setPerformanceData(prev => ({
        ...prev,
        nodesCount: data.nodes.length || 0,
        linksCount: data.links.length || 0,
      }));
    }
  }, [data]);

  // 处理图表渲染完成事件
  useEffect(() => {
    const handleGraphRendered = event => {
      if (event.data && event.data.type === 'GRAPH_RENDERED') {
        setPerformanceData(prev => ({
          ...prev,
          renderTime: event.data.time || 0,
          nodesCount: event.data.nodeCount || prev.nodesCount,
          linksCount: event.data.linkCount || prev.linksCount,
        }));
      }
    };

    window.addEventListener('message', handleGraphRendered);
    return () => {
      window.removeEventListener('message', handleGraphRendered);
    };
  }, []);

  // 初始化网络图数据
  useEffect(() => {
    if (!data || data.length === 0) return;

    console.log('初始化网络图数据...');
    // 初始化逻辑

    // 清理函数
    return () => {
      // 清理逻辑
    };
  }, [data, width, height]);

  // 处理物理引擎状态变化
  useEffect(() => {
    if (!graphRef.current) return;

    // 根据enablePhysics处理不同的逻辑
    // 更新逻辑

    // 清理函数
    return () => {
      // 清理逻辑
    };
  }, [enablePhysics]);

  // 处理统计信息显示状态变化
  useEffect(() => {
    if (!graphRef.current) return;

    // 根据showStats处理不同的逻辑
    // 更新主题逻辑

    // 清理函数
    return () => {
      // 清理逻辑
    };
  }, [showStats]);

  // 切换物理引擎
  const togglePhysics = () => {
    setEnablePhysics(!enablePhysics);
  };

  // 切换统计显示
  const toggleStats = () => {
    setShowStats(!showStats);
  };

  // 处理节点点击
  const handleNodeClick = node => {
    console.log('NetworkGraph: 节点点击 ->', node);
    if (onNodeClick) {
      onNodeClick(node);
    }
  };

  // 数据验证
  const isValidData =
    data && data.nodes && Array.isArray(data.nodes) && data.links && Array.isArray(data.links);

  // 如果没有数据或格式不正确
  if (!isValidData && !loading) {
    console.error('NetworkGraph: 无效的数据格式', data);
    return (
      <div className={styles.errorContainer}>
        <p>无效的图形数据</p>
      </div>
    );
  }

  // 如果正在加载
  if (loading && (!data?.nodes || data.nodes.length === 0)) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>加载图表中...</p>
      </div>
    );
  }

  // 如果有错误
  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p>{error}</p>
      </div>
    );
  }

  // 确保有效的数据
  if (!isValidData || !data.nodes.length) {
    return (
      <div className={styles.emptyContainer}>
        <p>没有可显示的图表数据</p>
      </div>
    );
  }

  return (
    <div className={styles.graphContainer}>
      {/* 控制面板 */}
      <div className={styles.controlsContainer}>
        <button
          className={styles.controlButton}
          onClick={togglePhysics}
          title={enablePhysics ? '固定节点位置' : '启用物理引擎'}
        >
          {enablePhysics ? '🔒 固定节点' : '🔓 释放节点'}
        </button>
        <button
          className={styles.controlButton}
          onClick={toggleStats}
          title={showStats ? '隐藏统计信息' : '显示统计信息'}
        >
          📊 {showStats ? '隐藏统计' : '显示统计'}
        </button>
      </div>

      {/* 性能统计信息 */}
      {showStats && (
        <div className={styles.performanceInfo}>
          <p>节点数量: {performanceData.nodesCount}</p>
          <p>连接数量: {performanceData.linksCount}</p>
          <p>渲染时间: {performanceData.renderTime}ms</p>
        </div>
      )}

      {/* 主图表组件 */}
      <div className={styles.d3Container} ref={graphRef}>
        <D3NetworkGraph
          nodes={data.nodes}
          links={data.links}
          onNodeClick={handleNodeClick}
          selectedNode={selectedNode}
          width={width}
          height={height}
          enablePhysics={enablePhysics}
        />
      </div>
    </div>
  );
};

export default NetworkGraph;
