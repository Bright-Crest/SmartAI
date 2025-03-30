import React, { useState, useEffect, useCallback } from 'react';
import dynamic from 'next/dynamic';
import styles from '../../styles/AddressTracking.module.css';
import Head from 'next/head';

// 使用动态导入避免SSR渲染问题
const D3NetworkGraph = dynamic(
  () => import('../../components/network/D3NetworkGraph'),
  { ssr: false }
);

// 创建一个模拟的节点和连接数据函数
const generateGraphData = (address) => {
  // 创建一个模拟的节点和连接数据，模拟区块链网络图
  const nodes = [];
  const links = [];
  
  // 添加中心节点（输入的地址）
  const centerNode = {
    id: address,
    label: `${address.substring(0, 6)}...${address.substring(address.length - 4)}`,
    type: 'eoa', // 默认为普通外部账户
    riskScore: Math.random() > 0.7 ? Math.floor(Math.random() * 100) : Math.floor(Math.random() * 40),
    balance: `${(Math.random() * 10).toFixed(2)} ETH`,
    totalTx: Math.floor(Math.random() * 1000),
    // 中心节点较大
    radius: 25,
    value: 50,
  };
  nodes.push(centerNode);
  
  // 生成相关节点数量
  const relatedNodesCount = Math.floor(Math.random() * 5) + 4; // 生成4-8个相关节点
  
  // 预定义一些节点类型供随机选择
  const nodeTypes = ['exchange', 'defi', 'contract', 'eoa', 'whale', 'mixer'];
  
  // 添加相关节点
  for (let i = 0; i < relatedNodesCount; i++) {
    const nodeId = `0x${Math.random().toString(16).substring(2, 10)}${Math.random().toString(16).substring(2, 10)}`;
    const nodeType = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
    
    // 根据节点类型设置不同的风险评分分布
    let riskScore;
    switch (nodeType) {
      case 'mixer':
        riskScore = 80 + Math.floor(Math.random() * 20); // 混币器高风险
        break;
      case 'exchange':
        riskScore = 10 + Math.floor(Math.random() * 30); // 交易所低风险
        break;
      case 'whale':
        riskScore = 30 + Math.floor(Math.random() * 40); // 大户中等风险
        break;
      default:
        riskScore = Math.floor(Math.random() * 100); // 其他随机风险
    }
    
    // 创建节点
    const node = {
      id: nodeId,
      label: `${nodeId.substring(0, 6)}...${nodeId.substring(nodeId.length - 4)}`,
      type: nodeType,
      riskScore,
      balance: `${(Math.random() * 100).toFixed(2)} ETH`,
      totalTx: Math.floor(Math.random() * 500),
      radius: 15 + Math.floor(Math.random() * 10), // 半径15-25之间
      value: 20 + Math.floor(Math.random() * 30),
    };
    nodes.push(node);
    
    // 生成从中心节点到该节点的连接
    const sourceToTarget = {
      id: `${address}->${nodeId}`,
      source: address,
      target: nodeId,
      value: 1 + Math.floor(Math.random() * 5),
      type: Math.random() > 0.5 ? 'transfer' : 'contract',
      timestamp: new Date(Date.now() - Math.floor(Math.random() * 30) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    };
    links.push(sourceToTarget);
    
    // 有50%的概率生成从该节点到中心节点的连接
    if (Math.random() > 0.5) {
      const targetToSource = {
        id: `${nodeId}->${address}`,
        source: nodeId,
        target: address,
        value: 1 + Math.floor(Math.random() * 5),
        type: Math.random() > 0.7 ? 'defi' : 'transfer',
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 30) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      };
      links.push(targetToSource);
    }
  }
  
  // 生成相关节点之间的一些连接，使网络更真实
  for (let i = 0; i < relatedNodesCount; i++) {
    const sourceIndex = 1 + Math.floor(Math.random() * relatedNodesCount);
    let targetIndex = 1 + Math.floor(Math.random() * relatedNodesCount);
    
    // 确保不自己连自己
    while (targetIndex === sourceIndex) {
      targetIndex = 1 + Math.floor(Math.random() * relatedNodesCount);
    }
    
    // 确保索引在有效范围内
    if (sourceIndex < nodes.length && targetIndex < nodes.length) {
      const source = nodes[sourceIndex];
      const target = nodes[targetIndex];
      
      links.push({
        id: `${source.id}->${target.id}`,
        source: source.id,
        target: target.id,
        value: 1 + Math.floor(Math.random() * 3),
        type: Math.random() > 0.5 ? 'transfer' : 'contract',
        timestamp: new Date(Date.now() - Math.floor(Math.random() * 30) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      });
    }
  }
  
  return { nodes, links };
};

// 主组件
function AddressTrackingPage() {
  const [address, setAddress] = useState('');
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [enablePhysics, setEnablePhysics] = useState(true);
  const [showHints, setShowHints] = useState(true);

  // 处理地址搜索
  const handleSearch = useCallback(() => {
    if (!address.trim()) {
      setError('请输入有效的地址');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSelectedNode(null);
    
    // 模拟API调用延迟
    setTimeout(() => {
      try {
        const data = generateGraphData(address);
        setGraphData(data);
        setLoading(false);
      } catch (err) {
        console.error('获取数据出错:', err);
        setError('获取地址数据时出错');
        setLoading(false);
      }
    }, 1000);
  }, [address]);

  // 处理回车键搜索
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // 处理节点点击
  const handleNodeClick = useCallback((node) => {
    setSelectedNode(node);
  }, []);

  // 重置布局
  const resetLayout = () => {
    if (graphData && graphData.nodes.length > 0) {
      setGraphData({
        nodes: [...graphData.nodes],
        links: [...graphData.links]
      });
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>区块链地址追踪分析 | SmartAI</title>
      </Head>

      <h1 className={styles.title}>区块链地址追踪分析</h1>

      <div className={styles.searchSection}>
        <div className={styles.searchBox}>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="输入钱包地址、交易哈希或ENS名称..."
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button 
            className={styles.searchButton} 
            onClick={handleSearch}
            disabled={loading || !address.trim()}
          >
            {loading ? (
              <>
                <div className={styles.spinner}></div>
                <span>分析中...</span>
              </>
            ) : (
              <span>开始分析</span>
            )}
          </button>
        </div>

        {showHints && (
          <div className={styles.exampleAddresses}>
            <p>示例地址：</p>
            <button onClick={() => setAddress('0x28c6c06298d514db089934071355e5743bf21d60')}>
              Binance冷钱包
            </button>
            <button onClick={() => setAddress('0xdead000000000000000042069420694206942069')}>
              DeadWallet
            </button>
            <button onClick={() => setAddress('vitalik.eth')}>
              vitalik.eth
            </button>
            <button 
              className={styles.hintToggle}
              onClick={() => setShowHints(false)}
            >
              隐藏示例
            </button>
          </div>
        )}

        {!showHints && (
          <button 
            className={styles.hintToggle}
            onClick={() => setShowHints(true)}
          >
            显示示例地址
          </button>
        )}

        {error && (
          <div className={styles.errorMessage}>
            <p>{error}</p>
          </div>
        )}
      </div>

      <div className={styles.contentContainer}>
        {loading && !graphData.nodes.length && (
          <div className={styles.loadingOverlay}>
            <div className={styles.spinner}></div>
            <p className={styles.loadingText}>正在分析地址关系...</p>
            <p className={styles.loadingSubtext}>这可能需要几分钟时间，请耐心等待</p>
          </div>
        )}

        {!loading && !graphData.nodes.length && !error && (
          <div className={styles.emptyState}>
            <p>请输入地址开始追踪分析</p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <>
            <div className={styles.graphContainer}>
              <D3NetworkGraph
                data={graphData}
                onNodeClick={handleNodeClick}
                selectedNode={selectedNode ? selectedNode.id : null}
                enablePhysics={enablePhysics}
              />
              
              <div className={styles.graphControls}>
                <button 
                  className={styles.controlButton}
                  title="重新布局"
                  onClick={resetLayout}
                >
                  ↻
                </button>
                <button 
                  className={styles.controlButton}
                  title={enablePhysics ? "锁定节点位置" : "启用物理引擎"}
                  onClick={() => setEnablePhysics(!enablePhysics)}
                >
                  {enablePhysics ? "🔓" : "🔒"}
                </button>
              </div>
            </div>

            {selectedNode && (
              <div className={styles.nodeDetails}>
                <h3>{selectedNode.label || selectedNode.id}</h3>
                {selectedNode.type && (
                  <span className={styles.nodeType} data-type={selectedNode.type.toLowerCase()}>
                    {selectedNode.type}
                  </span>
                )}
                
                <div className={styles.detailsInfo}>
                  <div className={styles.detailItem}>
                    <strong>地址:</strong> {selectedNode.id}
                  </div>
                  
                  {selectedNode.balance && (
                    <div className={styles.detailItem}>
                      <strong>余额:</strong> {selectedNode.balance}
                    </div>
                  )}
                  
                  {selectedNode.totalTx && (
                    <div className={styles.detailItem}>
                      <strong>交易次数:</strong> {selectedNode.totalTx}
                    </div>
                  )}
                  
                  {selectedNode.riskScore !== undefined && (
                    <div className={styles.detailItem}>
                      <strong>风险评分:</strong> 
                      <div className={styles.riskScore}>
                        <div 
                          className={styles.riskBar} 
                          style={{
                            width: `${selectedNode.riskScore}%`,
                            backgroundColor: 
                              selectedNode.riskScore > 80 ? '#f44336' :
                              selectedNode.riskScore > 60 ? '#ff9800' :
                              selectedNode.riskScore > 40 ? '#ffc107' : '#4caf50'
                          }}
                        ></div>
                        <span>{selectedNode.riskScore}%</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default AddressTrackingPage; 