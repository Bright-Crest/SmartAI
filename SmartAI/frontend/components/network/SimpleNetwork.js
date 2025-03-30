import React from 'react';
import styles from './NetworkGraph.module.css';

/**
 * 简化版的网络图组件，不使用D3
 */
const SimpleNetwork = ({ data, onNodeClick, selectedNode, width = '100%', height = '600px' }) => {
  // 简单记录props信息
  console.log('SimpleNetwork组件接收数据:', {
    dataNodes: data?.nodes?.length || 0,
    dataLinks: data?.links?.length || 0,
    selectedNode,
  });

  // 简单检查数据是否有效
  if (!data || !data.nodes || !Array.isArray(data.nodes) || data.nodes.length === 0) {
    return (
      <div className={styles.errorContainer}>
        <p>无效的图形数据</p>
      </div>
    );
  }

  return (
    <div className={styles.container} style={{ width, height }}>
      <div className={styles.simpleNetwork}>
        <h3>简化版网络图 (不使用D3)</h3>
        <p>
          已加载 {data.nodes.length} 个节点和 {data.links.length} 个连接
        </p>

        <div className={styles.nodeList}>
          <h4>节点列表：</h4>
          <ul>
            {data.nodes.slice(0, 5).map(node => (
              <li
                key={node.id}
                className={selectedNode === node.id ? styles.selectedNode : ''}
                onClick={() => onNodeClick && onNodeClick(node)}
              >
                {node.label}
                {node.type && <span className={styles.nodeType}>({node.type})</span>}
                {node.riskScore !== undefined && (
                  <span className={styles.riskScore}>风险: {node.riskScore}</span>
                )}
              </li>
            ))}
            {data.nodes.length > 5 && <li>...还有 {data.nodes.length - 5} 个节点</li>}
          </ul>
        </div>

        <div className={styles.linkList}>
          <h4>连接列表：</h4>
          <ul>
            {data.links.slice(0, 5).map(link => (
              <li key={link.id}>
                {link.source} → {link.target}
                {link.type && <span className={styles.linkType}>({link.type})</span>}
              </li>
            ))}
            {data.links.length > 5 && <li>...还有 {data.links.length - 5} 个连接</li>}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SimpleNetwork;
