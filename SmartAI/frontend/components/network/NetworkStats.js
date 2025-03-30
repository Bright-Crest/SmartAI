import React, { useMemo } from 'react';
import Card from '../common/Card';
import PropTypes from 'prop-types';

/**
 * 网络图统计信息组件
 * 显示网络图的关键统计数据，包括节点数量、连接数量、风险分布等
 *
 * @param {Object} props - 组件属性
 * @param {Object} props.data - 网络图数据，包含nodes和links数组
 * @returns {JSX.Element} 网络图统计信息组件
 */
const NetworkStats = ({ data }) => {
  // 计算统计数据
  const stats = useMemo(() => {
    if (!data || !data.nodes || !data.links) {
      return {
        nodeCount: 0,
        linkCount: 0,
        nodeTypeDistribution: {},
        riskDistribution: {},
        highestValue: 0,
        avgConnections: 0,
      };
    }

    // 节点数量
    const nodeCount = data.nodes.length;
    // 连接数量
    const linkCount = data.links.length;

    // 节点类型分布
    const nodeTypeDistribution = data.nodes.reduce((acc, node) => {
      acc[node.type] = (acc[node.type] || 0) + 1;
      return acc;
    }, {});

    // 风险分布
    const riskDistribution = data.nodes.reduce((acc, node) => {
      if (node.riskScore !== undefined) {
        let riskLevel = '未知';
        if (node.riskScore >= 75) riskLevel = '高';
        else if (node.riskScore >= 40) riskLevel = '中';
        else riskLevel = '低';

        acc[riskLevel] = (acc[riskLevel] || 0) + 1;
      }
      return acc;
    }, {});

    // 最高交易值
    const highestValue = data.nodes.reduce((max, node) => {
      if (node.value !== undefined && node.value > max) {
        return node.value;
      }
      return max;
    }, 0);

    // 平均连接数
    const connectionsMap = {};
    data.links.forEach(link => {
      connectionsMap[link.source] = (connectionsMap[link.source] || 0) + 1;
      connectionsMap[link.target] = (connectionsMap[link.target] || 0) + 1;
    });

    const avgConnections =
      Object.values(connectionsMap).reduce((sum, val) => sum + val, 0) / nodeCount || 0;

    return {
      nodeCount,
      linkCount,
      nodeTypeDistribution,
      riskDistribution,
      highestValue,
      avgConnections,
    };
  }, [data]);

  // 如果没有数据，显示默认状态
  if (!data || stats.nodeCount === 0) {
    return (
      <Card>
        <div className="text-center py-4 text-gray-500">无网络统计数据可显示</div>
      </Card>
    );
  }

  return (
    <Card>
      <h3 className="text-lg font-medium mb-4">网络统计</h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <h5 className="text-base font-semibold mb-2">基本数据</h5>
          <ul className="space-y-1">
            <li>节点总数: {stats.nodeCount}</li>
            <li>连接总数: {stats.linkCount}</li>
            <li>平均连接数: {stats.avgConnections.toFixed(2)}</li>
          </ul>
        </div>

        <div>
          <h5 className="text-base font-semibold mb-2">节点类型分布</h5>
          <ul className="space-y-1">
            {Object.entries(stats.nodeTypeDistribution).map(([type, count]) => (
              <li key={type}>
                {getNodeTypeName(type)}: {count} ({Math.round((count / stats.nodeCount) * 100)}%)
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h5 className="text-base font-semibold mb-2">风险等级分布</h5>
          <ul className="space-y-1">
            <li className="text-red-600">
              高风险: {stats.riskDistribution['高'] || 0} (
              {Math.round(((stats.riskDistribution['高'] || 0) / stats.nodeCount) * 100)}%)
            </li>
            <li className="text-yellow-600">
              中风险: {stats.riskDistribution['中'] || 0} (
              {Math.round(((stats.riskDistribution['中'] || 0) / stats.nodeCount) * 100)}%)
            </li>
            <li className="text-green-600">
              低风险: {stats.riskDistribution['低'] || 0} (
              {Math.round(((stats.riskDistribution['低'] || 0) / stats.nodeCount) * 100)}%)
            </li>
          </ul>
        </div>
      </div>
    </Card>
  );
};

NetworkStats.propTypes = {
  data: PropTypes.shape({
    nodes: PropTypes.array.isRequired,
    links: PropTypes.array.isRequired,
  }).isRequired,
};

export default NetworkStats;

/**
 * 将节点类型转换为易读的名称
 * @param {string} type - 节点类型
 * @returns {string} 易读的节点类型名称
 */
function getNodeTypeName(type) {
  const typeMap = {
    address: '普通地址',
    contract: '合约',
    exchange: '交易所',
    mixer: '混币器',
    high_risk: '高风险地址',
  };
  return typeMap[type] || type;
}

/**
 * 格式化数值，添加千位分隔符和缩写大数
 * @param {number} value - 需要格式化的数值
 * @returns {string} 格式化后的字符串
 */
function formatValue(value) {
  if (value >= 1000000000) {
    return (value / 1000000000).toFixed(2) + 'B';
  } else if (value >= 1000000) {
    return (value / 1000000).toFixed(2) + 'M';
  } else if (value >= 1000) {
    return (value / 1000).toFixed(2) + 'K';
  }
  return value.toString();
}
