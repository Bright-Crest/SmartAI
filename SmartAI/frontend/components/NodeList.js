import React, { useState, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import useVirtualizedList from '../hooks/useVirtualizedList';
import { useEnhancedMemo } from '../hooks/useMemo';

// 缓存键，用于增强性能优化
const CACHE_KEYS = {
  FILTERED_NODES: 'filteredNodes',
  SORTED_NODES: 'sortedNodes',
};

/**
 * 风险等级标签组件
 *
 * @param {Object} props - 组件属性
 * @param {string} props.risk - 风险等级
 * @returns {JSX.Element} 风险等级标签
 */
const RiskBadge = ({ risk }) => {
  const riskColors = {
    high: { bg: '#fde2e2', text: '#e74c3c', border: '#fabebe' },
    medium: { bg: '#fef3e2', text: '#f39c12', border: '#fce4be' },
    low: { bg: '#e2f0fd', text: '#3498db', border: '#bee1fc' },
  };

  const style = riskColors[risk] || { bg: '#f0f0f0', text: '#666', border: '#ddd' };

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        fontSize: '12px',
        borderRadius: '10px',
        backgroundColor: style.bg,
        color: style.text,
        border: `1px solid ${style.border}`,
      }}
    >
      {risk || '未知'}
    </span>
  );
};

/**
 * 单个节点项组件
 *
 * @param {Object} props - 组件属性
 * @param {Object} props.node - 节点数据
 * @param {Function} props.onClick - 点击事件处理函数
 * @param {boolean} props.isSelected - 是否被选中
 * @param {Function} props.measureRef - 测量高度的ref回调函数
 * @returns {JSX.Element} 节点项组件
 */
const NodeItem = React.memo(({ node, onClick, isSelected, measureRef }) => {
  const handleClick = useCallback(() => {
    onClick(node);
  }, [onClick, node]);

  return (
    <div
      ref={measureRef}
      className="node-item"
      style={{
        padding: '10px 15px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #eaeaea',
        backgroundColor: isSelected ? '#f2f9ff' : 'white',
        transition: 'background-color 0.2s ease',
      }}
      onClick={handleClick}
      data-id={node.id}
    >
      <div className="node-item-main" style={{ flex: 1, overflow: 'hidden' }}>
        <div
          className="node-item-title"
          style={{
            fontWeight: 500,
            marginBottom: '4px',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {node.label || node.id}
        </div>

        {node.address && (
          <div
            className="node-item-address"
            style={{
              fontSize: '12px',
              color: '#666',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {node.address}
          </div>
        )}
      </div>

      <div className="node-item-meta" style={{ marginLeft: '10px' }}>
        {node.risk && <RiskBadge risk={node.risk} />}
      </div>
    </div>
  );
});

NodeItem.displayName = 'NodeItem';

/**
 * 节点列表组件 - 使用虚拟列表渲染大量节点数据
 *
 * @param {Object} props - 组件属性
 * @param {Array} props.nodes - 节点数据数组
 * @param {Function} props.onNodeClick - 节点点击事件处理函数
 * @param {string} props.selectedNodeId - 当前选中的节点ID
 * @param {string} props.searchTerm - 搜索关键词
 * @param {string} props.selectedRiskLevel - 选择的风险级别筛选
 * @returns {JSX.Element} 节点列表组件
 */
const NodeList = ({
  nodes = [],
  onNodeClick = () => {},
  selectedNodeId = null,
  searchTerm = '',
  selectedRiskLevel = '',
}) => {
  // 排序选项
  const [sortField, setSortField] = useState('default');
  const [sortOrder, setSortOrder] = useState('asc');

  // 虚拟列表hook
  const {
    containerRef,
    getVisibleRange,
    getItemOffset,
    getContentHeight,
    measureItemHeight,
    scrollToIndex,
  } = useVirtualizedList({
    itemHeight: 60, // 预估的每个项目高度
    overscan: 5, // 可视区域外额外渲染的项目数
  });

  // 过滤节点基于搜索和风险级别
  const filteredNodes = useEnhancedMemo(
    () => {
      // 记录开始时间
      const startTime = performance.now();

      let result = [...nodes];

      // 搜索过滤
      if (searchTerm) {
        const lowerSearchTerm = searchTerm.toLowerCase();
        result = result.filter(
          node =>
            (node.id && node.id.toLowerCase().includes(lowerSearchTerm)) ||
            (node.label && node.label.toLowerCase().includes(lowerSearchTerm)) ||
            (node.address && node.address.toLowerCase().includes(lowerSearchTerm))
        );
      }

      // 风险级别过滤
      if (selectedRiskLevel) {
        result = result.filter(node => node.risk === selectedRiskLevel);
      }

      // 记录过滤耗时
      const filterTime = performance.now() - startTime;
      console.log(`节点过滤耗时: ${filterTime.toFixed(2)}ms`, {
        原始节点数: nodes.length,
        过滤后节点数: result.length,
      });

      return result;
    },
    [nodes, searchTerm, selectedRiskLevel],
    { key: CACHE_KEYS.FILTERED_NODES, debug: process.env.NODE_ENV === 'development' }
  );

  // 处理节点排序
  const sortedNodes = useEnhancedMemo(
    () => {
      // 如果没有指定排序或使用默认排序，直接返回过滤后的节点
      if (sortField === 'default') {
        return filteredNodes;
      }

      // 记录开始时间
      const startTime = performance.now();

      // 复制数组以避免修改原始数据
      const result = [...filteredNodes];

      // 排序函数
      const sortFunctions = {
        risk: (a, b) => {
          const riskOrder = { high: 3, medium: 2, low: 1, undefined: 0 };
          const aValue = riskOrder[a.risk] || 0;
          const bValue = riskOrder[b.risk] || 0;
          return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
        },
        label: (a, b) => {
          const aValue = (a.label || a.id || '').toLowerCase();
          const bValue = (b.label || b.id || '').toLowerCase();
          return sortOrder === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
        },
        value: (a, b) => {
          const aValue = a.value || 0;
          const bValue = b.value || 0;
          return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
        },
      };

      // 执行排序
      if (sortFunctions[sortField]) {
        result.sort(sortFunctions[sortField]);
      }

      // 记录排序耗时
      const sortTime = performance.now() - startTime;
      console.log(`节点排序耗时: ${sortTime.toFixed(2)}ms`);

      return result;
    },
    [filteredNodes, sortField, sortOrder],
    { key: CACHE_KEYS.SORTED_NODES, debug: process.env.NODE_ENV === 'development' }
  );

  // 处理排序变化
  const handleSortChange = useCallback(
    e => {
      const value = e.target.value;

      if (value === sortField) {
        // 如果字段相同，切换排序顺序
        setSortOrder(prev => (prev === 'asc' ? 'desc' : 'asc'));
      } else {
        // 更新排序字段，重置为升序
        setSortField(value);
        setSortOrder('asc');
      }
    },
    [sortField]
  );

  // 获取要渲染的节点范围
  const { startIndex, endIndex } = getVisibleRange(sortedNodes.length);

  // 计算虚拟列表内容高度
  const contentHeight = getContentHeight(sortedNodes.length);

  // 当选中节点变化时，滚动到选中节点
  React.useEffect(() => {
    if (selectedNodeId) {
      const nodeIndex = sortedNodes.findIndex(node => node.id === selectedNodeId);
      if (nodeIndex !== -1) {
        scrollToIndex(nodeIndex, 'auto', 10);
      }
    }
  }, [selectedNodeId, sortedNodes, scrollToIndex]);

  // 渲染节点项
  const renderNode = (node, index) => {
    const isSelected = node.id === selectedNodeId;

    return (
      <div
        key={node.id}
        style={{
          position: 'absolute',
          top: getItemOffset(index),
          left: 0,
          right: 0,
        }}
      >
        <NodeItem
          node={node}
          onClick={onNodeClick}
          isSelected={isSelected}
          measureRef={element => measureItemHeight(index, element)}
        />
      </div>
    );
  };

  return (
    <div className="node-list" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 列表头部 - 排序控制 */}
      <div
        className="node-list-header"
        style={{
          padding: '10px 15px',
          borderBottom: '1px solid #eaeaea',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <strong>节点列表</strong> ({sortedNodes.length})
        </div>

        <div className="node-list-sort">
          <select
            value={sortField}
            onChange={handleSortChange}
            style={{
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              fontSize: '12px',
            }}
          >
            <option value="default">默认排序</option>
            <option value="label">按名称</option>
            <option value="risk">按风险</option>
            <option value="value">按价值</option>
          </select>

          <button
            onClick={() => setSortOrder(prev => (prev === 'asc' ? 'desc' : 'asc'))}
            style={{
              marginLeft: '5px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              background: 'white',
              cursor: 'pointer',
              fontSize: '12px',
            }}
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {/* 虚拟滚动容器 */}
      <div
        ref={containerRef}
        className="node-list-container"
        style={{
          flex: 1,
          overflow: 'auto',
          position: 'relative',
        }}
      >
        {/* 内容高度占位 */}
        <div style={{ height: contentHeight, position: 'relative' }}>
          {/* 只渲染可见区域的节点 */}
          {sortedNodes.length > 0 ? (
            sortedNodes
              .slice(startIndex, endIndex + 1)
              .map((node, idx) => renderNode(node, startIndex + idx))
          ) : (
            <div
              style={{
                padding: '20px',
                textAlign: 'center',
                color: '#666',
              }}
            >
              没有符合条件的节点
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

NodeList.propTypes = {
  nodes: PropTypes.arrayOf(PropTypes.object),
  onNodeClick: PropTypes.func,
  selectedNodeId: PropTypes.string,
  searchTerm: PropTypes.string,
  selectedRiskLevel: PropTypes.string,
};

export default NodeList;
