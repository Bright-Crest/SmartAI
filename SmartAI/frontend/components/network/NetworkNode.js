import React, { memo } from 'react';
import PropTypes from 'prop-types';

/**
 * 获取节点颜色
 * 根据节点类型和风险级别返回对应的颜色
 *
 * @param {string} type - 节点类型
 * @param {string} riskLevel - 风险等级 (low, medium, high)
 * @returns {string} 颜色代码
 */
function getNodeColor(type, riskLevel) {
  // 首先检查风险等级
  if (riskLevel) {
    switch (riskLevel) {
      case 'high':
        return '#ef4444'; // 红色
      case 'medium':
        return '#f59e0b'; // 黄色
      case 'low':
        return '#10b981'; // 绿色
    }
  }

  // 如果没有风险等级，根据节点类型返回颜色
  switch (type) {
    case 'contract':
      return '#8b5cf6'; // 紫色
    case 'exchange':
      return '#3b82f6'; // 蓝色
    case 'mixer':
      return '#f43f5e'; // 玫红色
    case 'high_risk':
      return '#dc2626'; // 红色
    case 'kol':
      return '#f97316'; // 橙色
    case 'address':
    default:
      return '#6b7280'; // 灰色
  }
}

/**
 * 获取节点大小
 * 根据节点重要性和值返回节点大小
 *
 * @param {number} value - 节点值
 * @param {boolean} isHighlighted - 是否高亮显示
 * @param {boolean} isSelected - 是否被选中
 * @returns {number} 节点半径
 */
function getNodeSize(value, isHighlighted, isSelected) {
  // 基础大小
  let size = 5;

  // 根据值调整大小
  if (value !== undefined && value !== null) {
    size += Math.min(Math.sqrt(value) * 0.5, 10);
  }

  // 高亮节点略大
  if (isHighlighted) {
    size += 2;
  }

  // 选中节点更大
  if (isSelected) {
    size += 4;
  }

  return size;
}

/**
 * 网络节点组件
 * 渲染网络图中的单个节点，支持不同类型、状态和风险等级
 *
 * @param {Object} props - 组件属性
 * @param {Object} props.node - 节点数据
 * @param {boolean} props.selected - 是否被选中
 * @param {boolean} props.highlighted - 是否高亮
 * @param {boolean} props.expanded - 是否展开
 * @param {Function} props.onClick - 点击处理函数
 * @param {Function} props.onDoubleClick - 双击处理函数
 * @returns {JSX.Element} 网络节点组件
 */
const NetworkNode = memo(function NetworkNode({
  node,
  selected = false,
  highlighted = false,
  expanded = false,
  onClick,
  onDoubleClick,
}) {
  // 节点基础属性
  const { id, x, y, type = 'address', label, value, riskLevel } = node;

  // 计算节点大小和颜色
  const size = getNodeSize(value, highlighted, selected);
  const color = getNodeColor(type, riskLevel);

  // 标签显示逻辑
  const showLabel = selected || highlighted || expanded || size > 8;

  // 确保坐标存在
  const xPos = x || 0;
  const yPos = y || 0;

  // 鼠标事件处理
  const handleClick = event => {
    event.stopPropagation();
    if (onClick) onClick(node, event);
  };

  const handleDoubleClick = event => {
    event.stopPropagation();
    if (onDoubleClick) onDoubleClick(node, event);
  };

  return (
    <g
      transform={`translate(${xPos},${yPos})`}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      style={{ cursor: 'pointer' }}
    >
      {/* 节点基础圆形 */}
      <circle
        r={size}
        fill={color}
        stroke={selected ? '#000' : highlighted ? '#555' : color}
        strokeWidth={selected ? 2 : highlighted ? 1.5 : 1}
        opacity={highlighted || selected ? 1 : 0.8}
      />

      {/* 展开状态指示器 */}
      {expanded && (
        <circle
          r={size + 3}
          fill="none"
          stroke="#888"
          strokeWidth={1}
          strokeDasharray="3,3"
          opacity={0.6}
        />
      )}

      {/* 风险指示器 - 仅显示高风险和中风险 */}
      {(riskLevel === 'high' || riskLevel === 'medium') && (
        <circle
          r={size + 1.5}
          fill="none"
          stroke={riskLevel === 'high' ? '#ef4444' : '#f59e0b'}
          strokeWidth={1.5}
          opacity={0.8}
        />
      )}

      {/* 节点标签 */}
      {showLabel && label && (
        <text
          dy="-10"
          fontSize={selected ? 12 : 10}
          textAnchor="middle"
          fill={selected ? '#000' : '#333'}
          fontWeight={selected ? 'bold' : 'normal'}
          pointerEvents="none"
          style={{ userSelect: 'none' }}
        >
          {label}
        </text>
      )}
    </g>
  );
});

NetworkNode.propTypes = {
  node: PropTypes.object.isRequired,
  selected: PropTypes.bool,
  highlighted: PropTypes.bool,
  expanded: PropTypes.bool,
  onClick: PropTypes.func,
  onDoubleClick: PropTypes.func,
};

export default NetworkNode;
