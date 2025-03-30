import React, { useMemo } from 'react';

/**
 * 网络连接线组件
 * 用于渲染网络图中连接节点的线
 *
 * @param {Object} props - 组件属性
 * @param {Object} props.link - 连接数据
 * @param {Object} props.sourcePosition - 源节点位置 {x, y}
 * @param {Object} props.targetPosition - 目标节点位置 {x, y}
 * @param {boolean} props.isSelected - 是否被选中
 * @param {Function} props.onClick - 点击连接的处理函数
 * @param {number} props.strokeWidth - 线条宽度
 * @param {boolean} props.animated - 是否启用动画
 * @param {boolean} props.directed - 是否显示方向箭头
 * @returns {JSX.Element} 网络连接线组件
 */
export default function NetworkLink({
  link,
  sourcePosition,
  targetPosition,
  isSelected = false,
  onClick,
  strokeWidth = 1.5,
  animated = false,
  directed = true,
}) {
  // 根据连接类型确定颜色
  const linkColor = useMemo(() => {
    if (!link) return '#D1D5DB'; // 默认浅灰色

    switch (link.type) {
      case 'high_value':
        return '#3B82F6'; // 蓝色 - 高价值交易
      case 'suspicious':
        return '#EF4444'; // 红色 - 可疑交易
      case 'mixer':
        return '#F97316'; // 橙色 - 混币交易
      case 'regular':
        return '#10B981'; // 绿色 - 常规交易
      default:
        return '#9CA3AF'; // 默认灰色
    }
  }, [link]);

  // 如果没有连接数据或位置信息，不渲染
  if (!link || !sourcePosition || !targetPosition) return null;

  // 计算链接线的长度和角度
  const dx = targetPosition.x - sourcePosition.x;
  const dy = targetPosition.y - sourcePosition.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);

  // 计算箭头位置（目标节点前一点）
  const arrowDistance = 15; // 箭头与目标节点的距离
  const arrowX = targetPosition.x - (dx * arrowDistance) / length;
  const arrowY = targetPosition.y - (dy * arrowDistance) / length;

  // 动画样式
  const animationStyle = animated
    ? {
        strokeDasharray: '5,5',
        animation: 'dash 1s linear infinite',
      }
    : {};

  return (
    <g onClick={() => onClick && onClick(link)}>
      {/* 连接线 */}
      <line
        x1={sourcePosition.x}
        y1={sourcePosition.y}
        x2={targetPosition.x}
        y2={targetPosition.y}
        stroke={linkColor}
        strokeWidth={isSelected ? strokeWidth * 2 : strokeWidth}
        strokeOpacity={0.6}
        style={animationStyle}
      />

      {/* 方向箭头 */}
      {directed && (
        <polygon
          points={`${arrowX},${arrowY} ${arrowX - 5},${arrowY - 3} ${arrowX - 5},${arrowY + 3}`}
          transform={`rotate(${angle + 90}, ${arrowX}, ${arrowY})`}
          fill={linkColor}
          opacity={0.8}
        />
      )}

      {/* 选中状态指示器 */}
      {isSelected && (
        <line
          x1={sourcePosition.x}
          y1={sourcePosition.y}
          x2={targetPosition.x}
          y2={targetPosition.y}
          stroke="#3B82F6"
          strokeWidth={strokeWidth * 2.5}
          strokeOpacity={0.3}
        />
      )}

      {/* 连接值标签（可选） */}
      {link.value && (
        <text
          x={(sourcePosition.x + targetPosition.x) / 2}
          y={(sourcePosition.y + targetPosition.y) / 2 - 5}
          fill={linkColor}
          fontSize="10"
          textAnchor="middle"
          dominantBaseline="middle"
          pointerEvents="none"
          opacity={0.8}
        >
          {typeof link.value === 'number' ? link.value.toFixed(1) : link.value}
        </text>
      )}
    </g>
  );
}
