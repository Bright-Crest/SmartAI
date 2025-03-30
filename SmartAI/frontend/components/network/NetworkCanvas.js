import React, { useState, useRef, useEffect, useCallback } from 'react';
import NetworkNode from './NetworkNode';
import NetworkLink from './NetworkLink';
import ContextMenu from './ContextMenu';

/**
 * 网络画布组件
 * 用于渲染整个网络图并处理交互
 *
 * @param {Object} props - 组件属性
 * @param {Object} props.data - 网络图数据，包含nodes和links
 * @param {Function} props.onNodeClick - 节点点击事件处理函数
 * @param {Function} props.onLinkClick - 连接点击事件处理函数
 * @param {Function} props.onNodeDoubleClick - 节点双击事件处理函数
 * @param {Object} props.selectedNode - 当前选中的节点
 * @param {Object} props.selectedLink - 当前选中的连接
 * @param {Object} props.expandedNodes - 已展开的节点ID集合
 * @param {number} props.zoom - 缩放级别
 * @param {string} props.layout - 布局类型，可选值：'force', 'radial', 'hierarchical', 'circular', 'grid', 'concentric'
 * @returns {JSX.Element} 网络画布组件
 */
export default function NetworkCanvas({
  data,
  onNodeClick,
  onLinkClick,
  onNodeDoubleClick,
  selectedNode,
  selectedLink,
  expandedNodes = {},
  zoom = 1,
  layout = 'force',
}) {
  // 创建引用用于访问DOM元素
  const containerRef = useRef(null);
  const svgRef = useRef(null);

  // 状态管理
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [nodePositions, setNodePositions] = useState({});
  const [isDragging, setIsDragging] = useState(false);
  const [viewBox, setViewBox] = useState('0 0 800 600');
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  // 右键菜单状态
  const [contextMenu, setContextMenu] = useState({
    isOpen: false,
    position: { x: 0, y: 0 },
    node: null,
  });

  // 处理容器大小变化
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height });
        setViewBox(`${-panOffset.x} ${-panOffset.y} ${width / zoom} ${height / zoom}`);
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    return () => {
      window.removeEventListener('resize', updateDimensions);
    };
  }, [zoom, panOffset]);

  // 根据布局类型计算节点位置
  useEffect(() => {
    if (!data || !data.nodes || !dimensions) return;

    const { width, height } = dimensions;
    const positions = {};

    // 根据布局类型计算位置
    switch (layout) {
      case 'radial':
        // 径向布局
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.4;

        data.nodes.forEach((node, index) => {
          const angle = (index / data.nodes.length) * 2 * Math.PI;
          positions[node.id] = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle),
          };
        });
        break;

      case 'hierarchical':
        // 分层布局
        const levels = {};

        // 首先确定节点的层级
        data.nodes.forEach(node => {
          const level = node.level || 0;
          if (!levels[level]) levels[level] = [];
          levels[level].push(node);
        });

        // 计算每层的节点位置
        const levelKeys = Object.keys(levels).sort((a, b) => a - b);
        const levelCount = levelKeys.length;

        levelKeys.forEach((level, levelIndex) => {
          const nodesInLevel = levels[level];
          const levelY = (levelIndex + 1) * (height / (levelCount + 1));

          nodesInLevel.forEach((node, nodeIndex) => {
            const levelX = (nodeIndex + 1) * (width / (nodesInLevel.length + 1));
            positions[node.id] = { x: levelX, y: levelY };
          });
        });
        break;

      case 'circular':
        // 环形布局 - 根据连接关系将节点排列在多个同心圆上
        const circularCenterX = width / 2;
        const circularCenterY = height / 2;

        // 计算节点的连接数
        const connectionCounts = {};
        data.nodes.forEach(node => {
          connectionCounts[node.id] = 0;
        });

        data.links.forEach(link => {
          connectionCounts[link.source] = (connectionCounts[link.source] || 0) + 1;
          connectionCounts[link.target] = (connectionCounts[link.target] || 0) + 1;
        });

        // 根据连接数将节点分组
        const maxConnections = Math.max(...Object.values(connectionCounts), 1);
        const groups = Array(Math.min(5, maxConnections))
          .fill()
          .map(() => []);

        data.nodes.forEach(node => {
          const connections = connectionCounts[node.id] || 0;
          const groupIndex = Math.min(
            Math.floor((connections / maxConnections) * groups.length),
            groups.length - 1
          );
          groups[groupIndex].push(node);
        });

        // 将节点放置在对应的环上
        groups.forEach((group, groupIndex) => {
          const ringRadius = Math.min(width, height) * 0.4 * ((groupIndex + 1) / groups.length);

          group.forEach((node, nodeIndex) => {
            const angle = (nodeIndex / group.length) * 2 * Math.PI;
            positions[node.id] = {
              x: circularCenterX + ringRadius * Math.cos(angle),
              y: circularCenterY + ringRadius * Math.sin(angle),
            };
          });
        });
        break;

      case 'grid':
        // 网格布局 - 将节点排列在均匀的网格上
        const nodeCount = data.nodes.length;
        const aspectRatio = width / height;

        // 计算合适的行列数
        let cols = Math.ceil(Math.sqrt(nodeCount * aspectRatio));
        let rows = Math.ceil(nodeCount / cols);

        // 确保有足够的单元格
        if (cols * rows < nodeCount) {
          cols = Math.ceil(Math.sqrt(nodeCount));
          rows = Math.ceil(nodeCount / cols);
        }

        // 计算单元格大小
        const cellWidth = width / cols;
        const cellHeight = height / rows;

        // 放置节点
        data.nodes.forEach((node, index) => {
          const row = Math.floor(index / cols);
          const col = index % cols;

          positions[node.id] = {
            x: (col + 0.5) * cellWidth,
            y: (row + 0.5) * cellHeight,
          };
        });
        break;

      case 'concentric':
        // 同心圆布局 - 按照重要性或风险等级将节点按同心圆排列
        const concentricCenterX = width / 2;
        const concentricCenterY = height / 2;
        const maxRadius = Math.min(width, height) * 0.45;

        // 根据风险评分或值对节点进行分组
        const getNodeImportance = node => node.riskScore || node.value || 0;
        const sortedNodes = [...data.nodes].sort(
          (a, b) => getNodeImportance(b) - getNodeImportance(a)
        );

        // 划分成多个同心环
        const ringCount = Math.min(5, Math.ceil(sortedNodes.length / 10));
        const rings = Array(ringCount)
          .fill()
          .map(() => []);

        sortedNodes.forEach((node, index) => {
          const ringIndex = Math.min(
            Math.floor((index / sortedNodes.length) * ringCount),
            ringCount - 1
          );
          rings[ringIndex].push(node);
        });

        // 将节点放置在同心环上
        rings.forEach((ring, ringIndex) => {
          // 最重要的节点在最内圈
          const ringRadius = maxRadius * ((ringIndex + 1) / rings.length);

          ring.forEach((node, nodeIndex) => {
            const angle = (nodeIndex / ring.length) * 2 * Math.PI;
            positions[node.id] = {
              x: concentricCenterX + ringRadius * Math.cos(angle),
              y: concentricCenterY + ringRadius * Math.sin(angle),
            };
          });
        });
        break;

      case 'force':
      default:
        // 力导向布局（简化版）
        data.nodes.forEach(node => {
          positions[node.id] = node.position || {
            x: Math.random() * width * 0.8 + width * 0.1,
            y: Math.random() * height * 0.8 + height * 0.1,
          };
        });

        // 简单的位置调整（实际项目中可能会使用d3-force等库）
        for (let i = 0; i < 50; i++) {
          const forces = {};

          // 初始化力
          data.nodes.forEach(node => {
            forces[node.id] = { x: 0, y: 0 };
          });

          // 节点间的排斥力
          for (let i = 0; i < data.nodes.length; i++) {
            const nodeA = data.nodes[i];
            const posA = positions[nodeA.id];

            for (let j = i + 1; j < data.nodes.length; j++) {
              const nodeB = data.nodes[j];
              const posB = positions[nodeB.id];

              const dx = posB.x - posA.x;
              const dy = posB.y - posA.y;
              const distance = Math.sqrt(dx * dx + dy * dy) || 1;
              const force = 100 / distance;

              forces[nodeA.id].x -= (dx / distance) * force;
              forces[nodeA.id].y -= (dy / distance) * force;
              forces[nodeB.id].x += (dx / distance) * force;
              forces[nodeB.id].y += (dy / distance) * force;
            }
          }

          // 连接的吸引力
          data.links.forEach(link => {
            const sourcePos = positions[link.source];
            const targetPos = positions[link.target];

            if (sourcePos && targetPos) {
              const dx = targetPos.x - sourcePos.x;
              const dy = targetPos.y - sourcePos.y;
              const distance = Math.sqrt(dx * dx + dy * dy) || 1;
              const force = distance / 30;

              forces[link.source].x += (dx / distance) * force;
              forces[link.source].y += (dy / distance) * force;
              forces[link.target].x -= (dx / distance) * force;
              forces[link.target].y -= (dy / distance) * force;
            }
          });

          // 应用力并更新位置
          data.nodes.forEach(node => {
            const pos = positions[node.id];
            const force = forces[node.id];

            pos.x += force.x * 0.1;
            pos.y += force.y * 0.1;

            // 边界检查
            pos.x = Math.max(30, Math.min(width - 30, pos.x));
            pos.y = Math.max(30, Math.min(height - 30, pos.y));
          });
        }
        break;
    }

    setNodePositions(positions);
  }, [data, dimensions, layout]);

  // 处理节点拖动
  const handleNodeDragStart = useCallback(
    (nodeId, e) => {
      // 防止事件冒泡
      e.stopPropagation();

      // 标记为拖动状态
      setIsDragging(true);

      // 记录初始位置
      const initialX = e.clientX;
      const initialY = e.clientY;
      const initialPos = nodePositions[nodeId];

      // 处理鼠标移动
      const handleMouseMove = moveEvent => {
        if (!isDragging) return;

        const dx = (moveEvent.clientX - initialX) / zoom;
        const dy = (moveEvent.clientY - initialY) / zoom;

        setNodePositions(prev => ({
          ...prev,
          [nodeId]: {
            x: initialPos.x + dx,
            y: initialPos.y + dy,
          },
        }));
      };

      // 处理鼠标松开
      const handleMouseUp = () => {
        setIsDragging(false);
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      // 添加全局事件监听
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [nodePositions, isDragging, zoom]
  );

  // 处理画布平移
  const handleCanvasDragStart = useCallback(
    e => {
      // 忽略节点拖动
      if (e.target !== svgRef.current) return;

      // 标记为拖动状态
      setIsDragging(true);

      // 记录初始位置
      const initialX = e.clientX;
      const initialY = e.clientY;
      const initialPanOffset = { ...panOffset };

      // 处理鼠标移动
      const handleMouseMove = moveEvent => {
        if (!isDragging) return;

        const dx = (moveEvent.clientX - initialX) / zoom;
        const dy = (moveEvent.clientY - initialY) / zoom;

        setPanOffset({
          x: initialPanOffset.x - dx,
          y: initialPanOffset.y - dy,
        });

        setViewBox(
          `${initialPanOffset.x - dx} ${initialPanOffset.y - dy} ${dimensions.width / zoom} ${dimensions.height / zoom}`
        );
      };

      // 处理鼠标松开
      const handleMouseUp = () => {
        setIsDragging(false);
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };

      // 添加全局事件监听
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    },
    [panOffset, isDragging, zoom, dimensions]
  );

  // 处理节点双击
  const handleNodeDoubleClick = useCallback(
    node => {
      if (onNodeDoubleClick) {
        onNodeDoubleClick(node);
      }
    },
    [onNodeDoubleClick]
  );

  // 处理右键菜单打开
  const handleContextMenu = useCallback((node, e) => {
    e.preventDefault();
    e.stopPropagation();

    // 计算菜单位置，考虑缩放和平移
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setContextMenu({
      isOpen: true,
      position: { x, y },
      node,
    });
  }, []);

  // 关闭右键菜单
  const closeContextMenu = useCallback(() => {
    setContextMenu(prev => ({
      ...prev,
      isOpen: false,
    }));
  }, []);

  // 处理菜单操作
  const handleMenuActions = useCallback(
    {
      onView: node => {
        // 查看节点详情
        if (onNodeClick) onNodeClick(node);
      },
      onToggleExpand: node => {
        // 展开/收起节点
        if (onNodeDoubleClick) onNodeDoubleClick(node);
      },
      onHighlight: node => {
        // 高亮相关节点的实现
        console.log('高亮相关节点', node);
      },
      onTrack: node => {
        // 关注地址的实现
        console.log('关注地址', node);
      },
      onAnalyze: node => {
        // 深度分析的实现
        console.log('深度分析', node);
        // 这里可以导航到分析页面
        // window.location.href = `/address/${node.id}/analysis`;
      },
    },
    [onNodeClick, onNodeDoubleClick]
  );

  // 背景点击时关闭菜单
  const handleBackgroundClick = useCallback(() => {
    closeContextMenu();
  }, [closeContextMenu]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full relative overflow-hidden"
      onClick={handleBackgroundClick}
      onContextMenu={e => e.preventDefault()} // 禁用默认右键菜单
    >
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={viewBox}
        preserveAspectRatio="xMidYMid meet"
        className="cursor-move"
        onMouseDown={handleCanvasDragStart}
      >
        {/* 背景网格（可选） */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f0f0f0" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" pointerEvents="none" />

        {/* 渲染链接 */}
        {data &&
          data.links &&
          Object.keys(nodePositions).length > 0 &&
          data.links.map(link => {
            const sourcePos = nodePositions[link.source];
            const targetPos = nodePositions[link.target];

            if (!sourcePos || !targetPos) return null;

            return (
              <NetworkLink
                key={link.id || `${link.source}-${link.target}`}
                link={link}
                sourcePosition={sourcePos}
                targetPosition={targetPos}
                isSelected={selectedLink && selectedLink.id === link.id}
                onClick={() => onLinkClick && onLinkClick(link)}
              />
            );
          })}

        {/* 渲染节点 */}
        {data &&
          data.nodes &&
          Object.keys(nodePositions).length > 0 &&
          data.nodes.map(node => {
            const position = nodePositions[node.id];
            if (!position) return null;

            const isExpanded = expandedNodes && expandedNodes[node.id];

            // 使用外部组件来简化渲染逻辑
            return (
              <foreignObject
                key={node.id}
                x={position.x - 15}
                y={position.y - 15}
                width={30}
                height={30}
                onMouseDown={e => handleNodeDragStart(node.id, e)}
              >
                <NetworkNode
                  node={node}
                  isSelected={selectedNode && selectedNode.id === node.id}
                  isExpanded={isExpanded}
                  onClick={() => onNodeClick && onNodeClick(node)}
                  onDoubleClick={() => handleNodeDoubleClick(node)}
                  onContextMenu={e => handleContextMenu(node, e)}
                  size={30}
                  labelPosition="bottom"
                />
              </foreignObject>
            );
          })}
      </svg>

      {/* 控制面板（可选，也可以使用外部控件） */}
      <div className="absolute bottom-4 right-4 flex space-x-2">
        <button
          className="p-2 bg-white rounded-full shadow hover:shadow-md"
          onClick={() => setPanOffset({ x: 0, y: 0 })}
          title="重置视图"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zm0-2a6 6 0 100-12 6 6 0 000 12z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      </div>

      {/* 右键菜单 */}
      <ContextMenu
        isOpen={contextMenu.isOpen}
        position={contextMenu.position}
        node={contextMenu.node}
        onClose={closeContextMenu}
        actions={handleMenuActions}
      />
    </div>
  );
}
