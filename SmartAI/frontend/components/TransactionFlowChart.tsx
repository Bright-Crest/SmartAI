import React, { useState, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  ResponsiveContainer, 
  Tooltip, 
  Sankey, 
  Scatter,
  ScatterChart,
  ZAxis,
  Rectangle
} from 'recharts';
import { asJsxComponent } from '../utils/chartUtils';
import FocusableItem from './a11y/FocusableItem';
import useKeyboardNavigation from '../hooks/useKeyboardNavigation';
import { useA11y } from '../context/A11yContext';

// 定义节点类型
export interface Node {
  name: string;
  value?: number;
  address?: string;
  category?: string;
  riskLevel?: 'low' | 'medium' | 'high' | 'critical';
}

// 定义链接类型
export interface Link {
  source: number;
  target: number;
  value: number;
  timestamp?: string;
  txHash?: string;
}

// 定义组件属性类型
interface TransactionFlowChartProps {
  nodes: Node[];
  links: Link[];
  className?: string;
  onNodeClick?: (node: Node) => void;
  onLinkClick?: (link: Link) => void;
}

// 使用类型断言修复组件类型
const SafeResponsiveContainer = asJsxComponent(ResponsiveContainer);
const SafeTooltip = asJsxComponent(Tooltip);
const SafeSankey = asJsxComponent(Sankey);
const SafeScatter = asJsxComponent(Scatter);
const SafeScatterChart = asJsxComponent(ScatterChart);
const SafeZAxis = asJsxComponent(ZAxis);
const SafeRectangle = asJsxComponent(Rectangle);

/**
 * 交易流向图组件 - 使用 d3-sankey 实现交互式流向图
 */
const TransactionFlowChart: React.FC<TransactionFlowChartProps> = ({ 
  nodes, 
  links, 
  className = '',
  onNodeClick,
  onLinkClick
}) => {
  const { t } = useTranslation();
  const [activeNode, setActiveNode] = useState<number | null>(null);
  const [activeLink, setActiveLink] = useState<number | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const { handleEnterAndSpace, handleArrowKeys } = useKeyboardNavigation();
  const { announce } = useA11y();
  
  // 屏幕阅读器状态
  const [accessibilityStatus, setAccessibilityStatus] = useState<string>('');
  
  // 获取节点颜色
  const getNodeColor = useCallback((node: Node) => {
    // 根据风险等级或类别设置不同颜色
    if (node.riskLevel) {
      switch (node.riskLevel) {
        case 'low': return '#10b981';
        case 'medium': return '#f59e0b';
        case 'high': return '#ef4444';
        case 'critical': return '#7f1d1d';
        default: return '#6b7280';
      }
    }
    
    // 根据类别设置颜色
    if (node.category) {
      switch (node.category) {
        case 'exchange': return '#3b82f6';
        case 'defi': return '#8b5cf6';
        case 'wallet': return '#14b8a6';
        case 'contract': return '#f97316';
        default: return '#6b7280';
      }
    }
    
    return '#6b7280';
  }, []);
  
  // 处理节点激活
  const handleNodeActivation = useCallback((index: number, nodeData: Node) => {
    setActiveNode(index);
    // 更新屏幕阅读器状态
    const nodeStatus = `已选择节点: ${nodeData.name}${nodeData.category ? `，类别: ${nodeData.category}` : ''}${nodeData.riskLevel ? `，风险等级: ${nodeData.riskLevel}` : ''}`;
    setAccessibilityStatus(nodeStatus);
    announce(nodeStatus);
    
    if (onNodeClick && nodeData) {
      onNodeClick(nodeData);
    }
  }, [onNodeClick, announce]);
  
  // 处理链接激活
  const handleLinkActivation = useCallback((index: number, linkData: Link) => {
    setActiveLink(index);
    // 更新屏幕阅读器状态
    const sourceNode = nodes[linkData.source]?.name || '';
    const targetNode = nodes[linkData.target]?.name || '';
    const linkStatus = `已选择交易链接: 从 ${sourceNode} 到 ${targetNode}，金额: ${linkData.value}`;
    setAccessibilityStatus(linkStatus);
    announce(linkStatus);
    
    if (onLinkClick && linkData) {
      onLinkClick(linkData);
    }
  }, [onLinkClick, nodes, announce]);
  
  // 自定义节点组件
  const CustomNode = useCallback(({ x, y, width, height, index, payload }: any) => {
    const isActive = activeNode === index;
    const nodeData = payload as Node;
    const color = getNodeColor(nodeData);
    
    // 构造无障碍描述
    const ariaLabel = `节点: ${nodeData.name}${nodeData.category ? `，类别: ${nodeData.category}` : ''}${nodeData.riskLevel ? `，风险等级: ${nodeData.riskLevel}` : ''}`;
    const nodeId = `node-${index}`;
    
    // 处理键盘事件
    const handleKeyDown = (e: React.KeyboardEvent) => {
      // 处理Enter和Space键
      handleEnterAndSpace(() => handleNodeActivation(index, nodeData))(e);
      
      // 添加方向键导航
      handleArrowKeys({
        up: () => {
          // 查找上一个节点元素
          const prevNode = document.querySelector(`[data-node-id="${index - 1}"]`) as HTMLElement;
          if (prevNode) prevNode.focus();
        },
        down: () => {
          // 查找下一个节点元素
          const nextNode = document.querySelector(`[data-node-id="${index + 1}"]`) as HTMLElement;
          if (nextNode) nextNode.focus();
        }
      })(e);
    };
    
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={color}
          fillOpacity={isActive ? 1 : 0.9}
          stroke="#fff"
          strokeWidth={isActive ? 2 : 0}
          cursor="pointer"
          onClick={() => handleNodeActivation(index, nodeData)}
          onKeyDown={handleKeyDown}
          role="button"
          aria-pressed={isActive}
          tabIndex={0}
          aria-label={ariaLabel}
          data-node-id={index}
          id={nodeId}
          style={{ outline: isActive ? '2px solid #3b82f6' : 'none' }}
          focusable="true"
        />
        <title id={`node-desc-${index}`}>{ariaLabel}</title>
      </g>
    );
  }, [activeNode, getNodeColor, handleNodeActivation, handleEnterAndSpace, handleArrowKeys]);
  
  // 自定义链接组件
  const CustomLink = useCallback(({ sourceX, targetX, sourceY, targetY, sourceControlX, targetControlX, linkWidth, index, payload }: any) => {
    const isActive = activeLink === index;
    const linkData = payload as Link;
    
    // 构造无障碍描述
    const sourceNode = nodes[linkData.source]?.name || '';
    const targetNode = nodes[linkData.target]?.name || '';
    const ariaLabel = `交易链接: 从 ${sourceNode} 到 ${targetNode}，金额: ${linkData.value}`;
    const linkId = `link-${index}`;
    
    // 处理键盘事件
    const handleKeyDown = (e: React.KeyboardEvent) => {
      // 处理Enter和Space键
      handleEnterAndSpace(() => handleLinkActivation(index, linkData))(e);
      
      // 添加方向键导航
      handleArrowKeys({
        up: () => {
          // 查找上一个链接元素
          const prevLink = document.querySelector(`[data-link-id="${index - 1}"]`) as HTMLElement;
          if (prevLink) prevLink.focus();
        },
        down: () => {
          // 查找下一个链接元素
          const nextLink = document.querySelector(`[data-link-id="${index + 1}"]`) as HTMLElement;
          if (nextLink) nextLink.focus();
        }
      })(e);
    };
    
    // 创建路径
    const path = `
      M${sourceX},${sourceY}
      C${sourceControlX},${sourceY} ${targetControlX},${targetY} ${targetX},${targetY}
    `;
    
    return (
      <g>
        <path
          d={path}
          fill="none"
          stroke="#9CA3AF"
          strokeWidth={Math.max(1, linkWidth)}
          strokeOpacity={isActive ? 0.8 : 0.2}
          cursor="pointer"
          onClick={() => handleLinkActivation(index, linkData)}
          onKeyDown={handleKeyDown}
          role="button"
          aria-pressed={isActive}
          tabIndex={0}
          aria-label={ariaLabel}
          data-link-id={index}
          id={linkId}
          style={{ outline: isActive ? '2px solid #3b82f6' : 'none' }}
          focusable="true"
        />
        <title id={`link-desc-${index}`}>{ariaLabel}</title>
      </g>
    );
  }, [activeLink, handleLinkActivation, nodes, handleEnterAndSpace, handleArrowKeys]);
  
  // 自定义工具提示
  const CustomTooltip = useCallback(({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      
      if (data.source !== undefined && data.target !== undefined) {
        // 链接工具提示
        return (
          <div className="bg-white p-3 border border-gray-200 shadow-lg rounded-md">
            <p className="font-medium text-gray-800">{t('charts.transactionFlow.link')}</p>
            <div className="space-y-1 mt-2">
              <div className="flex items-center">
                <span className="text-gray-700">{t('charts.transactionFlow.from')}: </span>
                <span className="font-medium ml-1">{nodes[data.source]?.name}</span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-700">{t('charts.transactionFlow.to')}: </span>
                <span className="font-medium ml-1">{nodes[data.target]?.name}</span>
              </div>
              <div className="flex items-center">
                <span className="text-gray-700">{t('charts.transactionFlow.value')}: </span>
                <span className="font-medium ml-1">{data.value}</span>
              </div>
              {data.txHash && (
                <div className="flex items-center">
                  <span className="text-gray-700">{t('charts.transactionFlow.txHash')}: </span>
                  <span className="font-medium ml-1 text-xs">{`${data.txHash.substring(0, 10)}...`}</span>
                </div>
              )}
            </div>
          </div>
        );
      } else {
        // 节点工具提示
        return (
          <div className="bg-white p-3 border border-gray-200 shadow-lg rounded-md">
            <p className="font-medium text-gray-800">{data.name}</p>
            <div className="space-y-1 mt-2">
              {data.address && (
                <div className="flex items-center">
                  <span className="text-gray-700">{t('charts.transactionFlow.address')}: </span>
                  <span className="font-medium ml-1 text-xs">{`${data.address.substring(0, 10)}...`}</span>
                </div>
              )}
              {data.category && (
                <div className="flex items-center">
                  <span className="text-gray-700">{t('charts.transactionFlow.category')}: </span>
                  <span className="font-medium ml-1">{data.category}</span>
                </div>
              )}
              {data.riskLevel && (
                <div className="flex items-center">
                  <span className="text-gray-700">{t('charts.transactionFlow.riskLevel')}: </span>
                  <span className="font-medium ml-1">{data.riskLevel}</span>
                </div>
              )}
            </div>
          </div>
        );
      }
    }
    
    return null;
  }, [nodes, t]);
  
  // 创建键盘可导航节点列表（用于无障碍）
  const accessibleNodeList = useMemo(() => {
    return nodes.map((node, index) => ({
      ...node,
      index,
      type: 'node'
    }));
  }, [nodes]);
  
  // 创建键盘可导航链接列表（用于无障碍）
  const accessibleLinkList = useMemo(() => {
    return links.map((link, index) => ({
      ...link,
      index,
      type: 'link',
      from: nodes[link.source]?.name || '',
      to: nodes[link.target]?.name || ''
    }));
  }, [links, nodes]);
  
  // 组合节点和链接，形成可键盘导航的列表
  const accessibleElements = useMemo(() => {
    return [
      ...accessibleNodeList.map(node => ({
        id: `node-${node.index}`,
        label: `节点: ${node.name}${node.category ? `，类别: ${node.category}` : ''}${node.riskLevel ? `，风险等级: ${node.riskLevel}` : ''}`,
        onClick: () => handleNodeActivation(node.index, nodes[node.index]),
        isActive: activeNode === node.index,
        type: 'node'
      })),
      ...accessibleLinkList.map(link => ({
        id: `link-${link.index}`,
        label: `交易链接: 从 ${link.from} 到 ${link.to}，金额: ${link.value}`,
        onClick: () => handleLinkActivation(link.index, links[link.index]),
        isActive: activeLink === link.index,
        type: 'link'
      }))
    ];
  }, [accessibleNodeList, accessibleLinkList, activeNode, activeLink, handleNodeActivation, handleLinkActivation, nodes, links]);
  
  // 处理键盘导航
  const handleListKeyDown = useCallback((e: React.KeyboardEvent, index: number) => {
    handleArrowKeys({
      up: () => {
        const prevIndex = Math.max(0, index - 1);
        const prevElement = document.getElementById(`accessible-element-${prevIndex}`);
        if (prevElement) prevElement.focus();
      },
      down: () => {
        const nextIndex = Math.min(accessibleElements.length - 1, index + 1);
        const nextElement = document.getElementById(`accessible-element-${nextIndex}`);
        if (nextElement) nextElement.focus();
      },
      home: () => {
        const firstElement = document.getElementById(`accessible-element-0`);
        if (firstElement) firstElement.focus();
      },
      end: () => {
        const lastElement = document.getElementById(`accessible-element-${accessibleElements.length - 1}`);
        if (lastElement) lastElement.focus();
      }
    })(e);
  }, [accessibleElements.length, handleArrowKeys]);
  
  return (
    <div className={`bg-white p-4 rounded-lg shadow ${className}`} ref={chartRef}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800" id="flow-chart-title">
          {t('charts.transactionFlow.title')}
        </h2>
      </div>
      
      {/* 无障碍状态通知区域 */}
      <div 
        className="sr-only" 
        aria-live="polite"
      >
        {accessibilityStatus}
      </div>
      
      {/* 交易流图 */}
      <div 
        className="h-96" 
        role="application" 
        aria-labelledby="flow-chart-title"
      >
        <SafeResponsiveContainer width="100%" height="100%">
          <SafeSankey
            data={{ nodes, links }}
            node={CustomNode}
            link={CustomLink}
            nodePadding={50}
            margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
          >
            <SafeTooltip content={CustomTooltip} />
          </SafeSankey>
        </SafeResponsiveContainer>
      </div>
      
      {/* 无障碍键盘导航列表 */}
      <div className="mt-4 border-t pt-4 border-gray-200">
        <h3 className="text-md font-medium text-gray-800 mb-2" id="accessible-elements-heading">
          {t('charts.transactionFlow.accessibleElements')}
        </h3>
        <p className="text-sm text-gray-700 mb-3">
          {t('charts.transactionFlow.keyboardInstructions')}
        </p>
        <div className="flow-root">
          <ul className="divide-y divide-gray-200" role="listbox" aria-labelledby="accessible-elements-heading">
            {accessibleElements.map((element, idx) => (
              <li key={element.id}>
                <FocusableItem
                  id={`accessible-element-${idx}`}
                  onClick={element.onClick}
                  className={`block px-4 py-2 hover:bg-gray-50 transition-colors ${element.isActive ? 'bg-blue-50 font-medium' : ''}`}
                  role="option"
                  aria-selected={element.isActive}
                  onKeyDown={(e) => handleListKeyDown(e, idx)}
                >
                  <div className="flex items-center">
                    <span className={`w-3 h-3 rounded-full mr-2 ${element.type === 'node' ? 'bg-blue-500' : 'bg-green-500'}`} aria-hidden="true"></span>
                    <span>{element.label}</span>
                  </div>
                </FocusableItem>
              </li>
            ))}
          </ul>
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-700 border-t pt-4 border-gray-200">
        <p>{t('charts.transactionFlow.instructions')}</p>
      </div>
    </div>
  );
};

export default TransactionFlowChart; 