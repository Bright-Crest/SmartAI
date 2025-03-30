/**
 * 区块链网络图类型定义
 */

// 风险等级
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

// 网络节点类型
export interface NetworkNode {
  id: string;
  label?: string;
  x?: number;
  y?: number;
  fx?: number | null;  // 固定位置X（力导向布局中）
  fy?: number | null;  // 固定位置Y（力导向布局中）
  group?: string;      // 节点分组
  value?: number;      // 节点大小
  weight?: number;     // 节点权重（影响布局）
  color?: string;      // 节点颜色
  riskLevel?: RiskLevel; // 风险级别
  isCluster?: boolean; // 是否为聚合节点
  clusterSize?: number; // 聚合大小
  clusteredNodes?: NetworkNode[]; // 被聚合的节点
  [key: string]: any; // 允许添加其他属性
}

// 网络链接类型
export interface NetworkLink {
  id?: string;
  source: string | NetworkNode;
  target: string | NetworkNode;
  value?: number;   // 链接权重/粗细
  width?: number;   // 显示宽度
  color?: string;   // 链接颜色
  label?: string;   // 链接标签
  dashed?: boolean; // 是否为虚线
  [key: string]: any; // 允许添加其他属性
}

// 网络数据类型
export interface NetworkData {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

// 网络图配置选项
export interface NetworkOptions {
  // 布局相关
  width: number;
  height: number;
  forceStrength?: number;
  linkDistance?: number;
  
  // 渲染相关
  nodeSize?: number | ((node: NetworkNode) => number);
  nodeColor?: string | ((node: NetworkNode) => string);
  linkWidth?: number | ((link: NetworkLink) => number);
  linkColor?: string | ((link: NetworkLink) => string);
  
  // 性能优化相关
  useWebWorker?: boolean;
  enableClustering?: boolean;
  clusterThreshold?: number;
  maxNodesPerRender?: number;
  
  // 交互相关
  draggable?: boolean;
  zoomable?: boolean;
  highlightConnections?: boolean;
  
  // 移动设备相关
  mobileOptimization?: boolean;
  mobileModeBreakpoint?: number;
}

// 网络布局计算结果
export interface LayoutResult {
  nodes: NetworkNode[];
  finished?: boolean;
  progress?: number;
}

// 集群分析结果
export interface ClusteringResult {
  nodes: NetworkNode[];
  links: NetworkLink[];
  clusterMap?: Map<string, string>;
}

// 网络图渲染状态
export interface NetworkState {
  initialized: boolean;
  loading: boolean;
  error: string | null;
  progress: number;
  renderedNodes: number;
  renderedLinks: number;
  fps?: number;
}

// 定义基本组件属性类型
export interface BlockchainNetworkBaseProps {
  nodes: NetworkNode[];
  links: NetworkLink[];
  width?: number;
  height?: number;
  className?: string;
  onNodeClick?: (node: NetworkNode) => void;
  onLinkClick?: (link: NetworkLink) => void;
}

// 优化版组件的附加属性
export interface BlockchainNetworkOptimizedProps extends BlockchainNetworkBaseProps {
  // 性能相关选项
  enableCulling?: boolean;       // 启用视口裁剪
  enableClustering?: boolean;    // 启用节点聚合
  clusterDistance?: number;      // 聚合距离
  renderQuality?: 'high' | 'medium' | 'low';  // 渲染质量
}

// 视图状态类型
export interface ViewState {
  scale: number;
  offsetX: number;
  offsetY: number;
}

// 工具提示数据类型
export interface TooltipData {
  content: React.ReactNode;
  x: number;
  y: number;
  visible: boolean;
}

// 图数据类型
export interface GraphData {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

// 渲染模式类型
export type RenderMode = 'canvas' | 'svg' | '3d' | 'webgl' | 'auto'; 