import React, { Component } from 'react';
import PropTypes from 'prop-types';
import * as d3 from 'd3'; // 明确导入d3库，避免全局变量问题
import styles from './NetworkGraph.module.css';

// 重构为更严格的D3与React集成模式
class D3NetworkGraph extends Component {
  constructor(props) {
    super(props);
    this.container = React.createRef();
    this.state = {
      isLoading: true,
      error: null,
    };

    // 简化状态管理
    this.simulation = null;
    this.svg = null;
    this.zoomGroup = null;
    this._d3Initialized = false;

    // 引用追踪
    this.nodeElements = null;
    this.linkElements = null;

    // 减少绑定方法数量，只保留必要的
    this.clearGraph = this.clearGraph.bind(this);
    this.initializeGraph = this.initializeGraph.bind(this);
  }

  componentDidMount() {
    // 初始化一次，确保只创建一个实例
    if (!this._d3Initialized && this.container.current) {
      this._d3Initialized = true;
      this.initializeGraph();
    }
  }

  componentDidUpdate(prevProps) {
    const { data, width, height } = this.props;

    // 精确控制重新渲染的条件
    if (prevProps.data !== data || prevProps.width !== width || prevProps.height !== height) {
      // 完全重建图形而不是部分更新
      this.clearGraph();
      this.initializeGraph();
    }
  }

  componentWillUnmount() {
    // 确保彻底清理所有D3资源
    this.clearGraph();
  }

  // 清理图形资源
  clearGraph() {
    // 停止模拟并解除所有事件绑定
    if (this.simulation) {
      this.simulation.stop();
      this.simulation.on('tick', null);
      this.simulation = null;
    }

    // 完全移除SVG元素避免内存泄漏
    if (this.container.current) {
      const svg = d3.select(this.container.current).select('svg');
      if (!svg.empty()) {
        // 首先解除所有事件
        svg.selectAll('*').on('*', null);
        svg.on('*', null);
        svg.remove();
      }
    }

    // 重置所有引用
    this.svg = null;
    this.zoomGroup = null;
    this.nodeElements = null;
    this.linkElements = null;
  }

  // 将初始化和更新合并为同一个函数
  initializeGraph() {
    const { data, width, height, onNodeClick } = this.props;

    if (!data || !data.nodes || !data.links || !this.container.current) {
      return;
    }

    // 创建新的SVG
    this.svg = d3
      .select(this.container.current)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('viewBox', `0 0 ${width || 800} ${height || 600}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // 添加缩放功能
    const zoom = d3
      .zoom()
      .scaleExtent([0.1, 4])
      .on('zoom', event => {
        if (this.zoomGroup) {
          this.zoomGroup.attr('transform', event.transform);
        }
      });

    this.svg.call(zoom);

    // 创建主容器
    this.zoomGroup = this.svg.append('g');

    // 添加背景
    this.zoomGroup
      .append('rect')
      .attr('width', width || 800)
      .attr('height', height || 600)
      .attr('fill', '#fafafa');

    // 预处理数据
    const nodesData = [...data.nodes].map(node => ({
      ...node,
      _radius: node.size || 20,
      _color: this.getNodeColor(node),
      // 确保所有节点有初始位置
      x: node.x || width / 2 + (Math.random() - 0.5) * 100,
      y: node.y || height / 2 + (Math.random() - 0.5) * 100,
    }));

    const linksData = [...data.links].filter(link => link && link.source && link.target);

    // 创建力模拟
    this.simulation = d3
      .forceSimulation(nodesData)
      .alphaDecay(0.028) // 更温和的衰减
      .alphaMin(0.001) // 更低的停止阈值
      .velocityDecay(0.4) // 增加阻尼
      .force(
        'link',
        d3
          .forceLink(linksData)
          .id(d => d.id)
          .distance(100)
          .strength(0.3)
      )
      .force(
        'charge',
        d3
          .forceManyBody()
          .strength(-30) // 减小排斥力
          .distanceMax(300)
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force(
        'collision',
        d3
          .forceCollide()
          .radius(d => (d._radius || 5) + 8) // 增大碰撞半径避免重叠
          .strength(0.8)
      );

    // 创建连接线 - 使用简单线条提高性能
    this.linkElements = this.zoomGroup
      .selectAll('.link')
      .data(linksData)
      .enter()
      .append('line') // 使用line代替path简化渲染
      .attr('class', styles.link)
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.value || 1));

    // 创建节点组
    const nodeGroups = this.zoomGroup
      .selectAll('.node')
      .data(nodesData)
      .enter()
      .append('g')
      .attr('class', styles.node)
      .attr('transform', d => `translate(${d.x},${d.y})`);

    // 添加节点圆圈 - 简化事件处理
    this.nodeElements = nodeGroups
      .append('circle')
      .attr('r', d => d._radius)
      .attr('fill', d => d._color)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('cursor', 'pointer');

    // 简化事件处理 - 使用全局标记避免频繁更新DOM
    let activeNode = null;

    // 统一使用事件对象
    const dragBehavior = d3
      .drag()
      .on('start', (event, d) => {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) this.simulation.alphaTarget(0);
        // 根据设置决定是否固定节点
        if (this.props.enablePhysics) {
          d.fx = null;
          d.fy = null;
        }
      });

    nodeGroups.call(dragBehavior);

    // 重构悬停事件 - 使用标志位而非直接操作fx/fy
    nodeGroups
      .on('mouseover', (event, d) => {
        // 标记活动节点
        activeNode = d;

        // 视觉反馈
        d3.select(event.currentTarget)
          .select('circle')
          .transition()
          .duration(150)
          .attr('stroke', '#555')
          .attr('stroke-width', 2.5);

        // 暂停力模拟
        this.simulation.alpha(0.1).alphaTarget(0);
      })
      .on('mouseout', (event, d) => {
        // 清除活动节点
        activeNode = null;

        // 恢复视觉效果
        d3.select(event.currentTarget)
          .select('circle')
          .transition()
          .duration(150)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);

        // 无需重启模拟，让它自然衰减
      })
      .on('click', (event, d) => {
        if (onNodeClick) {
          event.stopPropagation();
          onNodeClick(d);
        }
      });

    // 优化tick函数 - 批量更新以提高性能
    this.simulation.on('tick', () => {
      // 优化连接线更新
      this.linkElements
        .attr('x1', d => {
          const sourceNode =
            typeof d.source === 'object' ? d.source : nodesData.find(n => n.id === d.source);
          return sourceNode ? sourceNode.x : 0;
        })
        .attr('y1', d => {
          const sourceNode =
            typeof d.source === 'object' ? d.source : nodesData.find(n => n.id === d.source);
          return sourceNode ? sourceNode.y : 0;
        })
        .attr('x2', d => {
          const targetNode =
            typeof d.target === 'object' ? d.target : nodesData.find(n => n.id === d.target);
          return targetNode ? targetNode.x : 0;
        })
        .attr('y2', d => {
          const targetNode =
            typeof d.target === 'object' ? d.target : nodesData.find(n => n.id === d.target);
          return targetNode ? targetNode.y : 0;
        });

      // 更新节点位置
      nodeGroups.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // 预热仿真以获得更稳定的初始布局
    for (let i = 0; i < 100; i++) {
      this.simulation.tick();
    }

    // 立即应用首次渲染
    this.simulation.tick();

    // 延迟停止模拟
    setTimeout(() => {
      if (this.simulation) {
        this.simulation.alpha(0).alphaTarget(0);
      }
    }, 2000);
  }

  // 获取节点颜色的辅助函数
  getNodeColor(node) {
    const colorMap = {
      exchange: '#4CAF50',
      defi: '#2196F3',
      contract: '#FF9800',
      eoa: '#9C27B0',
      whale: '#E91E63',
      mixer: '#F44336',
      default: '#607D8B',
    };

    return node.type && colorMap[node.type.toLowerCase()]
      ? colorMap[node.type.toLowerCase()]
      : colorMap.default;
  }

  render() {
    return (
      <div
        ref={this.container}
        className={styles.graphContainer || 'graph-container'}
        style={{ width: '100%', height: '100%' }}
      />
    );
  }
}

D3NetworkGraph.propTypes = {
  data: PropTypes.shape({
    nodes: PropTypes.arrayOf(PropTypes.object).isRequired,
    links: PropTypes.arrayOf(PropTypes.object).isRequired,
  }),
  onNodeClick: PropTypes.func,
  selectedNode: PropTypes.string,
  width: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  height: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  enablePhysics: PropTypes.bool,
  enableZoom: PropTypes.bool,
  onRenderComplete: PropTypes.func,
};

D3NetworkGraph.defaultProps = {
  width: 800,
  height: 600,
  enablePhysics: true,
  enableZoom: true,
};

export default D3NetworkGraph;
