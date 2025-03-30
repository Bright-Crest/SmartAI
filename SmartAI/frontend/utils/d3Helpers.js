/**
 * D3图表帮助工具
 * 提供优化的D3图表渲染功能
 */
import * as d3 from 'd3';

/**
 * 创建力导向图模拟实例
 *
 * @param {Array} nodes - 节点数组
 * @param {Array} links - 链接数组
 * @param {Object} options - 配置选项
 * @param {number} [options.width=800] - 图表宽度
 * @param {number} [options.height=600] - 图表高度
 * @param {number} [options.linkDistance=100] - 链接距离
 * @param {number} [options.chargeStrength=-300] - 节点电荷强度
 * @param {Function} [options.nodeRadiusFn] - 计算节点半径的函数
 * @returns {Object} 模拟实例和相关工具函数
 */
export function createForceSimulation(nodes, links, options = {}) {
  const {
    width = 800,
    height = 600,
    linkDistance = 100,
    chargeStrength = -300,
    nodeRadiusFn = () => 10,
  } = options;

  // 复制节点和链接数据，避免修改原始数据
  const nodesData = nodes.map(node => ({ ...node }));
  const linksData = links.map(link => {
    if (typeof link.source === 'object' && link.source !== null) {
      return { ...link, source: link.source.id };
    }
    if (typeof link.target === 'object' && link.target !== null) {
      return { ...link, target: link.target.id };
    }
    return { ...link };
  });

  // 创建模拟实例
  const simulation = d3
    .forceSimulation(nodesData)
    .force(
      'link',
      d3
        .forceLink(linksData)
        .id(d => d.id)
        .distance(linkDistance)
    )
    .force('charge', d3.forceManyBody().strength(chargeStrength))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force(
      'collide',
      d3.forceCollide().radius(d => nodeRadiusFn(d) * 1.2)
    )
    .alphaDecay(0.028) // 略微降低默认衰减率以获得更平滑的效果
    .velocityDecay(0.4); // 增加阻尼以减少振荡

  // 预热模拟器
  const warmup = (iterations = 20) => {
    for (let i = 0; i < iterations; ++i) {
      simulation.tick();
    }
    return simulation;
  };

  // 添加预热方法到模拟实例上，使其可以直接从simulation调用
  simulation.warmup = warmup;

  // 销毁模拟器，释放资源
  const destroy = () => {
    simulation.stop();
    return null;
  };

  // 手动重新开始模拟
  const restart = (alpha = 1) => {
    simulation.alpha(alpha).restart();
    return simulation;
  };

  // 更新数据
  const updateData = (newNodes, newLinks) => {
    // 停止当前模拟
    simulation.stop();

    // 准备新数据
    const updatedNodes = newNodes.map(node => ({ ...node }));
    const updatedLinks = newLinks.map(link => {
      if (typeof link.source === 'object' && link.source !== null) {
        return { ...link, source: link.source.id };
      }
      if (typeof link.target === 'object' && link.target !== null) {
        return { ...link, target: link.target.id };
      }
      return { ...link };
    });

    // 更新模拟数据
    simulation.nodes(updatedNodes);
    simulation.force('link').links(updatedLinks);

    // 重新启动模拟
    return restart();
  };

  return {
    simulation,
    nodes: nodesData,
    links: linksData,
    warmup,
    destroy,
    restart,
    updateData,
  };
}

/**
 * 创建缩放行为
 *
 * @param {d3.Selection} svg - SVG选择器
 * @param {d3.Selection} container - 内容容器选择器
 * @param {Object} options - 缩放选项
 * @param {Array} [options.extent=[0.1, 8]] - 缩放范围
 * @param {number} [options.duration=250] - 过渡动画持续时间
 * @param {boolean} [options.disableDoubleClick=true] - 是否禁用双击缩放
 * @returns {Object} 缩放实例和辅助函数
 */
export function createZoomBehavior(svg, container, options = {}) {
  const { extent = [0.1, 8], duration = 250, disableDoubleClick = true } = options;

  // 创建缩放行为
  const zoom = d3
    .zoom()
    .scaleExtent(extent)
    .on('zoom', event => {
      container.attr('transform', event.transform);
    });

  // 绑定到SVG
  svg.call(zoom);

  // 禁用双击缩放
  if (disableDoubleClick) {
    svg.on('dblclick.zoom', null);
  }

  // 复位缩放
  const resetZoom = (animate = true) => {
    if (animate) {
      svg.transition().duration(duration).call(zoom.transform, d3.zoomIdentity);
    } else {
      svg.call(zoom.transform, d3.zoomIdentity);
    }
  };

  // 缩放到特定元素
  const zoomToElement = (element, scale = 2) => {
    if (!element) return;

    // 获取元素的位置
    const bounds = element.getBBox();
    const dx = bounds.x;
    const dy = bounds.y;
    const svgBounds = svg.node().getBoundingClientRect();
    const centerX = svgBounds.width / 2;
    const centerY = svgBounds.height / 2;

    // 计算新的变换
    const transform = d3.zoomIdentity
      .translate(centerX - dx * scale, centerY - dy * scale)
      .scale(scale);

    // 应用变换
    svg.transition().duration(duration).call(zoom.transform, transform);
  };

  // 返回缩放实例和工具函数
  return {
    zoom,
    resetZoom,
    zoomToElement,
  };
}

/**
 * 创建拖拽行为
 *
 * @param {d3.Selection} selection - 要应用拖拽的选择器
 * @param {d3.ForceSimulation} simulation - 力导向图模拟
 * @param {Object} options - 拖拽选项
 * @param {boolean} [options.keepFixed=false] - 拖拽后是否保持节点位置固定
 * @param {Function} [options.onDragStart] - 拖拽开始回调
 * @param {Function} [options.onDrag] - 拖拽中回调
 * @param {Function} [options.onDragEnd] - 拖拽结束回调
 * @returns {d3.Drag} 拖拽行为实例
 */
export function createDragBehavior(selection, simulation, options = {}) {
  const { keepFixed = false, onDragStart = null, onDrag = null, onDragEnd = null } = options;

  // 创建拖拽行为
  const drag = d3
    .drag()
    .subject(d => d)
    .on('start', function (event, d) {
      event.sourceEvent.stopPropagation();

      if (!event.active) {
        simulation.alphaTarget(0.3).restart();
      }

      // 固定节点位置
      d.fx = d.x;
      d.fy = d.y;

      // 调用回调
      if (onDragStart) {
        onDragStart(event, d, this);
      }
    })
    .on('drag', function (event, d) {
      // 更新节点位置
      d.fx = event.x;
      d.fy = event.y;

      // 调用回调
      if (onDrag) {
        onDrag(event, d, this);
      }
    })
    .on('end', function (event, d) {
      if (!event.active) {
        simulation.alphaTarget(0);
      }

      // 根据选项决定是否保持固定位置
      if (!keepFixed) {
        d.fx = null;
        d.fy = null;
      }

      // 调用回调
      if (onDragEnd) {
        onDragEnd(event, d, this);
      }
    });

  // 绑定到选择器
  selection.call(drag);

  return drag;
}

/**
 * 创建图形渲染器，用于性能优化渲染
 *
 * @param {Object} options - 渲染器选项
 * @param {number} [options.batchSize=100] - 批量渲染的大小
 * @param {number} [options.debounceTime=50] - 渲染防抖时间(毫秒)
 * @returns {Object} 渲染工具函数
 */
export function createRenderer(options = {}) {
  const { batchSize = 100, debounceTime = 50 } = options;

  let renderTimer = null;
  let animationFrame = null;

  // 批量渲染函数
  const batchRender = (items, renderFn, onComplete) => {
    let index = 0;

    const renderBatch = () => {
      const limit = Math.min(index + batchSize, items.length);

      for (let i = index; i < limit; i++) {
        renderFn(items[i], i);
      }

      index = limit;

      if (index < items.length) {
        animationFrame = requestAnimationFrame(renderBatch);
      } else if (onComplete) {
        onComplete();
      }
    };

    renderBatch();
  };

  // 防抖渲染函数
  const debouncedRender = renderFn => {
    if (renderTimer) {
      clearTimeout(renderTimer);
    }

    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }

    renderTimer = setTimeout(() => {
      renderFn();
    }, debounceTime);
  };

  // 清理所有渲染计时器
  const cleanup = () => {
    if (renderTimer) {
      clearTimeout(renderTimer);
      renderTimer = null;
    }

    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
  };

  return {
    batchRender,
    debouncedRender,
    cleanup,
  };
}

/**
 * 创建节点标签，优化文本渲染
 *
 * @param {d3.Selection} container - 标签容器
 * @param {Array} nodes - 节点数据
 * @param {Object} options - 标签选项
 * @param {Function} [options.getText] - 获取标签文本的函数
 * @param {Function} [options.getOffset] - 获取标签偏移的函数
 * @param {boolean} [options.addBackground=true] - 是否添加背景
 * @param {string} [options.fontFamily='Arial'] - 字体
 * @param {number} [options.fontSize=10] - 字体大小
 * @param {string} [options.textAnchor='start'] - 文本锚点
 * @returns {d3.Selection} 标签选择器
 */
export function createNodeLabels(container, nodes, options = {}) {
  const {
    getText = d => d.label || d.id,
    getOffset = d => [15, 0],
    addBackground = true,
    fontFamily = 'Arial',
    fontSize = 10,
    textAnchor = 'start',
  } = options;

  // 创建标签组
  const labelGroups = container
    .selectAll('.node-label')
    .data(nodes)
    .join('g')
    .attr('class', 'node-label')
    .attr('text-anchor', textAnchor)
    .attr('font-family', fontFamily)
    .attr('font-size', fontSize)
    .attr('pointer-events', 'none');

  // 如果需要，添加文本背景
  if (addBackground) {
    labelGroups
      .append('text')
      .attr('class', 'label-background')
      .attr('stroke', 'white')
      .attr('stroke-width', 3)
      .attr('paint-order', 'stroke')
      .text(getText)
      .attr('x', d => getOffset(d)[0])
      .attr('y', d => getOffset(d)[1]);
  }

  // 添加文本
  labelGroups
    .append('text')
    .attr('class', 'label-text')
    .attr('fill', '#333')
    .text(getText)
    .attr('x', d => getOffset(d)[0])
    .attr('y', d => getOffset(d)[1]);

  // 更新标签位置的函数
  const updatePositions = () => {
    labelGroups.attr('transform', d => `translate(${d.x},${d.y})`);
  };

  return {
    labels: labelGroups,
    updatePositions,
  };
}

/**
 * 为大数据集创建四叉树优化
 *
 * @param {Array} nodes - 节点数据
 * @param {Object} options - 四叉树选项
 * @param {number} [options.width=1000] - 边界宽度
 * @param {number} [options.height=1000] - 边界高度
 * @param {number} [options.nodeRadius=10] - 节点半径，用于碰撞检测
 * @returns {Object} 四叉树实例和辅助函数
 */
export function createQuadtree(nodes, options = {}) {
  const { width = 1000, height = 1000, nodeRadius = 10 } = options;

  // 创建四叉树
  const quadtree = d3
    .quadtree()
    .extent([
      [0, 0],
      [width, height],
    ])
    .x(d => d.x)
    .y(d => d.y)
    .addAll(nodes);

  // 查找最近的节点
  const findClosestNode = (x, y, radius = Infinity) => {
    let closestNode = null;
    let closestDistance = radius;

    quadtree.visit((quad, x1, y1, x2, y2) => {
      if (!quad.length) {
        const dx = quad.data.x - x;
        const dy = quad.data.y - y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < closestDistance) {
          closestDistance = distance;
          closestNode = quad.data;
        }
      }

      // 如果当前四叉树节点与查询点的最小距离大于已找到的最近距离
      // 则不需要继续搜索这个分支
      const closestPossibleDistance = pointRectDistance(x, y, x1, y1, x2, y2);
      return closestPossibleDistance > closestDistance;
    });

    return closestNode;
  };

  // 检测碰撞
  const detectCollisions = () => {
    const collisions = [];

    nodes.forEach(node => {
      quadtree.visit((quad, x1, y1, x2, y2) => {
        if (!quad.length) {
          const otherNode = quad.data;
          if (node !== otherNode) {
            const dx = node.x - otherNode.x;
            const dy = node.y - otherNode.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const minDistance = nodeRadius * 2;

            if (distance < minDistance) {
              collisions.push([node, otherNode]);
            }
          }
        }

        // 如果当前四叉树节点与节点的最小距离大于碰撞检测距离
        // 则不需要继续搜索这个分支
        return (
          x1 > node.x + nodeRadius ||
          x2 < node.x - nodeRadius ||
          y1 > node.y + nodeRadius ||
          y2 < node.y - nodeRadius
        );
      });
    });

    return collisions;
  };

  // 查找可见节点
  const findVisibleNodes = (x, y, width, height) => {
    const visibleNodes = [];

    quadtree.visit((quad, x1, y1, x2, y2) => {
      if (!quad.length) {
        const node = quad.data;
        if (node.x >= x && node.x <= x + width && node.y >= y && node.y <= y + height) {
          visibleNodes.push(node);
        }
      }

      // 如果当前四叉树节点与可见区域没有交集
      // 则不需要继续搜索这个分支
      return x1 > x + width || x2 < x || y1 > y + height || y2 < y;
    });

    return visibleNodes;
  };

  // 计算点到矩形的最近距离
  function pointRectDistance(px, py, x1, y1, x2, y2) {
    const dx = Math.max(x1 - px, 0, px - x2);
    const dy = Math.max(y1 - py, 0, py - y2);
    return Math.sqrt(dx * dx + dy * dy);
  }

  // 更新四叉树
  const update = () => {
    quadtree.removeAll(nodes);
    quadtree.addAll(nodes);
  };

  return {
    quadtree,
    findClosestNode,
    detectCollisions,
    findVisibleNodes,
    update,
  };
}

/**
 * 创建颜色比例尺
 *
 * @param {Array} domain - 域值
 * @param {Array} range - 范围值
 * @param {string} [type='categorical'] - 比例尺类型: 'categorical', 'sequential', 'diverging'
 * @returns {Function} 颜色比例尺函数
 */
export function createColorScale(domain, range, type = 'categorical') {
  switch (type) {
    case 'sequential':
      return d3.scaleLinear().domain(domain).range(range).interpolate(d3.interpolateRgb);

    case 'diverging':
      return d3.scaleDiverging().domain(domain).interpolator(d3.interpolateRgbBasis(range));

    case 'categorical':
    default:
      return d3.scaleOrdinal().domain(domain).range(range);
  }
}

export default {
  createForceSimulation,
  createZoomBehavior,
  createDragBehavior,
  createRenderer,
  createNodeLabels,
  createQuadtree,
  createColorScale,
};
