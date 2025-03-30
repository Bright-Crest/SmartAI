.force('link', d3.forceLink<Node, Link>(sampleData.links)
  .id(d => d.id)
  .distance(100))
.force('charge', d3.forceManyBody().strength(-200))
.force('center', d3.forceCenter(width / 2, height / 2))
// @ts-ignore - value属性在我们的自定义Node接口上存在，但不在SimulationNodeDatum上
.force('collision', d3.forceCollide().radius(d => (d.value || 5) + 10)); 