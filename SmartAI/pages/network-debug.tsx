import React, { useEffect, useRef, useState } from 'react';
import Head from 'next/head';
import * as d3 from 'd3';

// 简化类型定义
interface Node {
  id: string;
  label: string;
  risk: 'low' | 'medium' | 'high' | 'critical';
  value: number;
}

type Link = {
  source: string;
  target: string;
  value: number;
};

export default function NetworkDebugPage() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [error, setError] = useState<string | null>(null);
  
  // 示例数据 - 内联定义避免外部依赖
  const sampleData = {
    nodes: [
      { id: '1', label: '钱包A', risk: 'high', value: 30 },
      { id: '2', label: '交易所', risk: 'low', value: 25 },
      { id: '3', label: '钱包B', risk: 'medium', value: 28 },
      { id: '4', label: '混币器', risk: 'critical', value: 35 },
      { id: '5', label: '未知节点', risk: 'low', value: 22 },
    ] as Node[],
    
    links: [
      { source: '1', target: '2', value: 2 },
      { source: '1', target: '3', value: 1 },
      { source: '2', target: '4', value: 3 },
      { source: '3', target: '5', value: 2 },
      { source: '4', target: '5', value: 1 },
    ] as Link[]
  };
  
  // 风险级别对应的颜色
  const riskColors = {
    low: '#2ecc71',
    medium: '#f1c40f',
    high: '#f39c12',
    critical: '#e74c3c'
  };
  
  // 第一阶段：只创建SVG和静态元素，不使用复杂的D3功能
  useEffect(() => {
    try {
      if (!svgRef.current) return;
      
      // 清理之前的SVG内容
      d3.select(svgRef.current).selectAll('*').remove();
      
      const width = 600;
      const height = 400;
      
      // 设置SVG
      const svg = d3.select(svgRef.current)
        .attr('width', width)
        .attr('height', height);
        
      // 添加一个背景色
      svg.append('rect')
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#f8f9fa');
      
      // 手动绘制节点到固定位置
      const positions = {
        '1': { x: 150, y: 150 },
        '2': { x: 300, y: 100 },
        '3': { x: 450, y: 150 },
        '4': { x: 300, y: 250 },
        '5': { x: 450, y: 300 }
      };
      
      // 首先绘制连接线
      sampleData.links.forEach(link => {
        const source = positions[link.source];
        const target = positions[link.target];
        
        svg.append('line')
          .attr('x1', source.x)
          .attr('y1', source.y)
          .attr('x2', target.x)
          .attr('y2', target.y)
          .attr('stroke', '#999')
          .attr('stroke-width', link.value);
      });
      
      // 然后绘制节点
      sampleData.nodes.forEach(node => {
        const pos = positions[node.id];
        
        // 绘制圆形
        svg.append('circle')
          .attr('cx', pos.x)
          .attr('cy', pos.y)
          .attr('r', node.value)
          .attr('fill', riskColors[node.risk])
          .attr('stroke', '#fff')
          .attr('stroke-width', 2);
          
        // 添加标签
        svg.append('text')
          .attr('x', pos.x)
          .attr('y', pos.y)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', 'white')
          .attr('font-weight', 'bold')
          .attr('font-size', '12px')
          .text(node.label);
      });
      
    } catch (err) {
      console.error("Error creating network graph:", err);
      setError(`绘制网络图时出错: ${err.message}`);
    }
  }, []);
  
  return (
    <div style={{ padding: '20px' }}>
      <Head>
        <title>网络图调试页面</title>
      </Head>
      
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
        网络图调试页面（无交互版）
      </h1>
      
      <div style={{ 
        backgroundColor: 'white', 
        border: '1px solid #ddd', 
        borderRadius: '8px', 
        padding: '20px', 
        marginBottom: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        {error ? (
          <div style={{ color: 'red', padding: '20px' }}>
            {error}
          </div>
        ) : (
          <svg 
            ref={svgRef} 
            style={{ 
              border: '1px solid #eee',
              borderRadius: '4px',
              width: '100%',
              height: '400px'
            }}
          ></svg>
        )}
      </div>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
        gap: '20px',
        marginBottom: '20px' 
      }}>
        <div style={{ 
          backgroundColor: 'white', 
          border: '1px solid #ddd', 
          borderRadius: '8px', 
          padding: '20px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px' }}>
            风险等级图例
          </h2>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ 
                width: '12px', 
                height: '12px', 
                backgroundColor: riskColors.critical, 
                borderRadius: '50%', 
                display: 'inline-block', 
                marginRight: '8px' 
              }}></span>
              <span>严重风险</span>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ 
                width: '12px', 
                height: '12px', 
                backgroundColor: riskColors.high, 
                borderRadius: '50%', 
                display: 'inline-block', 
                marginRight: '8px' 
              }}></span>
              <span>高风险</span>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ 
                width: '12px', 
                height: '12px', 
                backgroundColor: riskColors.medium, 
                borderRadius: '50%', 
                display: 'inline-block', 
                marginRight: '8px' 
              }}></span>
              <span>中风险</span>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ 
                width: '12px', 
                height: '12px', 
                backgroundColor: riskColors.low, 
                borderRadius: '50%', 
                display: 'inline-block', 
                marginRight: '8px' 
              }}></span>
              <span>低风险</span>
            </div>
          </div>
        </div>
        
        <div style={{ 
          backgroundColor: 'white', 
          border: '1px solid #ddd', 
          borderRadius: '8px', 
          padding: '20px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)' 
        }}>
          <h2 style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px' }}>
            调试信息
          </h2>
          
          <p style={{ marginBottom: '10px' }}>
            这是一个简化的网络图调试页面，只使用基本的D3.js功能：
          </p>
          
          <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
            <li>静态节点位置 (不使用力导向模拟)</li>
            <li>基本的SVG元素绘制</li>
            <li>没有缩放或拖拽交互</li>
            <li>没有动画效果</li>
          </ul>
        </div>
      </div>
      
      <div>
        <a 
          href="/network-test-fixed" 
          style={{ 
            display: 'inline-block', 
            backgroundColor: '#3498db', 
            color: 'white', 
            padding: '8px 16px', 
            borderRadius: '4px', 
            textDecoration: 'none',
            marginRight: '10px'
          }}
        >
          查看静态版本
        </a>
        
        <a 
          href="/test" 
          style={{ 
            display: 'inline-block', 
            backgroundColor: '#2ecc71', 
            color: 'white', 
            padding: '8px 16px', 
            borderRadius: '4px', 
            textDecoration: 'none',
            marginRight: '10px'
          }}
        >
          查看测试页面
        </a>
        
        <a 
          href="/" 
          style={{ 
            display: 'inline-block', 
            backgroundColor: '#95a5a6', 
            color: 'white', 
            padding: '8px 16px', 
            borderRadius: '4px', 
            textDecoration: 'none' 
          }}
        >
          返回首页
        </a>
      </div>
    </div>
  );
} 