import React, { useEffect, useRef, useState } from 'react';
import Head from 'next/head';

// 简单的静态网络图组件
export default function NetworkTestFixedPage() {
  return (
    <div style={{ padding: '20px' }}>
      <Head>
        <title>网络图测试 - 修复版</title>
      </Head>
      
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
        网络图测试页面 (修复版)
      </h1>
      
      <div style={{ 
        backgroundColor: 'white', 
        border: '1px solid #ddd', 
        borderRadius: '8px', 
        padding: '20px', 
        marginBottom: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <p style={{ marginBottom: '15px' }}>这是一个简化的静态网络图：</p>
        
        <svg 
          width="600" 
          height="400"
          style={{ 
            border: '1px solid #eee',
            borderRadius: '4px',
            backgroundColor: '#f8f9fa'
          }}
        >
          {/* 钱包A - 高风险 */}
          <circle cx="150" cy="150" r="30" fill="#f39c12" />
          <text x="150" y="150" textAnchor="middle" fill="white" fontWeight="bold">钱包A</text>
          
          {/* 交易所 - 低风险 */}
          <circle cx="300" cy="100" r="25" fill="#2ecc71" />
          <text x="300" y="100" textAnchor="middle" fill="white" fontWeight="bold">交易所</text>
          
          {/* 钱包B - 中风险 */}
          <circle cx="450" cy="150" r="28" fill="#f1c40f" />
          <text x="450" y="150" textAnchor="middle" fill="white" fontWeight="bold">钱包B</text>
          
          {/* 混币器 - 严重风险 */}
          <circle cx="300" cy="250" r="35" fill="#e74c3c" />
          <text x="300" y="250" textAnchor="middle" fill="white" fontWeight="bold">混币器</text>
          
          {/* 未知节点 - 低风险 */}
          <circle cx="450" cy="300" r="22" fill="#2ecc71" />
          <text x="450" y="300" textAnchor="middle" fill="white" fontWeight="bold">未知</text>
          
          {/* 连接线 */}
          <line x1="150" y1="150" x2="300" y2="100" stroke="#999" strokeWidth="2" />
          <line x1="150" y1="150" x2="300" y2="250" stroke="#999" strokeWidth="2" />
          <line x1="300" y1="100" x2="450" y2="150" stroke="#999" strokeWidth="2" />
          <line x1="300" y1="250" x2="450" y2="300" stroke="#999" strokeWidth="3" />
          <line x1="450" y1="150" x2="450" y2="300" stroke="#999" strokeWidth="2" strokeDasharray="5,5" />
        </svg>
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
                backgroundColor: '#e74c3c', 
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
                backgroundColor: '#f39c12', 
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
                backgroundColor: '#f1c40f', 
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
                backgroundColor: '#2ecc71', 
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
            关于此页面
          </h2>
          
          <p style={{ marginBottom: '10px' }}>
            这是一个简化的静态网络图，去除了D3.js的交互功能，包括：
          </p>
          
          <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
            <li>预定义的节点和连接</li>
            <li>基于风险级别的颜色编码</li>
            <li>简单的图例和说明</li>
          </ul>
        </div>
      </div>
      
      <div>
        <a 
          href="/test" 
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
          访问简单测试页面
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