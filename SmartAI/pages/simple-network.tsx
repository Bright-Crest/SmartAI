import React from 'react';
import Head from 'next/head';

export default function SimpleNetworkPage() {
  return (
    <div style={{ padding: '20px' }}>
      <Head>
        <title>简单网络图测试</title>
      </Head>
      
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '20px' }}>
        简单网络图测试页面
      </h1>
      
      <div style={{ 
        backgroundColor: 'white', 
        border: '1px solid #ddd', 
        borderRadius: '8px', 
        padding: '20px', 
        marginBottom: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        minHeight: '300px'
      }}>
        <p>这是一个测试页面，用于验证基本渲染功能是否正常。</p>
        <svg 
          width="400" 
          height="200" 
          style={{ border: '1px solid #eee', marginTop: '20px' }}
        >
          <circle cx="100" cy="100" r="50" fill="#3498db" />
          <circle cx="300" cy="100" r="30" fill="#e74c3c" />
          <line x1="100" y1="100" x2="300" y2="100" stroke="#333" strokeWidth="2" />
          <text x="100" y="150" textAnchor="middle" fill="#333">节点A</text>
          <text x="300" y="150" textAnchor="middle" fill="#333">节点B</text>
        </svg>
      </div>
      
      <div>
        <a 
          href="/" 
          style={{ 
            display: 'inline-block', 
            backgroundColor: '#3498db', 
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