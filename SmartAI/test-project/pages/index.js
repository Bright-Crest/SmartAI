import React, { useState, useEffect } from 'react';

export default function Home() {
  // 使用客户端状态来存储时间，避免服务器渲染和客户端渲染不一致
  const [currentTime, setCurrentTime] = useState('');
  
  // 使用useEffect确保只在客户端执行
  useEffect(() => {
    // 在客户端设置时间
    setCurrentTime(new Date().toLocaleString());
    
    // 可选：每秒更新一次时间
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleString());
    }, 1000);
    
    // 清理定时器
    return () => clearInterval(timer);
  }, []);

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ color: "blue" }}>测试页面 - 首页</h1>
      <p>这是一个简单的Next.js测试页面</p>
      <div
        style={{
          marginTop: "20px",
          padding: "15px",
          border: "1px solid #ddd",
          borderRadius: "5px",
        }}
      >
        <p>当前时间: {currentTime}</p>
      </div>
    </div>
  );
}
