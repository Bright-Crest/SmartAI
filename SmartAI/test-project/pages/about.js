import React from "react";

export default function About() {
  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1 style={{ color: "green" }}>测试页面 - 关于</h1>
      <p>这是一个简单的关于页面</p>
      <div
        style={{
          marginTop: "20px",
          padding: "15px",
          border: "1px solid #ddd",
          borderRadius: "5px",
        }}
      >
        <p>Next.js测试项目</p>
      </div>
      <div style={{ marginTop: "20px" }}>
        <a href="/" style={{ color: "blue" }}>
          返回首页
        </a>
      </div>
    </div>
  );
}
