import fs from 'fs';
import path from 'path';

/**
 * API路由：代理网络图HTML内容
 * 这个API路由会读取静态网络图HTML文件并返回，同时设置适当的HTTP响应头
 */
export default function handler(req, res) {
  try {
    // 读取HTML文件
    const filePath = path.join(process.cwd(), 'public', 'network-graph.html');
    const htmlContent = fs.readFileSync(filePath, 'utf8');

    // 设置响应头
    res.setHeader('Content-Type', 'text/html');
    res.setHeader('X-Frame-Options', 'SAMEORIGIN'); // 允许同源iframe
    res.setHeader(
      'Content-Security-Policy',
      "default-src * 'self' 'unsafe-inline' 'unsafe-eval' data: gap:"
    );
    res.setHeader('Access-Control-Allow-Origin', '*');

    // 返回HTML内容
    res.status(200).send(htmlContent);
  } catch (error) {
    console.error('加载网络图HTML失败:', error);
    res.status(500).json({ error: '无法加载网络图内容' });
  }
}
