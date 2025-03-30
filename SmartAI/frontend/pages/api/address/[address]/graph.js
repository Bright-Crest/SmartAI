import { generateMockData } from '../../../../utils/mockData';

/**
 * 地址图谱API端点
 * 返回地址的关系图谱数据
 *
 * @param {Object} req - Next.js API请求对象
 * @param {Object} res - Next.js API响应对象
 */
export default async function handler(req, res) {
  // 只允许GET请求
  if (req.method !== 'GET') {
    return res.status(405).json({ message: '请求方法不允许' });
  }

  try {
    const { address } = req.query;

    // 验证地址格式
    if (!address || !/^0x[a-fA-F0-9]{40}$/.test(address)) {
      return res.status(400).json({ message: '无效的以太坊地址' });
    }

    // 获取查询参数
    const depth = parseInt(req.query.depth || '1', 10);
    const limit = parseInt(req.query.limit || '100', 10);

    // 目前使用模拟数据
    // 实际项目中会转发到后端服务
    const mockData = generateMockData(address);

    // 延迟1.5秒模拟API延迟
    await new Promise(resolve => setTimeout(resolve, 1500));

    // 返回网络图谱部分
    return res.status(200).json(mockData.networkData);
  } catch (error) {
    console.error('地址图谱API错误:', error);
    return res.status(500).json({ message: '服务器内部错误', error: error.message });
  }
}
