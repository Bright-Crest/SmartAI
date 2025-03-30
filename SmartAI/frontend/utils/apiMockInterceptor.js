/**
 * API模拟拦截器
 *
 * 用于开发环境，拦截API请求并返回模拟数据
 */
import mockData from './mockDataService';

// 判断是否启用模拟API
const enableMockApi =
  process.env.NEXT_PUBLIC_ENABLE_MOCK_API === 'true' || process.env.NODE_ENV === 'development';

// 模拟响应配置
const mockConfig = {
  minDelay: 300, // 最小延迟时间(ms)
  maxDelay: 1000, // 最大延迟时间(ms)
  errorRate: 0.05, // 模拟错误率(0-1)
};

/**
 * 模拟API响应
 *
 * @param {*} data - 响应数据
 * @param {Object} [config] - 配置
 * @returns {Promise<*>} 模拟响应结果
 */
async function mockResponse(data, config = {}) {
  return mockData.mockApiResponse(data, {
    ...mockConfig,
    ...config,
  });
}

/**
 * API请求拦截器
 *
 * @param {string} endpoint - API端点
 * @param {Object} options - 请求选项
 * @returns {Promise<Object|null>} 如果被拦截则返回模拟响应，否则返回null
 */
export async function interceptApiRequest(endpoint, options) {
  // 如果未启用模拟API，直接返回null
  if (!enableMockApi) {
    return null;
  }

  // 解析路径和查询参数
  const url = new URL(endpoint.startsWith('http') ? endpoint : `http://example.com${endpoint}`);
  const path = url.pathname;
  const params = Object.fromEntries(url.searchParams.entries());

  console.log(`[Mock API] 拦截请求: ${options.method || 'GET'} ${path}`);

  // 根据路径和方法模拟不同的响应
  try {
    // 网络分析API
    if (path.startsWith('/api/network')) {
      return await handleNetworkEndpoints(path, params, options);
    }

    // 交易分析API
    if (path.startsWith('/api/transactions')) {
      return await handleTransactionEndpoints(path, params, options);
    }

    // 地址分析API
    if (path.startsWith('/api/addresses')) {
      return await handleAddressEndpoints(path, params, options);
    }

    // 仪表盘API
    if (path.startsWith('/api/dashboard')) {
      return await handleDashboardEndpoints(path, params, options);
    }

    // 未匹配的路径，返回null表示不拦截
    console.log(`[Mock API] 未匹配的路径: ${path}`);
    return null;
  } catch (error) {
    console.error('[Mock API] 拦截处理错误:', error);
    throw error;
  }
}

/**
 * 处理网络分析相关的API请求
 *
 * @param {string} path - API路径
 * @param {Object} params - 查询参数
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} 模拟响应
 */
async function handleNetworkEndpoints(path, params, options) {
  // 获取网络图数据
  if (path.includes('/graph')) {
    const nodeCount = parseInt(params.limit) || 30;
    return mockResponse(mockData.generateNetworkData({ nodeCount }));
  }

  // 获取节点详情
  if (path.match(/\/nodes\/[^/]+$/)) {
    const nodeId = path.split('/').pop();
    const includeTransactions = params.includeTransactions === 'true';

    // 从节点ID提取地址
    const address = mockData.getTestAddresses()[parseInt(nodeId) % 5];
    const addressData = mockData.generateAddressData(address);

    // 如果不包含交易，移除交易数据减小响应大小
    if (!includeTransactions) {
      delete addressData.recentTransactions;
    }

    return mockResponse(addressData);
  }

  // 获取网络统计
  if (path.includes('/stats')) {
    const stats = {
      totalNodes: mockData.getRandomInt(1000, 5000),
      totalLinks: mockData.getRandomInt(2000, 10000),
      riskDistribution: {
        high: mockData.getRandomInt(100, 500),
        medium: mockData.getRandomInt(500, 2000),
        low: mockData.getRandomInt(2000, 4000),
        unknown: mockData.getRandomInt(0, 100),
      },
      nodeTypes: {
        exchange: mockData.getRandomInt(100, 500),
        contract: mockData.getRandomInt(500, 1500),
        address: mockData.getRandomInt(1000, 3000),
        mixer: mockData.getRandomInt(10, 50),
      },
      lastUpdated: new Date().toISOString(),
    };

    return mockResponse(stats);
  }

  // 未匹配的网络分析请求
  return null;
}

/**
 * 处理交易相关的API请求
 *
 * @param {string} path - API路径
 * @param {Object} params - 查询参数
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} 模拟响应
 */
async function handleTransactionEndpoints(path, params, options) {
  // 获取交易列表
  if (path === '/api/transactions') {
    const limit = parseInt(params.limit) || 20;
    const offset = parseInt(params.offset) || 0;
    const riskLevel = params.riskLevel;

    let transactions = mockData.generateTransactionData(limit + 10);

    // 根据风险等级筛选
    if (riskLevel) {
      transactions = transactions.filter(tx => tx.risk === riskLevel);
    }

    // 根据地址筛选
    if (params.address) {
      transactions = transactions.filter(
        tx => tx.from === params.address || tx.to === params.address
      );
    }

    // 分页处理
    const paginatedTransactions = transactions.slice(offset, offset + limit);

    return mockResponse({
      transactions: paginatedTransactions,
      total: 156 + mockData.getRandomInt(1, 100),
      page: Math.floor(offset / limit) + 1,
      pageSize: limit,
      totalPages: Math.ceil((156 + mockData.getRandomInt(1, 100)) / limit),
    });
  }

  // 获取交易详情
  if (path.match(/\/api\/transactions\/0x[a-fA-F0-9]+$/)) {
    const txHash = path.split('/').pop();
    const includeRelated = params.includeRelated === 'true';

    const txDetail = {
      ...mockData.generateTransactionData(1)[0],
      hash: txHash,
      gasLimit: mockData.getRandomInt(21000, 200000),
      nonce: mockData.getRandomInt(1, 1000),
      input: `0x${Array.from({ length: 32 }, () =>
        Math.floor(Math.random() * 16).toString(16)
      ).join('')}`,
      events: [],
      decodedInput: {
        method: 'transfer',
        params: [
          { name: 'to', value: mockData.generateRandomAddress(), type: 'address' },
          {
            name: 'amount',
            value: mockData.getRandomFloat(1, 1000, 2).toString(),
            type: 'uint256',
          },
        ],
      },
    };

    // 添加相关交易
    if (includeRelated) {
      txDetail.relatedTransactions = mockData.generateTransactionData(5);
    }

    return mockResponse(txDetail);
  }

  // 获取异常交易
  if (path.includes('/anomalies')) {
    const limit = parseInt(params.limit) || 10;
    const anomalousTransactions = mockData.generateTransactionData(limit).map(tx => ({
      ...tx,
      risk: 'high',
      riskScore: mockData.getRandomInt(75, 100),
      anomalyFactors: ['短时间内大额转账', '与已知风险地址交互', '异常的交易模式'].slice(
        0,
        mockData.getRandomInt(1, 3)
      ),
      confidence: mockData.getRandomInt(70, 95),
    }));

    return mockResponse(anomalousTransactions);
  }

  // 获取交易统计
  if (path.includes('/stats')) {
    const timeRange = params.timeRange || '24h';

    const stats = {
      count: mockData.getRandomInt(10000, 50000),
      totalValue: mockData.getRandomFloat(1000, 10000, 2),
      avgValue: mockData.getRandomFloat(0.5, 5, 2),
      avgGasPrice: mockData.getRandomFloat(10, 100, 6),
      riskDistribution: {
        high: mockData.getRandomInt(100, 500),
        medium: mockData.getRandomInt(500, 2000),
        low: mockData.getRandomInt(5000, 20000),
      },
      timeDistribution: {},
      typeDistribution: {
        standard: mockData.getRandomInt(5000, 20000),
        contract_call: mockData.getRandomInt(2000, 10000),
        token_transfer: mockData.getRandomInt(1000, 5000),
      },
    };

    // 根据时间范围生成时间分布
    const pointCount =
      timeRange === '24h' ? 24 : timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : 12;

    for (let i = 0; i < pointCount; i++) {
      const key = timeRange === '24h' ? `${i}h` : `day${i + 1}`;
      stats.timeDistribution[key] = mockData.getRandomInt(100, 1000);
    }

    return mockResponse(stats);
  }

  // 搜索交易
  if (path.includes('/search')) {
    const query = params.query || '';
    const limit = parseInt(params.limit) || 10;

    // 生成搜索结果
    const searchResults = mockData.generateTransactionData(limit).map(tx => ({
      hash: tx.hash,
      timestamp: tx.timestamp,
      from: tx.from,
      to: tx.to,
      value: tx.value,
      type: tx.type,
      risk: tx.risk,
    }));

    return mockResponse(searchResults);
  }

  // 未匹配的交易请求
  return null;
}

/**
 * 处理地址相关的API请求
 *
 * @param {string} path - API路径
 * @param {Object} params - 查询参数
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} 模拟响应
 */
async function handleAddressEndpoints(path, params, options) {
  // 获取地址详情
  if (path.match(/\/api\/addresses\/0x[a-fA-F0-9]+$/)) {
    const address = path.split('/').pop();
    const includeTransactions = params.includeTransactions === 'true';

    const addressData = mockData.generateAddressData(address);

    // 如果不包含交易，移除交易数据减小响应大小
    if (!includeTransactions) {
      delete addressData.recentTransactions;
    }

    return mockResponse(addressData);
  }

  // 获取地址交易记录
  if (path.match(/\/api\/addresses\/0x[a-fA-F0-9]+\/transactions$/)) {
    const address = path.split('/')[3];
    const limit = parseInt(params.limit) || 20;
    const offset = parseInt(params.offset) || 0;

    // 生成与该地址相关的交易
    const transactions = mockData.generateTransactionData(limit + 10).map(tx => {
      if (Math.random() > 0.5) {
        tx.from = address;
      } else {
        tx.to = address;
      }
      return tx;
    });

    // 分页处理
    const paginatedTransactions = transactions.slice(offset, offset + limit);

    return mockResponse({
      transactions: paginatedTransactions,
      total: 127 + mockData.getRandomInt(1, 100),
      page: Math.floor(offset / limit) + 1,
      pageSize: limit,
      totalPages: Math.ceil((127 + mockData.getRandomInt(1, 100)) / limit),
    });
  }

  // 获取地址网络
  if (path.match(/\/api\/addresses\/0x[a-fA-F0-9]+\/network$/)) {
    const address = path.split('/')[3];
    const depth = parseInt(params.depth) || 1;
    const limit = parseInt(params.limit) || 50;

    const networkData = mockData.generateNetworkData({
      nodeCount: limit,
      edgeFactor: 1.5,
    });

    // 确保中心节点是请求的地址
    if (networkData.nodes.length > 0) {
      networkData.nodes[0].address = address;
      networkData.nodes[0].label = address.substring(0, 10) + '...';
    }

    return mockResponse(networkData);
  }

  // 获取地址风险评分
  if (path.match(/\/api\/addresses\/0x[a-fA-F0-9]+\/risk$/)) {
    const address = path.split('/')[3];

    const riskScore = mockData.getRandomInt(0, 100);
    let risk;
    if (riskScore >= 75) risk = 'high';
    else if (riskScore >= 50) risk = 'medium';
    else risk = 'low';

    const riskFactorPool = [
      '与已知风险地址有交易',
      '短期内大量交易',
      '交易模式异常',
      '使用了混币服务',
      '交易链路复杂',
      '与暗网相关地址有交互',
    ];

    const riskFactors = [];
    if (riskScore >= 50) {
      const factorCount = mockData.getRandomInt(1, 3);
      for (let i = 0; i < factorCount; i++) {
        const factor = mockData.getRandomArrayElement(riskFactorPool);
        if (!riskFactors.includes(factor)) {
          riskFactors.push(factor);
        }
      }
    }

    return mockResponse({
      address,
      riskScore,
      risk,
      riskFactors,
      detail: {
        transactionRisk: mockData.getRandomInt(0, 100),
        behaviorRisk: mockData.getRandomInt(0, 100),
        associationRisk: mockData.getRandomInt(0, 100),
        lastUpdated: new Date().toISOString(),
      },
    });
  }

  // 获取高风险地址列表
  if (path.includes('/high-risk')) {
    const limit = parseInt(params.limit) || 20;
    const offset = parseInt(params.offset) || 0;

    const addresses = [];
    const addressPool = mockData.getTestAddresses(limit + 10);

    for (let i = 0; i < limit; i++) {
      addresses.push({
        address: addressPool[i],
        type: mockData.getRandomArrayElement(['eoa', 'contract', 'exchange', 'mixer']),
        riskScore: mockData.getRandomInt(75, 100),
        riskFactors: [
          '与已知风险地址有交易',
          '短期内大量交易',
          '交易模式异常',
          '使用了混币服务',
        ].slice(0, mockData.getRandomInt(1, 3)),
        label:
          Math.random() > 0.7
            ? mockData.getRandomArrayElement(['交易所', '矿池', '混币器', '暗网市场'])
            : null,
        lastActivity: mockData.getRandomDate(30).toISOString(),
      });
    }

    return mockResponse({
      addresses: addresses.slice(offset, offset + limit),
      total: 89 + mockData.getRandomInt(1, 20),
      page: Math.floor(offset / limit) + 1,
      pageSize: limit,
      totalPages: Math.ceil((89 + mockData.getRandomInt(1, 20)) / limit),
    });
  }

  // 搜索地址
  if (path.includes('/search')) {
    const query = params.query || '';
    const limit = parseInt(params.limit) || 10;

    const results = [];
    const addressPool = mockData.getTestAddresses(limit);

    for (let i = 0; i < limit; i++) {
      results.push({
        address: addressPool[i],
        type: mockData.getRandomArrayElement(['eoa', 'contract', 'exchange', 'wallet']),
        risk: mockData.getRandomArrayElement(['high', 'medium', 'low']),
        label:
          Math.random() > 0.7
            ? mockData.getRandomArrayElement(['交易所', '矿池', '智能合约', '个人钱包'])
            : null,
        balance: mockData.getRandomFloat(0.1, 100, 4),
        transactionCount: mockData.getRandomInt(10, 1000),
        lastActivity: mockData.getRandomDate(30).toISOString(),
      });
    }

    return mockResponse(results);
  }

  // 未匹配的地址请求
  return null;
}

/**
 * 处理仪表盘相关的API请求
 *
 * @param {string} path - API路径
 * @param {Object} params - 查询参数
 * @param {Object} options - 请求选项
 * @returns {Promise<Object>} 模拟响应
 */
async function handleDashboardEndpoints(path, params, options) {
  // 获取总览数据
  if (path.includes('/overview')) {
    return mockResponse(mockData.generateDashboardOverview());
  }

  // 获取风险趋势
  if (path.includes('/risk-trends')) {
    const timeRange = params.timeRange || '7d';
    const days =
      timeRange === '1d'
        ? 1
        : timeRange === '7d'
          ? 7
          : timeRange === '30d'
            ? 30
            : timeRange === '90d'
              ? 90
              : 7;

    return mockResponse(mockData.generateRiskTrendsData(days));
  }

  // 获取热门地址
  if (path.includes('/top-addresses')) {
    const limit = parseInt(params.limit) || 5;
    const sortBy = params.sortBy || 'activity';

    const addresses = [];
    const addressPool = mockData.getTestAddresses(limit);

    for (let i = 0; i < limit; i++) {
      addresses.push({
        address: addressPool[i],
        type: mockData.getRandomArrayElement(['eoa', 'contract', 'exchange', 'wallet']),
        risk: mockData.getRandomArrayElement(['high', 'medium', 'low']),
        label:
          Math.random() > 0.7
            ? mockData.getRandomArrayElement(['交易所', '矿池', '智能合约', '个人钱包'])
            : null,
        value: mockData.getRandomFloat(10, 1000, 2),
        balance: mockData.getRandomFloat(1, 100, 4),
        transactionCount: mockData.getRandomInt(100, 5000),
        lastActive: mockData.getRandomDate(7).toISOString(),
      });
    }

    // 根据排序字段进行排序
    if (sortBy === 'activity') {
      addresses.sort((a, b) => b.transactionCount - a.transactionCount);
    } else if (sortBy === 'value') {
      addresses.sort((a, b) => b.value - a.value);
    } else if (sortBy === 'risk') {
      const riskWeight = { high: 3, medium: 2, low: 1 };
      addresses.sort((a, b) => riskWeight[b.risk] - riskWeight[a.risk]);
    }

    return mockResponse(addresses);
  }

  // 获取热门交易
  if (path.includes('/top-transactions')) {
    const limit = parseInt(params.limit) || 5;
    const sortBy = params.sortBy || 'value';

    let transactions = mockData.generateTransactionData(limit);

    // 根据排序字段进行排序
    if (sortBy === 'value') {
      transactions.sort((a, b) => b.value - a.value);
    } else if (sortBy === 'time') {
      transactions.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    } else if (sortBy === 'risk') {
      const riskWeight = { high: 3, medium: 2, low: 1, null: 0 };
      transactions.sort((a, b) => riskWeight[b.risk || 'null'] - riskWeight[a.risk || 'null']);
    }

    return mockResponse(transactions);
  }

  // 获取网络活动
  if (path.includes('/network-activity')) {
    const timeRange = params.timeRange || '24h';
    const interval = params.interval || 'hour';

    const hours =
      timeRange === '1h'
        ? 1
        : timeRange === '24h'
          ? 24
          : timeRange === '7d'
            ? 24 * 7
            : timeRange === '30d'
              ? 24 * 30
              : 24;

    // 生成数据点
    const dataPoints = [];
    const now = new Date();
    const intervalMs =
      interval === 'minute'
        ? 60 * 1000
        : interval === 'hour'
          ? 60 * 60 * 1000
          : interval === 'day'
            ? 24 * 60 * 60 * 1000
            : interval === 'week'
              ? 7 * 24 * 60 * 60 * 1000
              : 60 * 60 * 1000;

    const totalPoints = Math.min(hours, 100); // 限制最大点数

    for (let i = 0; i < totalPoints; i++) {
      const date = new Date(now);
      date.setTime(now.getTime() - (totalPoints - i - 1) * intervalMs);

      dataPoints.push({
        timestamp: date.toISOString(),
        transactions: mockData.getRandomInt(50, 500),
        activeAddresses: mockData.getRandomInt(20, 200),
        volume: mockData.getRandomFloat(5, 50, 2),
        avgGasPrice: mockData.getRandomFloat(10, 100, 6),
      });
    }

    return mockResponse(dataPoints);
  }

  // 未匹配的仪表盘请求
  return null;
}

export default interceptApiRequest;
