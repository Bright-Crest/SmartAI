/**
 * 模拟数据服务
 *
 * 提供生成模拟数据的工具函数，用于开发和演示环境
 */

/**
 * 生成随机整数
 *
 * @param {number} min - 最小值(包含)
 * @param {number} max - 最大值(包含)
 * @returns {number} 随机整数
 */
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * 生成随机浮点数
 *
 * @param {number} min - 最小值
 * @param {number} max - 最大值
 * @param {number} [decimals=2] - 小数位数
 * @returns {number} 随机浮点数
 */
function getRandomFloat(min, max, decimals = 2) {
  const random = Math.random() * (max - min) + min;
  const power = Math.pow(10, decimals);
  return Math.round(random * power) / power;
}

/**
 * 从数组中随机选择一个元素
 *
 * @param {Array} array - 输入数组
 * @returns {*} 随机选择的元素
 */
function getRandomArrayElement(array) {
  return array[Math.floor(Math.random() * array.length)];
}

/**
 * 生成随机ETH地址
 *
 * @returns {string} 随机ETH地址
 */
function generateRandomAddress() {
  let address = '0x';
  for (let i = 0; i < 40; i++) {
    address += '0123456789abcdef'[Math.floor(Math.random() * 16)];
  }
  return address;
}

/**
 * 生成随机交易哈希
 *
 * @returns {string} 随机交易哈希
 */
function generateRandomTxHash() {
  let hash = '0x';
  for (let i = 0; i < 64; i++) {
    hash += '0123456789abcdef'[Math.floor(Math.random() * 16)];
  }
  return hash;
}

/**
 * 生成固定的测试地址集合（确保数据一致性）
 *
 * @param {number} [count=10] - 地址数量
 * @returns {string[]} 地址数组
 */
function getTestAddresses(count = 10) {
  // 使用固定的测试地址
  const addresses = [
    '0x742d35Cc6634C0532925a3b844Bc454e4438f44e', // 高风险交易所地址
    '0x8D97689C9818892B700e27F316cc3E41e17fBeb9', // 中风险矿池地址
    '0x3B873a919aA0512D5A0F09E6dCCEb5343753AA24', // 低风险普通用户地址
    '0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326', // 高风险混币器地址
    '0x5a52E96BAcdaBb82fd05763E25335261B270Efcb', // 中风险智能合约地址
  ];

  // 生成剩余的随机地址
  while (addresses.length < count) {
    addresses.push(generateRandomAddress());
  }

  return addresses.slice(0, count);
}

/**
 * 生成随机日期
 *
 * @param {number} [daysBack=30] - 最大天数（往前）
 * @returns {Date} 随机日期
 */
function getRandomDate(daysBack = 30) {
  const now = new Date();
  const randomDaysBack = Math.random() * daysBack;
  const randomMsBack = randomDaysBack * 24 * 60 * 60 * 1000;
  return new Date(now.getTime() - randomMsBack);
}

/**
 * 模拟API响应延迟
 *
 * @param {number} [min=200] - 最小延迟(毫秒)
 * @param {number} [max=800] - 最大延迟(毫秒)
 * @returns {Promise<void>} 延迟Promise
 */
export function simulateApiDelay(min = 200, max = 800) {
  const delay = getRandomInt(min, max);
  return new Promise(resolve => setTimeout(resolve, delay));
}

/**
 * 模拟API响应
 *
 * @param {*} data - 响应数据
 * @param {Object} [options] - 选项
 * @param {number} [options.minDelay=200] - 最小延迟(毫秒)
 * @param {number} [options.maxDelay=800] - 最大延迟(毫秒)
 * @param {number} [options.errorRate=0] - 错误率(0-1)
 * @returns {Promise<*>} 响应数据
 */
export async function mockApiResponse(data, options = {}) {
  const { minDelay = 200, maxDelay = 800, errorRate = 0 } = options;

  // 模拟延迟
  await simulateApiDelay(minDelay, maxDelay);

  // 模拟错误
  if (Math.random() < errorRate) {
    throw new Error('模拟API错误');
  }

  return data;
}

/**
 * 生成模拟网络图数据
 *
 * @param {Object} [options={}] - 配置选项
 * @param {number} [options.nodeCount=30] - 节点数量
 * @param {number} [options.edgeFactor=2] - 边数与节点数的比例
 * @returns {Object} 模拟网络图数据
 */
export function generateNetworkData(options = {}) {
  const { nodeCount = 30, edgeFactor = 2 } = options;

  const riskLevels = ['high', 'medium', 'low'];
  const types = ['address', 'contract', 'exchange', 'wallet'];
  const addressPool = getTestAddresses(Math.max(20, nodeCount));

  // 生成节点
  const nodes = Array.from({ length: nodeCount }, (_, i) => {
    const address = addressPool[i % addressPool.length];
    const type = getRandomArrayElement(types);
    const risk = i < 3 ? riskLevels[i] : getRandomArrayElement(riskLevels);

    return {
      id: i.toString(),
      label: address.substring(0, 10) + '...',
      address,
      type,
      risk,
      value: getRandomFloat(0.1, 20, 4),
      metadata: {
        transactionCount: getRandomInt(1, 100),
        firstSeen: getRandomDate(90).toISOString(),
        lastSeen: getRandomDate(30).toISOString(),
      },
    };
  });

  // 生成边
  const linkCount = Math.floor(nodeCount * edgeFactor);
  const links = Array.from({ length: linkCount }, (_, i) => {
    const source = getRandomInt(0, nodeCount - 1).toString();
    let target;
    do {
      target = getRandomInt(0, nodeCount - 1).toString();
    } while (source === target);

    return {
      source,
      target,
      type: Math.random() > 0.7 ? 'contract_call' : 'transaction',
      value: getRandomFloat(0.1, 5, 4),
      timestamp: getRandomDate(30).toISOString(),
    };
  });

  return { nodes, links };
}

/**
 * 生成模拟交易数据
 *
 * @param {number} [count=20] - 交易数量
 * @returns {Array} 模拟交易数据
 */
export function generateTransactionData(count = 20) {
  const addressPool = getTestAddresses();
  const transactions = [];

  for (let i = 0; i < count; i++) {
    const fromIndex = getRandomInt(0, addressPool.length - 1);
    let toIndex;
    do {
      toIndex = getRandomInt(0, addressPool.length - 1);
    } while (fromIndex === toIndex);

    const timestamp = getRandomDate(30);
    const value = getRandomFloat(0.01, 10, 4);
    const risk =
      i < 3 ? ['high', 'medium', 'low'][i] : getRandomArrayElement(['high', 'medium', 'low', null]);
    const riskFactors = risk
      ? [
          '异常大额交易',
          '地址风险评分高',
          '交易频率异常',
          '与已知风险地址交互',
          '使用了混币服务',
        ].slice(0, getRandomInt(1, 3))
      : [];

    transactions.push({
      hash: generateRandomTxHash(),
      blockNumber: 17000000 + i,
      timestamp: timestamp.toISOString(),
      from: addressPool[fromIndex],
      to: addressPool[toIndex],
      value,
      gasPrice: getRandomFloat(10, 100, 6),
      gasUsed: getRandomInt(21000, 100000),
      type: getRandomArrayElement(['standard', 'contract_call', 'token_transfer']),
      status: getRandomArrayElement(['confirmed', 'confirmed', 'confirmed', 'pending', 'failed']),
      risk,
      riskFactors,
    });
  }

  return transactions;
}

/**
 * 生成模拟地址数据
 *
 * @param {string} [address] - 指定地址，如果不提供则随机生成
 * @returns {Object} 模拟地址数据
 */
export function generateAddressData(address) {
  address = address || getTestAddresses(1)[0];

  // 生成相关交易
  const transactions = generateTransactionData(10).map(tx => {
    if (Math.random() > 0.5) {
      tx.from = address;
    } else {
      tx.to = address;
    }
    return tx;
  });

  // 生成地址标签
  const labels = ['交易所', '矿池', '智能合约', '个人钱包', '混币器'];
  const label = Math.random() > 0.7 ? getRandomArrayElement(labels) : null;

  // 生成随机标签
  const tags = ['高交易量', 'DEX用户', '合约创建者', 'NFT交易', '稳定币用户'];
  const selectedTags = [];
  const tagCount = getRandomInt(0, 3);
  for (let i = 0; i < tagCount; i++) {
    const randomTag = getRandomArrayElement(tags);
    if (!selectedTags.includes(randomTag)) {
      selectedTags.push(randomTag);
    }
  }

  // 生成风险分数和因素
  const riskScore = getRandomInt(0, 100);
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
    const factorCount = getRandomInt(1, 3);
    for (let i = 0; i < factorCount; i++) {
      const factor = getRandomArrayElement(riskFactorPool);
      if (!riskFactors.includes(factor)) {
        riskFactors.push(factor);
      }
    }
  }

  // 生成模拟余额和交易统计
  const balance = getRandomFloat(0.1, 100, 4);
  const transactionCount = getRandomInt(10, 1000);
  const firstActivity = getRandomDate(365);
  const lastActivity = getRandomDate(30);

  // 生成代币余额
  const tokens = {
    USDT: getRandomFloat(100, 10000, 2),
    USDC: getRandomFloat(100, 5000, 2),
    WETH: getRandomFloat(0.1, 10, 4),
    LINK: getRandomFloat(10, 500, 2),
  };

  // 生成活动统计
  const activity = {
    transactionsByTime: {
      daily: getRandomInt(0, 10),
      weekly: getRandomInt(10, 50),
      monthly: getRandomInt(50, 200),
    },
    transactionsByType: {
      send: getRandomInt(1, 100),
      receive: getRandomInt(1, 100),
      contract: getRandomInt(0, 50),
    },
    valueByTime: {
      daily: getRandomFloat(0, 10, 2),
      weekly: getRandomFloat(10, 50, 2),
      monthly: getRandomFloat(50, 200, 2),
    },
    riskDistribution: {
      high: getRandomInt(0, 10),
      medium: getRandomInt(5, 20),
      low: getRandomInt(10, 100),
    },
  };

  // 生成相关地址
  const relatedCount = getRandomInt(3, 8);
  const relatedAddresses = Array.from({ length: relatedCount }, () => {
    const relatedAddress = generateRandomAddress();
    return {
      address: relatedAddress,
      transactionCount: getRandomInt(1, 50),
      totalValue: getRandomFloat(0.1, 50, 4),
      lastInteraction: getRandomDate(60).toISOString(),
      relationshipType: getRandomArrayElement(['交易对手', '合约调用', '代币转账']),
    };
  });

  // 返回最终地址数据
  return {
    address,
    type: getRandomArrayElement(['eoa', 'contract', 'exchange', 'wallet']),
    risk,
    riskScore,
    riskFactors,
    balance,
    tokens,
    transactionCount,
    label,
    tags: selectedTags,
    firstActivity: firstActivity.toISOString(),
    lastActivity: lastActivity.toISOString(),
    recentTransactions: transactions,
    activity,
    relatedAddresses,
    metadata: {
      createdAt: getRandomDate(365).toISOString(),
      lastUpdated: new Date().toISOString(),
    },
  };
}

/**
 * 生成仪表盘概览数据
 *
 * @returns {Object} 仪表盘概览数据
 */
export function generateDashboardOverview() {
  return {
    transactionCount: getRandomInt(10000, 50000),
    activeAddresses: getRandomInt(5000, 20000),
    totalValue: getRandomFloat(1000, 5000, 2),
    avgTransactionValue: getRandomFloat(0.5, 5, 2),
    riskDistribution: {
      high: getRandomInt(100, 500),
      medium: getRandomInt(500, 2000),
      low: getRandomInt(5000, 20000),
    },
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * 生成风险趋势数据
 *
 * @param {number} [days=7] - 天数
 * @returns {Array} 风险趋势数据
 */
export function generateRiskTrendsData(days = 7) {
  const data = [];
  const now = new Date();

  for (let i = 0; i < days; i++) {
    const date = new Date(now);
    date.setDate(now.getDate() - (days - i - 1));

    data.push({
      date: date.toISOString().split('T')[0],
      highRisk: getRandomInt(10, 50),
      mediumRisk: getRandomInt(50, 200),
      lowRisk: getRandomInt(200, 1000),
      total: 0, // 将在下面计算
    });

    // 计算每天的总数
    data[i].total = data[i].highRisk + data[i].mediumRisk + data[i].lowRisk;
  }

  return data;
}

// 导出所有基础工具函数
export {
  getRandomInt,
  getRandomFloat,
  getRandomArrayElement,
  generateRandomAddress,
  generateRandomTxHash,
  getTestAddresses,
  getRandomDate,
};

// 主导出
export default {
  // 基础工具
  getRandomInt,
  getRandomFloat,
  getRandomArrayElement,
  generateRandomAddress,
  generateRandomTxHash,
  getTestAddresses,
  getRandomDate,

  // 数据生成函数
  generateNetworkData,
  generateTransactionData,
  generateAddressData,
  generateDashboardOverview,
  generateRiskTrendsData,

  // API模拟功能
  simulateApiDelay,
  mockApiResponse,
};
