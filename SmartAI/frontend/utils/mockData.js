/**
 * 模拟数据生成工具
 * 提供生成网络分析页面所需的模拟数据
 */

/**
 * 生成随机整数
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
 * 生成随机ETH地址
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
 * 生成模拟网络数据
 * @param {number} nodeCount - 节点数量
 * @param {number} linkCount - 连接数量
 * @returns {Object} 包含nodes和links的网络数据
 */
export function generateMockNetworkData(nodeCount = 100, linkCount = 200) {
  const nodeTypes = ['address', 'contract', 'exchange', 'mixer', 'high_risk'];

  // 生成节点
  const nodes = Array.from({ length: nodeCount }, (_, i) => {
    const type = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
    const riskScore =
      type === 'high_risk'
        ? getRandomInt(70, 100)
        : type === 'mixer'
          ? getRandomInt(50, 90)
          : getRandomInt(10, 70);

    return {
      id: `node-${i}`,
      address: generateRandomAddress(),
      type,
      label: `${type.charAt(0).toUpperCase() + type.slice(1)} ${i}`,
      value: Math.random() * 20 + 1,
      riskScore,
      connections: 0,
    };
  });

  // 生成连接
  const links = [];
  for (let i = 0; i < linkCount; i++) {
    const source = getRandomInt(0, nodeCount - 1);
    let target;
    do {
      target = getRandomInt(0, nodeCount - 1);
    } while (source === target);

    links.push({
      id: `link-${i}`,
      source: `node-${source}`,
      target: `node-${target}`,
      value: Math.random() * 5 + 0.5,
      type: Math.random() > 0.7 ? 'high_value' : 'normal',
    });

    // 更新连接计数
    const sourceNode = nodes.find(n => n.id === `node-${source}`);
    const targetNode = nodes.find(n => n.id === `node-${target}`);
    if (sourceNode) sourceNode.connections = (sourceNode.connections || 0) + 1;
    if (targetNode) targetNode.connections = (targetNode.connections || 0) + 1;
  }

  return { nodes, links };
}

/**
 * 生成模拟数据工具
 * 用于前端开发阶段，生成假的地址分析数据
 */

// 示例地址列表
export const sampleAddresses = [
  '0x28c6c06298d514db089934071355e5743bf21d60', // 价值投资者
  '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0', // DeFi策略师
  '0xdac17f958d2ee523a2206206994597c13d831ec7', // 交易员/市场择时者
];

// 行为类型列表
const behaviorTypes = ['套利型', '价值持有者', '交易者', '鲸鱼', '开发者'];

// 行为标签列表
const behaviorTags = [
  'DeFi用户',
  'NFT收藏家',
  '频繁交易',
  '长期持有',
  '稳定币偏好',
  '跨链交互',
  '合约部署者',
  'MEV机器人',
  '去中心化交易所用户',
  '借贷平台用户',
];

// 随机生成行为标签
const generateRandomTags = (count = 3) => {
  const shuffled = [...behaviorTags].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, Math.min(count, behaviorTags.length));
};

// 随机生成洞察
const generateRandomInsights = (count = 3) => {
  const insights = [
    {
      title: '交易频率高于平均水平',
      description: '在过去30天内交易频率高出平均水平127%，显示出活跃的交易行为。',
    },
    {
      title: '偏好去中心化交易',
      description: '90%的交易来自去中心化交易所，表明用户偏好非托管交易。',
    },
    {
      title: 'Gas优化明显',
      description: '交易Gas使用比同类用户低23%，显示了良好的链上操作优化。',
    },
    {
      title: '跨链活跃度高',
      description: '活跃于5个不同的区块链网络，表明用户具有丰富的跨链经验。',
    },
    {
      title: '稳定币储备比例大',
      description: '资产组合中稳定币占比65%，表明对市场波动持谨慎态度。',
    },
    {
      title: 'NFT交易活跃',
      description: '在NFT市场活跃度排名前10%，展示了对数字收藏品的兴趣。',
    },
    {
      title: 'DeFi借贷频繁',
      description: '频繁使用借贷协议，杠杆率控制在合理范围内。',
    },
    {
      title: '流动性提供者',
      description: '为多个AMM协议提供流动性，获取平台收益。',
    },
  ];

  const shuffled = [...insights].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, Math.min(count, insights.length));
};

/**
 * 生成SmartScore数据
 * @param {string} address - 钱包地址
 * @returns {Object} SmartScore数据对象
 */
export const generateSmartScoreData = address => {
  // 基于地址生成一个伪随机分数，使相同地址总是得到相同结果
  const addressSeed = Array.from(address).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const normalizedSeed = (addressSeed % 100) / 100;

  // 生成60-95之间的分数
  const score = Math.floor(60 + normalizedSeed * 35);

  // 选择行为类型
  const behaviorTypeIndex = Math.floor(normalizedSeed * behaviorTypes.length);
  const behaviorType = behaviorTypes[behaviorTypeIndex];

  // 生成标签
  const tags = generateRandomTags(3 + Math.floor(normalizedSeed * 2));

  return {
    score,
    behaviorType,
    tags,
  };
};

/**
 * 生成五维雷达图数据
 * @param {string} address - 钱包地址
 * @returns {Object} 五维雷达图数据对象
 */
export const generateRadarChartData = address => {
  // 基于地址生成伪随机数据
  const addressSeed = Array.from(address).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const normalizedSeed = (addressSeed % 100) / 100;

  // 生成五个维度的随机分数（范围30-100）
  const getRandomScore = (offset = 0) => Math.floor(30 + ((normalizedSeed + offset) % 1) * 70);

  return {
    activityScore: getRandomScore(0.1),
    protocolInteractionScore: getRandomScore(0.2),
    profitabilityScore: getRandomScore(0.3),
    fundFlowScore: getRandomScore(0.4),
    distillationScore: getRandomScore(0.5),
  };
};

/**
 * 生成行为分析数据
 * @param {string} address - 钱包地址
 * @returns {Object} 行为分析数据对象
 */
export const generateBehaviorData = address => {
  // 基于地址生成伪随机数据
  const addressSeed = Array.from(address).reduce((sum, char) => sum + char.charCodeAt(0), 0);
  const normalizedSeed = (addressSeed % 100) / 100;

  // 选择行为类型
  const behaviorTypeIndex = Math.floor(normalizedSeed * behaviorTypes.length);
  const behaviorType = behaviorTypes[behaviorTypeIndex];

  // 生成标签
  const behaviorTags = generateRandomTags(3 + Math.floor(normalizedSeed * 2));

  // 生成行为总结
  const summaries = [
    `该地址展现出典型的${behaviorType}特征，偏好使用去中心化交易所和DeFi协议。交易活跃度高，但资金流动较为谨慎，常持有主流资产和稳定币。`,
    `作为${behaviorType}，该地址在市场波动中表现出较高的抗风险能力，倾向于逆势操作。经常参与新协议的早期流动性挖矿，并持有多种代币以分散风险。`,
    `该地址是典型的${behaviorType}，通常在低位建仓，高位减持。持有周期较长，偏好蓝筹项目，较少参与高风险项目，展现出成熟的投资策略。`,
    `此${behaviorType}地址经常与知名DeFi协议交互，展示出对新兴金融工具的掌握。具有较强的资金效率意识，常使用借贷协议优化资产配置。`,
  ];

  const summaryIndex = Math.floor(normalizedSeed * summaries.length);
  const summary = summaries[summaryIndex];

  // 生成洞察
  const insights = generateRandomInsights(3 + Math.floor(normalizedSeed * 2));

  return {
    behaviorType,
    behaviorTags,
    summary,
    insights,
  };
};

/**
 * 生成网络图谱数据
 * @param {string} address - 钱包地址
 * @returns {Object} 网络图谱数据对象
 */
export const generateNetworkData = address => {
  // 生成5-10个随机节点
  const nodeCount = 5 + Math.floor(Math.random() * 5);
  const nodes = [];
  const links = [];

  // 添加中心节点（当前地址）
  nodes.push({
    id: address,
    label: `${address.slice(0, 6)}...${address.slice(-4)}`,
    type: 'address',
    value: 100,
  });

  // 生成随机节点和连接
  for (let i = 0; i < nodeCount; i++) {
    // 生成随机地址
    const randomAddrByte = Math.floor(Math.random() * 16).toString(16);
    const nodeAddress = `0x${randomAddrByte.repeat(40)}`.slice(0, 42);

    // 节点类型
    const nodeTypes = ['address', 'contract', 'exchange', 'mixer'];
    const nodeType = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];

    // 风险级别
    const riskLevels = [undefined, 'low', 'medium', 'high'];
    const riskLevel = riskLevels[Math.floor(Math.random() * riskLevels.length)];

    // 添加节点
    nodes.push({
      id: nodeAddress,
      label: `${nodeAddress.slice(0, 6)}...${nodeAddress.slice(-4)}`,
      type: nodeType,
      value: 30 + Math.floor(Math.random() * 70),
      riskLevel,
    });

    // 添加连接
    const ethValue = (1 + Math.random() * 9).toFixed(2);
    links.push({
      id: `link-${i}`,
      source: Math.random() > 0.3 ? address : nodeAddress,
      target: Math.random() > 0.3 ? nodeAddress : address,
      value: 1 + Math.random() * 2,
      label: `${ethValue} ETH`,
    });
  }

  return { nodes, links };
};

/**
 * 综合生成全部模拟数据
 * @param {string} address - 钱包地址
 * @returns {Object} 全部模拟数据对象
 */
export const generateMockData = address => {
  return {
    address,
    smartScore: generateSmartScoreData(address),
    radarData: generateRadarChartData(address),
    behaviorData: generateBehaviorData(address),
    networkData: generateNetworkData(address),
    timestamp: new Date().toISOString(),
  };
};
