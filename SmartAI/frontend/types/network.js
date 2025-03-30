/**
 * 网络图相关类型定义
 * 使用JSDoc进行类型说明，方便后续迁移到TypeScript
 */

/**
 * 风险等级枚举
 * @typedef {'high' | 'medium' | 'low'} RiskLevel
 */

/**
 * 网络节点类型
 * @typedef {Object} NetworkNode
 * @property {string} id - 节点唯一标识
 * @property {string} label - 节点显示标签
 * @property {RiskLevel} risk - 风险等级
 * @property {string} [address] - 区块链地址（可选）
 * @property {number} [value] - 节点值/大小（可选）
 * @property {string} [type] - 节点类型（钱包、交易所等）（可选）
 * @property {Object} [metadata] - 额外元数据（可选）
 */

/**
 * 网络连接类型
 * @typedef {Object} NetworkLink
 * @property {string} source - 源节点ID
 * @property {string} target - 目标节点ID
 * @property {string} [type] - 连接类型（交易、合同等）（可选）
 * @property {number} [value] - 连接值/权重（可选）
 * @property {number} [timestamp] - 连接时间戳（可选）
 * @property {Object} [metadata] - 额外元数据（可选）
 */

/**
 * 网络数据类型
 * @typedef {Object} NetworkData
 * @property {NetworkNode[]} nodes - 节点数组
 * @property {NetworkLink[]} links - 连接数组
 */

/**
 * 网络图配置选项
 * @typedef {Object} NetworkOptions
 * @property {number} [nodeSize=10] - 节点默认大小
 * @property {number} [linkDistance=100] - 连接默认距离
 * @property {number} [chargeStrength=-300] - 节点间斥力强度
 * @property {boolean} [enableZoom=true] - 是否启用缩放
 * @property {boolean} [enableDrag=true] - 是否启用拖拽
 * @property {Object} [colors] - 自定义颜色配置
 */

// 导出类型（虽然JS中不需要，但为了与TS语法保持一致且便于迁移）
export const NetworkTypes = {
  // 这里只是为了导出一个空对象，因为JS中没有真正的类型导出
};

// 供调试与演示使用的示例数据
export const sampleNetworkData = {
  nodes: [
    {
      id: '1',
      label: '主要钱包',
      risk: 'high',
      type: 'wallet',
      address: '0x1234567890abcdef1234567890abcdef12345678',
      value: 5.2,
      metadata: {
        firstSeen: '2023-01-15T10:30:00Z',
        lastSeen: '2023-06-20T14:45:00Z',
        transactionCount: 47,
        totalValue: 132.5,
        relatedAddresses: 12,
      },
    },
    {
      id: '2',
      label: '币安交易所',
      risk: 'low',
      type: 'exchange',
      address: '0x2345678901abcdef2345678901abcdef23456789',
      value: 8.7,
      metadata: {
        firstSeen: '2022-11-05T08:15:00Z',
        lastSeen: '2023-06-18T22:10:00Z',
        transactionCount: 156,
        totalValue: 543.2,
        relatedAddresses: 89,
      },
    },
    {
      id: '3',
      label: '二级钱包',
      risk: 'medium',
      type: 'wallet',
      address: '0x3456789012abcdef3456789012abcdef34567890',
      value: 3.1,
      metadata: {
        firstSeen: '2023-02-20T16:45:00Z',
        lastSeen: '2023-06-15T11:30:00Z',
        transactionCount: 23,
        totalValue: 67.8,
        relatedAddresses: 5,
      },
    },
    {
      id: '4',
      label: 'Tornado混币器',
      risk: 'high',
      type: 'mixer',
      address: '0x4567890123abcdef4567890123abcdef45678901',
      value: 6.4,
      metadata: {
        firstSeen: '2022-09-10T13:20:00Z',
        lastSeen: '2023-06-19T09:15:00Z',
        transactionCount: 421,
        totalValue: 1247.6,
        relatedAddresses: 312,
      },
    },
    {
      id: '5',
      label: '火币交易所',
      risk: 'low',
      type: 'exchange',
      address: '0x5678901234abcdef5678901234abcdef56789012',
      value: 7.9,
      metadata: {
        firstSeen: '2022-10-25T19:30:00Z',
        lastSeen: '2023-06-20T10:45:00Z',
        transactionCount: 189,
        totalValue: 678.3,
        relatedAddresses: 76,
      },
    },
    {
      id: '6',
      label: '可疑钱包A',
      risk: 'high',
      type: 'wallet',
      address: '0x6789012345abcdef6789012345abcdef67890123',
      value: 4.5,
      metadata: {
        firstSeen: '2023-03-05T07:30:00Z',
        lastSeen: '2023-06-17T21:10:00Z',
        transactionCount: 31,
        totalValue: 98.7,
        relatedAddresses: 9,
      },
    },
    {
      id: '7',
      label: '矿池账户',
      risk: 'medium',
      type: 'miner',
      address: '0x7890123456abcdef7890123456abcdef78901234',
      value: 9.2,
      metadata: {
        firstSeen: '2022-08-15T11:45:00Z',
        lastSeen: '2023-06-20T16:30:00Z',
        transactionCount: 214,
        totalValue: 876.4,
        relatedAddresses: 28,
      },
    },
    {
      id: '8',
      label: '智能合约',
      risk: 'low',
      type: 'contract',
      address: '0x8901234567abcdef8901234567abcdef89012345',
      value: 2.8,
      metadata: {
        firstSeen: '2023-01-30T14:15:00Z',
        lastSeen: '2023-06-18T18:45:00Z',
        transactionCount: 67,
        totalValue: 134.2,
        relatedAddresses: 41,
      },
    },
    {
      id: '9',
      label: 'DeFi协议',
      risk: 'medium',
      type: 'defi',
      address: '0x9012345678abcdef9012345678abcdef90123456',
      value: 5.7,
      metadata: {
        firstSeen: '2022-12-10T09:20:00Z',
        lastSeen: '2023-06-19T15:10:00Z',
        transactionCount: 118,
        totalValue: 432.9,
        relatedAddresses: 56,
      },
    },
    {
      id: '10',
      label: '可疑钱包B',
      risk: 'high',
      type: 'wallet',
      address: '0xa123456789abcdefa123456789abcdefa1234567',
      value: 3.6,
      metadata: {
        firstSeen: '2023-04-05T10:30:00Z',
        lastSeen: '2023-06-15T08:45:00Z',
        transactionCount: 19,
        totalValue: 57.3,
        relatedAddresses: 7,
      },
    },
  ],
  links: [
    { source: '1', target: '2', type: 'transaction', value: 2.5, timestamp: 1671618000000 },
    { source: '1', target: '3', type: 'transaction', value: 1.2, timestamp: 1672227600000 },
    { source: '1', target: '4', type: 'transaction', value: 3.8, timestamp: 1673091600000 },
    { source: '2', target: '5', type: 'transaction', value: 4.1, timestamp: 1675166400000 },
    { source: '3', target: '6', type: 'transaction', value: 0.9, timestamp: 1676376000000 },
    { source: '4', target: '7', type: 'transaction', value: 2.7, timestamp: 1677585600000 },
    { source: '4', target: '10', type: 'transaction', value: 1.5, timestamp: 1678795200000 },
    { source: '5', target: '8', type: 'transaction', value: 3.2, timestamp: 1680004800000 },
    { source: '6', target: '9', type: 'transaction', value: 1.8, timestamp: 1681214400000 },
    { source: '7', target: '2', type: 'transaction', value: 5.6, timestamp: 1682424000000 },
    { source: '8', target: '3', type: 'transaction', value: 2.3, timestamp: 1683633600000 },
    { source: '9', target: '1', type: 'transaction', value: 3.9, timestamp: 1684843200000 },
    { source: '10', target: '5', type: 'transaction', value: 1.1, timestamp: 1686052800000 },
    { source: '2', target: '6', type: 'transaction', value: 2.8, timestamp: 1687262400000 },
    { source: '3', target: '9', type: 'transaction', value: 1.7, timestamp: 1688472000000 },
    { source: '6', target: '10', type: 'transaction', value: 0.7, timestamp: 1689681600000 },
    { source: '7', target: '9', type: 'transaction', value: 4.5, timestamp: 1690891200000 },
    { source: '8', target: '10', type: 'transaction', value: 2.1, timestamp: 1692100800000 },
  ],
};

// 创建一个更简单的示例数据用于调试
export const simpleSampleData = {
  nodes: [
    { id: 'A1', label: '节点A1', risk: 'high', value: 5 },
    { id: 'A2', label: '节点A2', risk: 'medium', value: 4 },
    { id: 'A3', label: '节点A3', risk: 'low', value: 3 },
    { id: 'A4', label: '节点A4', risk: 'high', value: 6 },
    { id: 'A5', label: '节点A5', risk: 'medium', value: 4 },
  ],
  links: [
    { source: 'A1', target: 'A2', value: 3 },
    { source: 'A1', target: 'A3', value: 2 },
    { source: 'A2', target: 'A4', value: 1 },
    { source: 'A3', target: 'A4', value: 2 },
    { source: 'A4', target: 'A5', value: 3 },
  ],
};
