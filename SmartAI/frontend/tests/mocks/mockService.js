/**
 * 模拟服务 - 为前端测试提供模拟数据和API响应
 * 这个服务可以被所有组件测试使用，提供一致的模拟行为
 */

// 创建通用的模拟响应
export const createMockResponse = (data, status = 200, statusText = 'OK') => {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: async () => data,
    text: async () => JSON.stringify(data),
    headers: new Headers({
      'Content-Type': 'application/json',
    }),
  };
};

// 模拟API错误
export const createMockError = (status = 500, statusText = 'Internal Server Error') => {
  return {
    ok: false,
    status,
    statusText,
    json: async () => ({ error: statusText }),
    text: async () => JSON.stringify({ error: statusText }),
    headers: new Headers({
      'Content-Type': 'application/json',
    }),
  };
};

// 模拟延迟
export const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// 模拟交易数据
export const mockTransactions = [
  {
    id: 'tx1',
    hash: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
    from: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
    to: '0x8Ba1f109551bD432803012645Ac136ddd64DBA72',
    amount: '1.5 ETH',
    value: '1.5 ETH',
    timestamp: Date.now() - 1000 * 60 * 5,
    riskLevel: 'low',
    status: 'confirmed',
  },
  {
    id: 'tx2',
    hash: '0x2932b7A2355D6fecc4b5c0B6BD44cC31df247a2e',
    from: '0x2932b7A2355D6fecc4b5c0B6BD44cC31df247a2e',
    to: '0x6B175474E89094C44Da98b954EedeAC495271d0F',
    amount: '5000 USDC',
    value: '5000 USDC',
    timestamp: Date.now() - 1000 * 60 * 15,
    riskLevel: 'medium',
    riskScore: 45,
    status: 'confirmed',
  },
  {
    id: 'tx3',
    hash: '0x5B38Da6a701c568545dCfcB03FcB875f56beddC4',
    from: '0x5B38Da6a701c568545dCfcB03FcB875f56beddC4',
    to: '0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B',
    amount: '10 ETH',
    value: '10 ETH',
    timestamp: Date.now() - 1000 * 60 * 30,
    riskLevel: 'high',
    riskScore: 75,
    status: 'pending',
  },
];

// 模拟地址数据
export const mockAddresses = [
  {
    id: 'addr1',
    address: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
    label: '交易所钱包',
    balance: '125.45 ETH',
    transactionCount: 342,
    riskLevel: 'low',
    tags: ['交易所', '已验证'],
  },
  {
    id: 'addr2',
    address: '0x6B175474E89094C44Da98b954EedeAC495271d0F',
    label: '代币合约',
    balance: '0 ETH',
    transactionCount: 1254,
    riskLevel: 'medium',
    riskScore: 45,
    tags: ['合约', 'ERC20'],
  },
];

// 模拟风险警报数据
export const mockRiskAlerts = [
  {
    id: 'alert1',
    type: 'transaction',
    severity: 'high',
    timestamp: Date.now() - 1000 * 60 * 30,
    description: '大额交易警报: 100 ETH',
    address: '0x5B38Da6a701c568545dCfcB03FcB875f56beddC4',
    transactionHash: '0x5B38Da6a701c568545dCfcB03FcB875f56beddC4',
  },
  {
    id: 'alert2',
    type: 'address',
    severity: 'critical',
    timestamp: Date.now() - 1000 * 60 * 120,
    description: '风险地址互动',
    address: '0xdD870fA1b7C4700F2BD7f44238821C26f7392148',
  },
];

// 设置模拟API调用
export const setupMockApi = () => {
  // 重置并设置全局fetch模拟
  global.fetch = jest.fn();

  // 模拟交易相关API
  const mockTransactionsApi = () => {
    fetch.mockImplementation(url => {
      if (url.includes('/api/transactions')) {
        return Promise.resolve(
          createMockResponse({
            transactions: mockTransactions,
            total: mockTransactions.length,
          })
        );
      }

      if (url.match(/\/api\/transactions\/tx\d+/)) {
        const txId = url.split('/').pop();
        const tx = mockTransactions.find(t => t.id === txId) || mockTransactions[0];
        return Promise.resolve(createMockResponse({ transaction: tx }));
      }

      return Promise.resolve(createMockError(404, 'Not Found'));
    });
  };

  // 模拟地址相关API
  const mockAddressesApi = () => {
    fetch.mockImplementation(url => {
      if (url.includes('/api/addresses')) {
        return Promise.resolve(
          createMockResponse({
            addresses: mockAddresses,
            total: mockAddresses.length,
          })
        );
      }

      if (url.match(/\/api\/addresses\/0x[a-fA-F0-9]+/)) {
        const address = url.split('/').pop();
        const addr = mockAddresses.find(a => a.address === address) || mockAddresses[0];
        return Promise.resolve(createMockResponse({ address: addr }));
      }

      return Promise.resolve(createMockError(404, 'Not Found'));
    });
  };

  // 模拟风险警报API
  const mockAlertsApi = () => {
    fetch.mockImplementation(url => {
      if (url.includes('/api/alerts')) {
        return Promise.resolve(
          createMockResponse({
            alerts: mockRiskAlerts,
            total: mockRiskAlerts.length,
          })
        );
      }

      return Promise.resolve(createMockError(404, 'Not Found'));
    });
  };

  return {
    mockTransactionsApi,
    mockAddressesApi,
    mockAlertsApi,
    reset: () => {
      fetch.mockClear();
    },
  };
};

// 导出用于测试的通用函数
export const testUtils = {
  // 等待元素出现
  waitForElement: async (getElement, maxAttempts = 10, interval = 100) => {
    let attempts = 0;
    let element;

    while (attempts < maxAttempts) {
      try {
        element = getElement();
        if (element) return element;
      } catch (error) {
        // 忽略错误，元素可能尚未出现
      }

      await delay(interval);
      attempts++;
    }

    throw new Error('Element not found within timeout');
  },
};

export default {
  createMockResponse,
  createMockError,
  delay,
  mockTransactions,
  mockAddresses,
  mockRiskAlerts,
  setupMockApi,
  testUtils,
};
