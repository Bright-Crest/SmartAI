import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import TransactionList, { fetchTransactions } from '../../components/TransactionList';
import userEvent from '@testing-library/user-event';
import { mockTransactions, setupMockApi, createMockResponse } from '../mocks/mockService';

// 设置模拟API
const mockApi = setupMockApi();

// 在每个测试前重置模拟
beforeEach(() => {
  global.fetch = jest.fn();
  jest.clearAllMocks();
});

describe('TransactionList组件', () => {
  // 测试1：正确渲染交易数据
  test('正确渲染交易数据', async () => {
    // 使用模拟服务提供的数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: mockTransactions,
        total: mockTransactions.length
      }))
    );

    render(<TransactionList />);

    // 等待加载完成，检查发送方地址（格式化后）
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
      expect(screen.getByText('1.5 ETH')).toBeInTheDocument();
    });
  });

  // 测试2：测试加载状态
  test('显示加载状态', () => {
    // 模拟延迟加载
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 1000))
    );

    render(<TransactionList />);
    
    // 检查加载状态元素是否存在
    const loadingElement = screen.getByRole('progressbar');
    expect(loadingElement).toBeInTheDocument();
    expect(loadingElement).toHaveAttribute('aria-busy', 'true');
  });

  // 测试3：测试错误状态
  test('显示错误状态和重试按钮', async () => {
    // 模拟加载失败
    jest.spyOn(global, 'fetch').mockRejectedValue(new Error('加载失败'));

    render(<TransactionList />);
    
    // 等待错误状态显示
    await waitFor(() => {
      expect(screen.getByText('加载交易数据失败')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /重试/ })).toBeInTheDocument();
    });
  });

  // 测试4：搜索交易
  test('搜索交易', async () => {
    // 使用模拟服务提供的数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: mockTransactions,
        total: mockTransactions.length
      }))
    );

    render(<TransactionList />);

    // 等待加载完成
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
    });

    // 搜索交易
    const searchInput = screen.getByPlaceholderText('搜索交易');
    fireEvent.change(searchInput, { target: { value: '0x742d' } });
    fireEvent.click(screen.getByRole('button', { name: '搜索按钮' }));

    // 验证搜索结果
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
      expect(screen.queryByText('0x5B38...ddC4')).not.toBeInTheDocument();
    });
  });

  // 测试5：筛选交易状态
  test('筛选交易状态', async () => {
    // 使用模拟服务提供的数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: mockTransactions,
        total: mockTransactions.length
      }))
    );

    render(<TransactionList />);

    // 等待加载完成
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
    });

    // 筛选待处理状态
    fireEvent.click(screen.getByText('处理中'));

    // 验证只显示待处理交易
    await waitFor(() => {
      expect(screen.queryByText('0x742d...f44e')).not.toBeInTheDocument(); // confirmed状态，不应显示
      expect(screen.getByText('0x5B38...ddC4')).toBeInTheDocument(); // pending状态，应显示
    });
  });

  // 边缘情况测试
  test('当没有交易数据时应显示占位符', async () => {
    // 模拟空数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: [],
        total: 0
      }))
    );

    render(<TransactionList />);
    
    await waitFor(() => {
      expect(screen.getByText('没有找到交易数据')).toBeInTheDocument();
    });
  });

  test('搜索结果为空时应显示未找到结果提示', async () => {
    // 使用模拟服务提供的数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: mockTransactions,
        total: mockTransactions.length
      }))
    );

    render(<TransactionList />);
    
    // 等待加载完成
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
    });
    
    // 搜索不存在的交易
    const searchInput = screen.getByPlaceholderText('搜索交易');
    fireEvent.change(searchInput, { target: { value: '不存在的交易' } });
    fireEvent.click(screen.getByRole('button', { name: '搜索按钮' }));
    
    await waitFor(() => {
      expect(screen.getByText('没有找到匹配的交易')).toBeInTheDocument();
    });
  });

  // 测试无障碍支持
  test('交易行应支持键盘导航', async () => {
    // 使用模拟服务提供的数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: mockTransactions,
        total: mockTransactions.length
      }))
    );

    render(<TransactionList />);
    
    // 等待数据加载完成
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
    });

    // 检查交易行是否可以获取焦点
    const rows = document.querySelectorAll('[role="row"]');
    expect(rows.length).toBeGreaterThan(1); // 表头 + 至少一行数据
    
    // 可聚焦的元素应该有tabIndex属性
    const transactionRows = Array.from(rows).filter(row => row.getAttribute('tabIndex') === '0');
    expect(transactionRows.length).toBeGreaterThan(0);
  });

  test('应包含正确的ARIA属性', async () => {
    // 使用模拟服务提供的数据
    jest.spyOn(global, 'fetch').mockImplementation(() => 
      Promise.resolve(createMockResponse({ 
        transactions: mockTransactions,
        total: mockTransactions.length
      }))
    );

    render(<TransactionList />);
    
    // 等待数据加载完成
    await waitFor(() => {
      expect(screen.getByText('0x742d...f44e')).toBeInTheDocument();
    });

    // 检查表格是否有正确的ARIA角色
    expect(screen.getByRole('table')).toHaveAttribute('aria-label', '交易列表');
    
    // 检查单元格是否有正确的角色
    const cells = screen.getAllByRole('cell');
    expect(cells.length).toBeGreaterThan(0);
    
    // 检查状态是否有描述性标签
    const statusDescriptions = document.querySelectorAll('[aria-label^="状态:"]');
    expect(statusDescriptions.length).toBeGreaterThan(0);
  });
}); 