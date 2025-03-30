import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import GlobalSearch from '../../components/GlobalSearch';

// 模拟next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

// 模拟react-i18next
jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: { [key: string]: string } = {
        'globalSearch.placeholder': '搜索地址或交易哈希...',
        'globalSearch.label': '搜索'
      };
      return translations[key] || key;
    },
  }),
}));

describe('GlobalSearch组件', () => {
  // 渲染测试
  test('初始状态下渲染搜索图标', () => {
    render(<GlobalSearch />);
    
    // 验证初始状态是否渲染了搜索按钮
    const searchButton = screen.getByRole('button', { name: '搜索' });
    expect(searchButton).toBeInTheDocument();
    
    // 验证搜索输入框还未显示
    const searchInput = screen.queryByPlaceholderText('搜索地址或交易哈希...');
    expect(searchInput).not.toBeInTheDocument();
  });
  
  // 交互测试
  test('点击搜索图标展开搜索框', () => {
    render(<GlobalSearch />);
    
    // 点击搜索按钮
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 验证搜索框是否展开
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    expect(searchInput).toBeInTheDocument();
  });
  
  test('输入并提交搜索查询', () => {
    const { useRouter } = require('next/navigation');
    const pushMock = jest.fn();
    useRouter.mockImplementation(() => ({
      push: pushMock,
    }));
    
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 输入地址
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    fireEvent.change(searchInput, { target: { value: '0x1234567890abcdef1234567890abcdef12345678' } });
    
    // 点击搜索按钮提交查询
    const submitButton = screen.getByRole('button');
    fireEvent.click(submitButton);
    
    // 验证路由是否被正确调用
    expect(pushMock).toHaveBeenCalledWith('/addresses?address=0x1234567890abcdef1234567890abcdef12345678');
  });
  
  test('处理交易哈希搜索', () => {
    const { useRouter } = require('next/navigation');
    const pushMock = jest.fn();
    useRouter.mockImplementation(() => ({
      push: pushMock,
    }));
    
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 输入交易哈希
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    fireEvent.change(searchInput, { target: { value: '0x123' } });
    
    // 点击搜索按钮提交查询
    const submitButton = screen.getByRole('button');
    fireEvent.click(submitButton);
    
    // 验证路由是否被正确调用
    expect(pushMock).toHaveBeenCalledWith('/transactions?hash=0x123');
  });
  
  // 键盘交互测试
  test('按下Enter键提交搜索', () => {
    const { useRouter } = require('next/navigation');
    const pushMock = jest.fn();
    useRouter.mockImplementation(() => ({
      push: pushMock,
    }));
    
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 输入地址
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    fireEvent.change(searchInput, { target: { value: '0x1234567890abcdef1234567890abcdef12345678' } });
    
    // 按下Enter键
    fireEvent.keyDown(searchInput, { key: 'Enter', code: 'Enter' });
    
    // 验证路由是否被正确调用
    expect(pushMock).toHaveBeenCalledWith('/addresses?address=0x1234567890abcdef1234567890abcdef12345678');
  });
  
  test('按下Escape键关闭搜索框', () => {
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 验证搜索框已展开
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    expect(searchInput).toBeInTheDocument();
    
    // 按下Escape键
    fireEvent.keyDown(searchInput, { key: 'Escape', code: 'Escape' });
    
    // 验证搜索框已关闭
    expect(screen.queryByPlaceholderText('搜索地址或交易哈希...')).not.toBeInTheDocument();
  });
  
  // 边缘情况测试
  test('不提交空查询', () => {
    const { useRouter } = require('next/navigation');
    const pushMock = jest.fn();
    useRouter.mockImplementation(() => ({
      push: pushMock,
    }));
    
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 不输入任何内容，直接提交
    const submitButton = screen.getByRole('button');
    fireEvent.click(submitButton);
    
    // 验证路由没有被调用
    expect(pushMock).not.toHaveBeenCalled();
  });
  
  test('搜索后重置查询状态', async () => {
    const { useRouter } = require('next/navigation');
    const pushMock = jest.fn();
    useRouter.mockImplementation(() => ({
      push: pushMock,
    }));
    
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 输入地址
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    fireEvent.change(searchInput, { target: { value: '0x1234567890abcdef1234567890abcdef12345678' } });
    
    // 点击搜索按钮提交查询
    const submitButton = screen.getByRole('button');
    fireEvent.click(submitButton);
    
    // 验证搜索框已关闭
    await waitFor(() => {
      expect(screen.queryByPlaceholderText('搜索地址或交易哈希...')).not.toBeInTheDocument();
    });
    
    // 再次点击打开搜索框
    fireEvent.click(screen.getByRole('button'));
    
    // 验证输入框已重置
    const newSearchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    expect(newSearchInput).toHaveValue('');
  });
  
  // 无障碍测试
  test('搜索按钮有适当的aria标签', () => {
    render(<GlobalSearch />);
    
    const searchButton = screen.getByRole('button');
    expect(searchButton).toHaveAttribute('aria-label', '搜索');
  });
  
  test('搜索框获得焦点', () => {
    render(<GlobalSearch />);
    
    // 点击搜索按钮展开搜索框
    const searchButton = screen.getByRole('button');
    fireEvent.click(searchButton);
    
    // 验证搜索框获得了焦点
    const searchInput = screen.getByPlaceholderText('搜索地址或交易哈希...');
    expect(document.activeElement).toBe(searchInput);
  });
}); 