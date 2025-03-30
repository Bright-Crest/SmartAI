import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import {
  FaHome,
  FaChartBar,
  FaWallet,
  FaExchangeAlt,
  FaNetworkWired,
  FaExclamationTriangle,
  FaUserAlt,
  FaBell,
  FaSearch,
  FaSun,
  FaMoon,
  FaBars,
  FaTimes,
} from 'react-icons/fa';

// 导航链接
const navLinks = [
  { href: '/', label: '首页', icon: <FaHome /> },
  { href: '/dashboard', label: '仪表盘', icon: <FaChartBar /> },
  { href: '/addresses', label: '地址查询', icon: <FaWallet /> },
  { href: '/transactions', label: '交易分析', icon: <FaExchangeAlt /> },
  { href: '/network', label: '关系网络', icon: <FaNetworkWired /> },
  { href: '/blacklist', label: '黑名单', icon: <FaExclamationTriangle /> },
  { href: '/kol', label: 'KOL追踪', icon: <FaUserAlt /> },
  { href: '/alerts', label: '预警消息', icon: <FaBell /> },
];

/**
 * 主导航栏组件
 *
 * @returns {JSX.Element} 导航栏组件
 */
export default function Navbar() {
  const router = useRouter();
  const [darkMode, setDarkMode] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // 初始化暗黑模式
  useEffect(() => {
    // 检查用户之前的主题偏好
    const isDarkMode =
      localStorage.getItem('darkMode') === 'true' ||
      (!('darkMode' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);

    setDarkMode(isDarkMode);
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  // 切换暗黑模式
  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);

    // 保存用户偏好
    localStorage.setItem('darkMode', newDarkMode.toString());

    // 更新DOM
    if (newDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  // 处理搜索
  const handleSearch = e => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // 如果输入是有效的以太坊地址，则跳转到地址详情页
      if (/^0x[a-fA-F0-9]{40}$/.test(searchQuery.trim())) {
        router.push(`/address/${searchQuery.trim()}`);
      }
      // 如果输入是有效的交易哈希，则跳转到交易详情页
      else if (/^0x[a-fA-F0-9]{64}$/.test(searchQuery.trim())) {
        router.push(`/transaction/${searchQuery.trim()}`);
      }
      // 否则跳转到搜索结果页
      else {
        router.push(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
      }
      setSearchQuery('');
      setMobileMenuOpen(false);
    }
  };

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm dark:bg-gray-800 dark:border-gray-700">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo and Desktop Navigation */}
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/" className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                ChainIntelAI
              </Link>
            </div>
            <div className="hidden md:ml-6 md:flex md:space-x-1">
              {navLinks.map(link => (
                <Link
                  href={link.href}
                  key={link.href}
                  className={`inline-flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                    router.pathname === link.href
                      ? 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-300'
                      : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-blue-300 dark:hover:bg-gray-700'
                  }`}
                >
                  <span className="mr-1.5">{link.icon}</span>
                  {link.label}
                </Link>
              ))}
            </div>
          </div>

          {/* Search and Actions */}
          <div className="flex items-center">
            <div className="hidden md:block">
              <form onSubmit={handleSearch} className="relative">
                <input
                  type="text"
                  placeholder="搜索地址、交易..."
                  className="w-64 px-4 py-2 pl-10 pr-4 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                />
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <FaSearch className="text-gray-400" />
                </div>
              </form>
            </div>

            {/* Theme Toggle */}
            <button
              onClick={toggleDarkMode}
              className="ml-4 p-2 rounded-full text-gray-500 hover:text-blue-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-blue-400 dark:hover:bg-gray-700 focus:outline-none"
              aria-label={darkMode ? '切换到亮色模式' : '切换到暗色模式'}
            >
              {darkMode ? <FaSun className="h-5 w-5" /> : <FaMoon className="h-5 w-5" />}
            </button>

            {/* Mobile menu button */}
            <div className="flex items-center md:hidden">
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-blue-600 hover:bg-gray-100 dark:text-gray-400 dark:hover:text-blue-400 dark:hover:bg-gray-700 focus:outline-none"
                aria-expanded="false"
              >
                <span className="sr-only">{mobileMenuOpen ? '关闭菜单' : '打开菜单'}</span>
                {mobileMenuOpen ? (
                  <FaTimes className="block h-6 w-6" />
                ) : (
                  <FaBars className="block h-6 w-6" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div className={`md:hidden ${mobileMenuOpen ? 'block' : 'hidden'}`}>
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
          {navLinks.map(link => (
            <Link
              href={link.href}
              key={`mobile-${link.href}`}
              className={`block px-3 py-2 rounded-md text-base font-medium ${
                router.pathname === link.href
                  ? 'text-blue-600 bg-blue-50 dark:bg-blue-900 dark:text-blue-300'
                  : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-blue-300 dark:hover:bg-gray-700'
              }`}
              onClick={() => setMobileMenuOpen(false)}
            >
              <span className="inline-flex items-center">
                <span className="mr-2">{link.icon}</span>
                {link.label}
              </span>
            </Link>
          ))}

          <div className="pt-3 px-3">
            <form onSubmit={handleSearch} className="relative">
              <input
                type="text"
                placeholder="搜索地址、交易..."
                className="w-full px-4 py-2 pl-10 pr-4 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
              />
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <FaSearch className="text-gray-400" />
              </div>
            </form>
          </div>
        </div>
      </div>
    </nav>
  );
}
