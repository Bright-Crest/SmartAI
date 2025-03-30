'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';
import { MagnifyingGlassIcon } from '@heroicons/react/24/solid';

export default function GlobalSearch() {
  const { t } = useTranslation();
  const [query, setQuery] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputId = 'global-search-input';
  const searchResultsId = 'global-search-results';
  const searchLabelId = 'global-search-label';
  const searchDescId = 'global-search-description';

  // 处理搜索
  const handleSearch = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;
    
    // 根据输入内容判断是地址还是交易哈希
    if (query.startsWith('0x') && query.length >= 40) {
      // 如果以0x开头且长度大于等于40，认为是地址
      router.push(`/addresses?address=${query}`);
    } else {
      // 否则认为是交易哈希
      router.push(`/transactions?hash=${query}`);
    }
    
    // 搜索后收起搜索框
    setIsExpanded(false);
    setQuery('');
    // 将焦点返回到搜索按钮
    setTimeout(() => {
      buttonRef.current?.focus();
    }, 100);
  };

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsExpanded(false);
      buttonRef.current?.focus();
      e.preventDefault();
    }
  };

  // 处理搜索按钮点击
  const handleSearchButtonClick = () => {
    if (isExpanded) {
      handleSearch();
    } else {
      setIsExpanded(true);
    }
  };

  // 点击外部区域关闭搜索框
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsExpanded(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // 当展开时自动聚焦输入框
  useEffect(() => {
    if (isExpanded && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isExpanded]);

  const searchLabel = t('globalSearch.label') || '搜索';
  const searchPlaceholder = t('globalSearch.placeholder') || '搜索地址或交易哈希...';
  const searchButtonLabel = t('globalSearch.searchButton') || '执行搜索';

  return (
    <div className="relative" ref={containerRef}>
      {isExpanded ? (
        <form 
          onSubmit={handleSearch} 
          className="absolute right-0 top-0 flex items-center" 
          role="search" 
          aria-labelledby={searchLabelId}
          onKeyDown={handleKeyDown}
        >
          <label 
            id={searchLabelId} 
            htmlFor={searchInputId} 
            className="sr-only"
          >
            {searchLabel}
          </label>
          <div className="relative">
            <input
              id={searchInputId}
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={searchPlaceholder}
              className="w-64 px-4 py-2 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              autoFocus
              autoComplete="off"
              aria-autocomplete="list"
              aria-controls={searchResultsId}
              aria-expanded={isExpanded}
              aria-describedby={searchDescId}
              ref={inputRef}
            />
            <span id={searchDescId} className="sr-only">
              输入地址或交易哈希后按回车键进行搜索，或使用Escape键关闭搜索框
            </span>
          </div>
          <button
            type="submit"
            ref={buttonRef}
            className="bg-blue-700 hover:bg-blue-800 text-white px-3 py-2 rounded-r-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus-visible:ring-2 focus-visible:ring-offset-2"
            aria-label={searchButtonLabel}
          >
            <MagnifyingGlassIcon className="h-5 w-5" />
          </button>
        </form>
      ) : (
        <button
          onClick={handleSearchButtonClick}
          ref={buttonRef}
          className="text-gray-700 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus-visible:ring-2 focus-visible:ring-offset-2 rounded-full p-1"
          aria-label={searchLabel}
          aria-expanded={isExpanded}
          aria-haspopup="true"
          aria-controls={searchInputId}
        >
          <MagnifyingGlassIcon className="h-6 w-6" />
        </button>
      )}
      {/* 添加一个隐藏的搜索结果区域，用于屏幕阅读器 */}
      <div id={searchResultsId} className="sr-only" aria-live="polite">
        {query ? `正在搜索 ${query}` : ''}
      </div>
    </div>
  );
} 