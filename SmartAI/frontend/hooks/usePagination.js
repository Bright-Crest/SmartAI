import { useState, useEffect, useMemo } from 'react';

/**
 * 分页 hook，用于处理列表分页功能
 * @param {Array} data 需要分页的数据数组
 * @param {number} itemsPerPage 每页显示的项目数，默认为10
 * @returns {Object} 分页状态和函数
 */
export default function usePagination(data = [], itemsPerPage = 10) {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(itemsPerPage);

  // 计算总页数
  const totalPages = useMemo(() => {
    return data ? Math.ceil(data.length / pageSize) : 0;
  }, [data, pageSize]);

  // 当数据变化时，重置为第一页
  useEffect(() => {
    setCurrentPage(1);
  }, [data]);

  // 获取当前页的数据
  const currentData = useMemo(() => {
    if (!data) return [];
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return data.slice(startIndex, endIndex);
  }, [data, currentPage, pageSize]);

  // 跳转到下一页
  const nextPage = () => {
    setCurrentPage(prev => (prev < totalPages ? prev + 1 : prev));
  };

  // 跳转到上一页
  const prevPage = () => {
    setCurrentPage(prev => (prev > 1 ? prev - 1 : prev));
  };

  // 跳转到指定页
  const goToPage = pageNumber => {
    const page = Math.max(1, Math.min(pageNumber, totalPages));
    setCurrentPage(page);
  };

  // 改变每页显示数量
  const changePageSize = size => {
    setPageSize(size);
    setCurrentPage(1); // 重置到第一页
  };

  return {
    currentPage,
    totalPages,
    pageSize,
    currentData,
    nextPage,
    prevPage,
    goToPage,
    changePageSize,
  };
}
