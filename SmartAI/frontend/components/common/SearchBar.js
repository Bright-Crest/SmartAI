import React, { useState } from 'react';
import PropTypes from 'prop-types';

/**
 * 搜索栏组件
 *
 * @param {Object} props - 组件属性
 * @param {string} props.placeholder - 占位符文本
 * @param {Function} props.onSearch - 搜索回调函数
 * @param {string} props.className - 额外的CSS类名
 * @returns {JSX.Element} 搜索栏组件
 */
const SearchBar = ({ placeholder = '搜索...', onSearch, className = '' }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSubmit = e => {
    e.preventDefault();
    if (onSearch && searchTerm.trim()) {
      onSearch(searchTerm.trim());
    }
  };

  return (
    <form className={`relative ${className}`} onSubmit={handleSubmit}>
      <input
        type="text"
        className="w-full py-2 px-4 rounded-md border border-gray-300 focus:border-blue-500 focus:ring-blue-500 focus:outline-none"
        placeholder={placeholder}
        value={searchTerm}
        onChange={e => setSearchTerm(e.target.value)}
      />
      <button
        type="submit"
        className="absolute inset-y-0 right-0 px-3 flex items-center text-gray-500 hover:text-blue-500"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
      </button>
    </form>
  );
};

SearchBar.propTypes = {
  placeholder: PropTypes.string,
  onSearch: PropTypes.func.isRequired,
  className: PropTypes.string,
};

export default SearchBar;
