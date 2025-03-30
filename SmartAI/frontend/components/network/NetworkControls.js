import React, { useState } from 'react';
import Card from '../common/Card';
import Button from '../common/Button';
import PropTypes from 'prop-types';

/**
 * 网络图控制面板组件
 * 提供缩放控制和过滤选项
 *
 * @param {Object} props - 组件属性
 * @param {string} props.searchTerm - 搜索词
 * @param {function} props.setSearchTerm - 设置搜索词的函数
 * @param {string} props.selectedRisk - 当前选中的风险等级
 * @param {function} props.setSelectedRisk - 设置风险等级的函数
 * @param {function} props.resetFilters - 重置所有过滤器的函数
 * @param {number} props.zoom - 当前缩放级别
 * @param {Function} props.onZoomChange - 缩放级别变化处理函数
 * @param {Object} props.filterOptions - 过滤选项
 * @param {Function} props.onFilterChange - 过滤选项变化处理函数
 */
const NetworkControls = ({
  searchTerm,
  setSearchTerm,
  selectedRisk,
  setSelectedRisk,
  resetFilters,
  zoom = 1,
  onZoomChange,
  filterOptions = {},
  onFilterChange,
}) => {
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);

  // 风险等级选项
  const riskOptions = [
    { value: '', label: '全部' },
    { value: 'high', label: '高风险' },
    { value: 'medium', label: '中等风险' },
    { value: 'low', label: '低风险' },
  ];

  // 节点类型选项
  const nodeTypeOptions = [
    { value: 'address', label: '普通地址' },
    { value: 'contract', label: '合约' },
    { value: 'exchange', label: '交易所' },
    { value: 'mixer', label: '混币器' },
    { value: 'high_risk', label: '高风险地址' },
  ];

  // 处理节点类型过滤变化
  const handleNodeTypeChange = (type, checked) => {
    if (!onFilterChange) return;

    const nodeTypes = [...(filterOptions.nodeTypes || [])];

    if (checked && !nodeTypes.includes(type)) {
      nodeTypes.push(type);
    } else if (!checked && nodeTypes.includes(type)) {
      const index = nodeTypes.indexOf(type);
      nodeTypes.splice(index, 1);
    }

    onFilterChange({ nodeTypes });
  };

  return (
    <Card>
      <h3 className="text-lg font-medium mb-4">网络图控制</h3>

      <div className="space-y-4">
        {/* 搜索过滤 */}
        <div>
          <label htmlFor="search" className="block text-sm font-medium text-gray-700">
            搜索地址或标签
          </label>
          <div className="mt-1">
            <input
              type="text"
              id="search"
              className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md"
              placeholder="输入地址或标签"
              value={searchTerm || ''}
              onChange={e => setSearchTerm && setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* 风险等级过滤 */}
        <div>
          <label className="block text-sm font-medium text-gray-700">风险等级</label>
          <div className="mt-1 flex flex-wrap gap-2">
            {riskOptions.map(option => (
              <button
                key={option.value}
                className={`px-3 py-1 text-sm rounded-md ${
                  selectedRisk === option.value
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setSelectedRisk && setSelectedRisk(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* 缩放控制 */}
        {onZoomChange && (
          <div>
            <label className="block text-sm font-medium text-gray-700">
              缩放级别: {zoom.toFixed(1)}
            </label>
            <div className="mt-1 flex items-center space-x-2">
              <button
                className="p-1 rounded-md bg-gray-100 hover:bg-gray-200"
                onClick={() => onZoomChange(Math.max(0.5, zoom - 0.1))}
                aria-label="缩小"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
              </button>

              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={zoom}
                onChange={e => onZoomChange(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />

              <button
                className="p-1 rounded-md bg-gray-100 hover:bg-gray-200"
                onClick={() => onZoomChange(Math.min(2, zoom + 0.1))}
                aria-label="放大"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <line x1="12" y1="5" x2="12" y2="19"></line>
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* 高级过滤器切换 */}
        {onFilterChange && (
          <div>
            <button
              className="text-sm text-blue-500 hover:underline flex items-center"
              onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
            >
              {showAdvancedFilters ? '隐藏高级过滤器' : '显示高级过滤器'}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className={`ml-1 h-4 w-4 transform ${showAdvancedFilters ? 'rotate-180' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>

            {/* 高级过滤选项 */}
            {showAdvancedFilters && (
              <div className="mt-4 p-3 bg-gray-50 rounded-md space-y-4">
                {/* 节点类型过滤 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700">节点类型</label>
                  <div className="mt-1 grid grid-cols-2 gap-2">
                    {nodeTypeOptions.map(option => (
                      <label key={option.value} className="flex items-center text-sm">
                        <input
                          type="checkbox"
                          className="mr-1.5 h-4 w-4 text-blue-500 rounded"
                          checked={filterOptions.nodeTypes?.includes(option.value) || false}
                          onChange={e => handleNodeTypeChange(option.value, e.target.checked)}
                        />
                        {option.label}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 操作按钮 */}
        {resetFilters && (
          <div className="pt-2">
            <Button color="secondary" variant="outline" size="sm" onClick={resetFilters}>
              重置所有过滤器
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
};

NetworkControls.propTypes = {
  searchTerm: PropTypes.string,
  setSearchTerm: PropTypes.func,
  selectedRisk: PropTypes.string,
  setSelectedRisk: PropTypes.func,
  resetFilters: PropTypes.func,
  zoom: PropTypes.number,
  onZoomChange: PropTypes.func,
  filterOptions: PropTypes.object,
  onFilterChange: PropTypes.func,
};

export default NetworkControls;
