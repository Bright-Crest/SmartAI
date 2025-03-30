/**
 * 网络控制组件 - 提供搜索、过滤和刷新功能
 */
import React, { useCallback, useState } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import Card from './Card';
import Button from './Button';
import { RISK_LEVELS, RISK_COLORS } from '../utils/constants';

// 样式组件
const StyledSearchInput = styled.input`
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  width: 100%;
  margin-bottom: 10px;
  font-size: 14px;
  box-sizing: border-box;

  &:focus {
    outline: none;
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
  }
`;

const StyledSelect = styled.select`
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  width: 100%;
  margin-bottom: 10px;
  background-color: white;
  font-size: 14px;
  box-sizing: border-box;

  &:focus {
    outline: none;
    border-color: #3498db;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
  }
`;

const ButtonContainer = styled.div`
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-top: 5px;
`;

const StyledButton = styled.button`
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: ${props => (props.primary ? '#3498db' : 'white')};
  color: ${props => (props.primary ? 'white' : '#333')};
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
  flex: 1;

  &:hover {
    background-color: ${props => (props.primary ? '#2980b9' : '#f5f5f5')};
  }

  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
  }
`;

const layoutOptions = [
  {
    id: 'force',
    name: '力导向布局',
    icon: '🔄',
    description: '节点位置根据物理引力模型动态计算，适合展示复杂网络结构',
  },
  {
    id: 'radial',
    name: '径向布局',
    icon: '⭕',
    description: '节点围绕中心点均匀分布，适合强调中心与周边关系',
  },
  {
    id: 'grid',
    name: '网格布局',
    icon: '🔲',
    description: '节点整齐排列在网格中，适合清晰展示大量节点',
  },
  {
    id: 'circular',
    name: '环形布局',
    icon: '🔄',
    description: '根据连接数量将节点排列在不同环上，适合展示连接密度',
  },
  {
    id: 'concentric',
    name: '同心圆布局',
    icon: '🎯',
    description: '根据风险等级或重要性将节点排列在不同同心圆上',
  },
  {
    id: 'hierarchical',
    name: '分层布局',
    icon: '⬇️',
    description: '垂直排列显示节点间的层级关系，适合展示上下级关系',
  },
];

/**
 * 网络控制组件
 *
 * 提供搜索、过滤和刷新功能
 */
const NetworkControls = ({
  searchTerm,
  selectedRiskLevel,
  onSearch,
  onRiskFilterChange,
  onResetFilters,
  onRefresh,
  onLayoutChange,
  selectedLayout,
  loading,
  stats,
}) => {
  const [localSearchTerm, setLocalSearchTerm] = useState(searchTerm || '');

  // 提交搜索
  const handleSearchSubmit = useCallback(
    e => {
      e.preventDefault();
      onSearch(localSearchTerm);
    },
    [localSearchTerm, onSearch]
  );

  return (
    <Card className="mb-4">
      <div className="p-4">
        <h3 className="text-lg font-medium mb-4">网络控制</h3>

        {/* 搜索表单 */}
        <form onSubmit={handleSearchSubmit} className="mb-4">
          <div className="flex">
            <input
              type="text"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-l focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              placeholder="搜索地址、标签或交易"
              value={localSearchTerm}
              onChange={e => setLocalSearchTerm(e.target.value)}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-500 text-white rounded-r hover:bg-blue-600 focus:outline-none focus:ring-1 focus:ring-blue-500"
              disabled={loading}
            >
              搜索
            </button>
          </div>
        </form>

        {/* 风险等级过滤 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">风险等级过滤</label>
          <div className="flex space-x-2">
            {['all', 'high', 'medium', 'low'].map(level => (
              <button
                key={level}
                className={`px-3 py-1 text-sm rounded transition ${
                  selectedRiskLevel === level
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                }`}
                onClick={() => onRiskFilterChange(level)}
              >
                {level === 'all'
                  ? '全部'
                  : level === 'high'
                    ? '高风险'
                    : level === 'medium'
                      ? '中风险'
                      : '低风险'}
              </button>
            ))}
          </div>
        </div>

        {/* 布局选择 */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">布局选择</label>
          <div className="grid grid-cols-2 gap-2">
            {layoutOptions.map(layout => (
              <button
                key={layout.id}
                className={`px-3 py-2 text-sm rounded transition flex items-center justify-start ${
                  selectedLayout === layout.id
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                }`}
                onClick={() => onLayoutChange(layout.id)}
              >
                <span className="mr-1">{layout.icon}</span>
                <span>{layout.name}</span>
              </button>
            ))}
          </div>
          {selectedLayout && (
            <div className="mt-2 text-xs text-gray-600">
              {layoutOptions.find(l => l.id === selectedLayout)?.description}
            </div>
          )}
        </div>

        {/* 操作按钮 */}
        <div className="flex space-x-2">
          <button
            className="flex-1 px-3 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 focus:outline-none"
            onClick={onResetFilters}
            disabled={loading}
          >
            重置筛选
          </button>
          <button
            className="flex-1 px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 focus:outline-none"
            onClick={onRefresh}
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                刷新中
              </span>
            ) : (
              '刷新数据'
            )}
          </button>
        </div>

        {/* 网络概览 */}
        {stats && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <h4 className="text-sm font-medium mb-2">网络概览</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-gray-50 p-2 rounded flex justify-between">
                <span>节点数</span>
                <span className="font-medium">{stats.totalNodes || 0}</span>
              </div>
              <div className="bg-gray-50 p-2 rounded flex justify-between">
                <span>连接数</span>
                <span className="font-medium">{stats.totalLinks || 0}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

NetworkControls.propTypes = {
  searchTerm: PropTypes.string,
  selectedRiskLevel: PropTypes.string,
  selectedLayout: PropTypes.string,
  onSearch: PropTypes.func.isRequired,
  onRiskFilterChange: PropTypes.func.isRequired,
  onResetFilters: PropTypes.func.isRequired,
  onRefresh: PropTypes.func.isRequired,
  onLayoutChange: PropTypes.func.isRequired,
  loading: PropTypes.bool,
  stats: PropTypes.object,
};

export default NetworkControls;
