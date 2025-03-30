import React from 'react';
import PropTypes from 'prop-types';

/**
 * 网络图过滤控制组件
 * 提供筛选节点类型和风险等级的功能
 *
 * @param {Object} props - 组件属性
 * @param {Object} props.riskLevels - 选中的风险等级对象，格式为 {high: true, medium: true, low: true}
 * @param {Object} props.nodeTypes - 选中的节点类型对象，格式为 {address: true, contract: true, ...}
 * @param {Function} props.onRiskLevelChange - 风险等级选择变更处理函数
 * @param {Function} props.onNodeTypeChange - 节点类型选择变更处理函数
 * @returns {JSX.Element} 过滤控制组件
 */
const FilterControls = ({ riskLevels, nodeTypes, onRiskLevelChange, onNodeTypeChange }) => {
  return (
    <div>
      <div className="mb-4">
        <h6 className="text-sm font-medium mb-2">风险等级</h6>
        <div className="space-y-2">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="risk-high"
              className="mr-2"
              checked={riskLevels.high}
              onChange={e => onRiskLevelChange('high', e.target.checked)}
            />
            <label htmlFor="risk-high" className="text-sm text-red-600">
              高风险
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              id="risk-medium"
              className="mr-2"
              checked={riskLevels.medium}
              onChange={e => onRiskLevelChange('medium', e.target.checked)}
            />
            <label htmlFor="risk-medium" className="text-sm text-yellow-600">
              中风险
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              id="risk-low"
              className="mr-2"
              checked={riskLevels.low}
              onChange={e => onRiskLevelChange('low', e.target.checked)}
            />
            <label htmlFor="risk-low" className="text-sm text-green-600">
              低风险
            </label>
          </div>
        </div>
      </div>

      <div>
        <h6 className="text-sm font-medium mb-2">节点类型</h6>
        <div className="space-y-2">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="type-address"
              className="mr-2"
              checked={nodeTypes.address}
              onChange={e => onNodeTypeChange('address', e.target.checked)}
            />
            <label htmlFor="type-address" className="text-sm">
              地址
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              id="type-contract"
              className="mr-2"
              checked={nodeTypes.contract}
              onChange={e => onNodeTypeChange('contract', e.target.checked)}
            />
            <label htmlFor="type-contract" className="text-sm">
              合约
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              id="type-exchange"
              className="mr-2"
              checked={nodeTypes.exchange}
              onChange={e => onNodeTypeChange('exchange', e.target.checked)}
            />
            <label htmlFor="type-exchange" className="text-sm">
              交易所
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              id="type-mixer"
              className="mr-2"
              checked={nodeTypes.mixer}
              onChange={e => onNodeTypeChange('mixer', e.target.checked)}
            />
            <label htmlFor="type-mixer" className="text-sm">
              混币服务
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

FilterControls.propTypes = {
  riskLevels: PropTypes.shape({
    high: PropTypes.bool,
    medium: PropTypes.bool,
    low: PropTypes.bool,
  }).isRequired,
  nodeTypes: PropTypes.shape({
    address: PropTypes.bool,
    contract: PropTypes.bool,
    exchange: PropTypes.bool,
    mixer: PropTypes.bool,
  }).isRequired,
  onRiskLevelChange: PropTypes.func.isRequired,
  onNodeTypeChange: PropTypes.func.isRequired,
};

export default FilterControls;
