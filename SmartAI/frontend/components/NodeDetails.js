/**
 * 节点详情面板组件 - 展示选中节点的详细信息
 */
import React from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import Card from './Card';
import { RISK_COLORS } from '../utils/constants';
import { formatCurrency, formatNumber, shortenAddress } from '../utils/formatters';

// 样式组件
const DetailCard = styled(Card)`
  margin-bottom: 20px;
`;

const DetailTitle = styled.h3`
  margin-top: 0;
  margin-bottom: 15px;
  color: #333;
`;

const DetailItem = styled.div`
  margin-bottom: 12px;

  &:last-child {
    margin-bottom: 0;
  }
`;

const DetailLabel = styled.span`
  font-weight: 600;
  display: block;
  font-size: 12px;
  color: #666;
  margin-bottom: 2px;
`;

const DetailValue = styled.span`
  display: block;
  font-size: 14px;
  word-break: break-all;
`;

const RiskBadge = styled.span`
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  color: white;
  background-color: ${props => {
    switch (props.risk) {
      case 'high':
        return '#e74c3c';
      case 'medium':
        return '#f39c12';
      case 'low':
        return '#3498db';
      default:
        return '#95a5a6';
    }
  }};
`;

const ActionButton = styled.button`
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 6px 10px;
  margin-right: 8px;
  margin-top: 5px;
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #2980b9;
  }

  &:last-child {
    margin-right: 0;
  }
`;

const ButtonsContainer = styled.div`
  margin-top: 15px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;

const CloseButton = styled.button`
  position: absolute;
  top: 15px;
  right: 15px;
  background: transparent;
  border: none;
  font-size: 18px;
  cursor: pointer;
  color: #666;

  &:hover {
    color: #333;
  }
`;

/**
 * 节点详情组件
 *
 * @param {Object} props - 组件属性
 * @param {Object|null} props.node - 节点数据对象
 * @param {Function} props.onClose - 关闭详情回调函数
 * @returns {JSX.Element|null} 节点详情组件
 */
const NodeDetails = ({ node, loading, onClose }) => {
  if (!node && !loading) return null;

  // 如果正在加载
  if (loading) {
    return (
      <Card className="mb-4">
        <div className="p-4">
          <div className="flex justify-between mb-4">
            <h3 className="text-lg font-medium">节点详情</h3>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
              <span className="sr-only">关闭</span>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </div>
          <div className="flex justify-center items-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
            <span className="ml-2">加载中...</span>
          </div>
        </div>
      </Card>
    );
  }

  // 如果有节点数据
  const riskColor = RISK_COLORS[node.risk] || 'gray-500';

  return (
    <Card className="mb-4">
      <div className="p-4">
        <div className="flex justify-between mb-4">
          <h3 className="text-lg font-medium">节点详情</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <span className="sr-only">关闭</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        {/* 节点基本信息 */}
        <div className="mb-4 pb-4 border-b border-gray-200">
          <div className="flex items-center mb-2">
            <h4 className="text-xl font-semibold">{node.label || 'Unknown Node'}</h4>
            {node.risk && (
              <span className={`ml-2 px-2 py-1 text-xs rounded-full bg-${riskColor} text-white`}>
                {node.risk.toUpperCase()}
              </span>
            )}
          </div>

          {node.address && (
            <div className="flex items-center text-sm mb-1">
              <span className="font-medium mr-2">地址:</span>
              <span className="text-gray-600 font-mono">
                {shortenAddress(node.address)}
                <button
                  onClick={() => navigator.clipboard.writeText(node.address)}
                  className="ml-2 text-blue-500 hover:text-blue-700"
                  title="复制地址"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4 inline"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
                    <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
                  </svg>
                </button>
              </span>
            </div>
          )}

          {node.type && (
            <div className="text-sm mb-1">
              <span className="font-medium mr-2">类型:</span>
              <span className="text-gray-600">{node.type}</span>
            </div>
          )}

          {typeof node.value !== 'undefined' && (
            <div className="text-sm mb-1">
              <span className="font-medium mr-2">余额:</span>
              <span className="text-gray-600">{formatCurrency(node.value)}</span>
            </div>
          )}
        </div>

        {/* 交易统计 */}
        {node.transactions && (
          <div className="mb-4 pb-4 border-b border-gray-200">
            <h5 className="font-medium mb-2">交易统计</h5>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 p-2 rounded">
                <div className="text-xs text-gray-500">总交易数</div>
                <div className="font-semibold">{formatNumber(node.transactions.total)}</div>
              </div>
              <div className="bg-gray-50 p-2 rounded">
                <div className="text-xs text-gray-500">总交易额</div>
                <div className="font-semibold">{formatCurrency(node.transactions.totalValue)}</div>
              </div>
              <div className="bg-gray-50 p-2 rounded">
                <div className="text-xs text-gray-500">入账</div>
                <div className="font-semibold text-green-600">
                  {formatNumber(node.transactions.incoming)}
                </div>
              </div>
              <div className="bg-gray-50 p-2 rounded">
                <div className="text-xs text-gray-500">出账</div>
                <div className="font-semibold text-red-500">
                  {formatNumber(node.transactions.outgoing)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 连接节点信息 */}
        {node.connections && node.connections.length > 0 && (
          <div className="mb-4">
            <h5 className="font-medium mb-2">连接节点 ({node.connections.length})</h5>
            <div className="max-h-40 overflow-y-auto">
              <ul className="divide-y divide-gray-200">
                {node.connections.map((connection, index) => (
                  <li key={index} className="py-2">
                    <div className="flex items-center">
                      <div
                        className={`w-2 h-2 rounded-full bg-${RISK_COLORS[connection.risk] || 'gray-500'} mr-2`}
                      ></div>
                      <div>
                        <div className="font-medium">{connection.label || 'Unknown'}</div>
                        {connection.address && (
                          <div className="text-xs text-gray-500 font-mono">
                            {shortenAddress(connection.address)}
                          </div>
                        )}
                      </div>
                      {connection.value && (
                        <div className="ml-auto text-xs font-medium">
                          {formatCurrency(connection.value)}
                        </div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* 元数据或标签 */}
        {node.metadata && Object.keys(node.metadata).length > 0 && (
          <div>
            <h5 className="font-medium mb-2">其他信息</h5>
            <div className="text-sm">
              {Object.entries(node.metadata).map(([key, value]) => (
                <div key={key} className="mb-1">
                  <span className="font-medium mr-2">{key}:</span>
                  <span className="text-gray-600">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

NodeDetails.propTypes = {
  node: PropTypes.shape({
    id: PropTypes.string.isRequired,
    label: PropTypes.string,
    address: PropTypes.string,
    type: PropTypes.string,
    risk: PropTypes.string,
    value: PropTypes.number,
    metadata: PropTypes.object,
  }),
  onClose: PropTypes.func,
};

export default NodeDetails;
