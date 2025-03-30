/**
 * 网络统计组件 - 显示网络数据统计信息
 */
import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';
import Card from './Card';
import { formatNumber, formatDateTime } from '../utils/formatters';

// 样式组件
const StatsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const StatsTitle = styled.h3`
  font-size: 16px;
  margin: 0 0 10px 0;
  color: #333;
`;

const StatItem = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #eee;
  &:last-child {
    border-bottom: none;
  }
`;

const StatLabel = styled.span`
  font-size: 14px;
  color: #666;
`;

const StatValue = styled.span`
  font-size: 14px;
  font-weight: 500;
  color: #333;
`;

const RiskDistribution = styled.div`
  margin-top: 10px;
`;

const RiskBar = styled.div`
  height: 20px;
  width: 100%;
  background-color: #eee;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  margin-top: 5px;
`;

const RiskSegment = styled.div`
  height: 100%;
  background-color: ${props => props.color || '#ccc'};
  width: ${props => props.percentage || 0}%;
  transition: width 0.3s ease;
`;

const RiskLegend = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 5px;
  font-size: 12px;
  color: #666;
`;

/**
 * 网络统计组件
 *
 * 显示网络图的统计信息
 */
export default function NetworkStats({ data, loading, lastUpdated }) {
  // 如果没有数据且不在加载状态，则不显示
  if (!data && !loading) return null;

  return (
    <Card className="mb-4">
      <div className="p-4">
        <h3 className="text-lg font-medium mb-4">网络统计信息</h3>

        {loading ? (
          <div className="py-4 text-center text-gray-500">
            <div className="animate-pulse">加载中...</div>
          </div>
        ) : !data ? (
          <div className="py-4 text-center text-gray-500">没有可用的统计数据</div>
        ) : (
          <>
            <div className="space-y-4">
              {/* 基本统计信息 */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-gray-50 rounded-md text-center">
                  <div className="text-sm text-gray-500">总节点数</div>
                  <div className="text-xl font-semibold">{formatNumber(data.totalNodes || 0)}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-md text-center">
                  <div className="text-sm text-gray-500">总连接数</div>
                  <div className="text-xl font-semibold">{formatNumber(data.totalLinks || 0)}</div>
                </div>
              </div>

              {/* 风险分布信息 */}
              {data.riskDistribution && (
                <div>
                  <h4 className="text-sm font-medium mb-2">风险分布</h4>
                  <div className="flex h-4 rounded-full overflow-hidden mb-2">
                    {data.riskDistribution.high > 0 && (
                      <div
                        className="bg-red-500"
                        style={{
                          width: `${(data.riskDistribution.high / data.totalNodes) * 100}%`,
                        }}
                      />
                    )}
                    {data.riskDistribution.medium > 0 && (
                      <div
                        className="bg-yellow-500"
                        style={{
                          width: `${(data.riskDistribution.medium / data.totalNodes) * 100}%`,
                        }}
                      />
                    )}
                    {data.riskDistribution.low > 0 && (
                      <div
                        className="bg-green-500"
                        style={{ width: `${(data.riskDistribution.low / data.totalNodes) * 100}%` }}
                      />
                    )}
                    {data.riskDistribution.unknown > 0 && (
                      <div
                        className="bg-gray-300"
                        style={{
                          width: `${(data.riskDistribution.unknown / data.totalNodes) * 100}%`,
                        }}
                      />
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-red-500 rounded-full mr-2" />
                      <span>高风险: {data.riskDistribution.high || 0}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2" />
                      <span>中风险: {data.riskDistribution.medium || 0}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-green-500 rounded-full mr-2" />
                      <span>低风险: {data.riskDistribution.low || 0}</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-gray-300 rounded-full mr-2" />
                      <span>未知: {data.riskDistribution.unknown || 0}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* 网络结构信息 */}
              {data.networkStructure && (
                <div>
                  <h4 className="text-sm font-medium mb-2">网络结构</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span>平均连接度</span>
                      <span className="font-medium">
                        {data.networkStructure.averageDegree?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span>平均路径长度</span>
                      <span className="font-medium">
                        {data.networkStructure.averagePathLength?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span>聚类系数</span>
                      <span className="font-medium">
                        {data.networkStructure.clusteringCoefficient?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span>社区数量</span>
                      <span className="font-medium">
                        {data.networkStructure.communities || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* 最后更新时间 */}
            {lastUpdated && (
              <div className="mt-4 text-right text-xs text-gray-500">
                最后更新: {formatDateTime(lastUpdated)}
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
}

NetworkStats.propTypes = {
  data: PropTypes.shape({
    nodes: PropTypes.arrayOf(PropTypes.object),
    links: PropTypes.arrayOf(PropTypes.object),
    totalNodes: PropTypes.number,
    totalLinks: PropTypes.number,
    riskDistribution: PropTypes.shape({
      high: PropTypes.number,
      medium: PropTypes.number,
      low: PropTypes.number,
      unknown: PropTypes.number,
    }),
    networkStructure: PropTypes.shape({
      averageDegree: PropTypes.number,
      averagePathLength: PropTypes.number,
      clusteringCoefficient: PropTypes.number,
      communities: PropTypes.number,
    }),
  }),
  loading: PropTypes.bool,
  lastUpdated: PropTypes.string,
};
