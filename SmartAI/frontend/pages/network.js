import React, { useState, useEffect, useCallback } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, Spinner } from 'react-bootstrap';
import dynamic from 'next/dynamic';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { generateMockNetworkData } from '../utils/mockData';
import NetworkControls from '../components/network/NetworkControls';
import NetworkStats from '../components/network/NetworkStats';
import SearchBar from '../components/common/SearchBar';
import FilterControls from '../components/network/FilterControls';
import styles from '../styles/Network.module.css';
import ProfessionalMetricsDashboard from '../components/dashboard/ProfessionalMetricsDashboard';
import TradingStrategyAnalysis from '../components/dashboard/TradingStrategyAnalysis';
import TransactionStatistics from '../components/dashboard/TransactionStatistics';

// 使用dynamic import加载D3NetworkGraph组件，避免SSR问题
const D3NetworkGraph = dynamic(
  () => {
    console.log('开始加载D3NetworkGraph组件...');
    return import('../components/network/D3NetworkGraph')
      .then(mod => {
        console.log('D3NetworkGraph组件已加载成功');
        return mod;
      })
      .catch(err => {
        console.error('D3NetworkGraph组件加载失败:', err);
        // 返回一个fallback组件
        return () => (
          <div className="d3-load-error p-5 text-center">
            <Alert variant="danger">
              <p className="mb-3">网络图组件加载失败，请刷新页面重试。</p>
              <Button variant="outline-danger" onClick={() => window.location.reload()}>
                刷新页面
              </Button>
            </Alert>
          </div>
        );
      });
  },
  {
    loading: () => (
      <div className="text-center p-5">
        <Spinner animation="border" role="status" variant="primary" />
        <p className="mt-3">加载网络分析组件中...</p>
      </div>
    ),
    ssr: false, // 禁用服务器端渲染
  }
);

// 力导向布局描述
const layoutDescription = {
  title: '力导向布局',
  description: '节点间相互排斥，连接像弹簧一样拉近相关节点。适合查看整体网络结构和社区发现。',
  example: '寻找高度互联的账户群组，发现中心化交易所或集群',
};

/**
 * 网络分析页面
 */
export default function NetworkPage() {
  const router = useRouter();
  const [networkData, setNetworkData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [d3Status, setD3Status] = useState({ loaded: false });
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRiskLevels, setSelectedRiskLevels] = useState({
    high: true,
    medium: true,
    low: true,
  });
  const [selectedNodeTypes, setSelectedNodeTypes] = useState({
    address: true,
    contract: true,
    exchange: true,
    mixer: true,
  });
  const [selectedNode, setSelectedNode] = useState(null);
  const [d3LoadError, setD3LoadError] = useState(false);

  // 获取网络数据
  useEffect(() => {
    const fetchNetworkData = async () => {
      try {
        setLoading(true);
        setError(null);

        // 使用模拟数据
        const data = generateMockNetworkData(50, 3);

        // 应用过滤器
        const filteredData = filterNetworkData(data);

        setNetworkData(filteredData);
      } catch (err) {
        console.error('获取网络数据出错:', err);
        setError('加载网络数据失败，请重试');
      } finally {
        setLoading(false);
      }
    };

    fetchNetworkData();
  }, []);

  // 根据筛选条件过滤网络数据
  const filterNetworkData = useCallback(
    data => {
      if (!data || !data.nodes || !data.links) {
        return { nodes: [], links: [] };
      }

      // 根据风险级别和节点类型过滤节点
      const filteredNodes = data.nodes.filter(node => {
        // 风险级别筛选
        const riskLevel = node.riskScore > 70 ? 'high' : node.riskScore > 40 ? 'medium' : 'low';
        const passesRiskFilter = selectedRiskLevels[riskLevel];

        // 节点类型筛选
        const passesTypeFilter = selectedNodeTypes[node.type];

        // 搜索筛选
        const passesSearchFilter =
          !searchTerm ||
          node.id.includes(searchTerm) ||
          (node.label && node.label.includes(searchTerm)) ||
          (node.address && node.address.includes(searchTerm));

        return passesRiskFilter && passesTypeFilter && passesSearchFilter;
      });

      // 只保留与筛选后节点相关的连接
      const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
      const filteredLinks = data.links.filter(
        link =>
          filteredNodeIds.has(link.source.id || link.source) &&
          filteredNodeIds.has(link.target.id || link.target)
      );

      return { nodes: filteredNodes, links: filteredLinks };
    },
    [searchTerm, selectedRiskLevels, selectedNodeTypes]
  );

  // 处理风险级别变更
  const handleRiskLevelChange = (level, checked) => {
    setSelectedRiskLevels(prev => ({ ...prev, [level]: checked }));
    // 重新过滤数据
    if (networkData) {
      setNetworkData(filterNetworkData(networkData));
    }
  };

  // 处理节点类型变更
  const handleNodeTypeChange = (type, checked) => {
    setSelectedNodeTypes(prev => ({ ...prev, [type]: checked }));
    // 重新过滤数据
    if (networkData) {
      setNetworkData(filterNetworkData(networkData));
    }
  };

  // 处理刷新数据
  const handleRefreshData = () => {
    setLoading(true);
    setError(null);

    // 生成新的模拟数据
    setTimeout(() => {
      try {
        const newData = generateMockNetworkData(50, 3);
        setNetworkData(filterNetworkData(newData));
        setLoading(false);
      } catch (err) {
        console.error('生成数据出错:', err);
        setError('生成网络数据失败，请重试');
        setLoading(false);
      }
    }, 500);
  };

  // 处理搜索
  const handleSearch = term => {
    setSearchTerm(term);
    if (networkData) {
      setNetworkData(filterNetworkData(networkData));
    }
  };

  // 处理节点点击
  const handleNodeClick = useCallback(node => {
    console.log('节点点击:', node);
    setSelectedNode(node);
  }, []);

  // 处理节点双击
  const handleNodeDoubleClick = useCallback(
    node => {
      console.log('节点双击:', node);
      // 可以导航到节点详情页或展开节点
      router.push(`/address/${node.address || node.id}`);
    },
    [router]
  );

  // 处理连接点击
  const handleLinkClick = useCallback(link => {
    console.log('连接点击:', link);
    // 可以显示连接详情
  }, []);

  // 处理D3错误
  const handleD3Error = useCallback(err => {
    console.error('D3渲染错误:', err);
    setError('网络图渲染失败，请刷新页面重试。');
  }, []);

  // 处理D3状态更新
  const handleD3Status = useCallback(status => {
    console.log('D3状态更新:', status);
    setD3Status(prev => ({ ...prev, ...status }));
  }, []);

  // 处理图形渲染完成
  const handleGraphRendered = useCallback(renderInfo => {
    console.log('网络图渲染完成:', renderInfo);
    // 可以在这里记录或处理渲染统计信息
  }, []);

  return (
    <>
      <Head>
        <title>网络分析 | ChainIntel AI</title>
      </Head>
      <Container fluid className="py-4">
        <Row className="mb-4">
          <Col>
            <h1 className="h3 mb-2">区块链网络分析</h1>
            <p className="text-muted">可视化区块链交易和地址关系，探索网络结构，发现异常模式</p>
          </Col>
          <Col xs="auto">
            <Button variant="outline-primary" onClick={handleRefreshData} disabled={loading}>
              {loading ? (
                <>
                  <Spinner
                    as="span"
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                    className="me-2"
                  />
                  加载中...
                </>
              ) : (
                '刷新数据'
              )}
            </Button>
          </Col>
        </Row>

        <Row>
          {/* 左侧控制面板 */}
          <Col lg={3} className="mb-4">
            <Card className="mb-4">
              <Card.Header>搜索与过滤</Card.Header>
              <Card.Body>
                <SearchBar
                  placeholder="搜索地址、交易、标签..."
                  onSearch={handleSearch}
                  className="mb-4"
                />

                <FilterControls
                  riskLevels={selectedRiskLevels}
                  nodeTypes={selectedNodeTypes}
                  onRiskLevelChange={handleRiskLevelChange}
                  onNodeTypeChange={handleNodeTypeChange}
                />
              </Card.Body>
            </Card>

            {/* 布局信息 */}
            <Card className="mb-4">
              <Card.Header>布局信息</Card.Header>
              <Card.Body className="p-3">
                <h6>{layoutDescription.title}</h6>
                <p className="mb-2">{layoutDescription.description}</p>
                <small className="text-muted">
                  <strong>示例用途:</strong> {layoutDescription.example}
                </small>
              </Card.Body>
            </Card>

            {/* 节点详情 */}
            {selectedNode && (
              <Card>
                <Card.Header>节点详情</Card.Header>
                <Card.Body>
                  <h6>{selectedNode.type} 节点</h6>
                  <p className="mb-1">
                    <strong>地址:</strong>{' '}
                    <small className="text-break">{selectedNode.address}</small>
                  </p>
                  <p className="mb-1">
                    <strong>风险评分:</strong>{' '}
                    <span
                      className={
                        selectedNode.riskScore > 70
                          ? 'text-danger'
                          : selectedNode.riskScore > 40
                            ? 'text-warning'
                            : 'text-success'
                      }
                    >
                      {selectedNode.riskScore}
                    </span>
                  </p>
                  <p className="mb-1">
                    <strong>连接数:</strong> {selectedNode.connections || '计算中...'}
                  </p>
                  <div className="mt-3">
                    <Button
                      size="sm"
                      variant="outline-primary"
                      onClick={() => router.push(`/address/${selectedNode.address}`)}
                    >
                      查看详情
                    </Button>
                  </div>
                </Card.Body>
              </Card>
            )}
          </Col>

          {/* 主要网络图区域 */}
          <Col lg={9}>
            <Card className="mb-4 network-card">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <span>网络图 - {layoutDescription.title}</span>
                {!loading && !error && networkData && <NetworkControls />}
              </Card.Header>
              <Card.Body className={styles.networkContainer}>
                {loading ? (
                  <div className="text-center p-5">
                    <Spinner animation="border" role="status" variant="primary" />
                    <p className="mt-3">加载网络数据中...</p>
                  </div>
                ) : error ? (
                  <Alert variant="danger">
                    {error}
                    <Button
                      variant="outline-danger"
                      size="sm"
                      className="mt-2"
                      onClick={handleRefreshData}
                    >
                      重试
                    </Button>
                  </Alert>
                ) : d3LoadError ? (
                  <Alert variant="warning">
                    <p>网络图组件加载失败。可能是由于D3.js库未能正确加载。</p>
                    <Button
                      variant="primary"
                      onClick={() => window.location.reload()}
                      className="mt-2"
                    >
                      刷新页面
                    </Button>
                  </Alert>
                ) : networkData ? (
                  <D3NetworkGraph
                    data={networkData}
                    height={600}
                    onNodeClick={handleNodeClick}
                    onNodeDoubleClick={handleNodeDoubleClick}
                    onLinkClick={handleLinkClick}
                    onError={handleD3Error}
                    onD3Status={handleD3Status}
                    onGraphRendered={handleGraphRendered}
                    className={styles.networkGraph}
                  />
                ) : (
                  <div className="text-center p-5">
                    <p>无网络数据可显示</p>
                    <Button variant="primary" onClick={handleRefreshData}>
                      生成示例数据
                    </Button>
                  </div>
                )}
              </Card.Body>
            </Card>

            {/* 网络统计信息 */}
            {!loading && !error && networkData && (
              <Card>
                <Card.Header>网络统计</Card.Header>
                <Card.Body>
                  <NetworkStats data={networkData} />
                </Card.Body>
              </Card>
            )}
          </Col>
        </Row>

        <Row className="mt-4">
          <Col lg={12}>
            <Card>
              <Card.Header>专业指标分析</Card.Header>
              <Card.Body>
                <ProfessionalMetricsDashboard data={professionalMetricsData} />
                <TradingStrategyAnalysis data={tradingStrategyData} />
                <TransactionStatistics data={transactionStatisticsData} />
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
    </>
  );
}
