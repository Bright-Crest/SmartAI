import React from 'react';
import { Card, Row, Col, Badge, ProgressBar } from 'react-bootstrap';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

const TradingStrategyAnalysis = ({ data }) => {
  const {
    strategyType,
    entryTiming,
    exitTiming,
    positionSizing,
    riskRewardRatio,
    marketRegime,
    preferredMarketCap
  } = data;

  // 生成雷达图数据
  const radarData = {
    labels: ['入场时机', '出场时机', '仓位管理', '风险收益比', '市场适应性'],
    datasets: [
      {
        label: '策略表现',
        data: [
          entryTiming * 100,
          exitTiming * 100,
          positionSizing * 100,
          (riskRewardRatio / 3) * 100,
          80
        ],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2
      }
    ]
  };

  const radarOptions = {
    scales: {
      r: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 20
        }
      }
    }
  };

  // 生成市场状态标签
  const getMarketRegimeBadge = (regime) => {
    const variants = {
      '牛市': 'success',
      '熊市': 'danger',
      '震荡市': 'warning'
    };
    return <Badge bg={variants[regime]}>{regime}</Badge>;
  };

  // 生成市场市值标签
  const getMarketCapBadge = (cap) => {
    const variants = {
      '大市值': 'primary',
      '中市值': 'info',
      '小市值': 'secondary'
    };
    return <Badge bg={variants[cap]}>{cap}</Badge>;
  };

  return (
    <div className="trading-strategy-analysis">
      <h2 className="mb-4">交易策略分析</h2>

      <Row>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>策略概览</Card.Title>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span>策略类型：</span>
                <Badge bg="primary">{strategyType}</Badge>
              </div>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <span>市场状态：</span>
                {getMarketRegimeBadge(marketRegime)}
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span>偏好市值：</span>
                {getMarketCapBadge(preferredMarketCap)}
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>风险收益分析</Card.Title>
              <div className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span>风险收益比</span>
                  <span>{riskRewardRatio.toFixed(2)}</span>
                </div>
                <ProgressBar
                  variant="info"
                  now={(riskRewardRatio / 3) * 100}
                  className="mb-3"
                />
                <div className="d-flex justify-content-between mb-1">
                  <span>仓位管理</span>
                  <span>{(positionSizing * 100).toFixed(1)}%</span>
                </div>
                <ProgressBar
                  variant="success"
                  now={positionSizing * 100}
                />
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>交易执行分析</Card.Title>
              <div className="mb-3">
                <div className="d-flex justify-content-between mb-1">
                  <span>入场时机</span>
                  <span>{(entryTiming * 100).toFixed(1)}%</span>
                </div>
                <ProgressBar
                  variant="primary"
                  now={entryTiming * 100}
                  className="mb-3"
                />
                <div className="d-flex justify-content-between mb-1">
                  <span>出场时机</span>
                  <span>{(exitTiming * 100).toFixed(1)}%</span>
                </div>
                <ProgressBar
                  variant="warning"
                  now={exitTiming * 100}
                />
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>策略雷达图</Card.Title>
              <Radar data={radarData} options={radarOptions} />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TradingStrategyAnalysis; 