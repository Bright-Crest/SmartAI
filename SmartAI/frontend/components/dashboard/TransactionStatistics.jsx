import React from 'react';
import { Card, Row, Col, Table, Badge } from 'react-bootstrap';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TransactionStatistics = ({ data }) => {
  const {
    totalVolume,
    averageTradeSize,
    largestTrade,
    averageExecutionTime,
    averageSlippage,
    preferredTradingTime,
    preferredDex
  } = data;

  // 生成交易量分布图数据
  const volumeData = {
    labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
    datasets: [
      {
        label: '交易量',
        data: [120000, 150000, 180000, 200000, 220000, 250000],
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }
    ]
  };

  const volumeOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: '月度交易量分布'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            return '$' + value.toLocaleString();
          }
        }
      }
    }
  };

  // 生成交易时间分布图数据
  const timeData = {
    labels: ['亚洲时段', '欧洲时段', '美洲时段'],
    datasets: [
      {
        label: '交易频率',
        data: [45, 30, 25],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)'
        ],
        borderWidth: 1
      }
    ]
  };

  const timeOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: '交易时间分布'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value) {
            return value + '%';
          }
        }
      }
    }
  };

  // 生成DEX使用统计
  const dexUsage = {
    'Uniswap': 45,
    'SushiSwap': 30,
    'PancakeSwap': 25
  };

  return (
    <div className="transaction-statistics">
      <h2 className="mb-4">交易统计分析</h2>

      <Row>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>交易量分析</Card.Title>
              <Bar data={volumeData} options={volumeOptions} />
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>交易时间分布</Card.Title>
              <Bar data={timeData} options={timeOptions} />
            </Card.Body>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>关键指标</Card.Title>
              <Table striped bordered hover>
                <tbody>
                  <tr>
                    <td>总交易量</td>
                    <td>${totalVolume.toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td>平均交易规模</td>
                    <td>${averageTradeSize.toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td>最大单笔交易</td>
                    <td>${largestTrade.toLocaleString()}</td>
                  </tr>
                  <tr>
                    <td>平均执行时间</td>
                    <td>{averageExecutionTime}秒</td>
                  </tr>
                  <tr>
                    <td>平均滑点</td>
                    <td>{(averageSlippage * 100).toFixed(3)}%</td>
                  </tr>
                </tbody>
              </Table>
            </Card.Body>
          </Card>
        </Col>

        <Col md={6}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>交易偏好</Card.Title>
              <div className="mb-3">
                <h6>偏好交易时段</h6>
                <Badge bg="primary" className="me-2">{preferredTradingTime}</Badge>
              </div>
              <div>
                <h6>偏好DEX</h6>
                <div className="d-flex flex-wrap gap-2">
                  {Object.entries(dexUsage).map(([dex, percentage]) => (
                    <Badge key={dex} bg="info" className="me-2">
                      {dex}: {percentage}%
                    </Badge>
                  ))}
                </div>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default TransactionStatistics; 