import React from 'react';
import { Card, Row, Col, ProgressBar } from 'react-bootstrap';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const ProfessionalMetricsDashboard = ({ data }) => {
  const {
    alpha,
    beta,
    informationRatio,
    sortinoRatio,
    calmarRatio,
    omegaRatio,
    valueAtRisk,
    expectedShortfall
  } = data;

  // 计算指标状态
  const getMetricStatus = (value, type) => {
    switch (type) {
      case 'alpha':
        return value > 0 ? 'success' : 'danger';
      case 'beta':
        return value > 1 ? 'warning' : 'info';
      case 'informationRatio':
        return value > 1 ? 'success' : 'warning';
      case 'sortinoRatio':
        return value > 1.5 ? 'success' : 'warning';
      case 'calmarRatio':
        return value > 1 ? 'success' : 'warning';
      case 'omegaRatio':
        return value > 1.2 ? 'success' : 'warning';
      case 'valueAtRisk':
        return value < 0.05 ? 'success' : 'warning';
      case 'expectedShortfall':
        return value < 0.08 ? 'success' : 'warning';
      default:
        return 'secondary';
    }
  };

  // 生成指标卡片
  const MetricCard = ({ title, value, type, description }) => (
    <Card className="mb-4">
      <Card.Body>
        <Card.Title>{title}</Card.Title>
        <Card.Text className={`h3 text-${getMetricStatus(value, type)}`}>
          {value.toFixed(2)}
        </Card.Text>
        <Card.Text className="text-muted small">{description}</Card.Text>
        <ProgressBar
          variant={getMetricStatus(value, type)}
          now={Math.min(Math.abs(value) * 100, 100)}
          className="mt-2"
        />
      </Card.Body>
    </Card>
  );

  // 生成历史表现图表数据
  const chartData = {
    labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
    datasets: [
      {
        label: '投资组合收益',
        data: [0, 0.05, 0.12, 0.18, 0.25, 0.32],
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: '基准收益',
        data: [0, 0.03, 0.08, 0.15, 0.20, 0.28],
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: '历史表现对比'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            return (value * 100).toFixed(1) + '%';
          }
        }
      }
    }
  };

  return (
    <div className="professional-metrics-dashboard">
      <h2 className="mb-4">专业投资指标分析</h2>
      
      <Row>
        <Col md={6} lg={3}>
          <MetricCard
            title="Alpha (超额收益)"
            value={alpha}
            type="alpha"
            description="相对于基准的超额收益能力"
          />
        </Col>
        <Col md={6} lg={3}>
          <MetricCard
            title="Beta (市场敏感度)"
            value={beta}
            type="beta"
            description="相对于市场的波动性"
          />
        </Col>
        <Col md={6} lg={3}>
          <MetricCard
            title="信息比率"
            value={informationRatio}
            type="informationRatio"
            description="单位跟踪误差下的超额收益"
          />
        </Col>
        <Col md={6} lg={3}>
          <MetricCard
            title="索提诺比率"
            value={sortinoRatio}
            type="sortinoRatio"
            description="下行风险调整后的收益"
          />
        </Col>
      </Row>

      <Row>
        <Col md={6} lg={3}>
          <MetricCard
            title="卡玛比率"
            value={calmarRatio}
            type="calmarRatio"
            description="最大回撤调整后的年化收益"
          />
        </Col>
        <Col md={6} lg={3}>
          <MetricCard
            title="欧米伽比率"
            value={omegaRatio}
            type="omegaRatio"
            description="收益与损失的概率比"
          />
        </Col>
        <Col md={6} lg={3}>
          <MetricCard
            title="风险价值 (VaR)"
            value={valueAtRisk}
            type="valueAtRisk"
            description="在给定置信水平下的最大可能损失"
          />
        </Col>
        <Col md={6} lg={3}>
          <MetricCard
            title="预期损失"
            value={expectedShortfall}
            type="expectedShortfall"
            description="超过VaR的损失期望值"
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col>
          <Card>
            <Card.Body>
              <Card.Title>历史表现对比</Card.Title>
              <Line data={chartData} options={chartOptions} />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ProfessionalMetricsDashboard; 