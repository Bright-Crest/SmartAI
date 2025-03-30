import React, { useState, useEffect } from 'react';
import { Form, Button, Card, Alert } from 'react-bootstrap';

/**
 * API测试组件
 * 用于测试不同后端服务API集成
 */
const ApiTester = () => {
  const [address, setAddress] = useState('0x123456789012345678901234567890123456789a');
  const [apiType, setApiType] = useState('smartScore');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const apiTypes = [
    { value: 'smartScore', label: '智能投资评分' },
    { value: 'metrics', label: '投资行为指标' },
    { value: 'details', label: '地址详情' },
    { value: 'transactions', label: '交易历史' },
    { value: 'similar', label: '相似地址' },
    { value: 'batch', label: '批量评分' }
  ];

  const runTest = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      let result;
      const baseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

      switch (apiType) {
        case 'smartScore':
          result = await fetch(`${baseUrl}/api/smart-money/score/${address}`);
          break;
        case 'metrics':
          result = await fetch(`${baseUrl}/api/smart-money/metrics/${address}`);
          break;
        case 'details':
          result = await fetch(`${baseUrl}/api/smart-money/details/${address}`);
          break;
        case 'transactions':
          result = await fetch(`${baseUrl}/api/smart-money/transactions/${address}`);
          break;
        case 'similar':
          result = await fetch(`${baseUrl}/api/smart-money/similar/${address}`);
          break;
        case 'batch':
          result = await fetch(`${baseUrl}/api/smart-money/batch/scores`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              addresses: [address, '0x9876543210987654321098765432109876543210']
            })
          });
          break;
        default:
          throw new Error('未知的API类型');
      }

      const data = await result.json();
      setResponse(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="mb-4">
      <Card.Header>
        <h5 className="mb-0">API调试工具</h5>
      </Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>测试地址</Form.Label>
            <Form.Control
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="输入以太坊地址"
            />
          </Form.Group>

          <Form.Group className="mb-3">
            <Form.Label>API类型</Form.Label>
            <Form.Select
              value={apiType}
              onChange={(e) => setApiType(e.target.value)}
            >
              {apiTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </Form.Select>
          </Form.Group>

          <Button
            variant="primary"
            onClick={runTest}
            disabled={loading}
          >
            {loading ? '测试中...' : '运行测试'}
          </Button>
        </Form>

        {error && (
          <Alert variant="danger" className="mt-3">
            {error}
          </Alert>
        )}

        {response && (
          <div className="mt-3">
            <h6>响应结果：</h6>
            <pre className="bg-light p-3 rounded">
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default ApiTester; 