import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.network import NetworkRequest, NetworkResponse

client = TestClient(app)


def test_analyze_network_valid_request():
    """测试有效的网络分析请求"""
    payload = {
        "addresses": ["0x742d35Cc6634C0532925a3b844Bc454e4438f44e"],
        "depth": 1,
        "max_nodes": 10,
        "include_exchanges": True,
        "filter_by_type": None
    }
    
    response = client.post("/api/v1/network/analyze", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "nodes" in data
    assert "links" in data
    assert "statistics" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["links"], list)
    assert isinstance(data["statistics"], dict)


def test_analyze_network_invalid_depth():
    """测试深度超出范围的网络分析请求"""
    payload = {
        "addresses": ["0x742d35Cc6634C0532925a3b844Bc454e4438f44e"],
        "depth": 5,  # 超出最大深度限制
        "max_nodes": 10,
        "include_exchanges": True,
        "filter_by_type": None
    }
    
    response = client.post("/api/v1/network/analyze", json=payload)
    assert response.status_code == 422  # 验证请求


def test_analyze_network_invalid_address():
    """测试无效地址的网络分析请求"""
    payload = {
        "addresses": ["invalid_address"],
        "depth": 1,
        "max_nodes": 10,
        "include_exchanges": True,
        "filter_by_type": None
    }
    
    response = client.post("/api/v1/network/analyze", json=payload)
    assert response.status_code == 422  # 验证请求


def test_get_network_stats():
    """测试获取网络统计数据"""
    address = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    response = client.get(f"/api/v1/network/stats/{address}")
    assert response.status_code == 200
    
    data = response.json()
    assert "total_transactions" in data
    assert "unique_interactions" in data
    assert "high_risk_interactions" in data
    assert isinstance(data["total_transactions"], int)
    assert isinstance(data["unique_interactions"], int)
 