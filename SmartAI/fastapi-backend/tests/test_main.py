import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_read_root():
    """测试根路径响应是否正确"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()
    assert "docs" in response.json()


def test_health_check():
    """测试健康检查端点是否正确响应"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "services" in response.json()


def test_open_api_docs():
    """测试OpenAPI文档是否可访问"""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_open_api_json():
    """测试OpenAPI JSON是否可访问"""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    assert "application/json" in response.headers["content-type"]
    assert "openapi" in response.json()
    assert "paths" in response.json()
    assert "components" in response.json() 