# 区块链风险分析服务

本服务提供区块链地址的风险评分、用户画像分类和风险解释功能，基于 XGBoost 和 K-Means 机器学习模型实现。

## 功能特点

- **风险评分**：使用 XGBoost 模型评估地址风险程度，返回 0-100 的评分和风险等级
- **用户画像**：基于 K-Means 聚类算法，将地址分类为不同用户画像
- **风险解释**：提供基于规则的风险解释，指出风险因素和需要关注的点
- **特征提取**：自动从区块链 API 提取相关特征用于分析

## 技术架构

- **特征工程**：从区块链 API 获取数据，提取风险相关特征
- **风险评分**：使用 XGBoost 实现二分类风险评分模型
- **用户聚类**：使用 K-Means 实现用户画像分类
- **风险解释**：基于规则引擎实现风险解释
- **API 服务**：使用 FastAPI 提供 REST API 接口

## 安装与使用

### 依赖安装

```bash
# 安装依赖包
pip install xgboost scikit-learn pandas numpy joblib fastapi uvicorn requests
```

### 训练模型

```bash
# 训练风险评分和用户聚类模型
python run_risk_analysis.py --train
```

### 启动服务

```bash
# 启动风险分析API服务
python run_risk_analysis.py --serve

# 或指定端口
python run_risk_analysis.py --serve --port 8002
```

### 直接运行 API 服务

```bash
# 直接运行API服务
python risk_analysis_api.py
```

## API 接口

服务启动后，可以通过以下接口访问风险分析功能：

### 1. 获取风险评分

```
GET /risk/score/{address}
```

示例：

```bash
curl http://localhost:8002/risk/score/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

响应：

```json
{
  "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
  "risk_score": 35.5,
  "risk_level": "medium",
  "risk_description": "中等风险",
  "risk_explanation": "该地址表现为中等风险，存在一些需要关注的行为模式。主要风险因素包括：1. 该地址有大额转账行为，最大单笔交易金额超过2.0 ETH。 2. 该地址出账交易比例高达60%，存在资金快速流出特征。",
  "risk_factors": [...],
  "attention_points": [...],
  "features": {...}
}
```

### 2. 获取用户画像

```
GET /risk/profile/{address}
```

示例：

```bash
curl http://localhost:8002/risk/profile/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

响应：

```json
{
  "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
  "cluster": 1,
  "cluster_name": "活跃交易用户",
  "cluster_description": "该地址表现为活跃交易用户，有较高的交易频率和多样化的交易对象，但交易金额中等。",
  "features": {...}
}
```

### 3. 获取完整分析

```
GET /risk/analyze/{address}
```

示例：

```bash
curl http://localhost:8002/risk/analyze/0x742d35Cc6634C0532925a3b844Bc454e4438f44e
```

响应：

```json
{
  "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
  "risk_analysis": {...},
  "user_profile": {...}
}
```

## 模型说明

### 风险评分模型

采用 XGBoost 二分类算法，基于以下特征评估地址风险：

- ETH 余额
- 代币数量和价值
- 交易频率和金额
- 互动地址多样性
- 合约交互情况
- 高风险地址交互次数

### 用户聚类模型

采用 K-Means 聚类算法，将地址分为四种用户画像：

1. **低活跃小额用户**：交易频率低，持有资产少
2. **活跃交易用户**：交易频率高，交易对象多样化
3. **DeFi 重度用户**：大量合约交互，活跃于去中心化金融
4. **大额持币者**：高 ETH 余额，交易金额大

## 模块结构

```
fastapi-backend/
├── app/
│   ├── risk_analysis/
│   │   ├── __init__.py        # 初始化文件
│   │   ├── data/              # 数据处理模块
│   │   │   ├── __init__.py
│   │   │   └── feature_extractor.py  # 特征提取器
│   │   ├── models/            # 模型模块
│   │   │   ├── __init__.py
│   │   │   ├── risk_score.py  # 风险评分模型
│   │   │   └── user_clustering.py  # 用户聚类模型
│   │   ├── utils/             # 工具模块
│   │   │   ├── __init__.py
│   │   │   └── risk_explainer.py  # 风险解释器
│   │   ├── routes.py          # API路由
│   │   └── train_models.py    # 模型训练脚本
├── risk_analysis_api.py       # 风险分析API服务
└── run_risk_analysis.py       # 运行脚本
```

## 注意事项

1. 本服务依赖于区块链 API 提供的数据，确保区块链 API 服务正常运行
2. 首次使用需要训练模型，否则将使用简单的默认模型
3. 风险评分和用户画像仅供参考，不构成投资建议
4. 当前版本使用模拟数据进行训练，实际使用时应替换为真实数据
