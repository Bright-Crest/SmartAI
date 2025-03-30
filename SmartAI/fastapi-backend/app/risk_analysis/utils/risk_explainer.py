"""
风险解释器模块
提供基于规则的风险解释功能
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskExplainer:
    """
    基于规则的风险解释器
    """
    
    def __init__(self):
        """初始化风险解释器"""
        # 风险级别的阈值和描述
        self.risk_levels = {
            "low": {
                "threshold": 30,
                "description": "低风险",
                "explanation": "该地址表现为低风险，没有明显的可疑行为特征。"
            },
            "medium": {
                "threshold": 60,
                "description": "中等风险",
                "explanation": "该地址表现为中等风险，存在一些需要关注的行为模式。"
            },
            "high": {
                "threshold": 80,
                "description": "高风险",
                "explanation": "该地址表现为高风险，存在多项可疑行为特征。"
            },
            "very_high": {
                "threshold": 100,
                "description": "极高风险",
                "explanation": "该地址表现为极高风险，强烈建议避免与其交互。"
            }
        }
        
        # 风险因素规则
        self.risk_rules = [
            {
                "id": "high_value_transfers",
                "name": "大额转账",
                "description": "频繁进行大额转账",
                "threshold": 5.0,  # ETH
                "feature": "max_transaction_value",
                "risk_contribution": 10,
                "explanation": "该地址有大额转账行为，最大单笔交易金额超过{value} ETH。"
            },
            {
                "id": "high_interaction_frequency",
                "name": "高频交互",
                "description": "与大量不同地址交互",
                "threshold": 10,
                "feature": "unique_interaction_addresses",
                "risk_contribution": 5,
                "explanation": "该地址与超过{value}个不同地址有交互，可能是中继地址。"
            },
            {
                "id": "new_address",
                "name": "新建地址",
                "description": "地址创建时间较短",
                "threshold": 30,
                "feature": "days_since_first_tx",
                "risk_contribution": 5,
                "comparison": "less",
                "explanation": "该地址创建时间不足{value}天，属于较新的地址。"
            },
            {
                "id": "high_outgoing_ratio",
                "name": "高出账比例",
                "description": "出账交易比例高",
                "threshold": 0.75,
                "feature": "outgoing_tx_ratio",
                "risk_contribution": 10,
                "explanation": "该地址出账交易比例高达{value:.0%}，存在资金快速流出特征。"
            },
            {
                "id": "high_risk_interactions",
                "name": "高风险交互",
                "description": "与已知高风险地址交互",
                "threshold": 0,
                "feature": "high_risk_interaction_count",
                "risk_contribution": 20,
                "comparison": "greater",
                "explanation": "该地址与{value}个已知的高风险地址有交互记录。"
            },
            {
                "id": "token_diversity",
                "name": "代币多样性",
                "description": "持有多种不同代币",
                "threshold": 10,
                "feature": "token_count",
                "risk_contribution": 5,
                "explanation": "该地址持有{value}种不同代币，代币多样性较高。"
            }
        ]
    
    def get_risk_level(self, risk_score: float) -> Dict[str, Any]:
        """
        获取风险等级及其描述
        
        参数:
            risk_score: 风险评分(0-100)
            
        返回:
            风险等级信息
        """
        if risk_score < self.risk_levels["low"]["threshold"]:
            level = "low"
        elif risk_score < self.risk_levels["medium"]["threshold"]:
            level = "medium"
        elif risk_score < self.risk_levels["high"]["threshold"]:
            level = "high"
        else:
            level = "very_high"
        
        level_info = self.risk_levels[level]
        
        return {
            "level": level,
            "description": level_info["description"],
            "explanation": level_info["explanation"],
            "score": risk_score
        }
    
    def explain_risk(self, features: Dict[str, Any], risk_score: float) -> Dict[str, Any]:
        """
        解释风险评分
        
        参数:
            features: 特征字典
            risk_score: 风险评分
            
        返回:
            风险解释
        """
        risk_level = self.get_risk_level(risk_score)
        risk_factors = []
        
        # 应用风险规则
        for rule in self.risk_rules:
            feature = rule["feature"]
            if feature in features:
                feature_value = features[feature]
                threshold = rule["threshold"]
                comparison = rule.get("comparison", "greater")
                
                is_triggered = False
                if comparison == "greater":
                    is_triggered = feature_value > threshold
                elif comparison == "less":
                    is_triggered = feature_value < threshold
                else:
                    is_triggered = feature_value == threshold
                
                if is_triggered:
                    explanation = rule["explanation"].format(value=feature_value)
                    risk_factors.append({
                        "id": rule["id"],
                        "name": rule["name"],
                        "description": rule["description"],
                        "feature_value": feature_value,
                        "threshold": threshold,
                        "risk_contribution": rule["risk_contribution"],
                        "explanation": explanation
                    })
        
        # 按风险贡献排序
        risk_factors.sort(key=lambda x: x["risk_contribution"], reverse=True)
        
        # 生成综合解释
        factors_explanation = ""
        if risk_factors:
            factors_explanation = "主要风险因素包括："
            for i, factor in enumerate(risk_factors[:3]):  # 取前3个主要因素
                factors_explanation += f"{i+1}. {factor['explanation']} "
        
        # 构建完整解释
        complete_explanation = risk_level["explanation"]
        if factors_explanation:
            complete_explanation += " " + factors_explanation
        
        # 关注因素
        attention_points = []
        high_risk_triggers = [f for f in risk_factors if f["risk_contribution"] >= 10]
        if high_risk_triggers:
            for factor in high_risk_triggers:
                attention_points.append(factor["explanation"])
        
        # 构建返回结果
        result = {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "complete_explanation": complete_explanation,
            "attention_points": attention_points
        }
        
        return result 