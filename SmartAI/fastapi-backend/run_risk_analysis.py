#!/usr/bin/env python
"""
风险分析运行脚本
用于训练模型和启动服务
"""

import os
import sys
import argparse
import logging
import subprocess
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 目录设置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_models():
    """训练风险分析模型"""
    logger.info("开始训练风险分析模型...")
    
    # 导入训练模块
    try:
        from app.risk_analysis.train_models import train_risk_score_model, train_user_clustering_model
        
        # 训练风险评分模型
        risk_model, risk_metrics = train_risk_score_model()
        logger.info(f"风险评分模型训练完成，性能指标: {risk_metrics}")
        
        # 训练用户聚类模型
        cluster_model, cluster_metrics = train_user_clustering_model()
        logger.info(f"用户聚类模型训练完成，性能指标: {cluster_metrics}")
        
        return True
    except Exception as e:
        logger.error(f"模型训练出错: {str(e)}")
        return False

def run_service(port=8002):
    """启动风险分析服务"""
    logger.info(f"启动风险分析服务，端口: {port}...")
    
    try:
        # 设置环境变量
        env = os.environ.copy()
        env["PORT"] = str(port)
        
        # 启动服务
        service_script = os.path.join(CURRENT_DIR, "risk_analysis_api.py")
        subprocess.run([sys.executable, service_script], env=env)
        
        return True
    except Exception as e:
        logger.error(f"启动服务出错: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="风险分析模型训练和服务启动")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--serve", action="store_true", help="启动服务")
    parser.add_argument("--port", type=int, default=8002, help="服务端口")
    
    args = parser.parse_args()
    
    if args.train:
        success = train_models()
        if not success:
            logger.error("模型训练失败")
            sys.exit(1)
    
    if args.serve:
        success = run_service(port=args.port)
        if not success:
            logger.error("服务启动失败")
            sys.exit(1)
    
    if not args.train and not args.serve:
        parser.print_help()

if __name__ == "__main__":
    main() 