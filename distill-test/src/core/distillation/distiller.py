import numpy as np
import time
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import logger, log_function_call, log_step
from utils.metrics import ModelEvaluator as MetricsCalculator
from core.models.teacher import TeacherModel
from core.models.student import StudentModel
from core.evaluation.evaluator import ModelEvaluator
from utils.config import config

class Distiller:
    def __init__(self, teacher_model=None, student_model=None):
        """初始化蒸馏器
        
        Args:
            teacher_model: 教师模型实例
            student_model: 学生模型实例
        """
        self.teacher = teacher_model
        self.student = student_model
        self.metrics_calculator = MetricsCalculator()
        self.evaluator = ModelEvaluator()
        
        # 记录执行时间
        self.train_times = {
            'teacher': 0,
            'student': 0
        }
    
    @log_function_call
    def initialize_models(self):
        """初始化模型"""
        if self.teacher is None:
            self.teacher = TeacherModel(
                model_type="teacher",
                model_name=config.get_model_name("teacher"),
                model_params=config.get_model_params("teacher")
            )
        
        if self.student is None:
            self.student = StudentModel(
                model_type="student",
                model_name=config.get_model_name("student"),
                model_params=config.get_model_params("student")
            )
    
    @log_step("训练教师模型")
    def train_teacher(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """训练教师模型"""
        if self.teacher is None:
            self.initialize_models()
        
        # 记录开始时间
        start_time = time.time()
        
        # 创建评估集
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        # 训练模型
        if eval_set:
            kwargs['eval_set'] = eval_set
        self.teacher.train(X_train, y_train, **kwargs)
        
        # 记录训练时间
        self.train_times['teacher'] = time.time() - start_time
        logger.info(f"教师模型训练耗时: {self.train_times['teacher']:.2f} 秒")
        
        # 评估教师模型
        if X_val is not None and y_val is not None:
            y_pred = self.teacher.predict(X_val)
            y_prob = self.teacher.predict_proba(X_val)
            metrics = self.metrics_calculator.calculate_metrics(y_val, y_pred, y_prob)
            logger.info("教师模型验证集评估结果:\n" + self.metrics_calculator.format_metrics(metrics))
    
    @log_step("知识蒸馏")
    def distill(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """执行知识蒸馏"""
        if self.teacher is None or self.student is None:
            self.initialize_models()
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取教师模型的软标签
        logger.info("生成教师模型软标签")
        teacher_probs = self.teacher.predict_proba(X_train)
        
        # 准备参数，避免重复传递
        student_kwargs = kwargs.copy()
        
        # 只有在kwargs中没有这些参数时才添加默认值
        if 'alpha' not in student_kwargs:
            student_kwargs['alpha'] = config.train_config['alpha']
            
        if 'temperature' not in student_kwargs:
            student_kwargs['temperature'] = config.train_config['temperature']
        
        # 训练学生模型
        logger.info(f"使用温度参数 {student_kwargs['temperature']} 和软标签权重 {student_kwargs['alpha']} 训练学生模型")
        self.student.train(
            X_train, 
            y_train,
            teacher_probs=teacher_probs,
            **student_kwargs
        )
        
        # 记录训练时间
        self.train_times['student'] = time.time() - start_time
        logger.info(f"学生模型训练耗时: {self.train_times['student']:.2f} 秒")
        
        # 评估学生模型
        if X_val is not None and y_val is not None:
            logger.info("在验证集上评估和比较模型")
            self._evaluate_and_compare(X_val, y_val)
    
    @log_function_call
    def _evaluate_and_compare(self, X, y):
        """评估和比较两个模型"""
        # 教师模型预测
        teacher_pred = self.teacher.predict(X)
        teacher_prob = self.teacher.predict_proba(X)
        
        # 直接使用原始标签进行评估
        teacher_metrics = self.metrics_calculator.calculate_metrics(y, teacher_pred, teacher_prob)
        
        # 学生模型预测
        student_pred = self.student.predict(X)
        student_prob = self.student.predict_proba(X)
        student_metrics = self.metrics_calculator.calculate_metrics(y, student_pred, student_prob)
        
        # 计算预测一致性
        consistency_metrics = self.metrics_calculator.calculate_prediction_consistency(
            teacher_pred, student_pred
        )
        
        # 计算模型复杂度
        teacher_complexity = self.metrics_calculator.calculate_model_complexity(self.teacher.get_model())
        student_complexity = self.metrics_calculator.calculate_model_complexity(self.student.get_model())
        
        # 输出评估结果
        logger.info("模型评估结果:")
        logger.info("教师模型:\n" + self.metrics_calculator.format_metrics(teacher_metrics))
        logger.info("学生模型:\n" + self.metrics_calculator.format_metrics(student_metrics))
        logger.info("预测一致性:\n" + self.metrics_calculator.format_metrics(consistency_metrics))
        logger.info("模型复杂度:")
        logger.info("教师模型:\n" + self.metrics_calculator.format_metrics(teacher_complexity))
        logger.info("学生模型:\n" + self.metrics_calculator.format_metrics(student_complexity))
        
        # 使用评估器进行更详细的比较分析
        logger.info("使用评估器进行详细比较分析")
        self.evaluator.compare_models(self.teacher, self.student, X, y)
        
        # 分析模型复杂度
        self.evaluator.analyze_model_complexity(self.teacher, self.student)
        
        return {
            'teacher': teacher_metrics,
            'student': student_metrics,
            'consistency': consistency_metrics,
            'teacher_complexity': teacher_complexity,
            'student_complexity': student_complexity
        }
    
    @log_function_call
    def save_models(self):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型保存目录
        models_dir = config.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        teacher_filename = f"teacher_{self.teacher.model_name}_{timestamp}.joblib"
        student_filename = f"student_{self.student.model_name}_{timestamp}.joblib"
        
        self.teacher.save(teacher_filename)
        self.student.save(student_filename)
        
        # 保存训练时间和模型信息
        self._save_model_info(timestamp)
    
    @log_function_call
    def _save_model_info(self, timestamp):
        """保存模型信息
        
        Args:
            timestamp: 时间戳
        """
        import json
        
        info = {
            'timestamp': timestamp,
            'teacher': {
                'model_name': self.teacher.model_name,
                'model_params': self.teacher.model_params,
                'training_time': self.train_times['teacher']
            },
            'student': {
                'model_name': self.student.model_name,
                'model_params': self.student.model_params,
                'training_time': self.train_times['student']
            },
            'config': {
                'alpha': config.train_config['alpha'],
                'temperature': config.train_config['temperature']
            }
        }
        
        # 保存为JSON
        info_path = config.models_dir / f"model_info_{timestamp}.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"模型信息已保存至 {info_path}")
    
    @log_function_call
    def load_models(self, teacher_path=None, student_path=None):
        """加载模型
        
        Args:
            teacher_path: 教师模型路径（可选）
            student_path: 学生模型路径（可选）
        """
        if self.teacher is None or self.student is None:
            self.initialize_models()
        
        if teacher_path:
            self.teacher.load(teacher_path)
        
        if student_path:
            self.student.load(student_path)