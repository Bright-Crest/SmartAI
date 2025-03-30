import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import RiskBadge from '../../components/RiskBadge';

describe('RiskBadge组件', () => {
  // 基本渲染测试
  test('渲染低风险徽章', () => {
    render(<RiskBadge level="low" />);
    
    const badge = screen.getByText('低风险');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-green-100');
    expect(badge).toHaveClass('text-green-800');
  });
  
  test('渲染中风险徽章', () => {
    render(<RiskBadge level="medium" />);
    
    const badge = screen.getByText('中风险');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-yellow-100');
    expect(badge).toHaveClass('text-yellow-800');
  });
  
  test('渲染高风险徽章', () => {
    render(<RiskBadge level="high" />);
    
    const badge = screen.getByText('高风险');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-orange-100');
    expect(badge).toHaveClass('text-orange-800');
  });
  
  test('渲染严重风险徽章', () => {
    render(<RiskBadge level="critical" />);
    
    const badge = screen.getByText('严重风险');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveClass('bg-red-100');
    expect(badge).toHaveClass('text-red-800');
  });
  
  // 风险分数测试
  test('显示风险分数', () => {
    const score = 85;
    render(<RiskBadge level="high" score={score} showScore={true} />);
    
    expect(screen.getByText('高风险')).toBeInTheDocument();
    expect(screen.getByText(score.toString())).toBeInTheDocument();
  });
  
  test('不显示风险分数当showScore为false', () => {
    const score = 85;
    render(<RiskBadge level="high" score={score} showScore={false} />);
    
    expect(screen.getByText('高风险')).toBeInTheDocument();
    expect(screen.queryByText(score.toString())).not.toBeInTheDocument();
  });
  
  test('showScore默认值为false', () => {
    const score = 85;
    render(<RiskBadge level="high" score={score} />);
    
    expect(screen.getByText('高风险')).toBeInTheDocument();
    expect(screen.queryByText(score.toString())).not.toBeInTheDocument();
  });
  
  // 自定义类名测试
  test('应用自定义className', () => {
    const customClass = 'test-custom-class';
    render(<RiskBadge level="low" className={customClass} />);
    
    const badge = screen.getByText('低风险');
    expect(badge).toHaveClass(customClass);
  });
  
  // 无障碍测试
  test('包含适当的ARIA属性', () => {
    render(<RiskBadge level="medium" score={45} showScore={true} />);
    
    const badge = screen.getByRole('status');
    expect(badge).toHaveAttribute('aria-label', '风险等级: 中风险，分数: 45');
    expect(badge).toHaveAttribute('aria-level', '2');
  });
  
  test('风险分数在屏幕阅读器中不隐藏', () => {
    render(<RiskBadge level="high" score={75} showScore={true} />);
    
    const badge = screen.getByRole('status');
    expect(badge).toHaveAttribute('aria-label', expect.stringContaining('75'));
  });
  
  test('风险分数值在视觉上对屏幕阅读器隐藏', () => {
    render(<RiskBadge level="critical" score={95} showScore={true} />);
    
    const scoreElement = screen.getByText('95');
    expect(scoreElement).toHaveAttribute('aria-hidden', 'true');
  });
}); 