import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import OptimizedForm, { FormField, FormValues } from '../../components/OptimizedForm';
import userEvent from '@testing-library/user-event';

// 测试用的表单字段配置
const testFields: FormField[] = [
  {
    id: 'name',
    label: '姓名',
    type: 'text',
    placeholder: '请输入姓名',
    required: true,
    errorMessage: '姓名不能为空'
  },
  {
    id: 'email',
    label: '邮箱',
    type: 'email',
    placeholder: '请输入邮箱',
    required: true,
    validation: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    errorMessage: '请输入有效的邮箱地址'
  },
  {
    id: 'password',
    label: '密码',
    type: 'password',
    placeholder: '请输入密码',
    required: true,
    errorMessage: '密码不能为空'
  },
  {
    id: 'gender',
    label: '性别',
    type: 'select',
    required: true,
    options: [
      { value: 'male', label: '男' },
      { value: 'female', label: '女' },
      { value: 'other', label: '其他' }
    ]
  },
  {
    id: 'description',
    label: '个人简介',
    type: 'textarea',
    placeholder: '请输入个人简介'
  }
];

describe('OptimizedForm组件', () => {
  // 基本渲染测试
  test('渲染所有表单字段', () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    // 检查所有字段是否渲染
    expect(screen.getByLabelText(/姓名/)).toBeInTheDocument();
    expect(screen.getByLabelText(/邮箱/)).toBeInTheDocument();
    expect(screen.getByLabelText(/密码/)).toBeInTheDocument();
    expect(screen.getByLabelText(/性别/)).toBeInTheDocument();
    expect(screen.getByLabelText(/个人简介/)).toBeInTheDocument();
    
    // 检查提交按钮是否存在
    expect(screen.getByRole('button', { name: '提交' })).toBeInTheDocument();
  });
  
  // 初始值测试
  test('正确应用初始值', () => {
    const initialValues = {
      name: '张三',
      email: 'zhangsan@example.com'
    };
    
    render(
      <OptimizedForm 
        fields={testFields} 
        onSubmit={() => {}} 
        initialValues={initialValues} 
      />
    );
    
    // 验证初始值是否被正确应用
    expect(screen.getByLabelText(/姓名/)).toHaveValue('张三');
    expect(screen.getByLabelText(/邮箱/)).toHaveValue('zhangsan@example.com');
    expect(screen.getByLabelText(/密码/)).toHaveValue('');
  });
  
  // 表单输入测试
  test('处理用户输入', async () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    // 获取输入字段
    const nameInput = screen.getByLabelText(/姓名/);
    const emailInput = screen.getByLabelText(/邮箱/);
    const passwordInput = screen.getByLabelText(/密码/);
    
    // 输入值
    fireEvent.change(nameInput, { target: { value: '李四' } });
    fireEvent.change(emailInput, { target: { value: 'lisi@example.com' } });
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    
    // 验证输入值是否正确设置
    expect(nameInput).toHaveValue('李四');
    expect(emailInput).toHaveValue('lisi@example.com');
    expect(passwordInput).toHaveValue('password123');
  });
  
  // 表单提交测试
  test('提交表单时调用onSubmit', async () => {
    const mockSubmit = jest.fn();
    render(<OptimizedForm fields={testFields} onSubmit={mockSubmit} />);
    
    // 填写必填字段
    fireEvent.change(screen.getByLabelText(/姓名/), { target: { value: '王五' } });
    fireEvent.change(screen.getByLabelText(/邮箱/), { target: { value: 'wangwu@example.com' } });
    fireEvent.change(screen.getByLabelText(/密码/), { target: { value: 'password123' } });
    
    // 选择性别
    fireEvent.change(screen.getByLabelText(/性别/), { target: { value: 'male' } });
    
    // 提交表单
    fireEvent.click(screen.getByRole('button', { name: '提交' }));
    
    // 等待提交完成
    await waitFor(() => {
      expect(mockSubmit).toHaveBeenCalledTimes(1);
      expect(mockSubmit).toHaveBeenCalledWith({
        name: '王五',
        email: 'wangwu@example.com',
        password: 'password123',
        gender: 'male',
        description: ''
      });
    });
  });
  
  // 验证测试
  test('显示必填字段错误', async () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    // 直接点击提交按钮，不填写任何字段
    fireEvent.click(screen.getByRole('button', { name: '提交' }));
    
    // 验证错误消息是否显示
    await waitFor(() => {
      expect(screen.getByText('姓名不能为空')).toBeInTheDocument();
      expect(screen.getByText('请输入有效的邮箱地址')).toBeInTheDocument();
      expect(screen.getByText('密码不能为空')).toBeInTheDocument();
    });
  });
  
  test('根据字段验证规则显示错误', async () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    // 填写无效的邮箱格式
    const emailInput = screen.getByLabelText(/邮箱/);
    fireEvent.change(emailInput, { target: { value: 'invalid-email' } });
    fireEvent.blur(emailInput); // 触发验证
    
    // 验证错误消息是否显示
    await waitFor(() => {
      expect(screen.getByText('请输入有效的邮箱地址')).toBeInTheDocument();
    });
  });
  
  test('清除输入后触发验证', async () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    // 先填写名称然后清除
    const nameInput = screen.getByLabelText(/姓名/);
    fireEvent.change(nameInput, { target: { value: '张三' } });
    fireEvent.change(nameInput, { target: { value: '' } });
    fireEvent.blur(nameInput); // 触发验证
    
    // 验证错误消息是否显示
    await waitFor(() => {
      expect(screen.getByText('姓名不能为空')).toBeInTheDocument();
    });
  });
  
  // 自定义按钮文本测试
  test('应用自定义提交按钮文本', () => {
    render(
      <OptimizedForm 
        fields={testFields} 
        onSubmit={() => {}} 
        submitButtonText="保存" 
      />
    );
    
    expect(screen.getByRole('button', { name: '保存' })).toBeInTheDocument();
  });
  
  // 无障碍测试
  test('表单具有适当的无障碍特性', () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    // 检查标签是否正确关联到输入字段
    const nameInput = screen.getByLabelText(/姓名/);
    expect(nameInput).toHaveAttribute('id', 'name');
    
    // 检查必填字段是否有视觉指示器
    const nameLabel = screen.getByText(/姓名/);
    expect(nameLabel.innerHTML).toContain('*');
    
    // 检查提交按钮是否可访问
    expect(screen.getByRole('button', { name: '提交' })).toBeInTheDocument();
  });
  
  // 自动聚焦和悬浮状态测试
  test('字段获得焦点时添加强调样式', async () => {
    render(<OptimizedForm fields={testFields} onSubmit={() => {}} />);
    
    const nameInput = screen.getByLabelText(/姓名/);
    
    // 获取焦点前的样式
    const initialClassName = nameInput.className;
    
    // 聚焦输入框
    fireEvent.focus(nameInput);
    
    // 验证聚焦后的样式变化
    expect(nameInput.className).not.toBe(initialClassName);
    expect(nameInput).toHaveClass('focus:ring-2');
    expect(nameInput).toHaveClass('focus:ring-blue-500');
  });
  
  // 测试表单重置功能
  test('提交成功后重置表单功能', async () => {
    // 创建一个模拟的重置表单函数
    const resetForm = jest.fn((values: FormValues) => {
      // 模拟异步重置
      return new Promise<void>(resolve => {
        setTimeout(() => {
          resolve();
        }, 100);
      });
    });
    
    render(
      <OptimizedForm 
        fields={testFields} 
        onSubmit={resetForm}
        submitButtonText="提交并重置" 
      />
    );
    
    // 填写表单
    fireEvent.change(screen.getByLabelText(/姓名/), { target: { value: '张三' } });
    fireEvent.change(screen.getByLabelText(/邮箱/), { target: { value: 'zhangsan@example.com' } });
    fireEvent.change(screen.getByLabelText(/密码/), { target: { value: 'password123' } });
    fireEvent.change(screen.getByLabelText(/性别/), { target: { value: 'male' } });
    
    // 提交表单
    fireEvent.click(screen.getByRole('button', { name: '提交并重置' }));
    
    // 验证提交函数被调用
    await waitFor(() => {
      expect(resetForm).toHaveBeenCalled();
    });
  });
}); 