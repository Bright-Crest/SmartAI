import React, { forwardRef } from 'react';

export interface A11yFormInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  /**
   * 输入框标签文本
   */
  label: string;
  
  /**
   * 输入框ID（如果未提供将自动生成）
   */
  id?: string;
  
  /**
   * 错误消息（用于验证失败时）
   */
  error?: string;
  
  /**
   * 帮助文本（说明该字段的用途）
   */
  helpText?: string;
  
  /**
   * 标签位置
   */
  labelPosition?: 'top' | 'left' | 'right' | 'hidden';
  
  /**
   * 输入框包装器的类名
   */
  wrapperClassName?: string;
  
  /**
   * 标签的类名
   */
  labelClassName?: string;
  
  /**
   * 输入框的类名
   */
  inputClassName?: string;
  
  /**
   * 错误消息的类名
   */
  errorClassName?: string;
  
  /**
   * 帮助文本的类名
   */
  helpTextClassName?: string;
  
  /**
   * 是否必填
   */
  required?: boolean;
}

/**
 * 无障碍表单输入组件
 * 确保所有输入框都有关联的标签和适当的ARIA属性
 * 遵循WCAG 2.1 AA标准
 */
const A11yFormInput = forwardRef<HTMLInputElement, A11yFormInputProps>(
  ({
    label,
    id,
    error,
    helpText,
    labelPosition = 'top',
    wrapperClassName = '',
    labelClassName = '',
    inputClassName = '',
    errorClassName = '',
    helpTextClassName = '',
    required = false,
    type = 'text',
    disabled = false,
    className = '',
    ...props
  }, ref) => {
    // 生成唯一ID，如果未提供
    const inputId = id || `input-${Math.random().toString(36).substring(2, 9)}`;
    
    // 帮助文本和错误消息的ID
    const helpTextId = helpText ? `${inputId}-help` : undefined;
    const errorId = error ? `${inputId}-error` : undefined;
    
    // 决定使用哪个ID作为描述输入框的文本
    const ariaDescribedby = [helpTextId, errorId].filter(Boolean).join(' ') || undefined;
    
    // 基础样式
    const baseInputStyles = `form-input block w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 ${
      disabled ? 'bg-gray-100 cursor-not-allowed' : ''
    } ${error ? 'border-red-500' : ''}`;
    
    // 标签的条件渲染
    const renderLabel = () => {
      if (labelPosition === 'hidden') {
        return (
          <label
            htmlFor={inputId}
            className="sr-only"
          >
            {label}
            {required && <span aria-hidden="true"> *</span>}
          </label>
        );
      }
      
      return (
        <label
          htmlFor={inputId}
          className={`block mb-1 font-medium text-gray-700 ${labelClassName}`}
        >
          {label}
          {required && <span aria-hidden="true" className="ml-1 text-red-500">*</span>}
        </label>
      );
    };
    
    // 帮助文本的条件渲染
    const renderHelpText = () => {
      if (!helpText) return null;
      
      return (
        <p 
          id={helpTextId} 
          className={`mt-1 text-sm text-gray-500 ${helpTextClassName}`}
        >
          {helpText}
        </p>
      );
    };
    
    // 错误消息的条件渲染
    const renderError = () => {
      if (!error) return null;
      
      return (
        <p 
          id={errorId} 
          className={`mt-1 text-sm text-red-600 ${errorClassName}`}
          role="alert"
        >
          {error}
        </p>
      );
    };
    
    // 确定包装器类名基于标签位置
    const getWrapperClassName = () => {
      switch (labelPosition) {
        case 'left':
          return `flex items-center ${wrapperClassName}`;
        case 'right':
          return `flex flex-row-reverse items-center ${wrapperClassName}`;
        case 'top':
        case 'hidden':
        default:
          return wrapperClassName;
      }
    };
    
    // 确定输入框类名基于标签位置
    const getInputClassName = () => {
      switch (labelPosition) {
        case 'left':
          return `ml-2 ${inputClassName}`;
        case 'right':
          return `mr-2 ${inputClassName}`;
        case 'top':
        case 'hidden':
        default:
          return inputClassName;
      }
    };
    
    return (
      <div className={getWrapperClassName()}>
        {renderLabel()}
        <input
          ref={ref}
          id={inputId}
          type={type}
          disabled={disabled}
          className={`${baseInputStyles} ${getInputClassName()} ${className}`}
          aria-describedby={ariaDescribedby}
          aria-invalid={!!error}
          aria-required={required}
          required={required}
          {...props}
        />
        {renderHelpText()}
        {renderError()}
      </div>
    );
  }
);

A11yFormInput.displayName = 'A11yFormInput';

export default A11yFormInput; 