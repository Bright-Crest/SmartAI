import React, { forwardRef } from 'react';

/**
 * 输入框组件
 * @param {Object} props - 组件属性
 * @param {string} props.type - 输入框类型，如text, email, password等
 * @param {string} props.id - 输入框ID
 * @param {string} props.name - 输入框名称
 * @param {string} props.label - 输入框标签
 * @param {string} props.placeholder - 占位文本
 * @param {boolean} props.required - 是否必填
 * @param {boolean} props.disabled - 是否禁用
 * @param {boolean} props.error - 是否有错误
 * @param {string} props.errorText - 错误文本
 * @param {string} props.helperText - 帮助文本
 * @param {string} props.size - 输入框大小，可选值：sm, md, lg
 * @param {string} props.className - 自定义类名
 * @param {Function} props.onChange - 值变化事件处理函数
 * @param {Function} props.onBlur - 失焦事件处理函数
 * @returns {JSX.Element} 输入框组件
 */
const Input = forwardRef(
  (
    {
      type = 'text',
      id,
      name,
      label,
      placeholder,
      required = false,
      disabled = false,
      error = false,
      errorText,
      helperText,
      size = 'md',
      className = '',
      onChange,
      onBlur,
      ...rest
    },
    ref
  ) => {
    // 尺寸样式
    const sizeStyles = {
      sm: 'py-1 px-2 text-xs',
      md: 'py-2 px-3 text-sm',
      lg: 'py-3 px-4 text-base',
    };

    // 状态样式
    const stateStyles = error
      ? 'border-red-500 focus:ring-red-500 focus:border-red-500 dark:border-red-600 dark:focus:ring-red-600 dark:focus:border-red-600'
      : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:border-gray-600 dark:focus:ring-blue-500 dark:focus:border-blue-500';

    // 基础样式
    const baseStyle =
      'block w-full rounded-md shadow-sm bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 disabled:opacity-70 disabled:cursor-not-allowed';

    return (
      <div className="mb-4">
        {label && (
          <label
            htmlFor={id}
            className={`block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 ${required ? 'required' : ''}`}
          >
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </label>
        )}
        <input
          type={type}
          id={id}
          name={name}
          placeholder={placeholder}
          disabled={disabled}
          required={required}
          onChange={onChange}
          onBlur={onBlur}
          ref={ref}
          className={`${baseStyle} ${sizeStyles[size]} ${stateStyles} ${className}`}
          {...rest}
        />
        {error && errorText && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errorText}</p>
        )}
        {helperText && !error && (
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
