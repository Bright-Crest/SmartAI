/**
 * 按钮组件 - 通用UI组件
 */
import React from 'react';
import PropTypes from 'prop-types';
import styled, { css } from 'styled-components';

// 按钮类型样式
const buttonTypes = {
  primary: css`
    background-color: #3498db;
    color: white;
    &:hover {
      background-color: #2980b9;
    }
  `,
  secondary: css`
    background-color: #95a5a6;
    color: white;
    &:hover {
      background-color: #7f8c8d;
    }
  `,
  success: css`
    background-color: #2ecc71;
    color: white;
    &:hover {
      background-color: #27ae60;
    }
  `,
  danger: css`
    background-color: #e74c3c;
    color: white;
    &:hover {
      background-color: #c0392b;
    }
  `,
  warning: css`
    background-color: #f39c12;
    color: white;
    &:hover {
      background-color: #d35400;
    }
  `,
  info: css`
    background-color: #3498db;
    color: white;
    &:hover {
      background-color: #2980b9;
    }
  `,
  light: css`
    background-color: #f8f9fa;
    color: #343a40;
    &:hover {
      background-color: #e9ecef;
    }
  `,
  dark: css`
    background-color: #343a40;
    color: white;
    &:hover {
      background-color: #23272b;
    }
  `,
  link: css`
    background-color: transparent;
    color: #3498db;
    box-shadow: none;
    &:hover {
      text-decoration: underline;
      background-color: transparent;
    }
  `,
};

// 按钮尺寸样式
const buttonSizes = {
  small: css`
    padding: 5px 10px;
    font-size: 12px;
  `,
  medium: css`
    padding: 8px 15px;
    font-size: 14px;
  `,
  large: css`
    padding: 12px 20px;
    font-size: 16px;
  `,
};

// 基础按钮样式
const StyledButton = styled.button`
  /* 基础样式 */
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
  text-decoration: none;
  outline: none;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);

  /* 尺寸样式 */
  ${props => buttonSizes[props.size] || buttonSizes.medium}

  /* 类型样式 */
  ${props => buttonTypes[props.$buttonType] || buttonTypes.primary}
  
  /* 禁用状态 */
  ${props =>
    props.disabled &&
    css`
      opacity: 0.6;
      cursor: not-allowed;
      pointer-events: none;
    `}
    
  /* 全宽按钮 */
  ${props =>
    props.$fullWidth &&
    css`
      width: 100%;
    `}
  
  /* 圆角按钮 */
  ${props =>
    props.$rounded &&
    css`
      border-radius: 50px;
    `}
  
  /* 自定义样式 */
  ${props => props.customStyles}
  
  /* Icon 间距 */
  & > svg {
    margin-right: ${props => (props.children ? '5px' : '0')};
  }
`;

/**
 * 按钮组件 - 通用UI组件
 *
 * @param {Object} props - 组件属性
 * @param {React.ReactNode} [props.children] - 按钮内容
 * @param {('primary'|'secondary'|'success'|'danger'|'warning'|'info'|'light'|'dark'|'link')} [props.variant='primary'] - 按钮类型
 * @param {('small'|'medium'|'large'|'sm'|'md'|'lg')} [props.size='medium'] - 按钮尺寸
 * @param {boolean} [props.disabled=false] - 是否禁用
 * @param {Function} [props.onClick] - 点击回调函数
 * @param {string} [props.className] - 自定义类名
 * @param {React.ReactNode} [props.icon] - 按钮图标
 * @param {boolean} [props.fullWidth=false] - 是否占满父容器宽度
 * @param {boolean} [props.rounded=false] - 是否圆角
 * @param {Object} [props.style] - 自定义样式
 * @param {string} [props.type='button'] - HTML按钮类型
 * @returns {JSX.Element} 按钮组件
 */
const Button = ({
  children,
  variant = 'primary',
  type = 'button',
  size = 'medium',
  disabled = false,
  onClick,
  className,
  icon,
  fullWidth = false,
  rounded = false,
  style,
  ...rest
}) => {
  // 将size标准化
  const normalizedSize =
    size === 'sm' ? 'small' : size === 'md' ? 'medium' : size === 'lg' ? 'large' : size;

  return (
    <StyledButton
      type={type}
      $buttonType={variant}
      size={normalizedSize}
      disabled={disabled}
      onClick={onClick}
      className={className}
      $fullWidth={fullWidth}
      $rounded={rounded}
      style={style}
      {...rest}
    >
      {icon && icon}
      {children}
    </StyledButton>
  );
};

Button.propTypes = {
  children: PropTypes.node,
  variant: PropTypes.oneOf([
    'primary',
    'secondary',
    'success',
    'danger',
    'warning',
    'info',
    'light',
    'dark',
    'link',
  ]),
  type: PropTypes.string,
  size: PropTypes.oneOf(['small', 'medium', 'large', 'sm', 'md', 'lg']),
  disabled: PropTypes.bool,
  onClick: PropTypes.func,
  className: PropTypes.string,
  icon: PropTypes.node,
  fullWidth: PropTypes.bool,
  rounded: PropTypes.bool,
  style: PropTypes.object,
};

export default Button;
