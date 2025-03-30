/**
 * 卡片组件 - 通用UI组件
 */
import React from 'react';
import PropTypes from 'prop-types';
import styled from 'styled-components';

// 样式组件
const CardContainer = styled.div`
  background-color: ${props => props.bgColor || '#fff'};
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  padding: ${props => props.padding || '15px'};
  margin-bottom: ${props => props.marginBottom || '0'};
  border: 1px solid ${props => props.borderColor || '#e0e0e0'};
  overflow: hidden;
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => (props.title ? '15px' : '0')};
  border-bottom: ${props => (props.title ? '1px solid #eee' : 'none')};
  padding-bottom: ${props => (props.title ? '10px' : '0')};
`;

const CardTitle = styled.h3`
  margin: 0;
  font-size: 16px;
  font-weight: 500;
  color: #333;
`;

const CardBody = styled.div`
  color: #333;
`;

/**
 * 卡片组件 - 通用UI组件
 *
 * @param {Object} props - 组件属性
 * @param {string} [props.title] - 卡片标题
 * @param {React.ReactNode} props.children - 卡片内容
 * @param {string} [props.className] - 自定义类名
 * @param {Object} [props.style] - 自定义样式
 * @param {string} [props.bgColor] - 背景颜色
 * @param {string} [props.borderColor] - 边框颜色
 * @param {string} [props.padding] - 内边距
 * @param {string} [props.marginBottom] - 下边距
 * @param {React.ReactNode} [props.headerRight] - 标题栏右侧内容
 * @returns {JSX.Element} 卡片组件
 */
const Card = ({
  title,
  children,
  className,
  style,
  bgColor,
  borderColor,
  padding,
  marginBottom,
  headerRight,
}) => {
  return (
    <CardContainer
      className={className}
      style={style}
      bgColor={bgColor}
      borderColor={borderColor}
      padding={padding}
      marginBottom={marginBottom}
    >
      {(title || headerRight) && (
        <CardHeader title={title}>
          {title && <CardTitle>{title}</CardTitle>}
          {headerRight}
        </CardHeader>
      )}
      <CardBody>{children}</CardBody>
    </CardContainer>
  );
};

Card.propTypes = {
  title: PropTypes.node,
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  style: PropTypes.object,
  bgColor: PropTypes.string,
  borderColor: PropTypes.string,
  padding: PropTypes.string,
  marginBottom: PropTypes.string,
  headerRight: PropTypes.node,
};

export default Card;
