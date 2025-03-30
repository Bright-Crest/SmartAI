/**
 * 主布局组件 - 应用程序的基础布局
 */
import React from 'react';
import PropTypes from 'prop-types';
import Head from 'next/head';
import Link from 'next/link';
import styled from 'styled-components';

// 样式组件
const LayoutContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

const Header = styled.header`
  background-color: #2c3e50;
  color: white;
  padding: 15px 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const HeaderContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;

  @media (max-width: 768px) {
    flex-direction: column;
    align-items: flex-start;
  }
`;

const Logo = styled.div`
  font-size: 20px;
  font-weight: bold;

  a {
    color: white;
    text-decoration: none;
    display: flex;
    align-items: center;

    &:hover {
      text-decoration: none;
    }
  }
`;

const Navigation = styled.nav`
  @media (max-width: 768px) {
    margin-top: 15px;
    width: 100%;
  }
`;

const NavList = styled.ul`
  list-style: none;
  display: flex;
  gap: 20px;
  margin: 0;
  padding: 0;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 10px;
  }
`;

const NavItem = styled.li`
  a {
    color: #ecf0f1;
    text-decoration: none;
    padding: 5px 0;
    font-weight: 500;
    transition: color 0.2s;

    &:hover {
      color: #3498db;
    }

    &.active {
      color: #3498db;
      border-bottom: 2px solid #3498db;
    }
  }
`;

const Main = styled.main`
  flex: 1;
  padding: 30px 0;
`;

const ContentContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
`;

const Footer = styled.footer`
  background-color: #34495e;
  color: #ecf0f1;
  padding: 20px 0;
  margin-top: auto;
`;

const FooterContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;

  @media (max-width: 768px) {
    flex-direction: column;
    gap: 15px;
    text-align: center;
  }
`;

const Copyright = styled.p`
  margin: 0;
  font-size: 14px;
`;

const FooterLinks = styled.div`
  display: flex;
  gap: 20px;

  a {
    color: #ecf0f1;
    text-decoration: none;
    font-size: 14px;

    &:hover {
      text-decoration: underline;
      color: #3498db;
    }
  }
`;

/**
 * 主布局组件 - 应用程序的基础布局
 *
 * @param {Object} props - 组件属性
 * @param {React.ReactNode} props.children - 页面内容
 * @param {string} [props.title='ChainIntelAI | 区块链情报分析平台'] - 页面标题
 * @param {string} [props.description='ChainIntelAI是一个区块链交易分析和监控平台，提供区块链情报分析和可视化服务。'] - 页面描述
 * @returns {JSX.Element} 主布局组件
 */
const MainLayout = ({
  children,
  title = 'ChainIntelAI | 区块链情报分析平台',
  description = 'ChainIntelAI是一个区块链交易分析和监控平台，提供区块链情报分析和可视化服务。',
}) => {
  const navItems = [
    { name: '首页', href: '/' },
    { name: '聪明钱分析', href: '/smart-money' },
  ];

  return (
    <LayoutContainer>
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header>
        <HeaderContainer>
          <Logo>
            <Link href="/" passHref legacyBehavior>
              <a>SmartAI</a>
            </Link>
          </Logo>
          <Navigation>
            <NavList>
              {navItems.map(item => (
                <NavItem key={item.href}>
                  <Link href={item.href} passHref legacyBehavior>
                    <a>{item.name}</a>
                  </Link>
                </NavItem>
              ))}
            </NavList>
          </Navigation>
        </HeaderContainer>
      </Header>

      <Main>
        <ContentContainer>{children}</ContentContainer>
      </Main>

      <Footer>
        <FooterContainer>
          <Copyright>&copy; {new Date().getFullYear()} SmartAI. 保留所有权利。</Copyright>

          <FooterLinks>
            <Link href="/about" passHref legacyBehavior>
              <a>关于我们</a>
            </Link>
          </FooterLinks>
        </FooterContainer>
      </Footer>
    </LayoutContainer>
  );
};

MainLayout.propTypes = {
  children: PropTypes.node.isRequired,
  title: PropTypes.string,
  description: PropTypes.string,
};

export default MainLayout;
