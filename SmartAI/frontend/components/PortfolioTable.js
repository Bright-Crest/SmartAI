import React from 'react';
import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Box,
  Flex,
  Image,
  Tooltip,
  chakra,
} from '@chakra-ui/react';
import { formatNumber } from '../utils/formatting';

/**
 * 显示钱包持仓表格
 * @param {Object} props - 组件属性
 * @param {Array} props.portfolio - 持仓数组
 * @param {number} props.limit - 显示限制
 */
const PortfolioTable = ({ portfolio = [], limit = 10 }) => {
  // 如果没有持仓数据，不渲染任何内容
  if (!portfolio || portfolio.length === 0) {
    return (
      <Box borderWidth="1px" borderRadius="lg" p={4}>
        <Text textAlign="center">无持仓数据</Text>
      </Box>
    );
  }

  // 确保每个代币对象有所有必要字段
  const safePortfolio = portfolio.map(token => ({
    address: token.address || '0x0000000000000000000000000000000000000000',
    name: token.name || '未知代币',
    symbol: token.symbol || 'UNKNOWN',
    balance: Number(token.balance || 0),
    valueUSD: Number(token.valueUSD || 0),
    decimals: Number(token.decimals || 18),
    allocation: Number(token.allocation || 0),
  }));

  // 计算总价值（如果需要）
  const totalValue = safePortfolio.reduce((sum, token) => sum + (token.valueUSD || 0), 0);

  // 按照价值(valueUSD)降序排序
  const sortedPortfolio = [...safePortfolio].sort((a, b) => b.valueUSD - a.valueUSD);

  // 显示有限数量的代币
  const displayPortfolio = sortedPortfolio.slice(0, limit);

  return (
    <Box borderWidth="1px" borderRadius="lg" overflowX="auto" w="100%">
      <Table variant="simple" size="sm">
        <Thead>
          <Tr>
            <Th>代币</Th>
            <Th isNumeric>数量</Th>
            <Th isNumeric>价值(USD)</Th>
            <Th isNumeric>配置比例</Th>
          </Tr>
        </Thead>
        <Tbody>
          {displayPortfolio.map((token, index) => (
            <Tr key={token.address || index}>
              <Td>
                <Flex align="center">
                  <Image
                    src={`https://raw.githubusercontent.com/trustwallet/assets/master/blockchains/ethereum/assets/${token.address}/logo.png`}
                    fallbackSrc="/images/token-placeholder.png"
                    boxSize="20px"
                    mr={2}
                    borderRadius="full"
                  />
                  <Tooltip label={token.name || '未知代币'}>
                    <Text fontWeight="medium">{token.symbol || 'UNKNOWN'}</Text>
                  </Tooltip>
                </Flex>
              </Td>
              <Td isNumeric>{formatNumber(token.balance || 0)}</Td>
              <Td isNumeric>${formatNumber(token.valueUSD || 0)}</Td>
              <Td isNumeric>
                {totalValue > 0
                  ? `${formatNumber((token.valueUSD / totalValue) * 100 || 0)}%`
                  : `${formatNumber(token.allocation * 100 || 0)}%`}
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
      <Text fontSize="xs" textAlign="right" p={2}>
        显示前{Math.min(displayPortfolio.length, limit)}个持仓 (共 {portfolio.length} 个)
      </Text>
    </Box>
  );
};

export default PortfolioTable;
