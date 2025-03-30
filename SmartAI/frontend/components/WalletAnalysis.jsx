import React from 'react';
import {
  Box,
  Heading,
  Text,
  Flex,
  Stack,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Divider,
  Badge,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  Progress,
  HStack,
  Tag,
  SimpleGrid
} from '@chakra-ui/react';

/**
 * 格式化货币数值
 * @param {number} value - 金额
 * @param {string} symbol - 货币符号
 * @param {number} decimals - 小数位数
 * @returns {string} 格式化后的数值
 */
const formatCurrency = (value, symbol = '$', decimals = 2) => {
  return `${symbol}${Number(value).toLocaleString('zh-CN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  })}`;
};

/**
 * 格式化数字
 * @param {number} value - 数值
 * @param {number} decimals - 小数位数
 * @returns {string} 格式化后的数值
 */
const formatNumber = (value, decimals = 2) => {
  return Number(value).toLocaleString('zh-CN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
};

/**
 * 格式化日期时间
 * @param {number} timestamp - Unix时间戳
 * @returns {string} 格式化后的日期时间
 */
const formatDate = (timestamp) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
};

/**
 * 钱包分析组件
 * @param {Object} props - 组件属性
 * @param {Object} props.walletData - 钱包数据
 * @returns {JSX.Element} 
 */
const WalletAnalysis = ({ walletData }) => {
  // 处理空数据情况
  if (!walletData || !walletData.assets) {
    return (
      <Box p={5} borderWidth="1px" borderRadius="lg">
        <Text>暂无数据可显示</Text>
      </Box>
    );
  }
  
  const { totalValue, assets, transactions } = walletData;
  
  // 计算资产分布百分比
  const calculatePercentage = (assetValue) => {
    return (assetValue / totalValue) * 100;
  };
  
  return (
    <Stack spacing={6}>
      {/* 总资产概览 */}
      <Box p={5} borderWidth="1px" borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4}>资产概览</Heading>
        <Flex justify="space-between" flexWrap="wrap">
          <Stat>
            <StatLabel>总资产价值</StatLabel>
            <StatNumber>{formatCurrency(totalValue)}</StatNumber>
            <StatHelpText>多链资产总价值</StatHelpText>
          </Stat>
          
          <Stat>
            <StatLabel>资产数量</StatLabel>
            <StatNumber>{assets.length}</StatNumber>
            <StatHelpText>包含代币种类</StatHelpText>
          </Stat>
          
          <Stat>
            <StatLabel>交易记录</StatLabel>
            <StatNumber>{transactions ? transactions.length : 0}</StatNumber>
            <StatHelpText>历史交易总数</StatHelpText>
          </Stat>
        </Flex>
      </Box>
      
      {/* 资产详情列表 */}
      <Box p={5} borderWidth="1px" borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4}>资产详情</Heading>
        <TableContainer>
          <Table variant="simple">
            <Thead>
              <Tr>
                <Th>资产</Th>
                <Th>类型</Th>
                <Th isNumeric>数量</Th>
                <Th isNumeric>价值(USD)</Th>
                <Th isNumeric>分布比例</Th>
              </Tr>
            </Thead>
            <Tbody>
              {assets.map((asset, index) => (
                <Tr key={index}>
                  <Td>
                    <HStack>
                      <Text fontWeight="bold">{asset.symbol}</Text>
                      <Badge colorScheme={asset.type === 'native' ? 'green' : 'blue'}>
                        {asset.type === 'native' ? '原生币' : '代币'}
                      </Badge>
                    </HStack>
                  </Td>
                  <Td>{asset.type === 'native' ? '主链币' : 'ERC20'}</Td>
                  <Td isNumeric>{formatNumber(asset.amount)}</Td>
                  <Td isNumeric>{formatCurrency(asset.value)}</Td>
                  <Td>
                    <Box w="100%">
                      <Text textAlign="right" mb={1}>
                        {formatNumber(calculatePercentage(asset.value))}%
                      </Text>
                      <Progress 
                        value={calculatePercentage(asset.value)} 
                        size="sm" 
                        colorScheme={asset.type === 'native' ? 'green' : 'blue'}
                        borderRadius="md"
                      />
                    </Box>
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </TableContainer>
      </Box>
      
      {/* 资产分布图表 */}
      <Box p={5} borderWidth="1px" borderRadius="lg" boxShadow="md">
        <Heading size="md" mb={4}>资产分布</Heading>
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={10}>
          <Box>
            <Heading size="sm" mb={4}>按资产类型</Heading>
            {assets.map((asset, index) => (
              <Box key={index} mb={3}>
                <Flex justify="space-between" mb={1}>
                  <Text>{asset.symbol}</Text>
                  <Text>{formatNumber(calculatePercentage(asset.value))}%</Text>
                </Flex>
                <Progress 
                  value={calculatePercentage(asset.value)} 
                  size="md" 
                  colorScheme={
                    index % 4 === 0 ? 'green' : 
                    index % 4 === 1 ? 'blue' : 
                    index % 4 === 2 ? 'purple' : 
                    'orange'
                  }
                  borderRadius="md"
                />
              </Box>
            ))}
          </Box>
          <Box>
            <Heading size="sm" mb={4}>价值分布</Heading>
            <TableContainer>
              <Table variant="simple" size="sm">
                <Thead>
                  <Tr>
                    <Th>资产</Th>
                    <Th isNumeric>价值(USD)</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {assets
                    .sort((a, b) => b.value - a.value)
                    .map((asset, index) => (
                      <Tr key={index}>
                        <Td>{asset.symbol}</Td>
                        <Td isNumeric>{formatCurrency(asset.value)}</Td>
                      </Tr>
                    ))}
                </Tbody>
              </Table>
            </TableContainer>
          </Box>
        </SimpleGrid>
      </Box>
      
      {/* 近期交易 */}
      {transactions && transactions.length > 0 && (
        <Box p={5} borderWidth="1px" borderRadius="lg" boxShadow="md">
          <Heading size="md" mb={4}>近期交易</Heading>
          <TableContainer>
            <Table variant="simple">
              <Thead>
                <Tr>
                  <Th>类型</Th>
                  <Th>资产</Th>
                  <Th isNumeric>数量</Th>
                  <Th>状态</Th>
                  <Th>时间</Th>
                </Tr>
              </Thead>
              <Tbody>
                {transactions.map((tx, index) => (
                  <Tr key={index}>
                    <Td>
                      <Tag colorScheme={tx.type === 'send' ? 'red' : 'green'}>
                        {tx.type === 'send' ? '转出' : '转入'}
                      </Tag>
                    </Td>
                    <Td>{tx.asset}</Td>
                    <Td isNumeric>{formatNumber(tx.amount)}</Td>
                    <Td>
                      <Badge colorScheme={tx.status === 'completed' ? 'green' : 'yellow'}>
                        {tx.status === 'completed' ? '已完成' : '进行中'}
                      </Badge>
                    </Td>
                    <Td>{formatDate(tx.timestamp)}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Stack>
  );
};

export default WalletAnalysis; 