import React from 'react';
import { Box, Heading, Text, Flex, Badge, Divider } from '@chakra-ui/react';
import { formatNumber, formatTimestamp } from '../utils/formatting';

/**
 * 交易模式卡片组件
 * 用于显示交易模式和相关统计
 */
const TransactionPatternCard = ({ pattern }) => {
  if (!pattern) {
    return (
      <Box borderWidth="1px" borderRadius="lg" p={4} bg="white">
        <Text textAlign="center">无交易模式数据</Text>
      </Box>
    );
  }

  // 从模式数据中提取信息
  const { overview = {}, frequencyPatterns = {}, sizePatterns = {}, strategies = [] } = pattern;

  // 获取频率文本
  const getFrequencyText = frequency => {
    switch (frequency) {
      case 'high':
        return '高频';
      case 'medium':
        return '中频';
      case 'low':
        return '低频';
      default:
        return '未知';
    }
  };

  // 获取频率颜色
  const getFrequencyColor = frequency => {
    switch (frequency) {
      case 'high':
        return 'green';
      case 'medium':
        return 'blue';
      case 'low':
        return 'yellow';
      default:
        return 'gray';
    }
  };

  return (
    <Box borderWidth="1px" borderRadius="lg" p={4} bg="white">
      <Heading size="md" mb={4}>
        交易模式
      </Heading>

      {/* 概览 */}
      <Box mb={4}>
        <Heading size="sm" mb={2}>
          交易概览
        </Heading>
        <Flex justifyContent="space-between" mb={2}>
          <Text color="gray.600">交易总数</Text>
          <Text fontWeight="medium">{formatNumber(overview.transactionCount || 0)}</Text>
        </Flex>
        <Flex justifyContent="space-between" mb={2}>
          <Text color="gray.600">首次交易</Text>
          <Text fontWeight="medium">{formatTimestamp(overview.firstTransaction) || '未知'}</Text>
        </Flex>
        <Flex justifyContent="space-between">
          <Text color="gray.600">最后交易</Text>
          <Text fontWeight="medium">{formatTimestamp(overview.lastTransaction) || '未知'}</Text>
        </Flex>
      </Box>

      <Divider my={4} />

      {/* 频率模式 */}
      <Box mb={4}>
        <Heading size="sm" mb={2}>
          交易频率
        </Heading>
        <Flex justifyContent="space-between" mb={2}>
          <Text color="gray.600">交易频率</Text>
          <Badge colorScheme={getFrequencyColor(frequencyPatterns.averageFrequency)}>
            {getFrequencyText(frequencyPatterns.averageFrequency)}
          </Badge>
        </Flex>
        <Flex justifyContent="space-between" mb={2}>
          <Text color="gray.600">日均交易</Text>
          <Text fontWeight="medium">{formatNumber(frequencyPatterns.dailyAvg || 0)}</Text>
        </Flex>
        <Flex justifyContent="space-between">
          <Text color="gray.600">月均交易</Text>
          <Text fontWeight="medium">{formatNumber(frequencyPatterns.monthlyAvg || 0)}</Text>
        </Flex>
      </Box>

      <Divider my={4} />

      {/* 交易规模 */}
      <Box mb={4}>
        <Heading size="sm" mb={2}>
          交易规模
        </Heading>
        <Flex justifyContent="space-between" mb={2}>
          <Text color="gray.600">平均交易规模</Text>
          <Text fontWeight="medium">{formatNumber(sizePatterns.averageSize || 0)} ETH</Text>
        </Flex>
        <Flex justifyContent="space-between">
          <Text color="gray.600">最大交易规模</Text>
          <Text fontWeight="medium">{formatNumber(sizePatterns.maxSize || 0)} ETH</Text>
        </Flex>
      </Box>

      {/* 策略 */}
      {strategies && strategies.length > 0 && (
        <>
          <Divider my={4} />
          <Box>
            <Heading size="sm" mb={2}>
              检测到的策略
            </Heading>
            <Flex wrap="wrap" gap={2}>
              {strategies.map((strategy, index) => (
                <Badge key={index} colorScheme="purple">
                  {strategy}
                </Badge>
              ))}
            </Flex>
          </Box>
        </>
      )}
    </Box>
  );
};

export default TransactionPatternCard;
