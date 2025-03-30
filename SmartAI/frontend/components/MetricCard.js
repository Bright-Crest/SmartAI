import React from 'react';
import {
  Box,
  Text,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Flex,
  Icon,
  Tooltip,
} from '@chakra-ui/react';
import { InfoIcon } from '@chakra-ui/icons';

/**
 * 指标卡片组件
 * 用于展示数值型指标以及相关解释
 */
const MetricCard = ({
  title,
  value,
  description = '',
  helpText = '',
  change = null,
  colorScheme = 'blue',
  format = val => val,
}) => {
  // 根据colorScheme获取对应的颜色
  const getColor = () => {
    switch (colorScheme) {
      case 'green':
        return 'green.500';
      case 'red':
        return 'red.500';
      case 'yellow':
        return 'yellow.500';
      case 'orange':
        return 'orange.500';
      case 'purple':
        return 'purple.500';
      default:
        return 'blue.500';
    }
  };

  // 处理变化百分比的颜色
  const getChangeColor = () => {
    if (change === null) return 'gray.500';
    return change >= 0 ? 'green.500' : 'red.500';
  };

  // 格式化变化百分比
  const formatChange = () => {
    if (change === null) return '';
    const prefix = change >= 0 ? '+' : '';
    return `${prefix}${change}%`;
  };

  return (
    <Box borderWidth="1px" borderRadius="lg" p={4} boxShadow="sm" bg="white">
      <Stat>
        <Flex justify="space-between" align="center">
          <StatLabel fontSize="sm" color="gray.600">
            {title}
          </StatLabel>
          {description && (
            <Tooltip label={description} placement="top">
              <Icon as={InfoIcon} color="gray.400" boxSize={4} />
            </Tooltip>
          )}
        </Flex>
        <StatNumber fontSize="xl" fontWeight="bold" color={getColor()} mt={1}>
          {format(value)}
        </StatNumber>
        {(helpText || change !== null) && (
          <StatHelpText mb={0} fontSize="xs">
            {helpText}
            {change !== null && (
              <Text as="span" color={getChangeColor()} ml={1}>
                {formatChange()}
              </Text>
            )}
          </StatHelpText>
        )}
      </Stat>
    </Box>
  );
};

export default MetricCard;
