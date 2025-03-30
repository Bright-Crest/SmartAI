import React from 'react';
import { Box, Text, Flex, Progress, Tooltip } from '@chakra-ui/react';

/**
 * 特征值显示组件
 * 用于展示0-1之间的能力值，并附带解释文本
 */
const TraitMeter = ({ label, value = 0.5, description = '' }) => {
  // 确保值在0-1之间
  const safeValue = Math.max(0, Math.min(1, value));

  // 颜色映射
  const getColorScheme = val => {
    if (val >= 0.7) return 'green';
    if (val >= 0.4) return 'blue';
    return 'orange';
  };

  return (
    <Box mb={3}>
      <Flex justify="space-between" mb={1}>
        <Tooltip label={description}>
          <Text fontSize="sm" fontWeight="medium">
            {label}
          </Text>
        </Tooltip>
        <Text fontSize="sm" fontWeight="bold">
          {Math.round(safeValue * 100)}%
        </Text>
      </Flex>
      <Progress
        value={safeValue * 100}
        colorScheme={getColorScheme(safeValue)}
        height="8px"
        borderRadius="full"
      />
    </Box>
  );
};

export default TraitMeter;
