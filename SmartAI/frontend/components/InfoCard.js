import React from 'react';
import { Box, Heading, VStack, HStack, Text, Divider } from '@chakra-ui/react';

/**
 * 信息卡片组件，用于显示标题和键值对列表
 * @param {Object} props - 组件属性
 * @param {string} props.title - 卡片标题
 * @param {Array} props.items - 要显示的项目数组，每项包含label和value
 */
const InfoCard = ({ title, items = [] }) => {
  return (
    <Box borderWidth="1px" borderRadius="lg" overflow="hidden" p={4} boxShadow="sm" bg="white">
      <Heading size="md" mb={4}>
        {title}
      </Heading>
      <VStack align="stretch" spacing={3}>
        {items.map((item, index) => (
          <React.Fragment key={index}>
            <HStack justify="space-between">
              <Text color="gray.600">{item.label}</Text>
              <Text fontWeight="medium">{item.value}</Text>
            </HStack>
            {index < items.length - 1 && <Divider />}
          </React.Fragment>
        ))}
      </VStack>
    </Box>
  );
};

export default InfoCard;
