import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  Box,
  Container,
  Heading,
  Text,
  Flex,
  Button,
  Link,
  Divider,
  Image,
  Badge,
  SimpleGrid,
  Skeleton,
  SkeletonText,
  IconButton,
  HStack,
  VStack,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Tooltip,
  useClipboard,
  useToast,
  UnorderedList,
  ListItem,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
} from '@chakra-ui/react';
import { ExternalLinkIcon, CopyIcon, SearchIcon, InfoIcon } from '@chakra-ui/icons';
import axios from 'axios';
import Head from 'next/head';
import PortfolioTable from '../../../components/PortfolioTable';
import { formatAddress, formatNumber, formatTimestamp } from '../../../utils/formatting';
import InfoCard from '../../../components/InfoCard';
import TransactionPatternCard from '../../../components/TransactionPatternCard';

// 添加备用数据，以防API调用失败
const FALLBACK_DATA = {
  ethBalance: 'N/A',
  transactions: 0,
  tokenCount: 0,
  isSmartMoney: false,
  portfolioValue: 'N/A',
  profitability: 'N/A',
  riskScore: 'Medium',
  activityLevel: 'Low',
};

// 骨架屏组件，在加载时显示
const SkeletonAnalysis = () => (
  <Box>
    <Flex direction="column" gap={4} mb={6}>
      <Skeleton height="40px" width="200px" />
      <SkeletonText mt="4" noOfLines={3} spacing="4" skeletonHeight="2" />
    </Flex>
    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={6}>
      <Skeleton height="200px" />
      <Skeleton height="200px" />
    </SimpleGrid>
    <Skeleton height="300px" mb={6} />
  </Box>
);

// 错误显示组件
const AnalysisError = ({ error }) => (
  <Box borderWidth="1px" borderRadius="lg" p={6} bg="red.50">
    <Alert status="error" variant="solid" borderRadius="md">
      <AlertIcon />
      <Box>
        <AlertTitle fontSize="lg">分析失败</AlertTitle>
        <AlertDescription>{error?.message || '无法分析地址，请稍后再试'}</AlertDescription>
      </Box>
    </Alert>
    <Text mt={4} color="gray.600">
      可能的原因:
    </Text>
    <UnorderedList mt={2} spacing={2}>
      <ListItem>地址格式无效</ListItem>
      <ListItem>没有足够的交易历史</ListItem>
      <ListItem>服务器暂时不可用</ListItem>
    </UnorderedList>
    <Button
      mt={6}
      colorScheme="blue"
      leftIcon={<SearchIcon />}
      onClick={() => window.location.reload()}
    >
      重试分析
    </Button>
  </Box>
);

// 聪明钱档案组件
const SmartMoneyProfile = ({ analysis }) => {
  return (
    <Box>
      <Flex direction="column" gap={4} mb={6}>
        <Flex justify="space-between" align="center">
          <Heading as="h2" size="lg">
            聪明钱分析
          </Heading>
          <Badge colorScheme="green" fontSize="md" p={2} borderRadius="md">
            聪明钱
          </Badge>
        </Flex>

        <Alert status="success" borderRadius="md">
          <AlertIcon />
          <Box>
            <AlertTitle>智能资金确认</AlertTitle>
            <AlertDescription>
              {analysis.smartMoneyInfo?.reason || '此地址交易模式符合聪明钱特征'}{' '}
              <Text as="span" fontWeight="bold">
                信心指数: {formatNumber(analysis.smartMoneyInfo?.confidence * 100 || 0)}%
              </Text>
            </AlertDescription>
          </Box>
        </Alert>
      </Flex>

      {/* 基本信息和统计数据 */}
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={6}>
        <InfoCard
          title="基本信息"
          items={[
            {
              label: '以太坊余额',
              value: `${analysis.portfolio?.length > 0 ? formatNumber(analysis.portfolio[0]?.balance || 0) : '0'} ETH`,
            },
            { label: '投资者类型', value: analysis.smartMoneyInfo?.investorType || '未知' },
            {
              label: '总交易数',
              value: analysis.transactionPatterns?.overview?.transactionCount || 0,
            },
            { label: '智能评分', value: `${formatNumber(analysis.score * 100 || 0)}%` },
          ]}
        />

        <InfoCard
          title="交易统计"
          items={[
            {
              label: '平均交易金额',
              value: `${formatNumber(analysis.transactionPatterns?.sizePatterns?.averageSize || 0)} ETH`,
            },
            {
              label: '最大交易金额',
              value: `${formatNumber(analysis.transactionPatterns?.sizePatterns?.maxSize || 0)} ETH`,
            },
            {
              label: '交易频率',
              value:
                analysis.transactionPatterns?.frequencyPatterns?.averageFrequency === 'high'
                  ? '高'
                  : analysis.transactionPatterns?.frequencyPatterns?.averageFrequency === 'medium'
                    ? '中'
                    : '低',
            },
            {
              label: '日均交易',
              value: formatNumber(analysis.transactionPatterns?.frequencyPatterns?.dailyAvg || 0),
            },
          ]}
        />
      </SimpleGrid>

      {/* 投资特性 */}
      {analysis.smartMoneyInfo?.traits && (
        <Box mb={6}>
          <Heading as="h3" size="md" mb={4}>
            投资特性
          </Heading>
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
            <TraitBox
              title="入场时机"
              value={analysis.smartMoneyInfo.traits.entryTiming || 0.5}
              description="能否在价格低点买入的能力"
            />
            <TraitBox
              title="退出时机"
              value={analysis.smartMoneyInfo.traits.exitTiming || 0.5}
              description="能否在价格高点卖出的能力"
            />
            <TraitBox
              title="持币能力"
              value={analysis.smartMoneyInfo.traits.hodlStrength || 0.5}
              description="长期持有优质资产的能力"
            />
            <TraitBox
              title="多样化程度"
              value={analysis.smartMoneyInfo.traits.diversification || 0.5}
              description="投资组合多样化的程度"
            />
            <TraitBox
              title="逆势操作"
              value={analysis.smartMoneyInfo.traits.contrarian || 0.5}
              description="逆市场主流做出决策的能力"
            />
          </SimpleGrid>
        </Box>
      )}

      {/* 当前持仓 */}
      {analysis.portfolio && analysis.portfolio.length > 0 && (
        <Box mb={6}>
          <Heading as="h3" size="md" mb={4}>
            当前持仓
          </Heading>
          <PortfolioTable portfolio={analysis.portfolio} />
        </Box>
      )}

      {/* 交易模式 */}
      {analysis.transactionPatterns && (
        <Box mb={6}>
          <Heading as="h3" size="md" mb={4}>
            交易模式
          </Heading>
          <Flex direction="column" gap={4}>
            <Text>
              首次交易:{' '}
              {formatTimestamp(analysis.transactionPatterns?.overview?.firstTransaction || '') ||
                '未知'}
            </Text>
            <Text>
              最后交易:{' '}
              {formatTimestamp(analysis.transactionPatterns?.overview?.lastTransaction || '') ||
                '未知'}
            </Text>
          </Flex>
        </Box>
      )}
    </Box>
  );
};

// 特征显示组件
const TraitBox = ({ title, value, description }) => {
  // 将0-1的值映射到星星数量(1-5)
  const stars = Math.max(1, Math.min(5, Math.round(value * 5)));

  return (
    <Box borderWidth="1px" borderRadius="md" p={4} bg="white">
      <Flex justify="space-between" align="center" mb={2}>
        <Heading size="sm">{title}</Heading>
        <Tooltip label={description} placement="top">
          <InfoIcon color="gray.400" />
        </Tooltip>
      </Flex>
      <Flex color="orange.400">
        {[...Array(5)].map((_, i) => (
          <Box key={i} color={i < stars ? 'orange.400' : 'gray.200'}>
            ★
          </Box>
        ))}
      </Flex>
    </Box>
  );
};

// 非聪明钱展示组件
const NotSmartMoneyProfile = ({ analysis }) => {
  return (
    <Box>
      <Flex direction="column" gap={4} mb={6}>
        <Flex justify="space-between" align="center">
          <Heading as="h2" size="lg">
            地址分析结果
          </Heading>
          <Badge colorScheme="gray" fontSize="md" p={2} borderRadius="md">
            普通钱包
          </Badge>
        </Flex>

        <Alert status="info" borderRadius="md">
          <AlertIcon />
          <Box>
            <AlertTitle>非聪明钱钱包</AlertTitle>
            <AlertDescription>
              {analysis.reason || '此地址不符合聪明钱标准，可能交易历史不足或未展现出特殊交易模式'}
            </AlertDescription>
          </Box>
        </Alert>
      </Flex>

      {/* 仍然显示基本信息 */}
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={6}>
        <InfoCard
          title="基本信息"
          items={[
            {
              label: '以太坊余额',
              value: `${analysis.portfolio?.length > 0 ? formatNumber(analysis.portfolio[0]?.balance || 0) : '0'} ETH`,
            },
            {
              label: '交易数量',
              value: analysis.transactionPatterns?.overview?.transactionCount || 0,
            },
            {
              label: '首次交易',
              value:
                formatTimestamp(analysis.transactionPatterns?.overview?.firstTransaction || '') ||
                '未知',
            },
            {
              label: '最近交易',
              value:
                formatTimestamp(analysis.transactionPatterns?.overview?.lastTransaction || '') ||
                '未知',
            },
          ]}
        />

        <InfoCard
          title="交易统计"
          items={[
            {
              label: '平均交易金额',
              value: `${formatNumber(analysis.transactionPatterns?.sizePatterns?.averageSize || 0)} ETH`,
            },
            {
              label: '最大交易金额',
              value: `${formatNumber(analysis.transactionPatterns?.sizePatterns?.maxSize || 0)} ETH`,
            },
            {
              label: '交易频率',
              value:
                analysis.transactionPatterns?.frequencyPatterns?.averageFrequency === 'high'
                  ? '高'
                  : analysis.transactionPatterns?.frequencyPatterns?.averageFrequency === 'medium'
                    ? '中'
                    : '低',
            },
            {
              label: '日均交易',
              value: formatNumber(analysis.transactionPatterns?.frequencyPatterns?.dailyAvg || 0),
            },
          ]}
        />
      </SimpleGrid>

      {/* 显示持仓 */}
      {analysis.portfolio && analysis.portfolio.length > 0 && (
        <Box mb={6}>
          <Heading as="h3" size="md" mb={4}>
            当前持仓
          </Heading>
          <PortfolioTable portfolio={analysis.portfolio} />
        </Box>
      )}
    </Box>
  );
};

// 分析结果区域
const AnalysisSection = ({ analysis, isLoading, error }) => {
  if (isLoading) {
    return <SkeletonAnalysis />;
  }

  if (error) {
    return <AnalysisError error={error} />;
  }

  // 修复: 如果有分析结果但不是聪明钱，也应该显示结果
  if (!analysis || analysis.error === true) {
    return <AnalysisError />;
  }

  // 如果是分析结果表明不是聪明钱，但API调用成功，应该显示正常结果
  return (
    <Box>
      {analysis.isSmartMoney ? (
        <SmartMoneyProfile analysis={analysis} />
      ) : (
        <NotSmartMoneyProfile analysis={analysis} />
      )}
    </Box>
  );
};

// 地址详情页面
export default function AddressDetail() {
  const router = useRouter();
  const { address } = router.query;
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const { hasCopied, onCopy } = useClipboard(address || '');
  const toast = useToast();

  // 复制地址到剪贴板
  const handleCopy = () => {
    onCopy();
    toast({
      title: '地址已复制',
      status: 'success',
      duration: 2000,
      isClosable: true,
    });
  };

  // 获取分析数据
  useEffect(() => {
    if (!address) return;

    const fetchAnalysis = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await axios.get(`/api/smart-money/analyze/${address}`);
        setAnalysis(response.data);
      } catch (err) {
        console.error('获取分析数据失败:', err);
        setError({
          message: err.response?.data?.message || '分析请求失败，请稍后再试',
          code: err.response?.data?.code || 'UNKNOWN_ERROR',
        });
        setAnalysis(FALLBACK_DATA);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalysis();
  }, [address]);

  if (!address) {
    return (
      <Container maxW="container.xl" py={8}>
        加载中...
      </Container>
    );
  }

  return (
    <>
      <Head>
        <title>{address ? `${formatAddress(address)} | 聪明钱分析` : '聪明钱分析'}</title>
      </Head>

      <Container maxW="container.xl" py={8}>
        <Box mb={8}>
          <Heading as="h1" size="xl" mb={2}>
            地址分析
          </Heading>
          <Flex align="center" mb={6}>
            <Text fontSize="lg" fontWeight="medium" mr={2}>
              {formatAddress(address, 10, 8)}
            </Text>
            <IconButton
              aria-label="复制地址"
              icon={<CopyIcon />}
              size="sm"
              onClick={handleCopy}
              mr={2}
            />
            <Link href={`https://etherscan.io/address/${address}`} isExternal>
              <IconButton aria-label="在Etherscan查看" icon={<ExternalLinkIcon />} size="sm" />
            </Link>
          </Flex>
          <Divider mb={6} />

          {/* 分析结果部分 */}
          <AnalysisSection analysis={analysis} isLoading={isLoading} error={error} />
        </Box>
      </Container>
    </>
  );
}
