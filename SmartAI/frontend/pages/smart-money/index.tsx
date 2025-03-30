import React, { useState } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import { 
  Box, 
  Container, 
  Typography, 
  TextField, 
  Button, 
  Paper, 
  Grid,
  Card,
  CardContent,
  CardActions,
  Divider,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip
} from '@mui/material';
import {
  Search as SearchIcon,
  TrendingUp as TrendingUpIcon,
  Insights as InsightsIcon,
  Analytics as AnalyticsIcon,
  AccountCircle as AccountCircleIcon
} from '@mui/icons-material';
import { exampleAddresses } from '../../utils/smartMoneyExamples';

// 示例地址标签映射
const addressLabels: Record<string, {name: string, type: string}> = {
  '0x28c6c06298d514db089934071355e5743bf21d60': {
    name: '价值投资者',
    type: '长期持有者'
  },
  '0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0': {
    name: 'DeFi策略师',
    type: '收益农民'
  },
  '0xdac17f958d2ee523a2206206994597c13d831ec7': {
    name: '交易员',
    type: '市场择时者'
  }
};

// 格式化地址
const formatAddress = (address: string): string => {
  return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
};

export default function SmartMoneyHome() {
  const router = useRouter();
  const [address, setAddress] = useState('');
  const [error, setError] = useState<string | null>(null);
  
  // 处理地址分析
  const handleAnalyzeAddress = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!address || address.trim() === '') {
      setError('请输入有效的区块链地址');
      return;
    }
    
    // 简单的地址验证（以太坊类地址格式）
    if (!/^0x[a-fA-F0-9]{40}$/.test(address)) {
      setError('地址格式无效，请输入有效的区块链地址');
      return;
    }
    
    // 导航到分析页面
    router.push(`/smart-money/address/${address}`);
  };
  
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 8 }}>
      <Head>
        <title>聪明钱分析 | SmartAI</title>
        <meta name="description" content="识别和追踪区块链上的聪明钱，获取交易洞察" />
      </Head>
      
      {/* 页面标题 */}
      <Box sx={{ textAlign: 'center', my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          聪明钱分析
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: 700, mx: 'auto' }}>
          识别、追踪并分析区块链上的"聪明钱"地址，学习成功投资者的交易策略
        </Typography>
      </Box>
      
      {/* 地址搜索表单 */}
      <Paper 
        component="form" 
        elevation={3} 
        sx={{ p: 4, mb: 6, maxWidth: 800, mx: 'auto' }}
        onSubmit={handleAnalyzeAddress}
      >
        <Typography variant="h5" gutterBottom>
          分析区块链地址
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          输入区块链地址，发现它是否属于"聪明钱"，并获取详细分析
        </Typography>
        
        <Box sx={{ display: 'flex', mb: 2 }}>
          <TextField
            fullWidth
            variant="outlined"
            label="区块链地址"
            placeholder="0x..."
            value={address}
            onChange={(e) => {
              setAddress(e.target.value);
              setError(null);
            }}
            error={!!error}
            helperText={error}
            sx={{ mr: 2 }}
          />
          <Button 
            type="submit" 
            variant="contained" 
            size="large" 
            startIcon={<SearchIcon />}
          >
            分析
          </Button>
        </Box>
        
        <Typography variant="body2" color="text.secondary">
          示例: 0x28c6c06298d514db089934071355e5743bf21d60
        </Typography>
      </Paper>
      
      {/* 功能介绍卡片 */}
      <Grid container spacing={4} sx={{ mb: 6 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SearchIcon color="primary" sx={{ fontSize: 30, mr: 1 }} />
                <Typography variant="h6">
                  识别聪明钱
                </Typography>
              </Box>
              <Typography variant="body2" paragraph>
                使用先进的算法和多维度指标，识别区块链上的聪明钱地址，了解他们的投资特征和交易行为模式。
              </Typography>
              <Typography variant="body2">
                我们分析：
              </Typography>
              <ul>
                <li>投资收益表现</li>
                <li>风险管理能力</li>
                <li>市场时机把握</li>
                <li>投资组合管理</li>
              </ul>
            </CardContent>
            <CardActions>
              <Button size="small" component={Link} href="/smart-money/methodology">
                了解识别方法
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon color="primary" sx={{ fontSize: 30, mr: 1 }} />
                <Typography variant="h6">
                  追踪投资组合
                </Typography>
              </Box>
              <Typography variant="body2" paragraph>
                追踪聪明钱的投资组合变化，了解他们持有哪些代币，如何分配资产，以及如何管理风险。
              </Typography>
              <Typography variant="body2">
                追踪内容：
              </Typography>
              <ul>
                <li>当前持仓分布</li>
                <li>资产配置变化</li>
                <li>投资组合多样化</li>
                <li>风险暴露分析</li>
              </ul>
            </CardContent>
            <CardActions>
              <Button size="small" component={Link} href="/smart-money/leaderboard">
                查看聪明钱排行榜
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <InsightsIcon color="primary" sx={{ fontSize: 30, mr: 1 }} />
                <Typography variant="h6">
                  分析交易模式
                </Typography>
              </Box>
              <Typography variant="body2" paragraph>
                深入分析聪明钱的交易模式和策略，包括他们的入场/退出时机、持仓时间、头寸管理等。
              </Typography>
              <Typography variant="body2">
                分析维度：
              </Typography>
              <ul>
                <li>交易时机模式</li>
                <li>价格入场点特征</li>
                <li>头寸规模管理</li>
                <li>交易频率分析</li>
              </ul>
            </CardContent>
            <CardActions>
              <Button size="small" component={Link} href="/smart-money/strategies">
                了解交易策略
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>
      
      {/* 示例地址列表 */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h5" gutterBottom>
          示例聪明钱地址
        </Typography>
        <Divider sx={{ mb: 2 }} />
        
        <Alert severity="info" sx={{ mb: 3 }}>
          以下示例地址可用于离线演示，代表不同类型的聪明钱投资者
        </Alert>
        
        <List>
          {exampleAddresses.map((addr) => (
            <ListItem key={addr} 
              sx={{ 
                border: '1px solid #e0e0e0', 
                borderRadius: 1, 
                mb: 1,
                '&:hover': {
                  backgroundColor: '#f5f5f5'
                }
              }}
              component={Link}
              href={`/smart-money/address/${addr}`}
            >
              <ListItemIcon>
                <AccountCircleIcon color="primary" />
              </ListItemIcon>
              <ListItemText 
                primary={addressLabels[addr]?.name || '聪明钱地址'} 
                secondary={formatAddress(addr)}
              />
              <Chip 
                label={addressLabels[addr]?.type || '聪明钱'} 
                size="small" 
                color="primary" 
                variant="outlined" 
              />
            </ListItem>
          ))}
        </List>
      </Paper>
      
      {/* 底部说明 */}
      <Typography variant="body2" color="text.secondary" sx={{ mt: 4, textAlign: 'center' }}>
        注意：聪明钱分析仅供参考，不构成投资建议。请自行进行研究并承担投资风险。
      </Typography>
    </Container>
  );
} 