import {
  analyzeAddress,
  getAddressBalance,
  getTokenBalances,
  getTransactionHistory,
  getEthPrice,
} from '../../../../utils/etherscan';

/**
 * 分析区块链地址，确定是否为聪明钱并提供详细分析
 *
 * @param {object} req - HTTP请求对象
 * @param {object} res - HTTP响应对象
 */
export default async function handler(req, res) {
  try {
    // 获取要分析的地址
    const { address } = req.query;

    if (!address || !/^0x[a-fA-F0-9]{40}$/.test(address)) {
      return res.status(400).json({
        error: true,
        message: '无效的以太坊地址',
        code: 'INVALID_ADDRESS',
      });
    }

    console.log(`开始分析地址: ${address}`);

    // 不同方式分析地址
    let analysisResult;

    // 前端方式 - 直接调用Etherscan API
    try {
      // 直接获取实际数据
      const [balance, transactions, tokens] = await Promise.all([
        getAddressBalance(address),
        getTransactionHistory(address),
        getTokenBalances(address),
      ]);

      // 实际持仓数据
      let portfolio = [];

      // 如果成功获取了代币数据
      if (tokens && tokens.length > 0) {
        portfolio = tokens.map(token => ({
          ...token,
          balance: token.balance || 0,
          valueUSD: token.valueUSD || 0,
          allocation: (token.valueUSD || 0) / 100 || 0, // 默认分配比例
        }));
      }

      // 如果获取ETH余额成功，添加ETH到持仓列表
      if (balance && balance > 0) {
        // 获取ETH价格
        const ethPrice = await getEthPrice();
        const ethUsdPrice = parseFloat(ethPrice?.ethusd || '2000');

        portfolio.unshift({
          address: '0x0000000000000000000000000000000000000000',
          name: 'Ethereum',
          symbol: 'ETH',
          decimals: 18,
          balance: balance,
          valueUSD: balance * ethUsdPrice,
          allocation: 0.5, // 将在UI端重新计算
        });
      }

      // 调用分析函数获取完整分析结果
      analysisResult = await analyzeAddress(address);

      // 使用实际持仓数据替换分析结果中的持仓
      if (portfolio.length > 0) {
        analysisResult.portfolio = portfolio;
      }
    } catch (error) {
      console.error('Etherscan API调用失败:', error);
      // 返回基本错误响应
      analysisResult = {
        address,
        isSmartMoney: false,
        reason: 'API调用失败',
        score: 0.3,
        apiCallFailed: true,
        error: true,
        message: error.message || '未知错误',
        portfolio: [], // 不使用默认数据，使用空数组
      };
    }

    // 尝试后端API方式
    try {
      const backendResponse = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/smart-money/analyze/${address}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 15000, // 15秒超时
        }
      );

      if (backendResponse.ok) {
        const backendData = await backendResponse.json();

        // 如果后端返回有效数据，并且前端没有成功获取数据或前端数据质量较低，使用后端数据
        if (
          backendData &&
          !backendData.error &&
          (!analysisResult || analysisResult.apiCallFailed || analysisResult.error)
        ) {
          analysisResult = backendData;

          // 确保持仓数据格式正确
          if (analysisResult && Array.isArray(analysisResult.portfolio)) {
            analysisResult.portfolio = analysisResult.portfolio.map(token => {
              // 确保所有币种都有正确的值
              return {
                ...token,
                balance: token.balance || 0,
                valueUSD: token.valueUSD || 0,
                allocation: (token.valueUSD || 0) / 100 || 0, // 默认分配比例
              };
            });
          }
        }
      }
    } catch (error) {
      console.error('后端API请求失败:', error.message);
      // 继续使用前端分析结果
    }

    // 确保最终结果有标准格式
    if (!analysisResult) {
      analysisResult = {
        address,
        isSmartMoney: false,
        reason: '无法获取数据',
        score: 0.3,
        analysisTimestamp: new Date().toISOString(),
        apiCallFailed: true,
        portfolio: [], // 使用空数组，不使用示例数据
      };
    }

    // 返回分析结果
    return res.status(200).json(analysisResult);
  } catch (error) {
    console.error('处理请求失败:', error);
    return res.status(500).json({
      error: true,
      message: '处理请求失败',
      details: error.message,
    });
  }
}
