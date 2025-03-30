export class DataOptimizer {
  // 优化智能投资评分数据
  optimizeSmartScore(data: any) {
    return {
      ...data,
      score: Math.round(data.score),
      confidence: Number(data.confidence.toFixed(2)),
      factors: Object.entries(data.factors).reduce((acc, [key, value]) => ({
        ...acc,
        [key]: Number((value as number).toFixed(2))
      }), {}),
      smartMoneyTraits: {
        ...data.smartMoneyTraits,
        tradingVolume: Number(data.smartMoneyTraits.tradingVolume.toFixed(2)),
        averageProfitPerTrade: Number(data.smartMoneyTraits.averageProfitPerTrade.toFixed(2))
      }
    };
  }

  // 优化投资行为指标数据
  optimizeMetrics(data: any) {
    return {
      ...data,
      metrics: {
        ...data.metrics,
        winRate: Number(data.metrics.winRate.toFixed(2)),
        averageReturn: Number(data.metrics.averageReturn.toFixed(2)),
        maxDrawdown: Number(data.metrics.maxDrawdown.toFixed(2)),
        sharpeRatio: Number(data.metrics.sharpeRatio.toFixed(2))
      },
      professionalMetrics: Object.entries(data.professionalMetrics).reduce((acc, [key, value]) => ({
        ...acc,
        [key]: Number((value as number).toFixed(2))
      }), {}),
      tradingStrategy: {
        ...data.tradingStrategy,
        entryTiming: Number(data.tradingStrategy.entryTiming.toFixed(2)),
        exitTiming: Number(data.tradingStrategy.exitTiming.toFixed(2)),
        positionSizing: Number(data.tradingStrategy.positionSizing.toFixed(2)),
        riskRewardRatio: Number(data.tradingStrategy.riskRewardRatio.toFixed(2))
      }
    };
  }

  // 优化地址详情数据
  optimizeDetails(data: any) {
    return {
      ...data,
      profile: {
        ...data.profile,
        totalVolume: Number(data.profile.totalVolume.toFixed(2))
      }
    };
  }

  // 优化交易历史数据
  optimizeTransactions(data: any) {
    return {
      ...data,
      transactions: data.transactions.map((tx: any) => ({
        ...tx,
        amount: Number(tx.amount.toFixed(4)),
        price: Number(tx.price.toFixed(2)),
        value: Number(tx.value.toFixed(2)),
        profitLoss: Number(tx.profitLoss.toFixed(2)),
        analysis: {
          ...tx.analysis,
          marketCap: Number(tx.analysis.marketCap.toFixed(2)),
          volume24h: Number(tx.analysis.volume24h.toFixed(2)),
          priceChange24h: Number(tx.analysis.priceChange24h.toFixed(2)),
          liquidity: Number(tx.analysis.liquidity.toFixed(2)),
          slippage: Number(tx.analysis.slippage.toFixed(3)),
          gasPrice: Number(tx.analysis.gasPrice.toFixed(2))
        }
      })),
      statistics: {
        ...data.statistics,
        totalVolume: Number(data.statistics.totalVolume.toFixed(2)),
        averageTradeSize: Number(data.statistics.averageTradeSize.toFixed(2)),
        largestTrade: Number(data.statistics.largestTrade.toFixed(2)),
        averageSlippage: Number(data.statistics.averageSlippage.toFixed(3))
      }
    };
  }

  // 优化相似地址数据
  optimizeSimilarAddresses(data: any) {
    return {
      ...data,
      similarAddresses: data.similarAddresses.map((addr: any) => ({
        ...addr,
        similarity: Number(addr.similarity.toFixed(2)),
        similarityAnalysis: Object.entries(addr.similarityAnalysis).reduce((acc, [key, value]) => ({
          ...acc,
          [key]: Number((value as number).toFixed(2))
        }), {})
      }))
    };
  }
}

export const dataOptimizer = new DataOptimizer(); 