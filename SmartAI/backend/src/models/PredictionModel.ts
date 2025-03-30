import * as tf from '@tensorflow/tfjs-node';
import { loggerWinston as logger } from '../utils/logger';
import { cache } from '../utils/cache';

/**
 * 智能资金行为预测模型
 * 使用TensorFlow.js构建的深度学习模型，用于预测智能资金的未来行为
 */
export class PredictionModel {
  private static instance: PredictionModel;
  private model: tf.LayersModel | null = null;
  private isModelLoaded: boolean = false;
  private readonly modelPath: string = './models/smart-money-prediction';
  private readonly featureColumns: string[] = [
    'entryTiming', 'exitTiming', 'hodlStrength', 'diversification', 'contrarian',
    'totalTrades', 'avgTradeValue', 'avgHoldingPeriod', 'winRate', 'profitFactor',
    'sharpeRatio', 'marketTrend', 'tokenMarketCap', 'tokenVolume', 'tokenPriceChange'
  ];
  
  /**
   * 获取单例实例
   */
  public static getInstance(): PredictionModel {
    if (!PredictionModel.instance) {
      PredictionModel.instance = new PredictionModel();
    }
    return PredictionModel.instance;
  }
  
  /**
   * 私有构造函数
   */
  private constructor() {}
  
  /**
   * 初始化并加载模型
   */
  public async init(): Promise<void> {
    try {
      // 尝试加载已训练的模型
      try {
        this.model = await tf.loadLayersModel(`file://${this.modelPath}/model.json`);
        this.isModelLoaded = true;
        logger.info('预测模型加载成功');
      } catch (error) {
        logger.warn('无法加载已有模型，将创建新模型', { error });
        this.model = this.createModel();
      }
    } catch (error) {
      logger.error('初始化预测模型失败', { error });
      throw error;
    }
  }
  
  /**
   * 创建模型架构
   * @private
   */
  private createModel(): tf.LayersModel {
    const numFeatures = this.featureColumns.length;
    
    // 创建序列模型
    const model = tf.sequential();
    
    // 添加LSTM层，用于捕捉时间序列特征
    model.add(tf.layers.lstm({
      units: 64,
      returnSequences: true,
      inputShape: [30, numFeatures], // 30天的历史数据
      activation: 'relu'
    }));
    
    // 添加Dropout层，防止过拟合
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // 添加第二个LSTM层
    model.add(tf.layers.lstm({
      units: 32,
      returnSequences: false,
      activation: 'relu'
    }));
    
    // 添加Dropout层
    model.add(tf.layers.dropout({ rate: 0.2 }));
    
    // 添加Dense层，用于输出预测结果
    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu'
    }));
    
    // 输出层 - 多目标预测（价格变化、交易概率、持有时间）
    model.add(tf.layers.dense({
      units: 3, // 价格变化百分比, 交易可能性, 预计持有时间
      activation: 'linear'
    }));
    
    // 编译模型
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse', 'mae']
    });
    
    logger.info('创建新的预测模型');
    
    return model;
  }
  
  /**
   * 准备训练数据
   * @param historicalData 历史交易数据
   * @private
   */
  private prepareTrainingData(historicalData: any[]): { xs: tf.Tensor, ys: tf.Tensor } {
    const windowSize = 30; // 使用30天的数据预测
    const xsData: number[][][] = [];
    const ysData: number[][] = [];
    
    for (let i = windowSize; i < historicalData.length; i++) {
      const window = historicalData.slice(i - windowSize, i);
      const target = historicalData[i];
      
      // 提取特征
      const windowFeatures = window.map(day => {
        return this.featureColumns.map(feature => {
          return day[feature] || 0;
        });
      });
      
      // 提取目标值 - 价格变化，交易可能性，持有时间
      const targetValues = [
        target.priceChangePercent || 0,
        target.tradeProbability || 0,
        target.holdingDuration || 0
      ];
      
      xsData.push(windowFeatures);
      ysData.push(targetValues);
    }
    
    return {
      xs: tf.tensor3d(xsData, [xsData.length, windowSize, this.featureColumns.length]),
      ys: tf.tensor2d(ysData, [ysData.length, 3])
    };
  }
  
  /**
   * 训练模型
   * @param historicalData 历史交易数据
   * @param epochs 训练轮数
   * @param batchSize 批量大小
   */
  public async train(
    historicalData: any[],
    epochs: number = 50,
    batchSize: number = 32
  ): Promise<tf.History> {
    if (!this.model) {
      throw new Error('模型未初始化');
    }
    
    logger.info('开始训练预测模型', { dataSize: historicalData.length, epochs, batchSize });
    
    // 准备训练数据
    const { xs, ys } = this.prepareTrainingData(historicalData);
    
    // 训练模型
    try {
      const history = await this.model.fit(xs, ys, {
        epochs,
        batchSize,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            logger.debug(`轮次 ${epoch + 1}/${epochs}, 损失: ${logs?.loss.toFixed(4)}, 验证损失: ${logs?.val_loss.toFixed(4)}`);
          }
        }
      });
      
      // 保存模型
      try {
        await this.model.save(`file://${this.modelPath}`);
        logger.info('预测模型保存成功');
      } catch (error) {
        logger.error('保存预测模型失败', { error });
      }
      
      this.isModelLoaded = true;
      logger.info('预测模型训练完成');
      
      // 释放训练数据
      xs.dispose();
      ys.dispose();
      
      return history;
    } catch (error) {
      logger.error('训练预测模型失败', { error });
      throw error;
    }
  }
  
  /**
   * 预测智能资金未来行为
   * @param address 智能资金地址
   * @param recentData 最近的交易数据
   */
  public async predict(address: string, recentData: any[]): Promise<any> {
    if (!this.isModelLoaded || !this.model) {
      throw new Error('模型未加载或初始化');
    }
    
    // 检查缓存
    const cacheKey = `prediction:${address}`;
    const cachedPrediction = cache.get<any>(cacheKey);
    
    if (cachedPrediction) {
      logger.debug('从缓存获取预测结果', { address });
      return cachedPrediction;
    }
    
    if (recentData.length < 30) {
      logger.warn('历史数据不足，无法进行准确预测', { address, dataPoints: recentData.length });
      
      // 返回简化预测
      const simplePrediction = {
        priceChangePrediction: 0,
        tradeProbability: 0.5,
        holdingDuration: 7,
        confidence: 0.3,
        nextActionEstimate: '无法准确预测，数据不足',
        predictionTimestamp: new Date()
      };
      
      cache.set(cacheKey, simplePrediction, 1800); // 缓存30分钟
      return simplePrediction;
    }
    
    logger.info('开始预测智能资金行为', { address });
    
    try {
      // 准备预测数据
      const recentWindow = recentData.slice(-30); // 取最近30天的数据
      
      const inputFeatures = recentWindow.map(day => {
        return this.featureColumns.map(feature => {
          return day[feature] || 0;
        });
      });
      
      // 转换为张量
      const inputTensor = tf.tensor3d([inputFeatures], [1, 30, this.featureColumns.length]);
      
      // 执行预测
      const predictionTensor = this.model.predict(inputTensor) as tf.Tensor;
      const predictionArray = await predictionTensor.array() as number[][];
      
      // 提取预测结果
      const [priceChangePrediction, tradeProbability, holdingDuration] = predictionArray[0];
      
      // 解释预测结果
      let nextActionEstimate = '';
      
      if (tradeProbability > 0.7) {
        if (priceChangePrediction > 0) {
          nextActionEstimate = '很可能在近期买入新资产';
        } else {
          nextActionEstimate = '很可能在近期卖出持有资产';
        }
      } else if (tradeProbability > 0.4) {
        nextActionEstimate = '可能会观望或小规模交易';
      } else {
        nextActionEstimate = '很可能继续持有当前资产';
      }
      
      // 构建预测结果
      const prediction = {
        priceChangePrediction,
        tradeProbability,
        holdingDuration: Math.max(1, Math.round(holdingDuration)), // 确保是正整数
        confidence: this.calculateConfidence(recentData),
        nextActionEstimate,
        predictionTimestamp: new Date()
      };
      
      // 缓存预测结果
      cache.set(cacheKey, prediction, 1800); // 缓存30分钟
      
      // 释放张量
      inputTensor.dispose();
      predictionTensor.dispose();
      
      logger.info('预测完成', { address });
      
      return prediction;
    } catch (error) {
      logger.error('预测智能资金行为失败', { error, address });
      throw error;
    }
  }
  
  /**
   * 计算预测置信度
   * @param recentData 近期数据
   * @private
   */
  private calculateConfidence(recentData: any[]): number {
    // 数据点越多，置信度越高
    const dataQuality = Math.min(1, recentData.length / 60);
    
    // 数据一致性检查
    let consistencyScore = 0;
    
    if (recentData.length > 0) {
      // 检查历史行为一致性
      const behaviourConsistency = this.checkBehaviourConsistency(recentData);
      consistencyScore = behaviourConsistency;
    }
    
    // 计算最终置信度，范围0-1
    return Math.min(1, (dataQuality * 0.6 + consistencyScore * 0.4));
  }
  
  /**
   * 检查行为一致性
   * @param data 历史数据
   * @private
   */
  private checkBehaviourConsistency(data: any[]): number {
    // 简化的一致性检查
    // 实际实现中应该使用更复杂的算法来评估行为模式
    
    if (data.length < 5) return 0.5; // 默认中等置信度
    
    // 计算关键指标的标准差，标准差越小表示行为越一致
    const tradeValues = data.map(d => d.tradeValue || 0);
    const holdingPeriods = data.map(d => d.holdingPeriod || 0);
    
    const tradeSd = this.standardDeviation(tradeValues);
    const holdingSd = this.standardDeviation(holdingPeriods);
    
    // 归一化标准差（越低越好）
    const normTradeSd = Math.exp(-tradeSd / (Math.max(...tradeValues) / 4));
    const normHoldingSd = Math.exp(-holdingSd / (Math.max(...holdingPeriods) / 4));
    
    // 合并各指标
    return (normTradeSd + normHoldingSd) / 2;
  }
  
  /**
   * 计算标准差
   * @param values 数值数组
   * @private
   */
  private standardDeviation(values: number[]): number {
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squareDiffs = values.map(value => Math.pow(value - avg, 2));
    const avgSquareDiff = squareDiffs.reduce((sum, val) => sum + val, 0) / squareDiffs.length;
    return Math.sqrt(avgSquareDiff);
  }
  
  /**
   * 评估模型性能
   * @param testData 测试数据
   */
  public async evaluate(testData: any[]): Promise<any> {
    if (!this.isModelLoaded || !this.model) {
      throw new Error('模型未加载或初始化');
    }
    
    if (testData.length < 31) { // 需要至少31条数据：30天的历史+1天的真实结果
      throw new Error('测试数据不足');
    }
    
    // 准备测试数据
    const { xs, ys } = this.prepareTrainingData(testData);
    
    // 评估模型
    const evaluation = await this.model.evaluate(xs, ys);
    
    // 提取评估结果
    const loss = Array.isArray(evaluation) ? evaluation[0].dataSync()[0] : evaluation.dataSync()[0];
    
    // 构建评估报告
    const report = {
      loss,
      mse: loss, // 均方误差与损失相同（因为我们使用MSE作为损失函数）
      rmse: Math.sqrt(loss), // 均方根误差
      testSamples: testData.length - 30,
      evaluationDate: new Date()
    };
    
    // 释放张量
    xs.dispose();
    ys.dispose();
    if (Array.isArray(evaluation)) {
      evaluation.forEach(tensor => tensor.dispose());
    } else {
      evaluation.dispose();
    }
    
    return report;
  }
} 