import { loggerWinston as logger } from '../utils/logger';
import { SmartMoneyTracker } from '../core/SmartMoneyTracker';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { NotificationService } from './NotificationService';
import { cache } from '../utils/cache';
import { EventEmitter } from 'events';

// 事件类型
enum AlertType {
  NEW_TRANSACTION = 'new_transaction',
  LARGE_TRANSACTION = 'large_transaction',
  PORTFOLIO_CHANGE = 'portfolio_change',
  PRICE_MOVEMENT = 'price_movement',
  SMART_MONEY_DETECTION = 'smart_money_detection',
  COMMON_BEHAVIOR = 'common_behavior',
  TRADING_OPPORTUNITY = 'trading_opportunity'
}

// 提醒优先级
enum AlertPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// 提醒接口
interface Alert {
  id: string;
  type: AlertType;
  priority: AlertPriority;
  title: string;
  description: string;
  address?: string;
  relatedAddresses?: string[];
  token?: string;
  transactionHash?: string;
  amount?: number;
  valueUSD?: number;
  priceChange?: number;
  timestamp: Date;
  read: boolean;
  dismissed: boolean;
  data?: Record<string, any>;
}

// 提醒订阅设置
interface AlertSubscription {
  userId: string;
  alertTypes: AlertType[];
  minPriority: AlertPriority;
  addresses?: string[];
  tokens?: string[];
  notificationChannels: {
    email: boolean;
    push: boolean;
    inApp: boolean;
    webhook?: string;
  };
  thresholds: {
    transactionValue: number; // 大额交易阈值（USD）
    portfolioChangePercent: number; // 投资组合变化阈值
    priceChangePercent: number; // 价格变化阈值
  };
}

/**
 * 实时提醒服务
 * 监控智能资金活动并生成提醒
 */
export class AlertService extends EventEmitter {
  private static instance: AlertService;
  private isRunning: boolean = false;
  private monitoringIntervalId: NodeJS.Timeout | null = null;
  private readonly MONITORING_INTERVAL = 5 * 60 * 1000; // 5分钟
  
  // 提醒缓存 - 避免重复提醒
  private alertCache: Map<string, boolean> = new Map();
  
  // 订阅列表 - 实际应用中应该从数据库加载
  private subscriptions: Map<string, AlertSubscription> = new Map();
  
  /**
   * 获取单例实例
   */
  public static getInstance(): AlertService {
    if (!AlertService.instance) {
      AlertService.instance = new AlertService();
    }
    return AlertService.instance;
  }
  
  /**
   * 私有构造函数
   */
  private constructor() {
    super();
    
    // 监听自己发出的提醒事件
    this.on('alert', this.handleAlert.bind(this));
  }
  
  /**
   * 初始化服务
   */
  public async init(): Promise<void> {
    try {
      logger.info('初始化实时提醒服务');
      
      // 加载订阅配置（实际项目中应从数据库加载）
      await this.loadSubscriptions();
      
      logger.info('实时提醒服务初始化完成');
    } catch (error) {
      logger.error('初始化实时提醒服务失败', { error });
      throw error;
    }
  }
  
  /**
   * 加载订阅配置
   * @private
   */
  private async loadSubscriptions(): Promise<void> {
    try {
      // 示例：从数据库加载用户订阅设置
      // 实际项目中应实现数据库查询
      
      // 模拟数据
      const mockSubscriptions: AlertSubscription[] = [
        {
          userId: 'user1',
          alertTypes: [
            AlertType.LARGE_TRANSACTION,
            AlertType.PORTFOLIO_CHANGE,
            AlertType.SMART_MONEY_DETECTION
          ],
          minPriority: AlertPriority.MEDIUM,
          addresses: ['0x123...', '0x456...'],
          notificationChannels: {
            email: true,
            push: true,
            inApp: true
          },
          thresholds: {
            transactionValue: 100000, // 10万美元
            portfolioChangePercent: 5, // 5%
            priceChangePercent: 10 // 10%
          }
        }
      ];
      
      // 清空并重新加载
      this.subscriptions.clear();
      
      mockSubscriptions.forEach(sub => {
        this.subscriptions.set(sub.userId, sub);
      });
      
      logger.info('加载提醒订阅成功', { count: this.subscriptions.size });
    } catch (error) {
      logger.error('加载提醒订阅失败', { error });
    }
  }
  
  /**
   * 启动监控
   */
  public start(): void {
    if (this.isRunning) {
      logger.warn('实时提醒服务已在运行');
      return;
    }
    
    logger.info('启动实时提醒服务');
    
    // 立即运行一次
    this.monitorSmartMoneyActivity();
    
    // 设置定时运行
    this.monitoringIntervalId = setInterval(
      () => this.monitorSmartMoneyActivity(),
      this.MONITORING_INTERVAL
    );
    
    this.isRunning = true;
  }
  
  /**
   * 停止监控
   */
  public stop(): void {
    if (!this.isRunning || !this.monitoringIntervalId) {
      logger.warn('实时提醒服务未在运行');
      return;
    }
    
    logger.info('停止实时提醒服务');
    
    clearInterval(this.monitoringIntervalId);
    this.monitoringIntervalId = null;
    this.isRunning = false;
  }
  
  /**
   * 监控智能资金活动
   * @private
   */
  private async monitorSmartMoneyActivity(): Promise<void> {
    try {
      logger.debug('开始监控智能资金活动');
      
      // 1. 监控最近活跃的智能资金地址
      await this.monitorRecentActivity();
      
      // 2. 监控被关注的特定地址
      await this.monitorWatchedAddresses();
      
      // 3. 监控大额交易
      await this.monitorLargeTransactions();
      
      // 4. 监控投资组合变化
      await this.monitorPortfolioChanges();
      
      // 5. 监控共同行为
      await this.monitorCommonBehavior();
      
      logger.debug('完成监控智能资金活动');
    } catch (error) {
      logger.error('监控智能资金活动失败', { error });
    }
  }
  
  /**
   * 监控最近活跃的智能资金
   * @private
   */
  private async monitorRecentActivity(): Promise<void> {
    try {
      // 获取最近活跃的智能资金
      const recentlyActive = await SmartMoneyTracker.getRecentlyActive(1, 20); // 最近1天内活跃的前20个地址
      
      for (const profile of recentlyActive) {
        // 检查是否有新交易
        const hasNewTransactions = await this.checkForNewTransactions(profile.address);
        
        if (hasNewTransactions) {
          // 生成提醒
          const alert: Alert = {
            id: `new_tx_${profile.address}_${Date.now()}`,
            type: AlertType.NEW_TRANSACTION,
            priority: AlertPriority.MEDIUM,
            title: '智能资金新交易',
            description: `智能资金地址 ${profile.address.substring(0, 8)}... 有新的交易活动`,
            address: profile.address,
            timestamp: new Date(),
            read: false,
            dismissed: false,
            data: {
              investorType: profile.investorTypes[0],
              score: profile.score.overall
            }
          };
          
          // 发出提醒事件
          this.emit('alert', alert);
        }
      }
    } catch (error) {
      logger.error('监控最近活跃的智能资金失败', { error });
    }
  }
  
  /**
   * 检查地址是否有新交易
   * @param address 地址
   * @private
   */
  private async checkForNewTransactions(address: string): Promise<boolean> {
    try {
      // 检查缓存，避免重复提醒
      const cacheKey = `last_tx_check:${address}`;
      const lastCheck = cache.get<number>(cacheKey);
      
      if (lastCheck) {
        // 模拟检查逻辑
        // 实际应该查询区块链API获取最近交易
        const mockHasNewTx = Math.random() > 0.7; // 30%概率有新交易
        
        if (mockHasNewTx) {
          // 更新缓存
          cache.set(cacheKey, Date.now(), 3600); // 1小时
          return true;
        }
      } else {
        // 首次检查，记录时间戳
        cache.set(cacheKey, Date.now(), 3600); // 1小时
      }
      
      return false;
    } catch (error) {
      logger.error('检查新交易失败', { error, address });
      return false;
    }
  }
  
  /**
   * 监控被关注的特定地址
   * @private
   */
  private async monitorWatchedAddresses(): Promise<void> {
    try {
      // 获取所有被关注的地址
      const watchedAddresses = new Set<string>();
      
      // 从所有订阅中提取被关注的地址
      this.subscriptions.forEach(sub => {
        if (sub.addresses && sub.addresses.length > 0) {
          sub.addresses.forEach(addr => watchedAddresses.add(addr));
        }
      });
      
      // 监控每个被关注的地址
      for (const address of watchedAddresses) {
        // 检查投资组合变化
        const portfolioChanged = await this.checkPortfolioChange(address);
        
        if (portfolioChanged) {
          // 获取投资组合信息
          const portfolio = await SmartMoneyTracker.trackPortfolioChanges(address);
          
          // 找出变化最大的持仓
          let maxChangeToken = '';
          let maxChangePercent = 0;
          
          if (portfolio && portfolio.changes) {
            portfolio.changes.forEach((change: any) => {
              if (Math.abs(change.percentChange) > Math.abs(maxChangePercent)) {
                maxChangeToken = change.token;
                maxChangePercent = change.percentChange;
              }
            });
          }
          
          // 生成提醒
          const alert: Alert = {
            id: `portfolio_${address}_${Date.now()}`,
            type: AlertType.PORTFOLIO_CHANGE,
            priority: AlertPriority.HIGH,
            title: '智能资金投资组合变化',
            description: `智能资金地址 ${address.substring(0, 8)}... 的投资组合发生显著变化`,
            address: address,
            token: maxChangeToken,
            priceChange: maxChangePercent,
            timestamp: new Date(),
            read: false,
            dismissed: false,
            data: {
              portfolio: portfolio.currentHoldings,
              changes: portfolio.changes
            }
          };
          
          // 发出提醒事件
          this.emit('alert', alert);
        }
      }
    } catch (error) {
      logger.error('监控被关注地址失败', { error });
    }
  }
  
  /**
   * 检查投资组合变化
   * @param address 地址
   * @private
   */
  private async checkPortfolioChange(address: string): Promise<boolean> {
    try {
      // 检查缓存
      const cacheKey = `portfolio_check:${address}`;
      const lastCheck = cache.get<number>(cacheKey);
      
      if (lastCheck) {
        // 模拟检查逻辑
        // 实际应该比较当前持仓与上次记录的持仓
        const mockChanged = Math.random() > 0.8; // 20%概率发生变化
        
        if (mockChanged) {
          // 更新缓存
          cache.set(cacheKey, Date.now(), 7200); // 2小时
          return true;
        }
      } else {
        // 首次检查，记录时间戳
        cache.set(cacheKey, Date.now(), 7200); // 2小时
      }
      
      return false;
    } catch (error) {
      logger.error('检查投资组合变化失败', { error, address });
      return false;
    }
  }
  
  /**
   * 监控大额交易
   * @private
   */
  private async monitorLargeTransactions(): Promise<void> {
    try {
      // 模拟获取最新大额交易
      // 实际应该查询区块链API或数据库
      
      // 生成随机示例大额交易
      if (Math.random() > 0.7) { // 30%概率发现大额交易
        const mockAddress = `0x${Math.random().toString(16).substr(2, 40)}`;
        const mockAmount = Math.floor(Math.random() * 1000000) + 100000; // 10-110万之间的随机数
        const mockToken = ['ETH', 'BTC', 'USDT', 'UNI', 'LINK'][Math.floor(Math.random() * 5)];
        const mockTxHash = `0x${Math.random().toString(16).substr(2, 64)}`;
        
        // 检查是否由已知的智能资金发起
        const isSmartMoney = await SmartMoneyProfileDAO.findByAddress(mockAddress) !== null;
        
        if (isSmartMoney) {
          // 检查缓存，避免重复提醒同一交易
          const cacheKey = `large_tx:${mockTxHash}`;
          
          if (!this.alertCache.has(cacheKey)) {
            // 生成提醒
            const alert: Alert = {
              id: `large_tx_${Date.now()}`,
              type: AlertType.LARGE_TRANSACTION,
              priority: AlertPriority.HIGH,
              title: '智能资金大额交易',
              description: `智能资金地址 ${mockAddress.substring(0, 8)}... 发生 ${mockAmount.toLocaleString()} ${mockToken} 的大额交易`,
              address: mockAddress,
              token: mockToken,
              transactionHash: mockTxHash,
              amount: mockAmount,
              valueUSD: mockAmount, // 简化示例，实际应转换为USD
              timestamp: new Date(),
              read: false,
              dismissed: false
            };
            
            // 发出提醒事件
            this.emit('alert', alert);
            
            // 添加到缓存
            this.alertCache.set(cacheKey, true);
            
            // 设置缓存过期（1小时后自动移除）
            setTimeout(() => {
              this.alertCache.delete(cacheKey);
            }, 60 * 60 * 1000);
          }
        }
      }
    } catch (error) {
      logger.error('监控大额交易失败', { error });
    }
  }
  
  /**
   * 监控投资组合变化
   * @private
   */
  private async monitorPortfolioChanges(): Promise<void> {
    try {
      // 获取所有活跃的智能资金地址
      const profiles = await SmartMoneyProfileDAO.getRecentlyActive(3, 30); // 最近3天活跃的前30个地址
      
      // 检查投资组合变化
      for (const profile of profiles) {
        // 模拟逻辑 - 实际应比较当前与历史投资组合
        if (Math.random() > 0.9) { // 10%概率发现显著变化
          // 模拟投资组合变化
          const mockToken = ['ETH', 'BTC', 'USDT', 'UNI', 'LINK'][Math.floor(Math.random() * 5)];
          const changeType = Math.random() > 0.5 ? '增持' : '减持';
          const changePercent = Math.floor(Math.random() * 30) + 10; // 10-40% 的变化
          
          // 检查缓存，避免重复提醒
          const cacheKey = `portfolio_change:${profile.address}:${mockToken}`;
          
          if (!this.alertCache.has(cacheKey)) {
            // 生成提醒
            const alert: Alert = {
              id: `portfolio_change_${Date.now()}`,
              type: AlertType.PORTFOLIO_CHANGE,
              priority: AlertPriority.MEDIUM,
              title: '智能资金持仓变化',
              description: `智能资金地址 ${profile.address.substring(0, 8)}... ${changeType}了 ${changePercent}% 的 ${mockToken}`,
              address: profile.address,
              token: mockToken,
              priceChange: changeType === '增持' ? changePercent : -changePercent,
              timestamp: new Date(),
              read: false,
              dismissed: false,
              data: {
                investorType: profile.investorTypes[0],
                score: profile.score.overall
              }
            };
            
            // 发出提醒事件
            this.emit('alert', alert);
            
            // 添加到缓存
            this.alertCache.set(cacheKey, true);
            
            // 设置缓存过期（4小时后自动移除）
            setTimeout(() => {
              this.alertCache.delete(cacheKey);
            }, 4 * 60 * 60 * 1000);
          }
        }
      }
    } catch (error) {
      logger.error('监控投资组合变化失败', { error });
    }
  }
  
  /**
   * 监控智能资金共同行为
   * @private
   */
  private async monitorCommonBehavior(): Promise<void> {
    try {
      // 获取顶级智能资金地址
      const topProfiles = await SmartMoneyProfileDAO.getLeaderboard({
        sortBy: 'score.overall',
        limit: 20
      });
      
      if (topProfiles.length < 5) {
        return; // 数据不足，跳过
      }
      
      // 随机选择5个顶级智能资金地址
      const sampleAddresses = topProfiles
        .sort(() => 0.5 - Math.random()) // 随机排序
        .slice(0, 5) // 取前5个
        .map(p => p.address);
      
      // 分析共同行为
      const commonBehavior = await SmartMoneyTracker.monitorCommonBehavior(sampleAddresses);
      
      // 检查是否有共同关注的代币
      if (commonBehavior.commonTokens && commonBehavior.commonTokens.length > 0) {
        // 取排名最靠前的共同代币
        const topToken = commonBehavior.commonTokens[0];
        
        // 检查缓存，避免重复提醒
        const cacheKey = `common_token:${topToken.token}`;
        
        if (!this.alertCache.has(cacheKey)) {
          // 生成提醒
          const alert: Alert = {
            id: `common_behavior_${Date.now()}`,
            type: AlertType.COMMON_BEHAVIOR,
            priority: AlertPriority.HIGH,
            title: '智能资金共同关注',
            description: `多个顶级智能资金地址共同持有 ${topToken.symbol || topToken.token}`,
            token: topToken.token,
            relatedAddresses: sampleAddresses,
            timestamp: new Date(),
            read: false,
            dismissed: false,
            data: {
              commonTokens: commonBehavior.commonTokens.slice(0, 3), // 前3个共同代币
              commonStrategies: commonBehavior.commonStrategies,
              addressCount: sampleAddresses.length
            }
          };
          
          // 发出提醒事件
          this.emit('alert', alert);
          
          // 添加到缓存
          this.alertCache.set(cacheKey, true);
          
          // 设置缓存过期（12小时后自动移除）
          setTimeout(() => {
            this.alertCache.delete(cacheKey);
          }, 12 * 60 * 60 * 1000);
        }
      }
    } catch (error) {
      logger.error('监控共同行为失败', { error });
    }
  }
  
  /**
   * 处理提醒事件
   * @param alert 提醒对象
   * @private
   */
  private async handleAlert(alert: Alert): Promise<void> {
    try {
      logger.info('处理新提醒', { alertId: alert.id, type: alert.type, priority: alert.priority });
      
      // 保存提醒到数据库（实际项目中实现）
      // await this.saveAlertToDatabase(alert);
      
      // 为每个订阅用户发送通知
      for (const [userId, subscription] of this.subscriptions.entries()) {
        // 检查该用户是否订阅了此类提醒
        if (
          subscription.alertTypes.includes(alert.type) &&
          this.isPriorityHighEnough(alert.priority, subscription.minPriority)
        ) {
          // 检查地址过滤器（如果存在）
          if (
            alert.address &&
            subscription.addresses &&
            subscription.addresses.length > 0 &&
            !subscription.addresses.includes(alert.address)
          ) {
            continue; // 跳过不在关注列表中的地址
          }
          
          // 检查代币过滤器（如果存在）
          if (
            alert.token &&
            subscription.tokens &&
            subscription.tokens.length > 0 &&
            !subscription.tokens.includes(alert.token)
          ) {
            continue; // 跳过不在关注列表中的代币
          }
          
          // 检查金额阈值（如果适用）
          if (
            alert.type === AlertType.LARGE_TRANSACTION &&
            alert.valueUSD &&
            alert.valueUSD < subscription.thresholds.transactionValue
          ) {
            continue; // 跳过低于阈值的交易
          }
          
          // 检查投资组合变化阈值（如果适用）
          if (
            alert.type === AlertType.PORTFOLIO_CHANGE &&
            alert.priceChange &&
            Math.abs(alert.priceChange) < subscription.thresholds.portfolioChangePercent
          ) {
            continue; // 跳过低于阈值的变化
          }
          
          // 发送通知
          await this.sendNotifications(userId, subscription, alert);
        }
      }
    } catch (error) {
      logger.error('处理提醒失败', { error, alertId: alert.id });
    }
  }
  
  /**
   * 检查优先级是否足够高
   * @param alertPriority 提醒优先级
   * @param minPriority 最低优先级
   * @private
   */
  private isPriorityHighEnough(alertPriority: AlertPriority, minPriority: AlertPriority): boolean {
    const priorityValues = {
      [AlertPriority.LOW]: 0,
      [AlertPriority.MEDIUM]: 1,
      [AlertPriority.HIGH]: 2,
      [AlertPriority.CRITICAL]: 3
    };
    
    return priorityValues[alertPriority] >= priorityValues[minPriority];
  }
  
  /**
   * 发送通知
   * @param userId 用户ID
   * @param subscription 订阅设置
   * @param alert 提醒内容
   * @private
   */
  private async sendNotifications(
    userId: string,
    subscription: AlertSubscription,
    alert: Alert
  ): Promise<void> {
    try {
      // 获取通知服务
      const notificationService = NotificationService.getInstance();
      
      // 发送应用内通知
      if (subscription.notificationChannels.inApp) {
        await notificationService.sendInAppNotification(userId, alert);
      }
      
      // 发送电子邮件通知
      if (subscription.notificationChannels.email) {
        await notificationService.sendEmailNotification(userId, alert);
      }
      
      // 发送推送通知
      if (subscription.notificationChannels.push) {
        await notificationService.sendPushNotification(userId, alert);
      }
      
      // 发送Webhook通知（如果配置了）
      if (subscription.notificationChannels.webhook) {
        await notificationService.sendWebhookNotification(
          subscription.notificationChannels.webhook,
          alert
        );
      }
      
      logger.debug('发送通知成功', { userId, alertId: alert.id });
    } catch (error) {
      logger.error('发送通知失败', { error, userId, alertId: alert.id });
    }
  }
  
  /**
   * 订阅提醒
   * @param userId 用户ID
   * @param subscription 订阅设置
   */
  public async subscribe(userId: string, subscription: Omit<AlertSubscription, 'userId'>): Promise<void> {
    try {
      const fullSubscription: AlertSubscription = {
        userId,
        ...subscription
      };
      
      // 保存订阅设置
      this.subscriptions.set(userId, fullSubscription);
      
      // 实际项目中应保存到数据库
      
      logger.info('用户订阅提醒成功', { userId });
    } catch (error) {
      logger.error('用户订阅提醒失败', { error, userId });
      throw error;
    }
  }
  
  /**
   * 取消订阅
   * @param userId 用户ID
   */
  public async unsubscribe(userId: string): Promise<void> {
    try {
      // 删除订阅设置
      this.subscriptions.delete(userId);
      
      // 实际项目中应从数据库删除
      
      logger.info('用户取消订阅提醒', { userId });
    } catch (error) {
      logger.error('用户取消订阅提醒失败', { error, userId });
      throw error;
    }
  }
  
  /**
   * 更新订阅设置
   * @param userId 用户ID
   * @param subscription 更新的订阅设置
   */
  public async updateSubscription(
    userId: string,
    subscription: Partial<Omit<AlertSubscription, 'userId'>>
  ): Promise<void> {
    try {
      // 获取现有的订阅设置
      const existingSubscription = this.subscriptions.get(userId);
      
      if (!existingSubscription) {
        throw new Error('用户未订阅提醒');
      }
      
      // 更新设置
      const updatedSubscription: AlertSubscription = {
        ...existingSubscription,
        ...subscription,
        userId // 确保userId不变
      };
      
      // 保存更新后的设置
      this.subscriptions.set(userId, updatedSubscription);
      
      // 实际项目中应更新到数据库
      
      logger.info('更新用户提醒订阅成功', { userId });
    } catch (error) {
      logger.error('更新用户提醒订阅失败', { error, userId });
      throw error;
    }
  }
  
  /**
   * 获取用户提醒列表
   * @param userId 用户ID
   * @param options 查询选项
   */
  public async getUserAlerts(
    userId: string,
    options: {
      unreadOnly?: boolean;
      limit?: number;
      offset?: number;
      types?: AlertType[];
      minPriority?: AlertPriority;
    } = {}
  ): Promise<Alert[]> {
    try {
      // 实际项目中应从数据库查询
      // 这里返回空数组作为示例
      
      logger.info('获取用户提醒列表', { userId, options });
      
      return [];
    } catch (error) {
      logger.error('获取用户提醒列表失败', { error, userId });
      throw error;
    }
  }
  
  /**
   * 标记提醒为已读
   * @param alertId 提醒ID
   */
  public async markAlertAsRead(alertId: string): Promise<void> {
    try {
      // 实际项目中应更新数据库
      
      logger.info('标记提醒为已读', { alertId });
    } catch (error) {
      logger.error('标记提醒为已读失败', { error, alertId });
      throw error;
    }
  }
  
  /**
   * 删除提醒
   * @param alertId 提醒ID
   */
  public async deleteAlert(alertId: string): Promise<void> {
    try {
      // 实际项目中应从数据库删除
      
      logger.info('删除提醒', { alertId });
    } catch (error) {
      logger.error('删除提醒失败', { error, alertId });
      throw error;
    }
  }
} 