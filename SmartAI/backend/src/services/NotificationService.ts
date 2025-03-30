import { loggerWinston as logger } from '../utils/logger';
import axios from 'axios';
import nodemailer from 'nodemailer';
import { EventEmitter } from 'events';

/**
 * 通知服务
 * 负责发送各种类型的通知（应用内、电子邮件、推送、Webhook）
 */
export class NotificationService extends EventEmitter {
  private static instance: NotificationService;
  private emailTransporter: any;
  private pushConfig: any;
  
  /**
   * 获取单例实例
   */
  public static getInstance(): NotificationService {
    if (!NotificationService.instance) {
      NotificationService.instance = new NotificationService();
    }
    return NotificationService.instance;
  }
  
  /**
   * 私有构造函数
   */
  private constructor() {
    super();
    
    // 设置邮件发送器
    this.setupEmailTransporter();
    
    // 设置推送配置
    this.setupPushConfig();
  }
  
  /**
   * 设置邮件发送器
   * @private
   */
  private setupEmailTransporter(): void {
    try {
      // 创建邮件发送器（使用环境变量配置）
      this.emailTransporter = nodemailer.createTransport({
        host: process.env.EMAIL_HOST || 'smtp.example.com',
        port: parseInt(process.env.EMAIL_PORT || '587', 10),
        secure: process.env.EMAIL_SECURE === 'true',
        auth: {
          user: process.env.EMAIL_USER || 'user@example.com',
          pass: process.env.EMAIL_PASSWORD || 'password'
        }
      });
      
      logger.info('邮件发送器设置成功');
    } catch (error) {
      logger.error('设置邮件发送器失败', { error });
    }
  }
  
  /**
   * 设置推送配置
   * @private
   */
  private setupPushConfig(): void {
    try {
      // 设置推送配置（使用环境变量）
      this.pushConfig = {
        fcmApiKey: process.env.FCM_API_KEY || '',
        apnsCertificate: process.env.APNS_CERTIFICATE || '',
        apnsKey: process.env.APNS_KEY || ''
      };
      
      logger.info('推送配置设置成功');
    } catch (error) {
      logger.error('设置推送配置失败', { error });
    }
  }
  
  /**
   * 发送应用内通知
   * @param userId 用户ID
   * @param notification 通知内容
   */
  public async sendInAppNotification(userId: string, notification: any): Promise<void> {
    try {
      logger.info('发送应用内通知', { userId });
      
      // 实际项目中应该将通知保存到数据库，然后通过WebSocket或轮询机制发送给客户端
      // 这里仅模拟发送过程
      const inAppNotification = {
        userId,
        notificationId: `notification_${Date.now()}`,
        title: notification.title,
        message: notification.description,
        type: notification.type,
        data: notification.data || {},
        createdAt: new Date(),
        read: false
      };
      
      // 模拟保存到数据库
      logger.debug('保存应用内通知', { notification: inAppNotification });
      
      // 触发事件，通知WebSocket服务器
      this.emit('inAppNotification', {
        userId,
        notification: inAppNotification
      });
      
    } catch (error) {
      logger.error('发送应用内通知失败', { error, userId });
      throw error;
    }
  }
  
  /**
   * 发送电子邮件通知
   * @param userId 用户ID
   * @param notification 通知内容
   */
  public async sendEmailNotification(userId: string, notification: any): Promise<void> {
    try {
      logger.info('发送电子邮件通知', { userId });
      
      // 获取用户邮箱（实际项目中应从数据库获取）
      const userEmail = this.getMockUserEmail(userId);
      
      if (!userEmail) {
        logger.warn('无法发送电子邮件，未找到用户邮箱', { userId });
        return;
      }
      
      // 准备邮件内容
      const emailContent = this.formatEmailContent(notification);
      
      // 发送邮件
      const mailOptions = {
        from: process.env.EMAIL_FROM || '"聪明钱提醒" <alerts@smartai.com>',
        to: userEmail,
        subject: `[SmartAI提醒] ${notification.title}`,
        html: emailContent
      };
      
      // 实际项目中取消此注释以发送邮件
      /*
      const info = await this.emailTransporter.sendMail(mailOptions);
      logger.info('电子邮件发送成功', { userId, messageId: info.messageId });
      */
      
      // 模拟发送
      logger.debug('模拟发送电子邮件', { to: userEmail, subject: mailOptions.subject });
      
    } catch (error) {
      logger.error('发送电子邮件通知失败', { error, userId });
      // 不抛出异常，以防止一个通知渠道失败影响其他渠道
    }
  }
  
  /**
   * 发送推送通知
   * @param userId 用户ID
   * @param notification 通知内容
   */
  public async sendPushNotification(userId: string, notification: any): Promise<void> {
    try {
      logger.info('发送推送通知', { userId });
      
      // 获取用户的设备令牌（实际项目中应从数据库获取）
      const deviceTokens = this.getMockUserDeviceTokens(userId);
      
      if (!deviceTokens || deviceTokens.length === 0) {
        logger.warn('无法发送推送通知，未找到用户设备令牌', { userId });
        return;
      }
      
      // 准备推送内容
      const pushContent = {
        title: notification.title,
        body: notification.description,
        data: {
          type: notification.type,
          id: notification.id,
          notificationData: notification.data || {}
        }
      };
      
      // 模拟向FCM发送推送
      // 实际项目中取消此注释以发送推送
      /*
      for (const token of deviceTokens) {
        if (token.type === 'fcm') {
          await this.sendFcmPush(token.token, pushContent);
        } else if (token.type === 'apns') {
          await this.sendApnsPush(token.token, pushContent);
        }
      }
      */
      
      // 模拟发送
      logger.debug('模拟发送推送通知', { 
        userId,
        deviceCount: deviceTokens.length,
        content: pushContent 
      });
      
    } catch (error) {
      logger.error('发送推送通知失败', { error, userId });
      // 不抛出异常，以防止一个通知渠道失败影响其他渠道
    }
  }
  
  /**
   * 发送Webhook通知
   * @param webhookUrl Webhook URL
   * @param notification 通知内容
   */
  public async sendWebhookNotification(webhookUrl: string, notification: any): Promise<void> {
    try {
      logger.info('发送Webhook通知', { webhookUrl });
      
      if (!webhookUrl) {
        logger.warn('无法发送Webhook通知，URL为空');
        return;
      }
      
      // 准备Webhook负载
      const webhookPayload = {
        event: 'smart_money_alert',
        timestamp: new Date().toISOString(),
        data: {
          id: notification.id,
          type: notification.type,
          priority: notification.priority,
          title: notification.title,
          description: notification.description,
          address: notification.address,
          token: notification.token,
          // 其他相关数据
          ...notification.data
        }
      };
      
      // 发送Webhook请求
      // 实际项目中取消此注释以发送Webhook
      /*
      const response = await axios.post(webhookUrl, webhookPayload, {
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'SmartAI/1.0'
        },
        timeout: 5000 // 5秒超时
      });
      
      if (response.status >= 200 && response.status < 300) {
        logger.info('Webhook通知发送成功', { webhookUrl });
      } else {
        logger.warn('Webhook通知发送失败', { webhookUrl, status: response.status });
      }
      */
      
      // 模拟发送
      logger.debug('模拟发送Webhook通知', { webhookUrl, payload: webhookPayload });
      
    } catch (error) {
      logger.error('发送Webhook通知失败', { error, webhookUrl });
      // 不抛出异常，以防止一个通知渠道失败影响其他渠道
    }
  }
  
  /**
   * 格式化邮件内容
   * @param notification 通知对象
   * @private
   */
  private formatEmailContent(notification: any): string {
    let detailsHtml = '';
    
    // 根据通知类型添加特定内容
    if (notification.address) {
      detailsHtml += `<p><strong>地址:</strong> ${notification.address}</p>`;
    }
    
    if (notification.token) {
      detailsHtml += `<p><strong>代币:</strong> ${notification.token}</p>`;
    }
    
    if (notification.amount) {
      detailsHtml += `<p><strong>数量:</strong> ${notification.amount.toLocaleString()}</p>`;
    }
    
    if (notification.valueUSD) {
      detailsHtml += `<p><strong>价值:</strong> $${notification.valueUSD.toLocaleString()}</p>`;
    }
    
    if (notification.priceChange) {
      const changePrefix = notification.priceChange > 0 ? '+' : '';
      detailsHtml += `<p><strong>变化:</strong> ${changePrefix}${notification.priceChange.toFixed(2)}%</p>`;
    }
    
    // 构建完整HTML
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background-color: #4a6cf7; color: white; padding: 10px 20px; border-radius: 5px 5px 0 0; }
          .content { background-color: #f9f9f9; padding: 20px; border-radius: 0 0 5px 5px; }
          .footer { margin-top: 20px; font-size: 12px; color: #666; text-align: center; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h2>${notification.title}</h2>
          </div>
          <div class="content">
            <p>${notification.description}</p>
            ${detailsHtml}
            <p style="margin-top: 20px;">
              <a href="${process.env.APP_URL || 'https://smartai.com'}/dashboard" style="background-color: #4a6cf7; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px;">
                查看详情
              </a>
            </p>
          </div>
          <div class="footer">
            <p>此邮件由系统自动发送，请勿回复。</p>
            <p>如果您不想再收到此类提醒，请在应用设置中<a href="${process.env.APP_URL || 'https://smartai.com'}/settings/notifications">修改通知设置</a>。</p>
          </div>
        </div>
      </body>
      </html>
    `;
  }
  
  /**
   * 向FCM发送推送
   * @param token FCM设备令牌
   * @param content 推送内容
   * @private
   */
  private async sendFcmPush(token: string, content: any): Promise<void> {
    try {
      // 实际项目中实现FCM推送发送
      const fcmPayload = {
        notification: {
          title: content.title,
          body: content.body
        },
        data: content.data,
        token: token
      };
      
      // 发送FCM请求
      // 实际使用firebase-admin或类似库
      logger.debug('FCM负载', { payload: fcmPayload });
      
    } catch (error) {
      logger.error('发送FCM推送失败', { error, token });
    }
  }
  
  /**
   * 向APNS发送推送
   * @param token APNS设备令牌
   * @param content 推送内容
   * @private
   */
  private async sendApnsPush(token: string, content: any): Promise<void> {
    try {
      // 实际项目中实现APNS推送发送
      const apnsPayload = {
        aps: {
          alert: {
            title: content.title,
            body: content.body
          },
          badge: 1,
          sound: 'default'
        },
        data: content.data
      };
      
      // 发送APNS请求
      // 实际使用apn或类似库
      logger.debug('APNS负载', { payload: apnsPayload });
      
    } catch (error) {
      logger.error('发送APNS推送失败', { error, token });
    }
  }
  
  /**
   * 获取模拟用户邮箱
   * @param userId 用户ID
   * @private
   */
  private getMockUserEmail(userId: string): string {
    // 实际项目中应从数据库获取
    return `user${userId}@example.com`;
  }
  
  /**
   * 获取模拟用户设备令牌
   * @param userId 用户ID
   * @private
   */
  private getMockUserDeviceTokens(userId: string): { type: string, token: string }[] {
    // 实际项目中应从数据库获取
    return [
      { type: 'fcm', token: 'fcm-token-' + userId },
      { type: 'apns', token: 'apns-token-' + userId }
    ];
  }
} 