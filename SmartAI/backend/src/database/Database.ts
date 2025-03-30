import { MongoClient, Db } from 'mongodb';
import { logger } from '../utils/logger';
import { config } from '../config';

/**
 * 数据库连接工具类
 * 负责管理MongoDB连接
 */
export class Database {
  private static client: MongoClient | null = null;
  private static db: Db | null = null;
  
  /**
   * 获取数据库连接
   * @returns 数据库实例
   */
  static async getConnection(): Promise<Db> {
    if (this.db) {
      return this.db;
    }
    
    try {
      // 连接字符串从配置中获取
      const url = config.mongodb.url;
      const dbName = config.mongodb.dbName;
      
      // 创建新连接
      this.client = await MongoClient.connect(url);
      this.db = this.client.db(dbName);
      
      logger.info('数据库连接成功', { dbName });
      
      return this.db;
    } catch (error) {
      logger.error('数据库连接失败', {
        error: error instanceof Error ? error.message : String(error),
      });
      throw new Error('数据库连接失败');
    }
  }
  
  /**
   * 关闭数据库连接
   */
  static async close(): Promise<void> {
    if (this.client) {
      try {
        await this.client.close();
        this.client = null;
        this.db = null;
        logger.info('数据库连接已关闭');
      } catch (error) {
        logger.error('关闭数据库连接失败', {
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }
  
  /**
   * 获取集合统计信息
   * @returns 集合统计信息
   */
  static async getCollectionStats(): Promise<Record<string, number>> {
    const stats: Record<string, number> = {};
    
    try {
      const db = await this.getConnection();
      const collections = await db.listCollections().toArray();
      
      for (const collection of collections) {
        const count = await db.collection(collection.name).countDocuments();
        stats[collection.name] = count;
      }
      
      return stats;
    } catch (error) {
      logger.error('获取集合统计失败', {
        error: error instanceof Error ? error.message : String(error),
      });
      return stats;
    }
  }
} 