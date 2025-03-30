import { Database } from '../Database';
import { logger } from '../../utils/logger';

/**
 * 交易记录接口
 */
export interface ITransaction {
  from: string;
  to: string;
  value: string;
  hash: string;
  blockNumber: number;
  timestamp: number;
  gas: string;
  gasPrice: string;
  input: string;
  status: boolean;
  chainId: number;
}

/**
 * 交易数据访问对象
 * 负责处理交易相关的数据库操作
 */
export class TransactionDAO {
  /**
   * 根据地址查找交易记录
   * @param address 地址（发送方或接收方）
   * @param limit 限制条数
   * @returns 交易记录数组
   */
  static async findByAddress(address: string, limit: number = 50): Promise<ITransaction[]> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('transactions');
      
      const transactions = await collection
        .find({
          $or: [
            { from: address },
            { to: address }
          ]
        })
        .sort({ timestamp: -1 })
        .limit(limit)
        .toArray();
      
      return transactions as ITransaction[];
    } catch (error) {
      logger.error('查询交易记录失败', {
        address,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }
  
  /**
   * 根据交易哈希查找交易记录
   * @param hash 交易哈希
   * @returns 交易记录或 null
   */
  static async findByHash(hash: string): Promise<ITransaction | null> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('transactions');
      
      const transaction = await collection.findOne({ hash });
      
      return transaction as ITransaction | null;
    } catch (error) {
      logger.error('查询交易记录失败', {
        hash,
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }
  
  /**
   * 根据地址和时间范围查找交易记录
   * @param address 地址
   * @param startTime 开始时间（UNIX时间戳）
   * @param endTime 结束时间（UNIX时间戳）
   * @param limit 限制条数
   * @returns 交易记录数组
   */
  static async findByAddressAndTimeRange(
    address: string,
    startTime: number,
    endTime: number,
    limit: number = 50
  ): Promise<ITransaction[]> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('transactions');
      
      const transactions = await collection
        .find({
          $or: [
            { from: address },
            { to: address }
          ],
          timestamp: {
            $gte: startTime,
            $lte: endTime
          }
        })
        .sort({ timestamp: -1 })
        .limit(limit)
        .toArray();
      
      return transactions as ITransaction[];
    } catch (error) {
      logger.error('查询交易时间范围记录失败', {
        address,
        startTime,
        endTime,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }
  
  /**
   * 保存交易记录
   * @param transaction 交易记录
   * @returns 保存结果
   */
  static async save(transaction: ITransaction): Promise<ITransaction | null> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('transactions');
      
      // 检查是否存在
      const existing = await collection.findOne({ hash: transaction.hash });
      if (existing) {
        return existing as ITransaction;
      }
      
      // 插入新记录
      await collection.insertOne({
        ...transaction,
        createdAt: new Date(),
        updatedAt: new Date()
      });
      
      return transaction;
    } catch (error) {
      logger.error('保存交易记录失败', {
        hash: transaction.hash,
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }
  
  /**
   * 批量保存交易记录
   * @param transactions 交易记录数组
   * @returns 成功插入的数量
   */
  static async bulkSave(transactions: ITransaction[]): Promise<number> {
    if (transactions.length === 0) return 0;
    
    try {
      const db = await Database.getConnection();
      const collection = db.collection('transactions');
      
      // 添加创建和更新时间
      const records = transactions.map(tx => ({
        ...tx,
        createdAt: new Date(),
        updatedAt: new Date()
      }));
      
      // 批量插入（忽略重复错误）
      const result = await collection.insertMany(records, { ordered: false });
      
      return result.insertedCount;
    } catch (error) {
      logger.error('批量保存交易记录失败', {
        count: transactions.length,
        error: error instanceof Error ? error.message : String(error),
      });
      return 0;
    }
  }
} 