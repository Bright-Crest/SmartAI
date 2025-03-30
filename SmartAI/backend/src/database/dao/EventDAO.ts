import { IEventRecord } from '../models/EventRecord';
import { EventRecord } from '../models/EventRecord';
import { cache } from '../redis';
import { createLogger } from '../../utils/logger';
import { NormalizedEvent } from '../../types/events';
import { Database } from '../Database';

const logger = createLogger({
  level: 'info',
  format: 'json',
  timestampFormat: 'YYYY-MM-DD HH:mm:ss',
});
const CACHE_TTL = 3600; // 1小时缓存

export class EventDAO {
  static async create(event: IEventRecord): Promise<IEventRecord> {
    try {
      const record = await EventRecord.create(event);
      await cache.set(`event:${record.traceId}`, record, CACHE_TTL);
      return record;
    } catch (error) {
      logger.error('Create event record error', { error });
      throw error;
    }
  }

  static async findByTraceId(traceId: string): Promise<IEventRecord | null> {
    try {
      // 先查缓存
      const cached = await cache.get<IEventRecord>(`event:${traceId}`);
      if (cached) {
        return cached;
      }

      // 缓存未命中，查数据库
      const record = await EventRecord.findOne({ traceId });
      if (record) {
        await cache.set(`event:${traceId}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Find event by traceId error', { traceId, error });
      throw error;
    }
  }

  static async findByTransactionHash(hash: string): Promise<IEventRecord | null> {
    try {
      return await EventRecord.findOne({ 'event.transactionHash': hash });
    } catch (error) {
      logger.error('Find event by transaction hash error', { hash, error });
      throw error;
    }
  }

  static async updateStatus(
    traceId: string,
    status: IEventRecord['status']
  ): Promise<IEventRecord | null> {
    try {
      const record = await EventRecord.findOneAndUpdate(
        { traceId },
        { $set: { status } },
        { new: true }
      );
      if (record) {
        await cache.set(`event:${traceId}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Update event status error', { traceId, status, error });
      throw error;
    }
  }

  static async updateRiskAnalysis(
    traceId: string,
    riskAnalysis: IEventRecord['riskAnalysis']
  ): Promise<IEventRecord | null> {
    try {
      const record = await EventRecord.findOneAndUpdate(
        { traceId },
        { $set: { riskAnalysis } },
        { new: true }
      );
      if (record) {
        await cache.set(`event:${traceId}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Update event risk analysis error', { traceId, error });
      throw error;
    }
  }

  static async delete(traceId: string): Promise<boolean> {
    try {
      const result = await EventRecord.deleteOne({ traceId });
      await cache.del(`event:${traceId}`);
      return result.deletedCount > 0;
    } catch (error) {
      logger.error('Delete event error', { traceId, error });
      throw error;
    }
  }

  static async deleteMany(filter: Record<string, any> = {}): Promise<number> {
    try {
      const result = await EventRecord.deleteMany(filter);
      return result.deletedCount;
    } catch (error) {
      logger.error('Delete many events error', { filter, error });
      throw error;
    }
  }

  /**
   * 查找地址的最近交易记录
   * @param address 地址
   * @param limit 限制数量
   * @returns 交易记录数组
   */
  static async findByAddress(address: string, limit: number = 20): Promise<IEventRecord[]> {
    try {
      // 查询发送方或接收方为指定地址的交易
      const records = await EventRecord.find({
        $or: [{ 'event.from': address.toLowerCase() }, { 'event.to': address.toLowerCase() }],
      })
        .sort({ 'event.timestamp': -1 }) // 按时间倒序
        .limit(limit);

      return records;
    } catch (error) {
      logger.error('Find events by address error', { address, limit, error });
      throw error;
    }
  }

  /**
   * 根据地址查找事件记录
   * @param address 地址
   * @param limit 限制条数
   * @returns 事件记录数组
   */
  static async findByAddress(address: string, limit: number = 10): Promise<IEventRecord[]> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('events');
      
      const events = await collection
        .find({
          $or: [
            { 'event.from': address },
            { 'event.to': address }
          ]
        })
        .sort({ 'event.timestamp': -1 })
        .limit(limit)
        .toArray();
      
      return events as IEventRecord[];
    } catch (error) {
      logger.error('查询事件记录失败', {
        address,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }
  
  /**
   * 根据地址和时间范围查找事件记录
   * @param address 地址
   * @param startTime 开始时间（UNIX时间戳）
   * @param endTime 结束时间（UNIX时间戳）
   * @param limit 限制条数
   * @returns 事件记录数组
   */
  static async findByAddressAndTimeRange(
    address: string,
    startTime: number,
    endTime: number,
    limit: number = 50
  ): Promise<IEventRecord[]> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('events');
      
      const events = await collection
        .find({
          $or: [
            { 'event.from': address },
            { 'event.to': address }
          ],
          'event.timestamp': {
            $gte: startTime,
            $lte: endTime
          }
        })
        .sort({ 'event.timestamp': -1 })
        .limit(limit)
        .toArray();
      
      return events as IEventRecord[];
    } catch (error) {
      logger.error('查询事件时间范围记录失败', {
        address,
        startTime,
        endTime,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }
  
  /**
   * 保存事件记录
   * @param event 事件
   * @returns 保存结果
   */
  static async save(event: NormalizedEvent): Promise<IEventRecord | null> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('events');
      
      const eventRecord: IEventRecord = {
        id: event.transactionHash,
        event,
        createdAt: new Date(),
        updatedAt: new Date()
      };
      
      await collection.insertOne(eventRecord);
      
      return eventRecord;
    } catch (error) {
      logger.error('保存事件记录失败', {
        transactionHash: event.transactionHash,
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }
  
  /**
   * 查找重复事件
   * @param transactionHash 交易哈希
   * @returns 事件记录或 null
   */
  static async findByTransactionHash(transactionHash: string): Promise<IEventRecord | null> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('events');
      
      const event = await collection.findOne({ id: transactionHash });
      
      return event as IEventRecord | null;
    } catch (error) {
      logger.error('查询事件记录失败', {
        transactionHash,
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }
  
  /**
   * 批量保存事件记录
   * @param events 事件数组
   * @returns 成功数量
   */
  static async bulkSave(events: NormalizedEvent[]): Promise<number> {
    if (events.length === 0) return 0;
    
    try {
      const db = await Database.getConnection();
      const collection = db.collection('events');
      
      const eventRecords = events.map(event => ({
        id: event.transactionHash,
        event,
        createdAt: new Date(),
        updatedAt: new Date()
      }));
      
      const result = await collection.insertMany(eventRecords, { ordered: false });
      
      return result.insertedCount;
    } catch (error) {
      logger.error('批量保存事件记录失败', {
        eventsCount: events.length,
        error: error instanceof Error ? error.message : String(error),
      });
      return 0;
    }
  }
}
