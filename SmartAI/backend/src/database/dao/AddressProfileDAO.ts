import { IAddressProfile } from '../models/AddressProfile';
import { AddressProfileModel } from '../models/AddressProfile';
import { cache } from '../redis';
import { createLogger } from '../../utils/logger';
import { AddressProfile } from '../../types/profile';
import { Database } from '../Database';
import { logger } from '../../utils/logger';

const CACHE_TTL = 3600; // 1小时缓存

export class AddressProfileDAO {
  static async create(profile: IAddressProfile): Promise<IAddressProfile> {
    try {
      const record = await AddressProfileModel.create(profile);
      await cache.set(`profile:${record.address}`, record, CACHE_TTL);
      return record;
    } catch (error) {
      logger.error('Create address profile error', { error });
      throw error;
    }
  }

  static async findByAddress(address: string): Promise<IAddressProfile | null> {
    try {
      // 先查缓存
      const cached = await cache.get<IAddressProfile>(`profile:${address}`);
      if (cached) {
        return cached;
      }

      // 缓存未命中，查数据库
      const record = await AddressProfileModel.findOne({ address });
      if (record) {
        await cache.set(`profile:${address}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Find address profile error', { address, error });
      throw error;
    }
  }

  static async update(
    address: string,
    updates: Partial<IAddressProfile>
  ): Promise<IAddressProfile | null> {
    try {
      const record = await AddressProfileModel.findOneAndUpdate(
        { address },
        { $set: updates },
        { new: true }
      );
      if (record) {
        await cache.set(`profile:${address}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Update address profile error', { address, error });
      throw error;
    }
  }

  static async updateRiskScore(
    address: string,
    riskScore: number
  ): Promise<IAddressProfile | null> {
    try {
      const record = await AddressProfileModel.findOneAndUpdate(
        { address },
        { $set: { riskScore, lastUpdated: new Date() } },
        { new: true }
      );
      if (record) {
        await cache.set(`profile:${address}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Update address risk score error', { address, riskScore, error });
      throw error;
    }
  }

  static async addTag(address: string, tag: string): Promise<IAddressProfile | null> {
    try {
      const record = await AddressProfileModel.findOneAndUpdate(
        { address },
        { $addToSet: { tags: tag } },
        { new: true }
      );
      if (record) {
        await cache.set(`profile:${address}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Add tag to address profile error', { address, tag, error });
      throw error;
    }
  }

  static async removeTag(address: string, tag: string): Promise<IAddressProfile | null> {
    try {
      const record = await AddressProfileModel.findOneAndUpdate(
        { address },
        { $pull: { tags: tag } },
        { new: true }
      );
      if (record) {
        await cache.set(`profile:${address}`, record, CACHE_TTL);
      }
      return record;
    } catch (error) {
      logger.error('Remove tag from address profile error', { address, tag, error });
      throw error;
    }
  }

  static async delete(address: string): Promise<boolean> {
    try {
      const result = await AddressProfileModel.deleteOne({ address });
      await cache.del(`profile:${address}`);
      return result.deletedCount > 0;
    } catch (error) {
      logger.error('Delete address profile error', { address, error });
      throw error;
    }
  }

  static async findByAddresses(addresses: string[]): Promise<AddressProfile[]> {
    if (addresses.length === 0) return [];
    
    try {
      const db = await Database.getConnection();
      const collection = db.collection('address_profiles');
      
      // 转换为小写
      const lowercaseAddresses = addresses.map(addr => addr.toLowerCase());
      
      const profiles = await collection
        .find({ address: { $in: lowercaseAddresses } })
        .toArray();
      
      return profiles as AddressProfile[];
    } catch (error) {
      logger.error('批量查询地址画像失败', {
        addressCount: addresses.length,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }

  static async save(profile: AddressProfile): Promise<AddressProfile | null> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('address_profiles');
      
      // 设置地址为小写以确保一致性
      const profileToSave = {
        ...profile,
        address: profile.address.toLowerCase(),
        updatedAt: new Date()
      };
      
      // 检查是否已存在
      const existingProfile = await collection.findOne({ address: profileToSave.address });
      
      if (existingProfile) {
        // 更新已有记录
        await collection.updateOne(
          { address: profileToSave.address },
          { $set: { ...profileToSave } }
        );
      } else {
        // 创建新记录
        profileToSave.createdAt = new Date();
        await collection.insertOne(profileToSave);
      }
      
      return profileToSave;
    } catch (error) {
      logger.error('保存地址画像失败', {
        address: profile.address,
        error: error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  static async batchUpdateRiskScores(
    addressScores: Record<string, number>
  ): Promise<number> {
    const addresses = Object.keys(addressScores);
    if (addresses.length === 0) return 0;
    
    try {
      const db = await Database.getConnection();
      const collection = db.collection('address_profiles');
      
      let successCount = 0;
      
      // 对每个地址创建更新操作
      const operations = addresses.map(address => {
        const score = addressScores[address];
        return {
          updateOne: {
            filter: { address: address.toLowerCase() },
            update: { 
              $set: { 
                riskScore: score,
                updatedAt: new Date()
              },
              $setOnInsert: {
                address: address.toLowerCase(),
                createdAt: new Date(),
                tags: [],
                firstSeen: new Date(),
                lastSeen: new Date(),
                transactionCount: 0
              }
            },
            upsert: true
          }
        };
      });
      
      // 执行批量更新
      const result = await collection.bulkWrite(operations);
      
      successCount = (result.upsertedCount || 0) + (result.modifiedCount || 0);
      
      logger.info('批量更新风险分数完成', {
        totalAddresses: addresses.length,
        successCount
      });
      
      return successCount;
    } catch (error) {
      logger.error('批量更新风险分数失败', {
        addressCount: addresses.length,
        error: error instanceof Error ? error.message : String(error),
      });
      return 0;
    }
  }

  static async findHighRiskAddresses(
    threshold: number = 0.7,
    limit: number = 100
  ): Promise<AddressProfile[]> {
    try {
      const db = await Database.getConnection();
      const collection = db.collection('address_profiles');
      
      const profiles = await collection
        .find({ riskScore: { $gte: threshold } })
        .sort({ riskScore: -1 })
        .limit(limit)
        .toArray();
      
      return profiles as AddressProfile[];
    } catch (error) {
      logger.error('查询高风险地址失败', {
        threshold,
        error: error instanceof Error ? error.message : String(error),
      });
      return [];
    }
  }
}
