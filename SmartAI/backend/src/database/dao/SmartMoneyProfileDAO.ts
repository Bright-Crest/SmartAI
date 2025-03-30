import { SmartMoneyProfile, ISmartMoneyProfile } from '../models/SmartMoneyProfile';
import { logger } from '../../utils/logger';
import { cache } from '../../utils/cache';

const CACHE_TTL = 3600; // 缓存1小时

export class SmartMoneyProfileDAO {
  /**
   * 创建新的聪明钱画像
   */
  static async create(profile: ISmartMoneyProfile): Promise<ISmartMoneyProfile> {
    try {
      const newProfile = new SmartMoneyProfile(profile);
      const savedProfile = await newProfile.save();
      // 保存到缓存
      cache.set(`smartmoney:${profile.address}`, savedProfile, CACHE_TTL);
      return savedProfile;
    } catch (error) {
      logger.error('创建聪明钱画像失败', { error, address: profile.address });
      throw error;
    }
  }

  /**
   * 通过地址获取聪明钱画像
   */
  static async findByAddress(address: string): Promise<ISmartMoneyProfile | null> {
    try {
      // 先尝试从缓存获取
      const cachedProfile = cache.get<ISmartMoneyProfile>(`smartmoney:${address}`);
      if (cachedProfile) {
        return cachedProfile;
      }

      // 从数据库获取
      const profile = await SmartMoneyProfile.findOne({ address });
      if (profile) {
        // 更新缓存
        cache.set(`smartmoney:${address}`, profile, CACHE_TTL);
      }
      return profile;
    } catch (error) {
      logger.error('查询聪明钱画像失败', { error, address });
      throw error;
    }
  }

  /**
   * 更新聪明钱画像
   */
  static async update(address: string, updates: Partial<ISmartMoneyProfile>): Promise<ISmartMoneyProfile | null> {
    try {
      const updatedProfile = await SmartMoneyProfile.findOneAndUpdate(
        { address },
        { $set: updates },
        { new: true }
      );
      
      if (updatedProfile) {
        // 更新缓存
        cache.set(`smartmoney:${address}`, updatedProfile, CACHE_TTL);
      }
      
      return updatedProfile;
    } catch (error) {
      logger.error('更新聪明钱画像失败', { error, address });
      throw error;
    }
  }

  /**
   * 删除聪明钱画像
   */
  static async delete(address: string): Promise<boolean> {
    try {
      const result = await SmartMoneyProfile.deleteOne({ address });
      // 删除缓存
      cache.del(`smartmoney:${address}`);
      return result.deletedCount > 0;
    } catch (error) {
      logger.error('删除聪明钱画像失败', { error, address });
      throw error;
    }
  }

  /**
   * 查询聪明钱排行榜
   */
  static async getLeaderboard(
    options: {
      sortBy?: 'score.overall' | 'performance.overallROI' | 'influence.followerCount' | 'activityStats.lastActive';
      investorTypes?: string[];
      minScore?: number;
      limit?: number;
      skip?: number;
    } = {}
  ): Promise<ISmartMoneyProfile[]> {
    try {
      const {
        sortBy = 'score.overall',
        investorTypes,
        minScore,
        limit = 20,
        skip = 0
      } = options;

      // 构建查询条件
      const query: any = {};
      
      if (investorTypes && investorTypes.length > 0) {
        query.investorTypes = { $in: investorTypes };
      }
      
      if (minScore) {
        query['score.overall'] = { $gte: minScore };
      }

      // 缓存键
      const cacheKey = `smartmoney:leaderboard:${sortBy}:${JSON.stringify(investorTypes)}:${minScore}:${limit}:${skip}`;
      
      // 先尝试从缓存获取
      const cachedLeaderboard = cache.get<ISmartMoneyProfile[]>(cacheKey);
      if (cachedLeaderboard) {
        return cachedLeaderboard;
      }

      // 从数据库查询
      const profiles = await SmartMoneyProfile.find(query)
        .sort({ [sortBy]: -1 })
        .limit(limit)
        .skip(skip);

      // 更新缓存
      cache.set(cacheKey, profiles, 300); // 缓存5分钟
      
      return profiles;
    } catch (error) {
      logger.error('查询聪明钱排行榜失败', { error });
      throw error;
    }
  }

  /**
   * 添加成功案例
   */
  static async addSuccessCase(address: string, successCase: any): Promise<ISmartMoneyProfile | null> {
    try {
      const updatedProfile = await SmartMoneyProfile.findOneAndUpdate(
        { address },
        { $push: { successCases: successCase } },
        { new: true }
      );
      
      if (updatedProfile) {
        // 更新缓存
        cache.set(`smartmoney:${address}`, updatedProfile, CACHE_TTL);
      }
      
      return updatedProfile;
    } catch (error) {
      logger.error('添加成功案例失败', { error, address });
      throw error;
    }
  }

  /**
   * 更新当前持仓
   */
  static async updateHoldings(address: string, holdings: any[]): Promise<ISmartMoneyProfile | null> {
    try {
      const updatedProfile = await SmartMoneyProfile.findOneAndUpdate(
        { address },
        { $set: { currentHoldings: holdings } },
        { new: true }
      );
      
      if (updatedProfile) {
        // 更新缓存
        cache.set(`smartmoney:${address}`, updatedProfile, CACHE_TTL);
      }
      
      return updatedProfile;
    } catch (error) {
      logger.error('更新当前持仓失败', { error, address });
      throw error;
    }
  }

  /**
   * 更新影响力指标
   */
  static async updateInfluence(address: string, influence: any): Promise<ISmartMoneyProfile | null> {
    try {
      const updatedProfile = await SmartMoneyProfile.findOneAndUpdate(
        { address },
        { $set: { influence } },
        { new: true }
      );
      
      if (updatedProfile) {
        // 更新缓存
        cache.set(`smartmoney:${address}`, updatedProfile, CACHE_TTL);
      }
      
      return updatedProfile;
    } catch (error) {
      logger.error('更新影响力指标失败', { error, address });
      throw error;
    }
  }

  /**
   * 查询投资者类型分布
   */
  static async getInvestorTypeDistribution(): Promise<Record<string, number>> {
    try {
      const cacheKey = 'smartmoney:investorTypeDistribution';
      
      // 先尝试从缓存获取
      const cachedDistribution = cache.get<Record<string, number>>(cacheKey);
      if (cachedDistribution) {
        return cachedDistribution;
      }

      // 从数据库聚合查询
      const distribution = await SmartMoneyProfile.aggregate([
        { $unwind: '$investorTypes' },
        { $group: { _id: '$investorTypes', count: { $sum: 1 } } },
        { $project: { _id: 0, type: '$_id', count: 1 } }
      ]);

      // 格式化结果
      const result = distribution.reduce((acc: Record<string, number>, item: any) => {
        acc[item.type] = item.count;
        return acc;
      }, {});

      // 更新缓存
      cache.set(cacheKey, result, 3600); // 缓存1小时
      
      return result;
    } catch (error) {
      logger.error('查询投资者类型分布失败', { error });
      throw error;
    }
  }

  /**
   * 查询近期活跃的聪明钱地址
   */
  static async getRecentlyActive(days: number = 7, limit: number = 10): Promise<ISmartMoneyProfile[]> {
    try {
      const cacheKey = `smartmoney:recentlyActive:${days}:${limit}`;
      
      // 先尝试从缓存获取
      const cachedProfiles = cache.get<ISmartMoneyProfile[]>(cacheKey);
      if (cachedProfiles) {
        return cachedProfiles;
      }

      // 计算时间范围
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);

      // 从数据库查询
      const profiles = await SmartMoneyProfile.find({
        'activityStats.lastActive': { $gte: cutoffDate }
      })
        .sort({ 'activityStats.lastActive': -1 })
        .limit(limit);

      // 更新缓存
      cache.set(cacheKey, profiles, 1800); // 缓存30分钟
      
      return profiles;
    } catch (error) {
      logger.error('查询近期活跃聪明钱失败', { error });
      throw error;
    }
  }

  /**
   * 查询按照ROI分组的聪明钱数量统计
   */
  static async getROIDistribution(): Promise<any[]> {
    try {
      const cacheKey = 'smartmoney:roiDistribution';
      
      // 先尝试从缓存获取
      const cachedDistribution = cache.get<any[]>(cacheKey);
      if (cachedDistribution) {
        return cachedDistribution;
      }

      // 从数据库聚合查询
      const distribution = await SmartMoneyProfile.aggregate([
        {
          $bucket: {
            groupBy: '$performance.overallROI',
            boundaries: [0, 50, 100, 200, 500, 1000, 5000, Infinity],
            default: 'Other',
            output: {
              count: { $sum: 1 },
              addresses: { $push: '$address' }
            }
          }
        }
      ]);

      // 更新缓存
      cache.set(cacheKey, distribution, 3600); // 缓存1小时
      
      return distribution;
    } catch (error) {
      logger.error('查询ROI分布失败', { error });
      throw error;
    }
  }
} 