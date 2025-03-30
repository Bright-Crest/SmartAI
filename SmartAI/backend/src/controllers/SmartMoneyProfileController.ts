import { Request, Response } from 'express';
import { loggerWinston as logger } from '../utils/logger';
import { SmartMoneyProfileDAO } from '../database/dao/SmartMoneyProfileDAO';
import { SmartMoneyScorer } from '../core/SmartMoneyScorer';

export class SmartMoneyProfileController {
  /**
   * 获取地址的聪明钱画像
   */
  static async getProfile(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 查询画像
      const profile = await SmartMoneyProfileDAO.findByAddress(address);
      
      if (!profile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      res.json({ success: true, data: profile });
    } catch (error) {
      logger.error('获取聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 创建聪明钱画像
   */
  static async createProfile(req: Request, res: Response): Promise<void> {
    try {
      const profileData = req.body;
      
      // 参数验证
      if (!profileData || !profileData.address) {
        res.status(400).json({ success: false, message: '画像数据缺失或无效' });
        return;
      }
      
      // 检查是否已存在
      const existingProfile = await SmartMoneyProfileDAO.findByAddress(profileData.address);
      if (existingProfile) {
        res.status(409).json({ success: false, message: '该地址的聪明钱画像已存在' });
        return;
      }
      
      // 创建画像
      const newProfile = await SmartMoneyProfileDAO.create(profileData);
      
      // 计算初始评分
      await SmartMoneyScorer.updateScore(profileData.address);
      
      res.status(201).json({ success: true, data: newProfile });
    } catch (error) {
      logger.error('创建聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 更新聪明钱画像
   */
  static async updateProfile(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      const updates = req.body;
      
      // 参数验证
      if (!address || !updates) {
        res.status(400).json({ success: false, message: '参数缺失或无效' });
        return;
      }
      
      // 更新画像
      const updatedProfile = await SmartMoneyProfileDAO.update(address, updates);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      // 更新评分
      await SmartMoneyScorer.updateScore(address);
      
      res.json({ success: true, data: updatedProfile });
    } catch (error) {
      logger.error('更新聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 删除聪明钱画像
   */
  static async deleteProfile(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 删除画像
      const result = await SmartMoneyProfileDAO.delete(address);
      
      if (!result) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      res.json({ success: true, message: '聪明钱画像已删除' });
    } catch (error) {
      logger.error('删除聪明钱画像失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取聪明钱排行榜
   */
  static async getLeaderboard(req: Request, res: Response): Promise<void> {
    try {
      const { 
        sortBy = 'score.overall',
        investorTypes,
        minScore,
        limit = 20,
        skip = 0
      } = req.query;
      
      // 处理查询参数
      const options: any = {
        sortBy: sortBy as string,
        minScore: minScore ? Number(minScore) : undefined,
        limit: Number(limit),
        skip: Number(skip)
      };
      
      // 处理投资者类型
      if (investorTypes) {
        options.investorTypes = Array.isArray(investorTypes) 
          ? investorTypes 
          : [investorTypes as string];
      }
      
      // 查询排行榜
      const leaderboard = await SmartMoneyProfileDAO.getLeaderboard(options);
      
      res.json({ 
        success: true, 
        data: leaderboard,
        meta: {
          total: leaderboard.length,
          limit: options.limit,
          skip: options.skip
        }
      });
    } catch (error) {
      logger.error('获取聪明钱排行榜失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取投资者类型分布
   */
  static async getInvestorTypeDistribution(req: Request, res: Response): Promise<void> {
    try {
      const distribution = await SmartMoneyProfileDAO.getInvestorTypeDistribution();
      
      res.json({ success: true, data: distribution });
    } catch (error) {
      logger.error('获取投资者类型分布失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取近期活跃的聪明钱
   */
  static async getRecentlyActive(req: Request, res: Response): Promise<void> {
    try {
      const { days = 7, limit = 10 } = req.query;
      
      const profiles = await SmartMoneyProfileDAO.getRecentlyActive(
        Number(days),
        Number(limit)
      );
      
      res.json({ success: true, data: profiles });
    } catch (error) {
      logger.error('获取近期活跃聪明钱失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取ROI分布
   */
  static async getROIDistribution(req: Request, res: Response): Promise<void> {
    try {
      const distribution = await SmartMoneyProfileDAO.getROIDistribution();
      
      res.json({ success: true, data: distribution });
    } catch (error) {
      logger.error('获取ROI分布失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 添加成功案例
   */
  static async addSuccessCase(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      const successCase = req.body;
      
      // 参数验证
      if (!address || !successCase) {
        res.status(400).json({ success: false, message: '参数缺失或无效' });
        return;
      }
      
      // 添加成功案例
      const updatedProfile = await SmartMoneyProfileDAO.addSuccessCase(address, successCase);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      // 更新评分
      await SmartMoneyScorer.updateScore(address);
      
      res.json({ success: true, data: updatedProfile });
    } catch (error) {
      logger.error('添加成功案例失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 更新持仓
   */
  static async updateHoldings(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      const holdings = req.body;
      
      // 参数验证
      if (!address || !Array.isArray(holdings)) {
        res.status(400).json({ success: false, message: '参数缺失或无效' });
        return;
      }
      
      // 更新持仓
      const updatedProfile = await SmartMoneyProfileDAO.updateHoldings(address, holdings);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      // 更新评分
      await SmartMoneyScorer.updateScore(address);
      
      res.json({ success: true, data: updatedProfile });
    } catch (error) {
      logger.error('更新持仓失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 获取评分
   */
  static async getScore(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 计算最新评分
      const score = await SmartMoneyScorer.calculateScore(address);
      
      res.json({ success: true, data: score });
    } catch (error) {
      logger.error('获取评分失败', { error });
      
      // 处理特定错误
      if (error instanceof Error && error.message.includes('未找到地址画像')) {
        res.status(404).json({ success: false, message: error.message });
        return;
      }
      
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
  
  /**
   * 更新评分
   */
  static async updateScore(req: Request, res: Response): Promise<void> {
    try {
      const { address } = req.params;
      
      // 参数验证
      if (!address) {
        res.status(400).json({ success: false, message: '地址参数缺失' });
        return;
      }
      
      // 更新评分
      const updatedProfile = await SmartMoneyScorer.updateScore(address);
      
      if (!updatedProfile) {
        res.status(404).json({ success: false, message: '未找到该地址的聪明钱画像' });
        return;
      }
      
      res.json({ success: true, data: updatedProfile.score });
    } catch (error) {
      logger.error('更新评分失败', { error });
      res.status(500).json({ success: false, message: '服务器错误' });
    }
  }
} 