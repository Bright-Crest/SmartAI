import { ApiResponse, PaginatedResponse } from '../types/api';
import crypto from 'crypto';

/**
 * API响应工具类
 * 用于生成标准化的API响应
 */
export class ApiResponseUtils {
  /**
   * 生成成功响应
   * @param data 响应数据
   * @param message 响应消息
   * @returns 标准化的API响应
   */
  static success<T>(data?: T, message: string = '操作成功'): ApiResponse<T> {
    return {
      code: 200,
      message,
      data,
      requestId: this.generateRequestId(),
      timestamp: Date.now()
    };
  }
  
  /**
   * 生成错误响应
   * @param code 错误代码
   * @param message 错误消息
   * @param data 错误数据
   * @returns 标准化的API响应
   */
  static error<T>(code: number = 500, message: string = '服务器错误', data?: T): ApiResponse<T> {
    return {
      code,
      message,
      data,
      requestId: this.generateRequestId(),
      timestamp: Date.now()
    };
  }
  
  /**
   * 生成分页响应
   * @param items 数据项列表
   * @param total 总数
   * @param page 当前页码
   * @param pageSize 每页大小
   * @param message 响应消息
   * @returns 标准化的分页响应
   */
  static paginated<T>(
    items: T[],
    total: number,
    page: number,
    pageSize: number,
    message: string = '查询成功'
  ): PaginatedResponse<T> {
    return {
      code: 200,
      message,
      data: {
        items,
        total,
        page,
        pageSize,
        totalPages: Math.ceil(total / pageSize)
      },
      requestId: this.generateRequestId(),
      timestamp: Date.now()
    };
  }
  
  /**
   * 生成业务错误响应（HTTP状态码仍为200，但code值表示业务错误）
   * @param code 业务错误代码
   * @param message 错误消息
   * @param data 错误数据
   * @returns 标准化的API响应
   */
  static businessError<T>(code: number = 400, message: string = '请求错误', data?: T): ApiResponse<T> {
    return {
      code,
      message,
      data,
      requestId: this.generateRequestId(),
      timestamp: Date.now()
    };
  }
  
  /**
   * 生成参数错误响应
   * @param message 错误消息
   * @returns 标准化的API响应
   */
  static paramError(message: string = '参数错误'): ApiResponse {
    return this.businessError(400, message);
  }
  
  /**
   * 生成授权错误响应
   * @param message 错误消息
   * @returns 标准化的API响应
   */
  static authError(message: string = '未授权'): ApiResponse {
    return this.businessError(401, message);
  }
  
  /**
   * 生成请求ID
   * @returns 请求ID
   */
  private static generateRequestId(): string {
    return crypto.randomBytes(16).toString('hex');
  }
} 