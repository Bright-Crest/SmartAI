import { Request, Response, NextFunction } from 'express';
import zlib from 'zlib';
import { logger } from './logger';

/**
 * 数据优化工具 - 用于提升API响应性能
 */
export class DataOptimizer {
  /**
   * 压缩API响应数据
   * 当响应数据较大时自动应用gzip压缩
   * @param req 请求对象
   * @param res 响应对象
   * @param data 响应数据
   * @param threshold 压缩阈值(字节)，默认1KB
   */
  public static compressResponse(
    req: Request, 
    res: Response, 
    data: any, 
    threshold: number = 1024
  ): void {
    // 检查客户端是否支持压缩
    const acceptEncoding = req.headers['accept-encoding'] || '';
    
    // 转换数据为JSON字符串
    const jsonData = JSON.stringify(data);
    
    // 小于阈值，无需压缩
    if (jsonData.length < threshold) {
      res.json(data);
      return;
    }
    
    // 根据客户端支持选择压缩方式
    if (acceptEncoding.includes('gzip')) {
      zlib.gzip(jsonData, (err, compressed) => {
        if (err) {
          logger.error('响应压缩失败', { error: err.message });
          res.json(data); // 压缩失败，返回未压缩数据
          return;
        }
        
        res.setHeader('Content-Encoding', 'gzip');
        res.setHeader('Content-Type', 'application/json');
        res.send(compressed);
      });
    } else if (acceptEncoding.includes('deflate')) {
      zlib.deflate(jsonData, (err, compressed) => {
        if (err) {
          logger.error('响应压缩失败', { error: err.message });
          res.json(data);
          return;
        }
        
        res.setHeader('Content-Encoding', 'deflate');
        res.setHeader('Content-Type', 'application/json');
        res.send(compressed);
      });
    } else {
      // 客户端不支持压缩
      res.json(data);
    }
  }
  
  /**
   * 响应数据精简
   * 根据query参数fields只返回需要的字段，减少数据量
   * @param data 原始数据对象或数组
   * @param fields 需要保留的字段，用逗号分隔
   */
  public static filterFields<T>(data: T, fields?: string): T {
    if (!fields || !data) {
      return data;
    }
    
    const fieldList = fields.split(',').map(f => f.trim());
    
    // 处理数组
    if (Array.isArray(data)) {
      return data.map(item => DataOptimizer.filterObjectFields(item, fieldList)) as T;
    }
    
    // 处理单个对象
    return DataOptimizer.filterObjectFields(data, fieldList);
  }
  
  /**
   * 过滤单个对象的字段
   * @param obj 原始对象
   * @param fields 需要保留的字段列表
   */
  private static filterObjectFields<T>(obj: T, fields: string[]): T {
    if (!obj || typeof obj !== 'object' || fields.length === 0) {
      return obj;
    }
    
    const result: Record<string, any> = {};
    
    for (const field of fields) {
      // 处理嵌套字段 (user.name)
      if (field.includes('.')) {
        const [parent, child] = field.split('.', 2);
        if (Object.prototype.hasOwnProperty.call(obj, parent)) {
          result[parent] = result[parent] || {};
          
          const parentObj = (obj as Record<string, any>)[parent];
          if (parentObj && typeof parentObj === 'object') {
            result[parent][child] = parentObj[child];
          }
        }
      } 
      // 处理普通字段
      else if (Object.prototype.hasOwnProperty.call(obj, field)) {
        result[field] = (obj as Record<string, any>)[field];
      }
    }
    
    return result as T;
  }
  
  /**
   * 字段选择中间件
   * 基于query参数fields过滤响应字段
   */
  public static fieldSelector() {
    return (req: Request, res: Response, next: NextFunction) => {
      const fields = req.query.fields as string;
      
      // 如果没有指定字段，跳过处理
      if (!fields) {
        return next();
      }
      
      // 保存原始的json方法
      const originalJson = res.json;
      
      // 重写json方法，添加字段过滤
      res.json = function(data: any) {
        // 恢复原始方法
        res.json = originalJson;
        
        // 过滤字段
        const filteredData = DataOptimizer.filterFields(data, fields);
        
        // 调用原始方法发送响应
        return originalJson.call(this, filteredData);
      };
      
      next();
    };
  }
  
  /**
   * 分页处理工具
   * @param data 原始数据数组
   * @param page 页码
   * @param pageSize 每页数量
   */
  public static paginateData<T>(data: T[], page: number = 1, pageSize: number = 20): {
    data: T[];
    pagination: {
      total: number;
      page: number;
      pageSize: number;
      pages: number;
    }
  } {
    const total = data.length;
    const pages = Math.ceil(total / pageSize);
    const validPage = Math.max(1, Math.min(page, pages || 1));
    
    const start = (validPage - 1) * pageSize;
    const end = Math.min(start + pageSize, total);
    const pagedData = data.slice(start, end);
    
    return {
      data: pagedData,
      pagination: {
        total,
        page: validPage,
        pageSize,
        pages
      }
    };
  }
  
  /**
   * 按照指定字段排序数据
   * @param data 原始数据数组
   * @param sortBy 排序字段
   * @param order 排序方向 ('asc'|'desc')
   */
  public static sortData<T>(data: T[], sortBy: string, order: 'asc' | 'desc' = 'asc'): T[] {
    if (!sortBy || !data || data.length === 0) {
      return data;
    }
    
    // 创建数据副本进行排序，避免修改原始数据
    const sortedData = [...data];
    
    sortedData.sort((a, b) => {
      // 处理嵌套属性
      const fields = sortBy.split('.');
      let valueA: any = a;
      let valueB: any = b;
      
      for (const field of fields) {
        if (valueA && typeof valueA === 'object') {
          valueA = valueA[field];
        } else {
          valueA = undefined;
          break;
        }
        
        if (valueB && typeof valueB === 'object') {
          valueB = valueB[field];
        } else {
          valueB = undefined;
          break;
        }
      }
      
      // 处理null或undefined值
      if (valueA === undefined || valueA === null) return order === 'asc' ? -1 : 1;
      if (valueB === undefined || valueB === null) return order === 'asc' ? 1 : -1;
      
      // 根据值类型进行比较
      if (typeof valueA === 'string' && typeof valueB === 'string') {
        return order === 'asc' 
          ? valueA.localeCompare(valueB) 
          : valueB.localeCompare(valueA);
      }
      
      return order === 'asc' 
        ? (valueA < valueB ? -1 : valueA > valueB ? 1 : 0)
        : (valueA < valueB ? 1 : valueA > valueB ? -1 : 0);
    });
    
    return sortedData;
  }
  
  /**
   * 深度克隆对象，移除敏感信息
   * @param data 原始数据
   * @param sensitiveFields 敏感字段列表
   */
  public static sanitizeData<T>(data: T, sensitiveFields: string[] = []): T {
    if (!data) return data;
    
    // 简单类型直接返回
    if (typeof data !== 'object') return data;
    
    // 数组处理
    if (Array.isArray(data)) {
      return data.map(item => DataOptimizer.sanitizeData(item, sensitiveFields)) as any;
    }
    
    // 对象处理
    const result: Record<string, any> = {};
    
    for (const [key, value] of Object.entries(data)) {
      // 跳过敏感字段
      if (sensitiveFields.includes(key)) {
        continue;
      }
      
      // 递归处理嵌套对象
      if (value && typeof value === 'object') {
        result[key] = DataOptimizer.sanitizeData(value, sensitiveFields);
      } else {
        result[key] = value;
      }
    }
    
    return result as T;
  }
} 