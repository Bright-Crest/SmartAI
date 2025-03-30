import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import { rateLimit } from 'express-rate-limit';
import { Database } from './database/Database';
import { logger } from './utils/logger';
import riskRoutes from './routes/riskRoutes';
import { configureExpress, configureResponseFormat } from './config/express';
import smartMoneyRoutes from './routes/smartMoneyRoutes';
import { errorHandler } from './middleware/errorHandler';
import { loggerWinston } from './utils/logger';
import { connectDB } from './database/connect';
import { KnexService } from './database/KnexService';
import { SmartMoneyTracker } from './core/SmartMoneyTracker';

// 路由导入
import userRoutes from './routes/userRoutes';
import authRoutes from './routes/authRoutes';
import dataRoutes from './routes/dataRoutes';

// 创建Express应用
const app = express();

// 应用配置
configureExpress(app);
configureResponseFormat(app);

// 中间件配置
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json());
app.use(morgan('combined', { stream: { write: (message) => loggerWinston.info(message.trim()) } }));

// 速率限制
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15分钟
  max: 100 // 限制每个IP 15分钟内最多100个请求
});
app.use(limiter);

// 数据库连接
connectDB().then(() => {
  loggerWinston.info('MongoDB连接成功');
}).catch(err => {
  loggerWinston.error('MongoDB连接失败', { error: err });
  process.exit(1);
});

// 初始化KnexService
const knexService = new KnexService();
knexService.init();

// 初始化SmartMoneyTracker
SmartMoneyTracker.init(knexService.getKnex());

// 路由
app.use('/api/risk', riskRoutes);
app.use('/api/users', userRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/data', dataRoutes);
app.use('/api/smart-money', smartMoneyRoutes);

// 根路由
app.get('/', (req, res) => {
  res.json({ message: 'SmartAI API 服务运行中' });
});

// 健康检查
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok', environment: process.env.NODE_ENV || 'development' });
});

// 错误处理
app.use(errorHandler);

// 404处理
app.use((req, res: any) => {
  res.fail(404, '未找到请求的资源');
});

// 启动服务器
const PORT = process.env.PORT || 3001;

async function startServer() {
  try {
    // 连接数据库
    await Database.getConnection();
    
    // 启动服务器
    const server = app.listen(PORT, () => {
      loggerWinston.info(`服务器已启动，监听端口: ${PORT}`);
    });

    // 处理未捕获的异常
    process.on('uncaughtException', (err) => {
      loggerWinston.error('未捕获的异常', { error: err.stack });
      // 给进程一些时间来完成日志记录
      setTimeout(() => {
        process.exit(1);
      }, 1000);
    });

    // 处理未处理的Promise拒绝
    process.on('unhandledRejection', (reason, promise) => {
      loggerWinston.error('未处理的Promise拒绝', { reason, promise });
    });

    // 处理进程终止信号
    process.on('SIGTERM', () => {
      loggerWinston.info('SIGTERM信号收到，优雅关闭中');
      server.close(() => {
        loggerWinston.info('进程已终止');
        process.exit(0);
      });
    });
  } catch (error) {
    loggerWinston.error('服务器启动失败', {
      error: error instanceof Error ? error.message : String(error)
    });
    process.exit(1);
  }
}

// 启动应用
startServer();

export default app; 