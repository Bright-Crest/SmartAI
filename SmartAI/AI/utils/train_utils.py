import os
import torch
import numpy as np
import random
import logging
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import matplotlib.pyplot as plt
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    设置随机种子，确保实验可重复性
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置为确定性运行模式（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"随机种子已设置为: {seed}")


def get_device(device_name: str = 'cuda') -> torch.device:
    """
    获取设备对象
    
    参数:
        device_name (str): 设备名称，'cuda'或'cpu'
        
    返回:
        torch.device: 设备对象
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")
    
    return device


def create_optimizer(
    model: torch.nn.Module, 
    config: Dict[str, Any]
) -> Optimizer:
    """
    创建优化器
    
    参数:
        model (nn.Module): 模型
        config (Dict): 优化器配置
        
    返回:
        Optimizer: 优化器实例
    """
    optimizer_name = config.get('name', 'adam').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0)
    
    if optimizer_name == 'adam':
        amsgrad = config.get('amsgrad', False)
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            amsgrad=amsgrad
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', False)
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay, 
            nesterov=nesterov
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def create_scheduler(
    optimizer: Optimizer, 
    config: Dict[str, Any], 
    epochs: int
) -> Optional[_LRScheduler]:
    """
    创建学习率调度器
    
    参数:
        optimizer (Optimizer): 优化器
        config (Dict): 调度器配置
        epochs (int): 总训练轮数
        
    返回:
        _LRScheduler or None: 学习率调度器实例
    """
    scheduler_name = config.get('name', 'cosine').lower()
    
    if scheduler_name == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    elif scheduler_name == 'cosine':
        min_lr = config.get('min_lr', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=min_lr
        )
    elif scheduler_name == 'plateau':
        factor = config.get('factor', 0.5)
        patience = config.get('patience', 5)
        min_lr = config.get('min_lr', 1e-6)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=factor, 
            patience=patience, 
            min_lr=min_lr
        )
    elif scheduler_name == 'none' or not scheduler_name:
        return None
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")


class AverageMeter:
    """
    跟踪和计算平均值和当前值
    """
    def __init__(self, name: str):
        """
        参数:
            name (str): 度量名称
        """
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """
        重置所有统计信息
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计信息
        
        参数:
            val (float): 当前值
            n (int): 批次大小
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    早停类，用于防止过拟合
    """
    def __init__(
        self, 
        patience: int = 10, 
        verbose: bool = True, 
        delta: float = 0, 
        mode: str = 'min', 
        save_path: Optional[str] = None
    ):
        """
        参数:
            patience (int): 停止前等待的轮数
            verbose (bool): 是否打印信息
            delta (float): 最小变化阈值
            mode (str): 监控模式，'min'或'max'
            save_path (str, 可选): 最佳模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        # 初始化最佳值
        if mode == 'min':
            self.best_val = float('inf')
        else:
            self.best_val = float('-inf')
    
    def __call__(
        self, 
        val: float, 
        model: torch.nn.Module, 
        epoch: int
    ) -> bool:
        """
        检查是否应该早停
        
        参数:
            val (float): 当前验证指标
            model (nn.Module): 当前模型
            epoch (int): 当前轮数
            
        返回:
            bool: 是否为最佳模型
        """
        score = -val if self.mode == 'min' else val
        
        if self.best_score is None:
            self.best_score = score
            self.best_val = val
            self.best_epoch = epoch
            self.save_checkpoint(model)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_val = val
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
            return True
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """
        保存最佳模型
        
        参数:
            model (nn.Module): 当前模型
        """
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                logger.info(f'模型保存到 {self.save_path}')


def plot_learning_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    save_path: Optional[str] = None
) -> None:
    """
    绘制训练和验证损失曲线
    
    参数:
        train_losses (List[float]): 训练损失列表
        val_losses (List[float]): 验证损失列表
        save_path (str, 可选): 图像保存路径
    """
    try:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='训练损失')
        plt.plot(epochs, val_losses, 'r-', label='验证损失')
        
        plt.title('训练和验证损失')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 保存或显示图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        logger.warning("未安装matplotlib，无法绘制学习曲线")


def print_config(config: DictConfig) -> None:
    """
    打印配置信息
    
    参数:
        config (DictConfig): Hydra配置对象
    """
    logger.info("配置信息:")
    
    # 基本配置
    logger.info(f"项目名称: {config.project_name}")
    logger.info(f"随机种子: {config.seed}")
    logger.info(f"设备: {config.device}")
    
    # 数据配置
    logger.info(f"数据目录: {config.data.graphs_dir}")
    logger.info(f"批大小: {config.data.batch_size}")
    logger.info(f"时间窗口大小: {config.data.time_window}秒")
    logger.info(f"最大时间窗口数: {config.data.max_time_windows}")
    
    # 模型配置
    logger.info(f"模型名称: {config.model.name}")
    logger.info(f"嵌入维度: {config.model.embedding_dim}")
    logger.info(f"隐藏层维度: {config.model.hidden_dim}")
    logger.info(f"输出维度: {config.model.output_dim}")
    
    # 训练配置
    if 'train' in config:
        logger.info(f"训练名称: {config.train.name}")
        logger.info(f"训练轮数: {config.train.epochs}")
        logger.info(f"优化器: {config.train.optimizer.name}, 学习率: {config.train.optimizer.lr}")
        logger.info(f"学习率调度器: {config.train.lr_scheduler.name}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    loss: float,
    path: str,
    args: Dict[str, Any]
) -> None:
    """
    保存检查点
    
    参数:
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        scheduler (_LRScheduler, 可选): 学习率调度器
        epoch (int): 当前轮数
        loss (float): 当前损失
        path (str): 保存路径
        args (Dict): 其他参数
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    logger.info(f"检查点已保存到 {path}")


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载检查点
    
    参数:
        model (nn.Module): 模型
        path (str): 检查点路径
        optimizer (Optimizer, 可选): 优化器
        scheduler (_LRScheduler, 可选): 学习率调度器
        device (torch.device, 可选): 设备
        
    返回:
        Dict: 检查点信息
    """
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"检查点已从 {path} 加载，轮数: {checkpoint['epoch']}")
    
    return checkpoint


def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """
    设置日志
    
    参数:
        log_dir (str): 日志目录
        level (int): 日志级别
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    log_file = os.path.join(log_dir, 'train.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logger.info(f"日志已设置，保存到 {log_file}")


def create_directories(config: DictConfig) -> None:
    """
    创建所需的目录结构
    
    参数:
        config (DictConfig): Hydra配置对象
    """
    dirs = [
        config.output.log_dir,
        config.output.model_dir,
        config.output.tensorboard_dir,
        config.output.results_dir,
        config.output.plots_dir
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")


def init_tensorboard(log_dir: str) -> Any:
    """
    初始化TensorBoard
    
    参数:
        log_dir (str): TensorBoard日志目录
        
    返回:
        SummaryWriter: TensorBoard写入器
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard已初始化，日志保存到 {log_dir}")
        return tb_writer
    except ImportError:
        logger.warning("未安装tensorboard，将不使用TensorBoard")
        return None 