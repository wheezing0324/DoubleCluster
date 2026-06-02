import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_dir='./logs', level=logging.INFO):
    """设置日志记录器"""
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir='./results/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        
    def log_metrics(self, iteration, metrics):
        """记录指标"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] Iteration {iteration}: "
        
        for key, value in metrics.items():
            log_line += f"{key}={value:.4f} "
            
            # 保存到字典
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        print(log_line)
        
        # 写入文件
        with open(self.log_dir / 'training.log', 'a') as f:
            f.write(log_line + '\n')
    
    def save_metrics(self, filename='metrics.npz'):
        """保存指标"""
        np.savez(self.log_dir / filename, **self.metrics)
    
    def get_metrics(self):
        """获取所有指标"""
        return self.metrics