import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 然后才是其他导入
import torch
import numpy as np
import random
import yaml
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time

from models.cnn_mnist import CNN_MNIST, CNN_MNIST_Simple
from models.lenet_cifar10 import LeNet_CIFAR10, LeNet_CIFAR10_Simple
from clients.heterogeneous_client import HeterogeneousClient
from server.eafl_server import EAFLServer
from utils.data_utils import NonIIDDataPartitioner
from utils.metrics import compute_accuracy, compute_loss
from utils.logger import setup_logger, TrainingLogger

def setup_logging(config):
    """设置日志"""
    log_dir = Path(config['experiment']['output_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name=config['experiment']['name'],
        log_dir=str(log_dir),
        level=logging.INFO
    )
    
    return logger

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_clients(client_datasets, config, device):
    """创建异构客户端"""
    clients = []
    
    # 模拟异构计算能力（Beta分布，大多数客户端速度较慢）
    num_clients = len(client_datasets)
    np.random.seed(config['experiment']['seed'])
    speed_factors = np.random.beta(2, 5, num_clients)
    
    # 创建模型
    if config['dataset']['name'] == 'mnist':
        if config['model']['name'] == 'cnn_mnist':
            model_class = CNN_MNIST
        else:
            model_class = CNN_MNIST_Simple
    else:
        if config['model']['name'] == 'lenet_cifar10':
            model_class = LeNet_CIFAR10
        else:
            model_class = LeNet_CIFAR10_Simple
    
    for i, dataset in enumerate(client_datasets):
        model = model_class()
        speed = speed_factors[i]
        
        client = HeterogeneousClient(
            client_id=i,
            model=model,
            dataset=dataset,
            device=device,
            speed_factor=speed
        )
        clients.append(client)
    
    logger = logging.getLogger(config['experiment']['name'])
    logger.info(f"Created {len(clients)} clients with speed factors: "
                f"mean={speed_factors.mean():.3f}, std={speed_factors.std():.3f}")
    
    return clients

def run_experiment(config):
    """运行实验"""
    # 设置
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(config['experiment']['seed'])
    logger = setup_logging(config)
    training_logger = TrainingLogger(log_dir=config['experiment']['output_dir'] + '/logs')
    
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    # 加载数据
    logger.info("Loading datasets...")
    partitioner = NonIIDDataPartitioner(
        dataset_name=config['dataset']['name'],
        num_clients=config['dataset']['num_clients'],
        non_iid_type=config['dataset']['non_iid_type'],
        non_iid_param=config['dataset']['non_iid_param']
    )
    
    train_dataset, test_dataset = partitioner.load_dataset()
    client_datasets = partitioner.partition_data(train_dataset)
    test_loader = partitioner.get_test_loader(test_dataset, batch_size=128)
    
    logger.info(f"Created {len(client_datasets)} clients")
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # 统计客户端数据分布
    data_sizes = [len(ds) for ds in client_datasets]
    logger.info(f"Client data sizes: min={min(data_sizes)}, max={max(data_sizes)}, "
                f"mean={np.mean(data_sizes):.1f}, std={np.std(data_sizes):.1f}")
    
    # 创建客户端
    clients = create_clients(client_datasets, config, device)
    
    # 创建全局模型
    if config['dataset']['name'] == 'mnist':
        if config['model']['name'] == 'cnn_mnist':
            global_model = CNN_MNIST().to(device)
        else:
            global_model = CNN_MNIST_Simple().to(device)
    else:
        if config['model']['name'] == 'lenet_cifar10':
            global_model = LeNet_CIFAR10().to(device)
        else:
            global_model = LeNet_CIFAR10_Simple().to(device)
    
    # 创建服务器
    server = EAFLServer(config, global_model, device)
    
    # 训练
    logger.info("Starting training...")
    start_time = time.time()
    
    results = {
        'iterations': [],
        'accuracy': [],
        'loss': [],
        'time': []
    }
    
    best_accuracy = 0.0
    best_iteration = 0
    
    for iteration in range(1, config['training']['max_iterations'] + 1):
        # 训练步骤
        server.train_step(clients)
        
        # 定期评估
        if iteration % config['evaluation']['test_frequency'] == 0 or iteration == 1:
            accuracy, loss = server.evaluate(test_loader)
            
            results['iterations'].append(iteration)
            results['accuracy'].append(accuracy)
            results['loss'].append(loss)
            
            # 记录训练时间
            if len(server.training_history['time']) > 0:
                avg_time = np.mean(server.training_history['time'][-10:])
            else:
                avg_time = 0
            results['time'].append(avg_time)
            
            # 记录到日志
            training_logger.log_metrics(iteration, {
                'accuracy': accuracy,
                'loss': loss,
                'time': avg_time
            })
            
            logger.info(f"Iteration {iteration}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_iteration = iteration
                checkpoint_dir = Path(config['experiment']['output_dir']) / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                server.save_checkpoint(checkpoint_dir / 'best_model.pt')
                logger.info(f"New best model at iteration {iteration}: Accuracy = {accuracy:.4f}")
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best accuracy: {best_accuracy:.4f} at iteration {best_iteration}")
    logger.info("=" * 60)
    
    # 保存结果
    save_results(results, config)
    
    return results

def save_results(results, config):
    """保存结果"""
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制准确率曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['iterations'], results['accuracy'], 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title(f"Test Accuracy - {config['experiment']['name']}")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['iterations'], results['loss'], 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f"Test Loss - {config['experiment']['name']}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{config['experiment']['name']}_results.png", dpi=150)
    plt.close()
    
    # 保存数值结果
    with open(output_dir / f"{config['experiment']['name']}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存最终模型
    logger = logging.getLogger(config['experiment']['name'])
    logger.info(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='EAFL Experiment Runner')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'ablation', 'hyperparam'], 
                        default='train', help='Experiment mode')
    args = parser.parse_args()
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} not found")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 运行实验
    if args.mode == 'train':
        run_experiment(config)
    elif args.mode == 'ablation':
        run_ablation_study(config)
    elif args.mode == 'hyperparam':
        run_hyperparameter_search(config)
    
    return 0

def run_ablation_study(config):
    """运行消融实验"""
    logger = logging.getLogger('ablation')
    logger.info("Running ablation study...")
    
    # 不同的配置变体
    variants = {
        'full': config.copy(),
        'no_gdc': config.copy(),
        'no_saa': config.copy(),
        'no_dsa': config.copy()
    }
    
    # 修改配置
    variants['no_gdc']['eafl']['re_clustering_interval'] = float('inf')  # 不重新聚类
    
    variants['no_saa']['eafl']['staleness_weight'] = 'uniform'  # 均匀权重
    
    variants['no_dsa']['eafl']['participation_ratio'] = 1.0  # 所有客户端参与
    
    results = {}
    for name, variant_config in variants.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {name} variant")
        logger.info(f"{'='*60}")
        
        # 修改实验名称
        variant_config['experiment']['name'] = f"{config['experiment']['name']}_{name}"
        
        # 运行实验
        result = run_experiment(variant_config)
        results[name] = result
    
    # 比较结果
    compare_ablation_results(results, config)

def compare_ablation_results(results, config):
    """比较消融实验结果"""
    output_dir = Path(config['experiment']['output_dir'])
    
    plt.figure(figsize=(10, 6))
    
    colors = {'full': 'blue', 'no_gdc': 'red', 'no_saa': 'green', 'no_dsa': 'orange'}
    markers = {'full': 'o', 'no_gdc': 's', 'no_saa': '^', 'no_dsa': 'D'}
    
    for name, result in results.items():
        if result and 'iterations' in result and 'accuracy' in result:
            plt.plot(result['iterations'], result['accuracy'], 
                    color=colors[name], marker=markers[name], 
                    markevery=len(result['iterations'])//10,
                    label=name.upper(), linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Ablation Study Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'ablation_study.png', dpi=150)
    plt.close()

def run_hyperparameter_search(config):
    """运行超参数搜索"""
    logger = logging.getLogger('hyperparam')
    logger.info("Running hyperparameter search...")
    
    # 搜索空间
    search_space = {
        'num_clusters': [3, 5, 7, 10],
        're_clustering_interval': [50, 100, 200],
        'participation_ratio': [0.05, 0.1, 0.2],
        'learning_rate': [0.001, 0.005, 0.01]
    }
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    # 简单网格搜索
    for num_clusters in search_space['num_clusters'][:2]:  # 减少搜索量
        for re_interval in search_space['re_clustering_interval'][:2]:
            for participation in search_space['participation_ratio'][:2]:
                for lr in search_space['learning_rate'][:2]:
                    # 修改配置
                    test_config = config.copy()
                    test_config['eafl']['num_clusters'] = num_clusters
                    test_config['eafl']['re_clustering_interval'] = re_interval
                    test_config['eafl']['participation_ratio'] = participation
                    test_config['training']['learning_rate'] = lr
                    test_config['training']['max_iterations'] = 200  # 减少迭代次数用于搜索
                    test_config['experiment']['name'] = f"search_{num_clusters}_{re_interval}_{participation}_{lr}"
                    
                    logger.info(f"\nTesting: N={num_clusters}, R={re_interval}, φ={participation}, lr={lr}")
                    
                    try:
                        result = run_experiment(test_config)
                        final_accuracy = result['accuracy'][-1] if result['accuracy'] else 0
                        
                        results.append({
                            'params': {
                                'num_clusters': num_clusters,
                                're_clustering_interval': re_interval,
                                'participation_ratio': participation,
                                'learning_rate': lr
                            },
                            'accuracy': final_accuracy
                        })
                        
                        if final_accuracy > best_accuracy:
                            best_accuracy = final_accuracy
                            best_params = test_config['eafl']
                            best_params['learning_rate'] = lr
                            
                    except Exception as e:
                        logger.error(f"Search failed: {e}")
    
    # 保存搜索结果
    output_dir = Path(config['experiment']['output_dir'])
    with open(output_dir / 'hyperparam_search.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'all_results': results
        }, f, indent=2)
    
    logger.info(f"\nBest parameters: {best_params}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    exit(main())