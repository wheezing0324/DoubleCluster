import torch
import numpy as np
import logging
import time
from collections import defaultdict
from pathlib import Path

from server.aggregation import TwoStageAggregation
from clustering.gradient_clustering import GradientClustering

class EAFLServer:
    """EAFL服务器实现"""
    
    def __init__(self, config, global_model, device):
        self.config = config
        self.global_model = global_model
        self.device = device
        
        # EAFL参数
        self.num_clusters = config['eafl']['num_clusters']
        self.re_clustering_interval = config['eafl']['re_clustering_interval']
        self.participation_ratio = config['eafl']['participation_ratio']
        
        # 状态变量
        self.clusters = None
        self.cluster_heads = None
        self.current_iteration = 0
        self.training_history = {
            'iterations': [],
            'accuracy': [],
            'loss': [],
            'time': []
        }
        
        # 聚合器
        self.aggregator = TwoStageAggregation(
            staleness_weight_func=config['eafl']['staleness_weight']
        )
        
        # 聚类器
        self.clustering = GradientClustering(
            num_clusters=self.num_clusters
        )
        
        self.logger = logging.getLogger(__name__)
        
    def perform_clustering(self, clients):
        """执行动态聚类"""
        self.logger.info(f"Performing clustering at iteration {self.current_iteration}")
        
        # 获取当前全局模型参数
        global_params = self.global_model.state_dict()
        
        # 收集所有客户端的梯度（使用新方法）
        all_gradients = []
        for client in clients:
            grad = client.get_gradients_with_global_model(global_params)
            all_gradients.append(grad)
        
        # 执行聚类
        clusters, cluster_heads = self.clustering.cluster_clients(all_gradients)
        
        self.clusters = clusters
        self.cluster_heads = cluster_heads
        
        return clusters, cluster_heads

    def select_participating_clients(self, cluster_clients, all_clients):
        """在每个聚类中选择参与训练的客户端"""
        if len(cluster_clients) == 0:
            return []
        
        # 选择最快的 φ 比例的客户端
        num_participants = max(1, int(len(cluster_clients) * self.participation_ratio))
        
        # 根据计算速度排序（速度因子越大越快）
        client_speeds = [(idx, all_clients[idx].speed_factor) for idx in cluster_clients]
        client_speeds.sort(key=lambda x: x[1], reverse=True)
        
        participating = [idx for idx, _ in client_speeds[:num_participants]]
        
        return participating
    
    # server/eafl_server.py
    def update_global_model(self, global_gradient):
        with torch.no_grad():
            for param, grad in zip(self.global_model.parameters(), global_gradient):
                if grad is not None:
                    param.data -= grad.to(self.device)   # 注意是减号
        return self.global_model.state_dict()
        
    def train_step(self, clients):
        """执行一轮训练"""
        start_time = time.time()
        self.current_iteration += 1
        t = self.current_iteration
        
        # 动态聚类（每R轮）
        if t % self.re_clustering_interval == 0 or t == 1:
            self.perform_clustering(clients)
        
        if self.clusters is None:
            self.logger.warning("No clusters available, skipping training step")
            return self.global_model.state_dict()
        
        # 对每个聚类进行异步聚合
        cluster_updates = []
        cluster_data_sizes = []
        
        for cluster_id, cluster_clients in self.clusters.items():
            # 选择参与本次迭代的客户端
            participating = self.select_participating_clients(cluster_clients, clients)
            
            if not participating:
                continue
            
            # 收集本地更新
            local_gradients = []
            staleness_list = []
            data_sizes = []
            
            for client_idx in participating:
                client = clients[client_idx]
                
                # 计算staleness
                staleness = t - client.last_participation_iter
                client.set_staleness(staleness)
                
                # 本地训练
                global_params = self.global_model.state_dict()
                grad, data_size = client.local_train(
                    global_params,
                    num_epochs=self.config['training']['local_epochs'],
                    lr=self.config['training']['learning_rate']
                )
                
                if data_size > 0:  # 只有有数据的客户端才参与
                    local_gradients.append(grad)
                    staleness_list.append(staleness)
                    data_sizes.append(data_size)
                    
                    # 更新参与时间
                    client.last_participation_iter = t
            
            # 聚类内聚合
            if local_gradients:
                aggregated_grad, weights = self.aggregator.intra_cluster_aggregation(
                    local_gradients, staleness_list, data_sizes
                )
                
                cluster_updates.append(aggregated_grad)
                cluster_data_sizes.append(sum(data_sizes))
        
        # 聚类间聚合并更新全局模型
        if cluster_updates:
            global_gradient = self.aggregator.inter_cluster_aggregation(
                cluster_updates, cluster_data_sizes
            )
            
            if global_gradient is not None:
                self.update_global_model(global_gradient)
        
        # 记录训练时间
        training_time = time.time() - start_time
        self.training_history['time'].append(training_time)
        
        return self.global_model.state_dict()
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                total_loss += loss.item() * data.size(0)
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        
        self.global_model.train()
        
        return accuracy, avg_loss
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_history
    
    def save_checkpoint(self, filepath):
        """保存检查点"""
        checkpoint = {
            'iteration': self.current_iteration,
            'model_state_dict': self.global_model.state_dict(),
            'clusters': self.clusters,
            'cluster_heads': self.cluster_heads,
            'training_history': self.training_history,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.clusters = checkpoint['clusters']
        self.cluster_heads = checkpoint['cluster_heads']
        self.training_history = checkpoint['training_history']
        self.logger.info(f"Checkpoint loaded from {filepath}")