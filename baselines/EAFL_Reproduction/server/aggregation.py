import torch
import numpy as np
import logging

class TwoStageAggregation:
    """两阶段聚合策略"""
    
    def __init__(self, staleness_weight_func='inverse'):
        self.staleness_weight_func = staleness_weight_func
        self.logger = logging.getLogger(__name__)
    
    def intra_cluster_aggregation(self, gradients_list, staleness_list, data_sizes):
        """聚类内聚合 - 半异步 + staleness-aware"""
        if not gradients_list:
            return None, None
        
        # 计算staleness权重
        weights = self._compute_staleness_weights(staleness_list)
        
        # 数据大小加权
        total_data = sum(data_sizes)
        if total_data > 0:
            data_weights = [size / total_data for size in data_sizes]
        else:
            data_weights = [1.0 / len(data_sizes)] * len(data_sizes)
        
        # 组合权重
        combined_weights = [w1 * w2 for w1, w2 in zip(weights, data_weights)]
        combined_weights = np.array(combined_weights)
        if combined_weights.sum() > 0:
            combined_weights = combined_weights / combined_weights.sum()
        
        # 加权聚合梯度
        aggregated_gradient = None
        for grad, weight in zip(gradients_list, combined_weights):
            if aggregated_gradient is None:
                aggregated_gradient = [g.clone() * weight for g in grad]
            else:
                for i, g in enumerate(grad):
                    aggregated_gradient[i] += g * weight
        
        return aggregated_gradient, combined_weights
    
    def inter_cluster_aggregation(self, cluster_gradients_list, cluster_data_sizes):
        """聚类间聚合 - 同步 + data size-aware"""
        if not cluster_gradients_list:
            return None
        
        # 数据大小加权
        total_data = sum(cluster_data_sizes)
        if total_data > 0:
            data_weights = [size / total_data for size in cluster_data_sizes]
        else:
            data_weights = [1.0 / len(cluster_data_sizes)] * len(cluster_data_sizes)
        
        # 加权聚合
        global_gradient = None
        for grad, weight in zip(cluster_gradients_list, data_weights):
            if global_gradient is None:
                global_gradient = [g.clone() * weight for g in grad]
            else:
                for i, g in enumerate(grad):
                    global_gradient[i] += g * weight
        
        return global_gradient
    
    def _compute_staleness_weights(self, staleness_list):
        """计算staleness权重"""
        weights = []
        for staleness in staleness_list:
            staleness = max(1, staleness)  # 避免除零
            if self.staleness_weight_func == 'inverse':
                weight = 1.0 / staleness
            elif self.staleness_weight_func == 'inverse_sqrt':
                weight = 1.0 / np.sqrt(staleness)
            elif self.staleness_weight_func == 'exponential':
                weight = np.exp(-staleness / 10.0)
            elif self.staleness_weight_func == 'linear':
                weight = max(0, 1 - staleness / 20.0)
            else:
                weight = 1.0 / staleness
            weights.append(weight)
        
        # 归一化
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
        
        return weights