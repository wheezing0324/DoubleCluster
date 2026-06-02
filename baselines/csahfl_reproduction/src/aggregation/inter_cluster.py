"""
Dynamic Distribution-Aware Inter-Cluster Aggregation (Section 3.5)

This module implements the inter-cluster aggregation mechanism for CSAHFL,
which combines quantity-based and distribution-based weighting to aggregate
edge server models into a global model.

Key Equations:
- Eq. 11: w_g = Σ_{k=1}^K λ_k w_edge^k (Global aggregation)
- Eq. 12: ω_k^qua = N_edge^k / Σ_{k=1}^K N_edge^k (Quantity weight)
- Eq. 13: ω_k^dis = 1 / (1 + D_KL(P_k || P_g)) (Distribution weight)
- Eq. 14: λ_k = (ω_k^qua × ω_k^dis) / Σ_{j=1}^K (ω_j^qua × ω_j^dis) (Combined weight)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class InterClusterAggregator:
    """
    Manages distribution-aware inter-cluster aggregation at cloud server level.
    
    Implements Eq. 11-14 from CSAHFL paper (Section 3.5):
    - Calculates quantity weights based on data distribution (Eq. 12)
    - Calculates distribution weights based on KL divergence (Eq. 13)
    - Combines weights for global aggregation (Eq. 14)
    - Performs weighted global model aggregation (Eq. 11)
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        Initialize inter-cluster aggregator.
        
        Args:
            config: Configuration dictionary containing:
                - inter_cluster.kl_epsilon: Epsilon for KL divergence stability
                - system.num_edge_servers: Number of edge servers (K)
                - training.num_classes: Number of classes for distribution estimation
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        self.kl_epsilon = config.get('inter_cluster', {}).get('kl_epsilon', 1e-10)
        self.num_classes = config.get('training', {}).get('num_classes', 10)
        
        # Track cluster information
        self.cluster_data_counts: Dict[int, int] = {}  # N_edge^k per cluster
        self.cluster_distributions: Dict[int, np.ndarray] = {}  # P_k per cluster
        self.global_distribution: Optional[np.ndarray] = None  # P_g (uniform)
        
        # Aggregation statistics
        self.aggregation_history: List[Dict] = []
        
    def set_global_distribution(self, distribution: np.ndarray):
        """
        Set the global target distribution P_g.
        
        Args:
            distribution: Array of shape (num_classes,) representing P_g
                         Typically uniform: [1/num_classes, ..., 1/num_classes]
        """
        self.global_distribution = distribution.astype(np.float64)
        
    def get_uniform_distribution(self) -> np.ndarray:
        """
        Get uniform distribution for global target P_g.
        
        Returns:
            Uniform distribution array of shape (num_classes,)
        """
        uniform = np.ones(self.num_classes) / self.num_classes
        return uniform
        
    def estimate_cluster_distribution(self, client_label_counts: Dict[int, Dict[int, int]]) -> np.ndarray:
        """
        Estimate label distribution P_k for a cluster from client label counts.
        
        Args:
            client_label_counts: Dict mapping client_id to {label: count}
                                Aggregated over all clients in the cluster
            
        Returns:
            Distribution array of shape (num_classes,) summing to 1.0
        """
        # Aggregate counts across all clients in cluster
        total_counts = np.zeros(self.num_classes, dtype=np.float64)
        
        for client_id, label_counts in client_label_counts.items():
            for label, count in label_counts.items():
                if 0 <= label < self.num_classes:
                    total_counts[label] += count
                    
        # Normalize to get distribution
        total_sum = total_counts.sum()
        if total_sum > 0:
            distribution = total_counts / total_sum
        else:
            # Fallback to uniform if no data
            distribution = self.get_uniform_distribution()
            
        return distribution
        
    def calculate_kl_divergence(self, p_k: np.ndarray, p_g: np.ndarray) -> float:
        """
        Calculate KL divergence D_KL(P_k || P_g).
        
        Args:
            p_k: Cluster distribution (P_k) of shape (num_classes,)
            p_g: Global distribution (P_g) of shape (num_classes,)
            
        Returns:
            KL divergence value (non-negative float)
        """
        # Add epsilon for numerical stability
        p_k_safe = np.clip(p_k, self.kl_epsilon, 1.0)
        p_g_safe = np.clip(p_g, self.kl_epsilon, 1.0)
        
        # Normalize to ensure valid probability distributions
        p_k_safe = p_k_safe / p_k_safe.sum()
        p_g_safe = p_g_safe / p_g_safe.sum()
        
        # Calculate KL divergence: D_KL(P_k || P_g) = Σ P_k(i) * log(P_k(i) / P_g(i))
        kl_div = np.sum(p_k_safe * np.log(p_k_safe / p_g_safe))
        
        # Ensure non-negative (numerical errors can cause small negative values)
        kl_div = max(0.0, kl_div)
        
        return kl_div
        
    def calculate_quantity_weight(self, cluster_data_counts: Dict[int, int]) -> Dict[int, float]:
        """
        Calculate quantity weights ω_k^qua for all clusters (Eq. 12).
        
        Args:
            cluster_data_counts: Dict mapping cluster_id to data count N_edge^k
            
        Returns:
            Dict mapping cluster_id to quantity weight ω_k^qua
        """
        total_data = sum(cluster_data_counts.values())
        
        if total_data == 0:
            # Fallback to uniform weights
            num_clusters = len(cluster_data_counts)
            return {k: 1.0 / num_clusters for k in cluster_data_counts}
            
        quantity_weights = {}
        for cluster_id, data_count in cluster_data_counts.items():
            quantity_weights[cluster_id] = data_count / total_data
            
        return quantity_weights
        
    def calculate_distribution_weight(self, cluster_distributions: Dict[int, np.ndarray],
                                     global_distribution: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Calculate distribution weights ω_k^dis for all clusters (Eq. 13).
        
        Args:
            cluster_distributions: Dict mapping cluster_id to distribution P_k
            global_distribution: Global target distribution P_g (default: uniform)
            
        Returns:
            Dict mapping cluster_id to distribution weight ω_k^dis
        """
        if global_distribution is None:
            global_distribution = self.get_uniform_distribution()
            
        distribution_weights = {}
        
        for cluster_id, p_k in cluster_distributions.items():
            kl_div = self.calculate_kl_divergence(p_k, global_distribution)
            # Eq. 13: ω_k^dis = 1 / (1 + D_KL(P_k || P_g))
            distribution_weights[cluster_id] = 1.0 / (1.0 + kl_div)
            
        return distribution_weights
        
    def calculate_combined_weights(self, quantity_weights: Dict[int, float],
                                   distribution_weights: Dict[int, float]) -> Dict[int, float]:
        """
        Calculate combined weights λ_k for all clusters (Eq. 14).
        
        Args:
            quantity_weights: Dict mapping cluster_id to ω_k^qua
            distribution_weights: Dict mapping cluster_id to ω_k^dis
            
        Returns:
            Dict mapping cluster_id to combined weight λ_k
        """
        # Calculate numerator: ω_k^qua × ω_k^dis
        numerators = {}
        for cluster_id in quantity_weights:
            qua_weight = quantity_weights.get(cluster_id, 0.0)
            dis_weight = distribution_weights.get(cluster_id, 0.0)
            numerators[cluster_id] = qua_weight * dis_weight
            
        # Calculate denominator: Σ_{j=1}^K (ω_j^qua × ω_j^dis)
        denominator = sum(numerators.values())
        
        if denominator == 0:
            # Fallback to uniform weights
            num_clusters = len(quantity_weights)
            return {k: 1.0 / num_clusters for k in quantity_weights}
            
        # Eq. 14: λ_k = (ω_k^qua × ω_k^dis) / Σ_{j=1}^K (ω_j^qua × ω_j^dis)
        combined_weights = {}
        for cluster_id, numerator in numerators.items():
            combined_weights[cluster_id] = numerator / denominator
            
        return combined_weights
        
    def perform_global_aggregation(self, edge_models: Dict[int, Dict[str, torch.Tensor]],
                                   combined_weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """
        Perform global model aggregation using combined weights (Eq. 11).
        
        Args:
            edge_models: Dict mapping cluster_id to model state dict
            combined_weights: Dict mapping cluster_id to weight λ_k
            
        Returns:
            Aggregated global model state dict
        """
        if not edge_models:
            raise ValueError("No edge models provided for aggregation")
            
        # Get model keys from first model
        first_model = next(iter(edge_models.values()))
        model_keys = list(first_model.keys())
        
        # Initialize global model
        global_model = {}
        
        for key in model_keys:
            # Eq. 11: w_g = Σ_{k=1}^K λ_k w_edge^k
            aggregated_tensor = None
            
            for cluster_id, model_state in edge_models.items():
                weight = combined_weights.get(cluster_id, 0.0)
                tensor = model_state[key].to(self.device)
                
                if aggregated_tensor is None:
                    aggregated_tensor = weight * tensor
                else:
                    aggregated_tensor += weight * tensor
                    
            global_model[key] = aggregated_tensor
            
        return global_model
        
    def aggregate(self, edge_models: Dict[int, Dict[str, torch.Tensor]],
                  cluster_data_counts: Dict[int, int],
                  cluster_distributions: Dict[int, np.ndarray],
                  global_distribution: Optional[np.ndarray] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Perform complete inter-cluster aggregation with all weight calculations.
        
        Args:
            edge_models: Dict mapping cluster_id to model state dict
            cluster_data_counts: Dict mapping cluster_id to data count N_edge^k
            cluster_distributions: Dict mapping cluster_id to distribution P_k
            global_distribution: Global target distribution P_g (default: uniform)
            
        Returns:
            Tuple of (global_model, aggregation_info) where:
                - global_model: Aggregated model state dict
                - aggregation_info: Dict with weights and statistics
        """
        if global_distribution is None:
            global_distribution = self.get_uniform_distribution()
            
        # Step 1: Calculate quantity weights (Eq. 12)
        quantity_weights = self.calculate_quantity_weight(cluster_data_counts)
        
        # Step 2: Calculate distribution weights (Eq. 13)
        distribution_weights = self.calculate_distribution_weight(
            cluster_distributions, global_distribution
        )
        
        # Step 3: Calculate combined weights (Eq. 14)
        combined_weights = self.calculate_combined_weights(
            quantity_weights, distribution_weights
        )
        
        # Step 4: Perform global aggregation (Eq. 11)
        global_model = self.perform_global_aggregation(edge_models, combined_weights)
        
        # Collect aggregation information
        aggregation_info = {
            'quantity_weights': quantity_weights,
            'distribution_weights': distribution_weights,
            'combined_weights': combined_weights,
            'global_distribution': global_distribution,
            'num_clusters': len(edge_models),
            'total_data': sum(cluster_data_counts.values())
        }
        
        # Store history
        self.aggregation_history.append(aggregation_info)
        
        return global_model, aggregation_info
        
    def get_weight_statistics(self) -> Dict:
        """
        Get statistics about aggregation weights across all rounds.
        
        Returns:
            Dict with weight statistics
        """
        if not self.aggregation_history:
            return {}
            
        quantity_weights_list = []
        distribution_weights_list = []
        combined_weights_list = []
        
        for info in self.aggregation_history:
            quantity_weights_list.extend(list(info['quantity_weights'].values()))
            distribution_weights_list.extend(list(info['distribution_weights'].values()))
            combined_weights_list.extend(list(info['combined_weights'].values()))
            
        return {
            'quantity_weights': {
                'mean': np.mean(quantity_weights_list),
                'std': np.std(quantity_weights_list),
                'min': np.min(quantity_weights_list),
                'max': np.max(quantity_weights_list)
            },
            'distribution_weights': {
                'mean': np.mean(distribution_weights_list),
                'std': np.std(distribution_weights_list),
                'min': np.min(distribution_weights_list),
                'max': np.max(distribution_weights_list)
            },
            'combined_weights': {
                'mean': np.mean(combined_weights_list),
                'std': np.std(combined_weights_list),
                'min': np.min(combined_weights_list),
                'max': np.max(combined_weights_list)
            },
            'total_aggregations': len(self.aggregation_history)
        }
        
    def reset(self):
        """Reset aggregator state for new experiment."""
        self.cluster_data_counts.clear()
        self.cluster_distributions.clear()
        self.aggregation_history.clear()


def create_inter_cluster_aggregator(config: Dict, device: str = 'cpu') -> InterClusterAggregator:
    """
    Factory function to create InterClusterAggregator instance.
    
    Args:
        config: Configuration dictionary
        device: Device for tensor operations
        
    Returns:
        Initialized InterClusterAggregator
    """
    return InterClusterAggregator(config, device)
