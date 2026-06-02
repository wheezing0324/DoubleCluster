"""
Cloud Server for CSAHFL Framework

Implements the cloud-level aggregation that coordinates all edge servers
and performs distribution-aware global model aggregation (Section 3.5).

This module handles:
- Global model initialization and management
- Collection of edge server models
- Distribution-aware inter-cluster aggregation (Eq. 11-14)
- Global model distribution to edge servers
- Global model evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

from src.aggregation.inter_cluster import InterClusterAggregator, create_inter_cluster_aggregator
from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class CloudServer:
    """
    Cloud server that manages global model aggregation across all edge servers.
    
    Implements distribution-aware inter-cluster aggregation (Section 3.5, Eq. 11-14):
    - Quantity weight (Eq. 12): ω_k^qua = N_edge^k / Σ N_edge^k
    - Distribution weight (Eq. 13): ω_k^dis = 1 / (1 + D_KL(P_k || P_g))
    - Combined weight (Eq. 14): λ_k = (ω_k^qua × ω_k^dis) / Σ (ω_j^qua × ω_j^dis)
    - Global aggregation (Eq. 11): w_g = Σ λ_k w_edge^k
    
    Attributes:
        server_id: Cloud server identifier (always 0)
        dataset_name: Name of the dataset being used
        config: Configuration dictionary
        device: Training device ('cuda' or 'cpu')
        global_model: Current global neural network model
        edge_models: Dictionary of edge server models {edge_id: state_dict}
        edge_data_quantities: Dictionary of edge server data quantities {edge_id: num_samples}
        edge_label_distributions: Dictionary of edge label distributions {edge_id: distribution}
        inter_cluster_aggregator: Inter-cluster aggregation handler
        aggregation_history: List of aggregation round statistics
    """
    
    def __init__(self, dataset_name: str, config: Dict, device: str = 'cpu'):
        """
        Initialize cloud server with global model and aggregation components.
        
        Args:
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            config: Configuration dictionary with model and aggregation settings
            device: Training device ('cuda' or 'cpu')
        """
        self.server_id = 0  # Single cloud server
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Initialize global model
        self.global_model = self._create_model()
        self.global_model.to(self.device)
        
        # Edge server tracking
        self.edge_models: Dict[int, Dict[str, torch.Tensor]] = {}
        self.edge_data_quantities: Dict[int, int] = {}
        self.edge_label_distributions: Dict[int, np.ndarray] = {}
        
        # Initialize inter-cluster aggregator for distribution-aware aggregation
        self.inter_cluster_aggregator = create_inter_cluster_aggregator(config, device)
        
        # Aggregation statistics
        self.aggregation_history: List[Dict] = []
        self.total_aggregation_time = 0.0
        self.num_aggregations = 0
        
        # Set global target distribution (uniform)
        num_classes = config.get('system', {}).get('num_classes', 10)
        self.global_target_distribution = np.ones(num_classes) / num_classes
        self.inter_cluster_aggregator.set_global_distribution(self.global_target_distribution)
    
    def _create_model(self) -> nn.Module:
        """
        Create global model based on dataset name.
        
        Returns:
            Initialized neural network model
        """
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(
                num_classes=self.config.get('system', {}).get('num_classes', 10),
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(
                num_classes=self.config.get('system', {}).get('num_classes', 10),
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'svhn':
            return create_svhn_model(
                num_classes=self.config.get('system', {}).get('num_classes', 10),
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def register_edge_server(self, edge_server_id: int, num_samples: int, 
                            label_distribution: Optional[np.ndarray] = None):
        """
        Register an edge server with the cloud server.
        
        Args:
            edge_server_id: Unique identifier for the edge server
            num_samples: Total number of data samples at this edge server
            label_distribution: Label distribution array (optional, can be estimated later)
        """
        self.edge_data_quantities[edge_server_id] = num_samples
        
        if label_distribution is not None:
            self.edge_label_distributions[edge_server_id] = label_distribution
        else:
            # Initialize with uniform distribution (will be updated later)
            num_classes = self.config.get('system', {}).get('num_classes', 10)
            self.edge_label_distributions[edge_server_id] = np.ones(num_classes) / num_classes
    
    def update_edge_label_distribution(self, edge_server_id: int, label_distribution: np.ndarray):
        """
        Update the label distribution for an edge server.
        
        Args:
            edge_server_id: Edge server identifier
            label_distribution: New label distribution array (should sum to 1.0)
        """
        # Normalize distribution
        if label_distribution.sum() > 0:
            label_distribution = label_distribution / label_distribution.sum()
        self.edge_label_distributions[edge_server_id] = label_distribution
    
    def receive_edge_model(self, edge_server_id: int, model_weights: Dict[str, torch.Tensor],
                          num_samples: int, label_distribution: Optional[np.ndarray] = None):
        """
        Receive aggregated model from an edge server.
        
        Args:
            edge_server_id: Edge server identifier
            model_weights: Model state dictionary from edge server
            num_samples: Number of samples used in edge aggregation
            label_distribution: Label distribution at this edge server
        """
        # Store edge model weights (move to CPU for storage)
        cpu_weights = {k: v.cpu() for k, v in model_weights.items()}
        self.edge_models[edge_server_id] = cpu_weights
        
        # Update data quantity
        self.edge_data_quantities[edge_server_id] = num_samples
        
        # Update label distribution if provided
        if label_distribution is not None:
            self.update_edge_label_distribution(edge_server_id, label_distribution)
    
    def perform_global_aggregation(self, cloud_round: int) -> Dict[str, torch.Tensor]:
        """
        Perform distribution-aware global aggregation across all edge servers.
        
        Implements Eq. 11-14 from Section 3.5:
        1. Calculate quantity weights (Eq. 12)
        2. Estimate cluster distributions and calculate distribution weights (Eq. 13)
        3. Combine weights (Eq. 14)
        4. Aggregate models (Eq. 11)
        
        Args:
            cloud_round: Current cloud aggregation round number
            
        Returns:
            Aggregated global model state dictionary
        """
        start_time = time.time()
        
        if len(self.edge_models) == 0:
            raise ValueError("No edge server models received for aggregation")
        
        # Get list of edge server IDs
        edge_ids = list(self.edge_models.keys())
        
        # Collect edge model weights
        edge_weights_list = [self.edge_models[eid] for eid in edge_ids]
        
        # Collect data quantities
        edge_quantities = [self.edge_data_quantities[eid] for eid in edge_ids]
        
        # Collect label distributions
        edge_distributions = [self.edge_label_distributions[eid] for eid in edge_ids]
        
        # Perform distribution-aware aggregation
        aggregated_weights, weight_stats = self.inter_cluster_aggregator.aggregate(
            edge_weights=edge_weights_list,
            edge_quantities=edge_quantities,
            edge_distributions=edge_distributions,
            cloud_round=cloud_round
        )
        
        # Update global model with aggregated weights
        self.global_model.load_state_dict(aggregated_weights)
        
        # Record aggregation statistics
        aggregation_time = time.time() - start_time
        self.total_aggregation_time += aggregation_time
        self.num_aggregations += 1
        
        round_stats = {
            'cloud_round': cloud_round,
            'num_edge_servers': len(edge_ids),
            'aggregation_time': aggregation_time,
            'weight_statistics': weight_stats,
            'total_data_samples': sum(edge_quantities)
        }
        self.aggregation_history.append(round_stats)
        
        return aggregated_weights
    
    def distribute_global_model(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model weights for distribution to edge servers.
        
        Returns:
            Global model state dictionary (on CPU for transmission)
        """
        return {k: v.cpu() for k, v in self.global_model.state_dict().items()}
    
    def evaluate_global_model(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate global model on test dataset.
        
        Args:
            test_loader: PyTorch DataLoader for test data
            
        Returns:
            Dictionary with 'loss' and 'accuracy' metrics
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.global_model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def get_global_model(self) -> nn.Module:
        """
        Get the current global model instance.
        
        Returns:
            Global neural network model
        """
        return self.global_model
    
    def set_global_model_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set global model weights (useful for initialization or resumption).
        
        Args:
            weights: Model state dictionary to load
        """
        self.global_model.load_state_dict(weights)
    
    def get_aggregation_statistics(self) -> Dict:
        """
        Get statistics about global aggregation operations.
        
        Returns:
            Dictionary with aggregation statistics
        """
        if self.num_aggregations == 0:
            return {
                'total_aggregations': 0,
                'total_time': 0.0,
                'average_time': 0.0
            }
        
        return {
            'total_aggregations': self.num_aggregations,
            'total_time': self.total_aggregation_time,
            'average_time': self.total_aggregation_time / self.num_aggregations,
            'latest_weights': self.get_weight_statistics()
        }
    
    def get_weight_statistics(self) -> Dict:
        """
        Get statistics about the latest aggregation weights.
        
        Returns:
            Dictionary with weight statistics from last aggregation
        """
        if len(self.aggregation_history) == 0:
            return {}
        
        return self.aggregation_history[-1].get('weight_statistics', {})
    
    def reset_for_new_experiment(self):
        """
        Reset cloud server state for a new experiment.
        """
        self.edge_models.clear()
        self.edge_data_quantities.clear()
        self.edge_label_distributions.clear()
        self.aggregation_history.clear()
        self.total_aggregation_time = 0.0
        self.num_aggregations = 0
        
        # Reinitialize global model
        self.global_model = self._create_model()
        self.global_model.to(self.device)
        
        # Reset inter-cluster aggregator
        self.inter_cluster_aggregator.reset()
        
        # Reset global target distribution
        num_classes = self.config.get('system', {}).get('num_classes', 10)
        self.global_target_distribution = np.ones(num_classes) / num_classes
        self.inter_cluster_aggregator.set_global_distribution(self.global_target_distribution)


def create_cloud_server(dataset_name: str, config: Dict, device: str = 'cpu') -> CloudServer:
    """
    Factory function to create a CloudServer instance.
    
    Args:
        dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
        config: Configuration dictionary
        device: Training device ('cuda' or 'cpu')
        
    Returns:
        Initialized CloudServer instance
    """
    return CloudServer(dataset_name=dataset_name, config=config, device=device)
