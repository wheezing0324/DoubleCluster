"""
Edge Server module for CSAHFL framework.
Implements edge aggregation management with adaptive semi-asynchronous aggregation (Section 3.4).
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.clients.client import Client
from src.aggregation.intra_cluster import IntraClusterAggregator


class EdgeServer:
    """
    Edge server that manages intra-cluster aggregation for a group of clients.
    Implements adaptive semi-asynchronous aggregation (Eq. 7-10).
    """
    
    def __init__(
        self,
        edge_server_id: int,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize edge server.
        
        Args:
            edge_server_id: Unique identifier for this edge server
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            config: Configuration dictionary
            device: Device for computation ('cuda' or 'cpu')
        """
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Client management
        self.clients: Dict[int, Client] = {}
        self.client_ids: List[int] = []
        
        # Aggregation state
        self.edge_model = None
        self.num_clients_in_cluster = 0
        self.total_data_samples = 0
        
        # Semi-asynchronous aggregation parameters
        self.aggregation_interval = None  # T_θ
        self.client_times: Dict[int, float] = {}  # Track client completion times
        self.staleness: Dict[int, int] = defaultdict(int)  # Track delayed rounds per client
        
        # Timing statistics
        self.round_times: List[float] = []
        
        # Initialize aggregator
        self.aggregator = IntraClusterAggregator(config, dataset_name, device)
        
    def add_client(self, client: Client):
        """
        Add a client to this edge server's cluster.
        
        Args:
            client: Client instance to add
        """
        self.clients[client.client_id] = client
        self.client_ids.append(client.client_id)
        self.num_clients_in_cluster += 1
        self.total_data_samples += len(client.train_data)
        
        # Initialize staleness tracking
        self.staleness[client.client_id] = 0
        
    def remove_client(self, client_id: int):
        """
        Remove a client from this edge server's cluster.
        
        Args:
            client_id: ID of client to remove
        """
        if client_id in self.clients:
            client = self.clients.pop(client_id)
            self.client_ids.remove(client_id)
            self.num_clients_in_cluster -= 1
            self.total_data_samples -= len(client.train_data)
            
    def get_num_clients(self) -> int:
        """Get number of clients in this edge server's cluster."""
        return len(self.clients)
    
    def get_total_samples(self) -> int:
        """Get total number of data samples across all clients."""
        return self.total_data_samples
    
    def initialize_edge_model(self, global_model: torch.nn.Module):
        """
        Initialize edge model from global model.
        
        Args:
            global_model: Global model from cloud server
        """
        # Create a copy of the global model
        self.edge_model = type(global_model)(
            num_classes=global_model.fc.out_features if hasattr(global_model, 'fc') else 10,
            dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
        )
        self.edge_model.load_state_dict(global_model.state_dict())
        self.edge_model.to(self.device)
        
    def calculate_aggregation_interval(self, client_times: Dict[int, float]) -> float:
        """
        Calculate adaptive aggregation interval T_θ (Eq. 8).
        
        T_θ = median(T_total) + α × IQR(T_total)
        
        Args:
            client_times: Dictionary mapping client_id to their total time
            
        Returns:
            Aggregation interval T_θ
        """
        times = list(client_times.values())
        
        if len(times) == 0:
            # Default interval if no times available
            return 10.0
            
        if len(times) == 1:
            return times[0]
        
        # Calculate median and IQR
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        # Median
        if n % 2 == 0:
            median = (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2
        else:
            median = sorted_times[n//2]
        
        # Quartiles
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        
        q1 = sorted_times[q1_idx]
        q3 = sorted_times[q3_idx]
        iqr = q3 - q1
        
        # Get alpha from config
        alpha = self.config.get('intra_cluster', {}).get('tolerance_alpha', 1.5)
        
        # Calculate T_θ
        t_theta = median + alpha * iqr
        
        self.aggregation_interval = t_theta
        return t_theta
    
    def perform_edge_aggregation(
        self,
        global_round: int,
        edge_round: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Perform semi-asynchronous intra-cluster aggregation (Eq. 10).
        
        w_edge^{k,t} = (1/N_edge^k) × (Σ_{i∈S_k} n_i w_i^t + Σ_{j∈A_k} τ_j n_j w_j^t)
        
        Args:
            global_round: Current global round number
            edge_round: Current edge round within global round
            
        Returns:
            Tuple of (aggregated_weights, aggregation_info)
        """
        start_time = time.time()
        
        # Collect client updates with timing
        client_updates = []
        synchronous_clients = []  # S_k: clients that reported within T_θ
        asynchronous_clients = []  # A_k: delayed clients
        
        # Get current aggregation interval
        if self.aggregation_interval is None:
            # First round: estimate from client capabilities
            self.aggregation_interval = self._estimate_initial_interval()
        
        t_theta = self.aggregation_interval
        
        # Collect updates from all clients
        for client_id, client in self.clients.items():
            # Get client's estimated completion time
            client_time = self.client_times.get(client_id, t_theta)
            
            # Determine if client is synchronous or asynchronous
            if client_time <= t_theta:
                synchronous_clients.append(client_id)
            else:
                asynchronous_clients.append(client_id)
            
            # Get client model weights and data size
            weights = client.get_model_weights()
            n_i = len(client.train_data)
            
            # Get staleness for this client
            gamma_j = self.staleness[client_id]
            
            client_updates.append({
                'client_id': client_id,
                'weights': weights,
                'data_size': n_i,
                'is_synchronous': client_id in synchronous_clients,
                'staleness': gamma_j
            })
        
        # Perform aggregation
        aggregated_weights, aggregation_info = self.aggregator.aggregate(
            client_updates=client_updates,
            synchronous_clients=synchronous_clients,
            asynchronous_clients=asynchronous_clients,
            staleness_dict=dict(self.staleness),
            aggregation_interval=t_theta,
            edge_server_id=self.edge_server_id,
            global_round=global_round,
            edge_round=edge_round
        )
        
        # Update edge model with aggregated weights
        if self.edge_model is not None:
            self.edge_model.load_state_dict(aggregated_weights)
        
        # Record timing
        aggregation_time = time.time() - start_time
        self.round_times.append(aggregation_time)
        
        # Update staleness for next round
        # Synchronous clients reset staleness, asynchronous clients increment
        for client_id in synchronous_clients:
            self.staleness[client_id] = 0
        for client_id in asynchronous_clients:
            self.staleness[client_id] += 1
        
        aggregation_info['aggregation_time'] = aggregation_time
        aggregation_info['num_synchronous'] = len(synchronous_clients)
        aggregation_info['num_asynchronous'] = len(asynchronous_clients)
        
        return aggregated_weights, aggregation_info
    
    def _estimate_initial_interval(self) -> float:
        """
        Estimate initial aggregation interval before client times are known.
        
        Returns:
            Estimated aggregation interval in seconds
        """
        # Use client-reported t_unit estimates
        client_times = []
        for client in self.clients.values():
            # Estimate based on client's data size and t_unit
            t_unit = client.t_unit
            n_bs = len(client.train_data) / self.config.get('training', {}).get('batch_size', 10)
            e_max = self.config.get('intra_cluster', {}).get('max_epochs', 10)
            
            t_cmp = e_max * n_bs * t_unit
            model_size = client.get_model_size()
            bandwidth = self.config.get('system', {}).get('bandwidth', 1e6)  # Default 1 Mbps
            t_com = model_size / bandwidth
            
            client_times.append(t_cmp + t_com)
        
        if len(client_times) > 0:
            return np.median(client_times)
        else:
            return 10.0  # Default fallback
    
    def update_client_times(self, client_id: int, completion_time: float):
        """
        Update recorded completion time for a client.
        
        Args:
            client_id: Client identifier
            completion_time: Time taken for client to complete local training
        """
        self.client_times[client_id] = completion_time
        
    def get_client_staleness(self, client_id: int) -> int:
        """
        Get current staleness value for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Staleness value (number of delayed rounds)
        """
        return self.staleness.get(client_id, 0)
    
    def reset_staleness(self):
        """Reset staleness tracking for all clients."""
        self.staleness = defaultdict(int)
        
    def get_edge_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current edge model weights.
        
        Returns:
            Dictionary of model weights
        """
        if self.edge_model is None:
            return {}
        return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
    
    def set_edge_model_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set edge model weights.
        
        Args:
            weights: Dictionary of model weights
        """
        if self.edge_model is None:
            return
        self.edge_model.load_state_dict(weights)
        
    def get_cluster_info(self) -> Dict:
        """
        Get information about this edge server's cluster.
        
        Returns:
            Dictionary with cluster statistics
        """
        return {
            'edge_server_id': self.edge_server_id,
            'num_clients': self.num_clients_in_cluster,
            'total_samples': self.total_data_samples,
            'client_ids': self.client_ids.copy(),
            'aggregation_interval': self.aggregation_interval,
            'average_round_time': np.mean(self.round_times) if self.round_times else 0.0,
            'staleness_distribution': dict(self.staleness)
        }
    
    def handle_client_mobility(
        self,
        departing_client_ids: List[int],
        arriving_clients: List[Client]
    ):
        """
        Handle client mobility events - clients leaving and joining.
        
        Args:
            departing_client_ids: List of client IDs leaving this edge server
            arriving_clients: List of Client instances joining this edge server
        """
        # Remove departing clients
        for client_id in departing_client_ids:
            self.remove_client(client_id)
            
        # Add arriving clients
        for client in arriving_clients:
            self.add_client(client)
            # Reset staleness for new clients
            self.staleness[client.client_id] = 0
            
    def get_aggregation_statistics(self) -> Dict:
        """
        Get statistics about aggregation performance.
        
        Returns:
            Dictionary with aggregation statistics
        """
        return {
            'edge_server_id': self.edge_server_id,
            'num_aggregations': len(self.round_times),
            'average_aggregation_time': np.mean(self.round_times) if self.round_times else 0.0,
            'min_aggregation_time': min(self.round_times) if self.round_times else 0.0,
            'max_aggregation_time': max(self.round_times) if self.round_times else 0.0,
            'current_aggregation_interval': self.aggregation_interval,
            'num_synchronous_clients': sum(1 for t in self.client_times.values() if t <= (self.aggregation_interval or 0)),
            'num_asynchronous_clients': sum(1 for t in self.client_times.values() if t > (self.aggregation_interval or 0))
        }


def create_edge_server(
    edge_server_id: int,
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> EdgeServer:
    """
    Factory function to create an EdgeServer instance.
    
    Args:
        edge_server_id: Unique identifier for the edge server
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Device for computation
        
    Returns:
        Initialized EdgeServer instance
    """
    return EdgeServer(
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        config=config,
        device=device
    )
