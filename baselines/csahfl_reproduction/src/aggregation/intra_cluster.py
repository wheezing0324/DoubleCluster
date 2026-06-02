"""
Adaptive Semi-Asynchronous Intra-Cluster Aggregation (Section 3.4)

This module implements the adaptive semi-asynchronous aggregation mechanism
for edge-level federated learning, including:
- Total time calculation (Eq. 7)
- Adaptive aggregation interval (Eq. 8)
- Workload adjustment (Eq. 9)
- Semi-asynchronous aggregation with staleness discounting (Eq. 10)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch


class IntraClusterAggregator:
    """
    Manages adaptive semi-asynchronous intra-cluster aggregation for edge servers.
    
    Implements Section 3.4 of CSAHFL paper:
    - Eq. 7: Total time calculation (t_total = t_cmp + t_com)
    - Eq. 8: Adaptive aggregation interval (T_θ = median + α × IQR)
    - Eq. 9: Workload adjustment (E_i based on T_θ)
    - Eq. 10: Semi-asynchronous aggregation with staleness discounting
    """
    
    def __init__(self, edge_server_id: int, config: Dict, device: str = 'cpu'):
        """
        Initialize intra-cluster aggregator.
        
        Args:
            edge_server_id: Unique identifier for this edge server
            config: Configuration dictionary with aggregation parameters
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.edge_server_id = edge_server_id
        self.config = config
        self.device = device
        
        # Configuration parameters from paper
        self.max_epochs = config.get('intra_cluster', {}).get('max_epochs', 10)  # E_max
        self.tolerance_alpha = config.get('intra_cluster', {}).get('tolerance_alpha', 1.5)  # α
        self.batch_size = config.get('training', {}).get('batch_size', 10)  # B
        self.t_unit = config.get('intra_cluster', {}).get('t_unit', 0.001)  # Base computation time
        
        # Client tracking
        self.client_times: Dict[int, float] = {}  # T_total for each client
        self.staleness: Dict[int, int] = defaultdict(int)  # γ_j: delayed rounds per client
        self.client_data_sizes: Dict[int, int] = {}  # n_i: data samples per client
        
        # Aggregation state
        self.aggregation_interval: float = 0.0  # T_θ
        self.current_round: int = 0
        self.synchronous_clients: List[int] = []  # S_k
        self.asynchronous_clients: List[int] = []  # A_k
        
        # Timing constants
        self.bandwidth = config.get('system', {}).get('bandwidth', 1e6)  # B_w: network bandwidth (bytes/s)
    
    def calculate_computation_time(self, data_size: int, epochs: int) -> float:
        """
        Calculate computation time for a client (t_cmp).
        
        Eq. 7: t_cmp = E × N_BS × t_unit
        where N_BS = |D_i| / B (number of batches)
        
        Args:
            data_size: Number of data samples for the client (|D_i|)
            epochs: Number of local training epochs (E)
            
        Returns:
            Computation time in seconds
        """
        num_batches = data_size / self.batch_size  # N_BS
        t_cmp = epochs * num_batches * self.t_unit
        return t_cmp
    
    def calculate_communication_time(self, model_size: int) -> float:
        """
        Calculate communication time for model upload (t_com).
        
        Eq. 7: t_com = M / B_w
        where M = model file size, B_w = network bandwidth
        
        Args:
            model_size: Model size in bytes (M)
            
        Returns:
            Communication time in seconds
        """
        t_com = model_size / self.bandwidth
        return t_com
    
    def calculate_total_time(self, data_size: int, epochs: int, model_size: int) -> float:
        """
        Calculate total time for a client to complete local training and upload.
        
        Eq. 7: t_total = t_cmp + t_com
        
        Args:
            data_size: Number of data samples
            epochs: Number of local epochs
            model_size: Model size in bytes
            
        Returns:
            Total time in seconds
        """
        t_cmp = self.calculate_computation_time(data_size, epochs)
        t_com = self.calculate_communication_time(model_size)
        t_total = t_cmp + t_com
        return t_total
    
    def update_client_time(self, client_id: int, data_size: int, epochs: int, model_size: int):
        """
        Update recorded time for a client.
        
        Args:
            client_id: Client identifier
            data_size: Client's data size
            epochs: Number of epochs used
            model_size: Model size in bytes
        """
        t_total = self.calculate_total_time(data_size, epochs, model_size)
        self.client_times[client_id] = t_total
        self.client_data_sizes[client_id] = data_size
    
    def calculate_aggregation_interval(self) -> float:
        """
        Calculate adaptive aggregation interval T_θ.
        
        Eq. 8: T_θ = median(T_total) + α × IQR(T_total)
        where IQR = Q75 - Q25 (interquartile range)
        
        Returns:
            Aggregation interval T_θ in seconds
        """
        if len(self.client_times) == 0:
            # Default interval if no client times recorded
            self.aggregation_interval = self.max_epochs * 10 * self.t_unit
            return self.aggregation_interval
        
        times = list(self.client_times.values())
        
        # Calculate median
        median_time = np.median(times)
        
        # Calculate IQR (Interquartile Range)
        q75 = np.percentile(times, 75)
        q25 = np.percentile(times, 25)
        iqr = q75 - q25
        
        # Eq. 8: T_θ = median + α × IQR
        self.aggregation_interval = median_time + self.tolerance_alpha * iqr
        
        return self.aggregation_interval
    
    def classify_clients(self, actual_times: Dict[int, float]) -> Tuple[List[int], List[int]]:
        """
        Classify clients into synchronous and asynchronous groups.
        
        Args:
            actual_times: Dictionary mapping client_id to actual completion time
            
        Returns:
            Tuple of (synchronous_clients, asynchronous_clients)
        """
        self.synchronous_clients = []
        self.asynchronous_clients = []
        
        for client_id, actual_time in actual_times.items():
            if actual_time <= self.aggregation_interval:
                self.synchronous_clients.append(client_id)
            else:
                self.asynchronous_clients.append(client_id)
                # Increment staleness for delayed clients
                self.staleness[client_id] += 1
        
        return self.synchronous_clients, self.asynchronous_clients
    
    def calculate_workload(self, client_id: int, t_com: float) -> int:
        """
        Calculate adjusted workload (epochs) for a client.
        
        Eq. 9: E_i = max(min((T_θ - t_com) / (N_BS × t_unit), E_max), 1)
        
        Args:
            client_id: Client identifier
            t_com: Communication time for this client
            
        Returns:
            Adjusted number of epochs (bounded between 1 and E_max)
        """
        data_size = self.client_data_sizes.get(client_id, 100)  # Default if not set
        num_batches = data_size / self.batch_size  # N_BS
        
        # Eq. 9 calculation
        available_time = self.aggregation_interval - t_com
        if available_time <= 0 or num_batches <= 0:
            adjusted_epochs = 1
        else:
            adjusted_epochs = available_time / (num_batches * self.t_unit)
        
        # Bound between 1 and E_max
        adjusted_epochs = max(min(adjusted_epochs, self.max_epochs), 1)
        
        return int(adjusted_epochs)
    
    def calculate_discount_factor(self, client_id: int) -> float:
        """
        Calculate staleness discount factor for asynchronous clients.
        
        τ_j = 1 / (1 + γ_j)
        where γ_j = number of delayed rounds for client j
        
        Args:
            client_id: Client identifier
            
        Returns:
            Discount factor τ_j (between 0 and 1)
        """
        gamma = self.staleness[client_id]  # γ_j
        tau = 1.0 / (1.0 + gamma)
        return tau
    
    def reset_staleness(self, client_id: int):
        """
        Reset staleness counter for a client after successful aggregation.
        
        Args:
            client_id: Client identifier
        """
        self.staleness[client_id] = 0
    
    def perform_aggregation(
        self,
        client_weights: Dict[int, Dict[str, torch.Tensor]],
        client_data_sizes: Dict[int, int],
        actual_times: Dict[int, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform semi-asynchronous intra-cluster aggregation.
        
        Eq. 10: w_edge^{k,t} = (1/N_edge^k) × (Σ_{i∈S_k} n_i w_i^t + Σ_{j∈A_k} τ_j n_j w_j^t)
        
        Args:
            client_weights: Dictionary mapping client_id to model weights dict
            client_data_sizes: Dictionary mapping client_id to data size (n_i)
            actual_times: Dictionary mapping client_id to actual completion time
            
        Returns:
            Aggregated edge model weights
        """
        # Classify clients based on actual completion times
        sync_clients, async_clients = self.classify_clients(actual_times)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        total_weighted_samples = 0.0
        
        # Get reference model structure from first client
        if len(client_weights) == 0:
            raise ValueError("No client weights provided for aggregation")
        
        first_client_id = list(client_weights.keys())[0]
        for key in client_weights[first_client_id].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[first_client_id][key])
        
        # Aggregate synchronous clients (full weight)
        # Σ_{i∈S_k} n_i w_i^t
        for client_id in sync_clients:
            if client_id not in client_weights:
                continue
            n_i = client_data_sizes.get(client_id, 1)
            weights = client_weights[client_id]
            
            for key in aggregated_weights.keys():
                if key in weights:
                    aggregated_weights[key] += weights[key] * n_i
            
            total_weighted_samples += n_i
            # Reset staleness for synchronous clients
            self.reset_staleness(client_id)
        
        # Aggregate asynchronous clients (with discount factor)
        # Σ_{j∈A_k} τ_j n_j w_j^t
        for client_id in async_clients:
            if client_id not in client_weights:
                continue
            n_j = client_data_sizes.get(client_id, 1)
            tau_j = self.calculate_discount_factor(client_id)  # τ_j
            weights = client_weights[client_id]
            
            for key in aggregated_weights.keys():
                if key in weights:
                    aggregated_weights[key] += weights[key] * tau_j * n_j
            
            total_weighted_samples += tau_j * n_j
            # Note: staleness continues to accumulate for async clients
        
        # Normalize by total weighted samples
        # w_edge^{k,t} = (1/N_edge^k) × ...
        if total_weighted_samples > 0:
            for key in aggregated_weights.keys():
                aggregated_weights[key] = aggregated_weights[key] / total_weighted_samples
        
        # Increment round counter
        self.current_round += 1
        
        return aggregated_weights
    
    def get_aggregation_statistics(self) -> Dict:
        """
        Get statistics about the current aggregation state.
        
        Returns:
            Dictionary with aggregation statistics
        """
        stats = {
            'edge_server_id': self.edge_server_id,
            'current_round': self.current_round,
            'aggregation_interval': self.aggregation_interval,
            'num_synchronous_clients': len(self.synchronous_clients),
            'num_asynchronous_clients': len(self.asynchronous_clients),
            'total_clients': len(self.client_times),
            'avg_staleness': np.mean(list(self.staleness.values())) if self.staleness else 0.0,
            'max_staleness': max(self.staleness.values()) if self.staleness else 0,
        }
        return stats
    
    def reset_for_new_round(self):
        """
        Reset aggregator state for a new global round.
        """
        self.synchronous_clients = []
        self.asynchronous_clients = []
        # Note: client_times and staleness are preserved across rounds
    
    def remove_client(self, client_id: int):
        """
        Remove a client from the aggregator (e.g., due to mobility).
        
        Args:
            client_id: Client identifier to remove
        """
        if client_id in self.client_times:
            del self.client_times[client_id]
        if client_id in self.client_data_sizes:
            del self.client_data_sizes[client_id]
        if client_id in self.staleness:
            del self.staleness[client_id]


def create_intra_cluster_aggregator(edge_server_id: int, config: Dict, device: str = 'cpu') -> IntraClusterAggregator:
    """
    Factory function to create an IntraClusterAggregator instance.
    
    Args:
        edge_server_id: Unique identifier for the edge server
        config: Configuration dictionary
        device: Device for tensor operations
        
    Returns:
        Initialized IntraClusterAggregator instance
    """
    return IntraClusterAggregator(edge_server_id, config, device)
