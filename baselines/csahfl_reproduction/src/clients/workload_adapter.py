"""
Workload Adapter for CSAHFL - Adaptive Semi-Asynchronous Intra-Cluster Aggregation (Section 3.4)

This module implements workload adjustment mechanism (Eq. 9) that adapts local training epochs
based on client heterogeneity and aggregation interval constraints.

Key Equations:
- Eq. 7: t_total = t_cmp + t_com (total time calculation)
- Eq. 8: T_θ = median(T_total) + α × IQR(T_total) (adaptive aggregation interval)
- Eq. 9: E_i = max(min((T_θ - t_com) / (N_BS × t_unit), E_max), 1) (workload adjustment)

Author: CSAHFL Reproduction
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict


class WorkloadAdapter:
    """
    Adaptive workload adjustment for semi-asynchronous federated learning.
    
    Adjusts local training epochs per client based on:
    - Client computation capability (t_unit estimation)
    - Communication time (t_com)
    - Adaptive aggregation interval (T_θ)
    - Maximum epoch constraint (E_max)
    
    Implements Eq. 9: E_i = max(min((T_θ - t_com) / (N_BS × t_unit), E_max), 1)
    
    Attributes:
        config (Dict): Configuration dictionary with intra_cluster settings
        max_epochs (int): Maximum local epochs (E_max, default=10)
        tolerance_alpha (float): Straggler tolerance parameter (α, default=1.5)
        batch_size (int): Local batch size (B, default=10)
        client_times (Dict[int, float]): Historical completion times per client
        client_data_sizes (Dict[int, int]): Data size per client (|D_i|)
        t_unit_estimates (Dict[int, float]): Computation time per batch per client
    """
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        """
        Initialize WorkloadAdapter with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - intra_cluster.max_epochs: E_max (default=10)
                - intra_cluster.tolerance_alpha: α (default=1.5)
                - training.batch_size: B (default=10)
            device (str): Device for computation ('cpu' or 'cuda')
        """
        self.config = config
        self.device = device
        
        # Extract configuration parameters
        intra_cluster_config = config.get('intra_cluster', {})
        training_config = config.get('training', {})
        
        self.max_epochs = intra_cluster_config.get('max_epochs', 10)  # E_max
        self.tolerance_alpha = intra_cluster_config.get('tolerance_alpha', 1.5)  # α
        self.batch_size = training_config.get('batch_size', 10)  # B
        
        # Client-specific tracking
        self.client_times: Dict[int, float] = {}  # Historical completion times
        self.client_data_sizes: Dict[int, int] = {}  # |D_i| per client
        self.t_unit_estimates: Dict[int, float] = {}  # t_unit per client
        
        # Aggregation interval tracking
        self.aggregation_interval: float = 0.0  # T_θ
        self.time_history: list = []  # History of T_total values
        
    def register_client(self, client_id: int, data_size: int, t_unit: float):
        """
        Register a client with its data size and computation capability.
        
        Args:
            client_id (int): Unique client identifier
            data_size (int): Size of client's local dataset (|D_i|)
            t_unit (float): Estimated computation time per batch (seconds)
        """
        self.client_data_sizes[client_id] = data_size
        self.t_unit_estimates[client_id] = t_unit
        
    def update_client_time(self, client_id: int, completion_time: float):
        """
        Update historical completion time for a client.
        
        Args:
            client_id (int): Client identifier
            completion_time (float): Actual completion time in seconds
        """
        self.client_times[client_id] = completion_time
        self.time_history.append(completion_time)
        
    def calculate_computation_time(self, client_id: int, epochs: Optional[int] = None) -> float:
        """
        Calculate computation time for a client (t_cmp component of Eq. 7).
        
        t_cmp = E × N_BS × t_unit
        
        Args:
            client_id (int): Client identifier
            epochs (int, optional): Number of epochs. If None, uses max_epochs
            
        Returns:
            float: Estimated computation time in seconds
        """
        if client_id not in self.t_unit_estimates:
            # Default estimate if not registered
            t_unit = 0.1  # Default 100ms per batch
        else:
            t_unit = self.t_unit_estimates[client_id]
            
        if epochs is None:
            epochs = self.max_epochs
            
        # N_BS = |D_i| / B (number of batches)
        data_size = self.client_data_sizes.get(client_id, 1000)  # Default 1000 samples
        n_batches = max(1, data_size // self.batch_size)
        
        # t_cmp = E × N_BS × t_unit
        t_cmp = epochs * n_batches * t_unit
        
        return t_cmp
    
    def calculate_communication_time(self, model_size_bytes: int, bandwidth_bps: float) -> float:
        """
        Calculate communication time (t_com component of Eq. 7).
        
        t_com = M / B_w
        
        Args:
            model_size_bytes (int): Model size in bytes (M)
            bandwidth_bps (float): Network bandwidth in bytes/second (B_w)
            
        Returns:
            float: Communication time in seconds
        """
        if bandwidth_bps <= 0:
            bandwidth_bps = 1e6  # Default 1 MB/s
            
        t_com = model_size_bytes / bandwidth_bps
        return t_com
    
    def calculate_total_time(self, client_id: int, model_size_bytes: int, 
                            bandwidth_bps: float, epochs: Optional[int] = None) -> float:
        """
        Calculate total time for a client (Eq. 7).
        
        t_total = t_cmp + t_com
        
        Args:
            client_id (int): Client identifier
            model_size_bytes (int): Model size in bytes
            bandwidth_bps (float): Network bandwidth in bytes/second
            epochs (int, optional): Number of epochs. If None, uses max_epochs
            
        Returns:
            float: Total estimated time in seconds
        """
        t_cmp = self.calculate_computation_time(client_id, epochs)
        t_com = self.calculate_communication_time(model_size_bytes, bandwidth_bps)
        
        t_total = t_cmp + t_com
        return t_total
    
    def calculate_aggregation_interval(self, client_ids: Optional[list] = None) -> float:
        """
        Calculate adaptive aggregation interval T_θ (Eq. 8).
        
        T_θ = median(T_total) + α × IQR(T_total)
        
        where IQR(T_total) = Q75 - Q25 (interquartile range)
        
        Args:
            client_ids (list, optional): List of client IDs to consider. 
                                         If None, uses all registered clients.
                                         
        Returns:
            float: Adaptive aggregation interval T_θ in seconds
        """
        if client_ids is None:
            client_ids = list(self.t_unit_estimates.keys())
            
        if len(client_ids) == 0:
            # Default interval if no clients
            self.aggregation_interval = 10.0  # Default 10 seconds
            return self.aggregation_interval
            
        # Calculate T_total for each client
        total_times = []
        for client_id in client_ids:
            # Use default bandwidth estimate (1 MB/s)
            t_total = self.calculate_total_time(
                client_id=client_id,
                model_size_bytes=1000000,  # 1 MB default
                bandwidth_bps=1e6  # 1 MB/s
            )
            total_times.append(t_total)
            
        if len(total_times) == 0:
            self.aggregation_interval = 10.0
            return self.aggregation_interval
            
        # Calculate median and IQR
        median_time = np.median(total_times)
        q75 = np.percentile(total_times, 75)
        q25 = np.percentile(total_times, 25)
        iqr = q75 - q25
        
        # Eq. 8: T_θ = median(T_total) + α × IQR(T_total)
        self.aggregation_interval = median_time + self.tolerance_alpha * iqr
        
        # Ensure minimum interval
        self.aggregation_interval = max(self.aggregation_interval, 1.0)  # At least 1 second
        
        return self.aggregation_interval
    
    def adjust_workload(self, client_id: int, model_size_bytes: int, 
                       bandwidth_bps: float, t_theta: Optional[float] = None) -> int:
        """
        Adjust local training epochs for a client (Eq. 9).
        
        E_i = max(min((T_θ - t_com) / (N_BS × t_unit), E_max), 1)
        
        This ensures:
        - Weak clients reduce epochs to finish within T_θ
        - Strong clients can use full E_max epochs
        - Epochs are bounded between 1 and E_max
        
        Args:
            client_id (int): Client identifier
            model_size_bytes (int): Model size in bytes
            bandwidth_bps (float): Network bandwidth in bytes/second
            t_theta (float, optional): Aggregation interval T_θ. 
                                       If None, uses stored aggregation_interval.
                                       
        Returns:
            int: Adjusted number of epochs E_i
        """
        if t_theta is None:
            t_theta = self.aggregation_interval
            
        # Get client-specific parameters
        if client_id not in self.t_unit_estimates:
            t_unit = 0.1  # Default 100ms per batch
        else:
            t_unit = self.t_unit_estimates[client_id]
            
        data_size = self.client_data_sizes.get(client_id, 1000)
        n_batches = max(1, data_size // self.batch_size)  # N_BS
        
        # Calculate communication time
        t_com = self.calculate_communication_time(model_size_bytes, bandwidth_bps)
        
        # Eq. 9: E_i = (T_θ - t_com) / (N_BS × t_unit)
        available_time = t_theta - t_com
        
        if available_time <= 0 or t_unit <= 0 or n_batches <= 0:
            # Not enough time for even one epoch
            adjusted_epochs = 1
        else:
            adjusted_epochs = available_time / (n_batches * t_unit)
            
        # Apply bounds: max(min(..., E_max), 1)
        adjusted_epochs = min(adjusted_epochs, self.max_epochs)
        adjusted_epochs = max(adjusted_epochs, 1)
        
        # Convert to integer
        adjusted_epochs = int(adjusted_epochs)
        
        return adjusted_epochs
    
    def get_client_workload_info(self, client_id: int, model_size_bytes: int,
                                 bandwidth_bps: float) -> Dict:
        """
        Get comprehensive workload information for a client.
        
        Args:
            client_id (int): Client identifier
            model_size_bytes (int): Model size in bytes
            bandwidth_bps (float): Network bandwidth in bytes/second
            
        Returns:
            Dict: Workload information including:
                - client_id: Client identifier
                - data_size: Local dataset size
                - t_unit: Computation time per batch
                - t_cmp: Computation time
                - t_com: Communication time
                - t_total: Total time
                - t_theta: Aggregation interval
                - adjusted_epochs: Adjusted epoch count
                - max_epochs: Maximum allowed epochs
        """
        t_unit = self.t_unit_estimates.get(client_id, 0.1)
        data_size = self.client_data_sizes.get(client_id, 1000)
        n_batches = max(1, data_size // self.batch_size)
        
        t_cmp = self.calculate_computation_time(client_id)
        t_com = self.calculate_communication_time(model_size_bytes, bandwidth_bps)
        t_total = t_cmp + t_com
        
        adjusted_epochs = self.adjust_workload(client_id, model_size_bytes, bandwidth_bps)
        
        return {
            'client_id': client_id,
            'data_size': data_size,
            't_unit': t_unit,
            'n_batches': n_batches,
            't_cmp': t_cmp,
            't_com': t_com,
            't_total': t_total,
            't_theta': self.aggregation_interval,
            'adjusted_epochs': adjusted_epochs,
            'max_epochs': self.max_epochs
        }
    
    def classify_clients_by_capability(self, client_ids: Optional[list] = None) -> Dict[str, list]:
        """
        Classify clients into capability groups based on their t_unit estimates.
        
        Groups:
        - 'fast': t_unit < 0.5 × median
        - 'normal': 0.5 × median <= t_unit <= 1.5 × median
        - 'slow': t_unit > 1.5 × median
        
        Args:
            client_ids (list, optional): List of client IDs to classify. 
                                         If None, uses all registered clients.
                                         
        Returns:
            Dict[str, list]: Dictionary with 'fast', 'normal', 'slow' client lists
        """
        if client_ids is None:
            client_ids = list(self.t_unit_estimates.keys())
            
        if len(client_ids) == 0:
            return {'fast': [], 'normal': [], 'slow': []}
            
        # Get t_unit values
        t_units = [self.t_unit_estimates.get(cid, 0.1) for cid in client_ids]
        median_t_unit = np.median(t_units)
        
        fast_threshold = 0.5 * median_t_unit
        slow_threshold = 1.5 * median_t_unit
        
        classification = {'fast': [], 'normal': [], 'slow': []}
        
        for client_id in client_ids:
            t_unit = self.t_unit_estimates.get(client_id, 0.1)
            
            if t_unit < fast_threshold:
                classification['fast'].append(client_id)
            elif t_unit > slow_threshold:
                classification['slow'].append(client_id)
            else:
                classification['normal'].append(client_id)
                
        return classification
    
    def get_statistics(self) -> Dict:
        """
        Get workload adapter statistics.
        
        Returns:
            Dict: Statistics including:
                - num_clients: Number of registered clients
                - avg_t_unit: Average computation time per batch
                - min_t_unit: Minimum t_unit (fastest client)
                - max_t_unit: Maximum t_unit (slowest client)
                - aggregation_interval: Current T_θ
                - max_epochs: E_max setting
                - tolerance_alpha: α setting
        """
        if len(self.t_unit_estimates) == 0:
            return {
                'num_clients': 0,
                'avg_t_unit': 0.0,
                'min_t_unit': 0.0,
                'max_t_unit': 0.0,
                'aggregation_interval': self.aggregation_interval,
                'max_epochs': self.max_epochs,
                'tolerance_alpha': self.tolerance_alpha
            }
            
        t_units = list(self.t_unit_estimates.values())
        
        return {
            'num_clients': len(self.t_unit_estimates),
            'avg_t_unit': float(np.mean(t_units)),
            'min_t_unit': float(np.min(t_units)),
            'max_t_unit': float(np.max(t_units)),
            'aggregation_interval': self.aggregation_interval,
            'max_epochs': self.max_epochs,
            'tolerance_alpha': self.tolerance_alpha,
            'batch_size': self.batch_size
        }
    
    def reset(self):
        """
        Reset workload adapter state for new experiment.
        """
        self.client_times.clear()
        self.client_data_sizes.clear()
        self.t_unit_estimates.clear()
        self.aggregation_interval = 0.0
        self.time_history.clear()


def create_workload_adapter(config: Dict, device: str = 'cpu') -> WorkloadAdapter:
    """
    Factory function to create WorkloadAdapter instance.
    
    Args:
        config (Dict): Configuration dictionary
        device (str): Device for computation ('cpu' or 'cuda')
        
    Returns:
        WorkloadAdapter: Configured workload adapter instance
    """
    return WorkloadAdapter(config, device)


def estimate_t_unit(model, train_data, train_labels, device: str = 'cpu', 
                   num_samples: int = 100) -> float:
    """
    Estimate computation time per batch (t_unit) for a client.
    
    This function runs a few training iterations to measure actual computation time.
    
    Args:
        model: PyTorch model to benchmark
        train_data: Training data (numpy array)
        train_labels: Training labels (numpy array)
        device (str): Device for computation
        num_samples (int): Number of samples to use for estimation
        
    Returns:
        float: Estimated t_unit in seconds per batch
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import time
    
    # Use subset of data for estimation
    num_samples = min(num_samples, len(train_data))
    
    # Convert to tensors
    data_tensor = torch.FloatTensor(train_data[:num_samples]).to(device)
    label_tensor = torch.LongTensor(train_labels[:num_samples]).to(device)
    
    # Create data loader with batch_size=1 for t_unit estimation
    dataset = TensorDataset(data_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Setup training
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Measure time for a few batches
    num_batches = min(10, len(loader))
    start_time = time.time()
    
    batch_count = 0
    for batch_idx, (data, target) in enumerate(loader):
        if batch_count >= num_batches:
            break
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        batch_count += 1
        
    elapsed_time = time.time() - start_time
    
    # Average time per batch
    t_unit = elapsed_time / max(1, batch_count)
    
    return t_unit
