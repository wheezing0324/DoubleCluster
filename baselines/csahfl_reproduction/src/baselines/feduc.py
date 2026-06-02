"""
FedUC: Time-Sharing Scheduling for Intra-Cluster Latency Baseline

This module implements FedUC, a federated learning baseline that uses time-sharing
scheduling to handle intra-cluster latency heterogeneity. FedUC divides clients into
groups and schedules them in a time-sharing manner to reduce the impact of stragglers.

Reference: FedUC baseline for CSAHFL comparison experiments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class FedUCClient:
    """
    Client for FedUC federated learning with time-sharing awareness.
    
    Implements local training with group assignment for time-sharing scheduling.
    """
    
    def __init__(
        self,
        client_id: int,
        edge_server_id: int,
        group_id: int,
        dataset_name: str,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedUC client.
        
        Args:
            client_id: Unique client identifier
            edge_server_id: Associated edge server ID
            group_id: Time-sharing group ID for scheduling
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            train_data: Training data (numpy array)
            train_labels: Training labels (numpy array)
            config: Configuration dictionary
            device: Training device ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.edge_server_id = edge_server_id
        self.group_id = group_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Create data loader
        self.train_loader = self._create_data_loader(train_data, train_labels)
        self.data_size = len(train_labels)
        
        # Training parameters
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.01)
        self.local_epochs = config.get('training', {}).get('local_epochs', 5)
        self.batch_size = config.get('training', {}).get('batch_size', 10)
        
        # Time-sharing parameters
        self.time_slot = config.get('feduc', {}).get('time_slot', 1.0)
        self.completion_time = 0.0
        self.is_straggler = False
        
        # Training history
        self.training_losses = []
        
    def _create_model(self) -> nn.Module:
        """Create model based on dataset name."""
        num_classes = self.config.get('system', {}).get('num_classes', 10)
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes, dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes, dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes, dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _create_data_loader(self, data: np.ndarray, labels: np.ndarray) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        # Convert to tensors
        data_tensor = torch.FloatTensor(data).permute(0, 3, 1, 2) / 255.0
        labels_tensor = torch.LongTensor(labels)
        
        # Create dataset and loader
        dataset = TensorDataset(data_tensor, labels_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return loader
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from global model."""
        self.model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def train_local(self, num_epochs: Optional[int] = None) -> Tuple[float, float]:
        """
        Train model locally for specified epochs.
        
        Args:
            num_epochs: Number of local epochs (overrides config if provided)
            
        Returns:
            Tuple of (average_loss, training_time)
        """
        if num_epochs is None:
            num_epochs = self.local_epochs
        
        start_time = time.time()
        
        # Create optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(num_epochs):
            batch_losses = []
            
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            avg_epoch_loss = np.mean(batch_losses)
            epoch_losses.append(avg_epoch_loss)
        
        training_time = time.time() - start_time
        self.completion_time = training_time
        avg_loss = np.mean(epoch_losses)
        self.training_losses.append(avg_loss)
        
        return avg_loss, training_time
    
    def evaluate_local(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on local test data.
        
        Args:
            test_data: Test data (numpy array)
            test_labels: Test labels (numpy array)
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        # Convert to tensors
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2) / 255.0
        labels_tensor = torch.LongTensor(test_labels)
        
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_client_info(self) -> Dict:
        """Get client information including group and timing."""
        return {
            'client_id': self.client_id,
            'edge_server_id': self.edge_server_id,
            'group_id': self.group_id,
            'data_size': self.data_size,
            'completion_time': self.completion_time,
            'is_straggler': self.is_straggler
        }


class FUCEdgeServer:
    """
    Edge server for FedUC with time-sharing scheduling.
    
    Manages client groups and performs time-sharing based aggregation.
    """
    
    def __init__(
        self,
        edge_server_id: int,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedUC edge server.
        
        Args:
            edge_server_id: Unique edge server identifier
            dataset_name: Name of dataset
            config: Configuration dictionary
            device: Training device
        """
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Client management
        self.clients: Dict[int, FedUCClient] = {}
        self.client_groups: Dict[int, List[int]] = defaultdict(list)
        
        # Time-sharing parameters
        self.num_groups = config.get('feduc', {}).get('num_groups', 3)
        self.time_slot = config.get('feduc', {}).get('time_slot', 1.0)
        self.straggler_threshold = config.get('feduc', {}).get('straggler_threshold', 2.0)
        
        # Edge model
        self.edge_model = self._create_model()
        self.edge_model.to(device)
        
        # Aggregation statistics
        self.aggregation_times = []
        self.group_completion_times = defaultdict(list)
        
    def _create_model(self) -> nn.Module:
        """Create model based on dataset name."""
        num_classes = self.config.get('system', {}).get('num_classes', 10)
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes, dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes, dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes, dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def add_client(self, client: FedUCClient):
        """Add client to edge server and assign to group."""
        self.clients[client.client_id] = client
        
        # Assign client to group based on group_id
        group_id = client.group_id % self.num_groups
        self.client_groups[group_id].append(client.client_id)
        client.group_id = group_id
    
    def remove_client(self, client_id: int):
        """Remove client from edge server."""
        if client_id in self.clients:
            client = self.clients[client_id]
            group_id = client.group_id
            if client_id in self.client_groups[group_id]:
                self.client_groups[group_id].remove(client_id)
            del self.clients[client_id]
    
    def initialize_edge_model(self, global_weights: Dict[str, torch.Tensor]):
        """Initialize edge model with global weights."""
        self.edge_model.load_state_dict({
            k: v.to(self.device) for k, v in global_weights.items()
        })
    
    def identify_stragglers(self, group_id: int) -> List[int]:
        """
        Identify straggler clients in a group based on completion times.
        
        Args:
            group_id: Group identifier
            
        Returns:
            List of straggler client IDs
        """
        client_ids = self.client_groups[group_id]
        if len(client_ids) == 0:
            return []
        
        # Get completion times
        completion_times = [
            self.clients[cid].completion_time
            for cid in client_ids
            if self.clients[cid].completion_time > 0
        ]
        
        if len(completion_times) == 0:
            return []
        
        # Calculate median time
        median_time = np.median(completion_times)
        threshold = median_time * self.straggler_threshold
        
        # Identify stragglers
        stragglers = [
            cid for cid in client_ids
            if self.clients[cid].completion_time > threshold
        ]
        
        # Mark stragglers
        for cid in stragglers:
            self.clients[cid].is_straggler = True
        
        return stragglers
    
    def perform_edge_aggregation(
        self,
        group_id: Optional[int] = None,
        skip_stragglers: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Perform time-sharing based edge aggregation.
        
        Args:
            group_id: Specific group to aggregate (None for all groups)
            skip_stragglers: Whether to skip straggler updates
            
        Returns:
            Aggregated edge model weights
        """
        start_time = time.time()
        
        # Determine which groups to aggregate
        if group_id is not None:
            groups_to_aggregate = [group_id]
        else:
            groups_to_aggregate = list(range(self.num_groups))
        
        # Collect weights from all groups
        all_weights = []
        all_counts = []
        
        for gid in groups_to_aggregate:
            client_ids = self.client_groups[gid]
            
            # Identify stragglers for this group
            if skip_stragglers:
                stragglers = self.identify_stragglers(gid)
            else:
                stragglers = []
            
            # Collect weights from non-straggler clients
            for cid in client_ids:
                if cid in stragglers and skip_stragglers:
                    continue
                
                client = self.clients[cid]
                weights = client.get_model_weights()
                count = client.data_size
                
                all_weights.append(weights)
                all_counts.append(count)
        
        if len(all_weights) == 0:
            # No clients to aggregate, return current edge model
            return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
        
        # Weighted average aggregation
        total_count = sum(all_counts)
        aggregated_weights = {}
        
        # Initialize with zeros
        for key in all_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(all_weights[0][key])
        
        # Accumulate weighted weights
        for weights, count in zip(all_weights, all_counts):
            weight_factor = count / total_count
            for key in weights.keys():
                aggregated_weights[key] += weight_factor * weights[key]
        
        # Update edge model
        self.edge_model.load_state_dict(aggregated_weights)
        
        aggregation_time = time.time() - start_time
        self.aggregation_times.append(aggregation_time)
        
        return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
    
    def perform_time_sharing_aggregation(self) -> Dict[str, torch.Tensor]:
        """
        Perform time-sharing aggregation across all groups sequentially.
        
        Each group gets a time slot, and aggregation happens group-by-group
        to reduce the impact of stragglers.
        
        Returns:
            Aggregated edge model weights
        """
        start_time = time.time()
        
        # Aggregate each group in sequence
        for group_id in range(self.num_groups):
            if len(self.client_groups[group_id]) > 0:
                # Aggregate this group
                self.perform_edge_aggregation(group_id=group_id, skip_stragglers=True)
                
                # Record group completion time
                group_time = time.time() - start_time
                self.group_completion_times[group_id].append(group_time)
        
        return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
    
    def get_edge_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current edge model weights."""
        return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
    
    def set_edge_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set edge model weights from global model."""
        self.edge_model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def get_server_info(self) -> Dict:
        """Get edge server information."""
        return {
            'edge_server_id': self.edge_server_id,
            'num_clients': len(self.clients),
            'num_groups': self.num_groups,
            'client_groups': {k: len(v) for k, v in self.client_groups.items()},
            'avg_aggregation_time': np.mean(self.aggregation_times) if self.aggregation_times else 0.0
        }


class FedUCCloudServer:
    """
    Cloud server for FedUC global aggregation.
    
    Aggregates edge server models using weighted averaging.
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedUC cloud server.
        
        Args:
            dataset_name: Name of dataset
            config: Configuration dictionary
            device: Training device
        """
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Edge server management
        self.edge_servers: Dict[int, FUCEdgeServer] = {}
        self.edge_data_sizes: Dict[int, int] = {}
        
        # Global model
        self.global_model = self._create_model()
        self.global_model.to(device)
        
        # Aggregation statistics
        self.aggregation_history = []
        
    def _create_model(self) -> nn.Module:
        """Create model based on dataset name."""
        num_classes = self.config.get('system', {}).get('num_classes', 10)
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes, dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes, dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes, dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def register_edge_server(self, edge_server: FUCEdgeServer, data_size: int):
        """
        Register edge server with cloud.
        
        Args:
            edge_server: Edge server instance
            data_size: Total data size at edge server
        """
        self.edge_servers[edge_server.edge_server_id] = edge_server
        self.edge_data_sizes[edge_server.edge_server_id] = data_size
    
    def receive_edge_model(
        self,
        edge_server_id: int,
        edge_weights: Dict[str, torch.Tensor],
        data_size: int
    ):
        """
        Receive edge model update.
        
        Args:
            edge_server_id: Edge server identifier
            edge_weights: Edge model weights
            data_size: Data size at edge server
        """
        self.edge_data_sizes[edge_server_id] = data_size
    
    def perform_global_aggregation(self) -> Dict[str, torch.Tensor]:
        """
        Perform global aggregation across all edge servers.
        
        Uses weighted averaging based on edge server data sizes.
        
        Returns:
            Aggregated global model weights
        """
        if len(self.edge_servers) == 0:
            return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        
        # Collect edge models and data sizes
        edge_weights = []
        edge_counts = []
        
        for edge_id, edge_server in self.edge_servers.items():
            weights = edge_server.get_edge_model_weights()
            count = self.edge_data_sizes.get(edge_id, 0)
            
            if count > 0:
                edge_weights.append(weights)
                edge_counts.append(count)
        
        if len(edge_weights) == 0:
            return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        
        # Weighted average aggregation
        total_count = sum(edge_counts)
        aggregated_weights = {}
        
        # Initialize with zeros
        for key in edge_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(edge_weights[0][key])
        
        # Accumulate weighted weights
        for weights, count in zip(edge_weights, edge_counts):
            weight_factor = count / total_count
            for key in weights.keys():
                aggregated_weights[key] += weight_factor * weights[key]
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        # Record aggregation
        self.aggregation_history.append({
            'num_edges': len(edge_weights),
            'total_samples': total_count
        })
        
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def distribute_global_model(self) -> Dict[str, torch.Tensor]:
        """
        Distribute global model to all edge servers.
        
        Returns:
            Global model weights
        """
        global_weights = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        
        # Send to all edge servers
        for edge_server in self.edge_servers.values():
            edge_server.set_edge_model_weights(global_weights)
        
        return global_weights
    
    def evaluate_global_model(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_data: Test data (numpy array)
            test_labels: Test labels (numpy array)
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        # Convert to tensors
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2) / 255.0
        labels_tensor = torch.LongTensor(test_labels)
        
        batch_size = self.config.get('training', {}).get('batch_size', 100)
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.global_model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_global_model(self) -> nn.Module:
        """Get global model instance."""
        return self.global_model
    
    def get_statistics(self) -> Dict:
        """Get aggregation statistics."""
        return {
            'num_edge_servers': len(self.edge_servers),
            'num_aggregations': len(self.aggregation_history),
            'aggregation_history': self.aggregation_history
        }


class FedUCTrainer:
    """
    Complete FedUC training orchestrator.
    
    Coordinates clients, edge servers, and cloud server for FedUC training
    with time-sharing scheduling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FedUC trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('system', {}).get('device', 'cpu')
        self.dataset_name = config.get('dataset', {}).get('name', 'fashion_mnist')
        
        # Training parameters
        self.total_rounds = config.get('experiment', {}).get('total_cloud_rounds', 50)
        self.edge_rounds_per_cloud = config.get('experiment', {}).get('edge_rounds_per_cloud', 3)
        self.clients_per_round = config.get('experiment', {}).get('clients_per_round', None)
        
        # Components (to be registered)
        self.clients: Dict[int, FedUCClient] = {}
        self.edge_servers: Dict[int, FUCEdgeServer] = {}
        self.cloud_server: Optional[FedUCCloudServer] = None
        
        # Training history
        self.training_history = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'time': []
        }
        
        # Timing
        self.start_time = None
        self.round_times = []
        
    def register_clients(self, clients: Dict[int, FedUCClient]):
        """Register all clients."""
        self.clients = clients
    
    def register_edge_servers(self, edge_servers: Dict[int, FUCEdgeServer]):
        """Register all edge servers."""
        self.edge_servers = edge_servers
        
        # Register edge servers with cloud
        if self.cloud_server is not None:
            for edge_id, edge_server in edge_servers.items():
                data_size = sum(
                    self.clients[cid].data_size
                    for group_clients in edge_server.client_groups.values()
                    for cid in group_clients
                )
                self.cloud_server.register_edge_server(edge_server, data_size)
    
    def register_cloud_server(self, cloud_server: FedUCCloudServer):
        """Register cloud server."""
        self.cloud_server = cloud_server
        
        # Register edge servers with cloud
        for edge_id, edge_server in self.edge_servers.items():
            data_size = sum(
                self.clients[cid].data_size
                for group_clients in edge_server.client_groups.values()
                for cid in group_clients
            )
            self.cloud_server.register_edge_server(edge_server, data_size)
    
    def train(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Run complete FedUC training.
        
        Args:
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        self.start_time = time.time()
        
        if verbose:
            print(f"Starting FedUC training for {self.total_rounds} cloud rounds...")
            print(f"Dataset: {self.dataset_name}, Device: {self.device}")
            print(f"Edge servers: {len(self.edge_servers)}, Clients: {len(self.clients)}")
        
        for cloud_round in range(self.total_rounds):
            round_start = time.time()
            
            # Distribute global model to edge servers
            if self.cloud_server is not None:
                global_weights = self.cloud_server.distribute_global_model()
                
                # Distribute to clients via edge servers
                for edge_server in self.edge_servers.values():
                    edge_server.set_edge_model_weights(global_weights)
                    
                    # Distribute to clients
                    for client_id in edge_server.client_groups.values():
                        for cid in (client_id if isinstance(client_id, list) else [client_id]):
                            if cid in self.clients:
                                self.clients[cid].set_model_weights(global_weights)
            
            # Edge aggregation rounds
            for edge_round in range(self.edge_rounds_per_cloud):
                # Local training for all clients
                for client in self.clients.values():
                    client.train_local()
                
                # Time-sharing aggregation at each edge server
                for edge_server in self.edge_servers.values():
                    edge_server.perform_time_sharing_aggregation()
            
            # Cloud aggregation
            if self.cloud_server is not None:
                self.cloud_server.perform_global_aggregation()
                
                # Evaluate global model
                metrics = self.cloud_server.evaluate_global_model(test_data, test_labels)
                
                round_time = time.time() - round_start
                self.round_times.append(round_time)
                
                # Record history
                self.training_history['round'].append(cloud_round)
                self.training_history['accuracy'].append(metrics['accuracy'])
                self.training_history['loss'].append(metrics['loss'])
                self.training_history['time'].append(round_time)
                
                if verbose and (cloud_round + 1) % 10 == 0:
                    print(f"Round {cloud_round + 1}/{self.total_rounds} - "
                          f"Accuracy: {metrics['accuracy']:.2f}%, "
                          f"Time: {round_time:.2f}s")
        
        total_time = time.time() - self.start_time
        
        if verbose:
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Final accuracy: {self.training_history['accuracy'][-1]:.2f}%")
        
        return self.training_history
    
    def get_training_statistics(self) -> Dict:
        """Get complete training statistics."""
        return {
            'total_rounds': self.total_rounds,
            'final_accuracy': self.training_history['accuracy'][-1] if self.training_history['accuracy'] else 0.0,
            'final_loss': self.training_history['loss'][-1] if self.training_history['loss'] else 0.0,
            'total_time': sum(self.round_times),
            'avg_round_time': np.mean(self.round_times) if self.round_times else 0.0,
            'training_history': self.training_history,
            'cloud_statistics': self.cloud_server.get_statistics() if self.cloud_server else {},
            'edge_statistics': {
                edge_id: server.get_server_info()
                for edge_id, server in self.edge_servers.items()
            }
        }


# Factory functions
def create_feduc_client(
    client_id: int,
    edge_server_id: int,
    group_id: int,
    dataset_name: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    config: Dict,
    device: str = 'cpu'
) -> FedUCClient:
    """
    Factory function to create FedUC client.
    
    Args:
        client_id: Unique client identifier
        edge_server_id: Associated edge server ID
        group_id: Time-sharing group ID
        dataset_name: Name of dataset
        train_data: Training data
        train_labels: Training labels
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedUCClient: Configured client instance
    """
    return FedUCClient(
        client_id=client_id,
        edge_server_id=edge_server_id,
        group_id=group_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_feduc_edge_server(
    edge_server_id: int,
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> FUCEdgeServer:
    """
    Factory function to create FedUC edge server.
    
    Args:
        edge_server_id: Unique edge server identifier
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FUCEdgeServer: Configured edge server instance
    """
    return FUCEdgeServer(
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_feduc_cloud_server(
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> FedUCCloudServer:
    """
    Factory function to create FedUC cloud server.
    
    Args:
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedUCCloudServer: Configured cloud server instance
    """
    return FedUCCloudServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_feduc_trainer(config: Dict) -> FedUCTrainer:
    """
    Factory function to create FedUC trainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FedUCTrainer: Configured trainer instance
    """
    return FedUCTrainer(config=config)
