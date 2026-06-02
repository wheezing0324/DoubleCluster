"""
HierFedAVG: Synchronous Hierarchical Federated Learning Baseline

Implements synchronous hierarchical federated learning with client-edge-cloud hierarchy.
This baseline performs synchronous aggregation at both edge and cloud levels,
providing a comparison point for CSAHFL's semi-asynchronous approach.

Reference: Standard hierarchical FL baseline for comparison with CSAHFL (Section 3.4)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

# Import model creation functions
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class HierFedAVGClient:
    """
    Client for synchronous hierarchical federated learning.
    
    Performs local training with fixed number of epochs and participates
    in synchronous edge-level aggregation.
    """
    
    def __init__(
        self,
        client_id: int,
        edge_server_id: int,
        dataset_name: str,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        config: Dict,
        device: str = 'cpu'
    ):
        self.client_id = client_id
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Create data loader
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_loader = self._create_data_loader()
        
        # Training configuration
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.01)
        self.local_epochs = config.get('intra_cluster', {}).get('max_epochs', 10)
        self.batch_size = config.get('training', {}).get('batch_size', 10)
        
        # Optimizer and criterion
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # Timing
        self.training_time = 0.0
        
    def _create_model(self) -> nn.Module:
        """Create CNN model based on dataset."""
        if self.dataset_name == 'fashion_mnist':
            return CNNFashionMNIST(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'cifar10':
            return CNNCIFAR10(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'svhn':
            return CNNSVHN(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _create_data_loader(self) -> DataLoader:
        """Create PyTorch DataLoader for client data."""
        # Convert numpy arrays to tensors
        # Handle different data formats (NHWC vs NCHW)
        if len(self.train_data.shape) == 4:
            # NHWC format: convert to NCHW
            data_tensor = torch.FloatTensor(self.train_data.transpose(0, 3, 1, 2))
        else:
            data_tensor = torch.FloatTensor(self.train_data)
        
        labels_tensor = torch.LongTensor(self.train_labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create SGD optimizer."""
        return optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current model weights as numpy arrays."""
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights from numpy arrays."""
        state_dict = {}
        for name, param in self.model.state_dict().items():
            if name in weights:
                state_dict[name] = torch.FloatTensor(weights[name])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
    
    def train_local(self, num_epochs: Optional[int] = None) -> Tuple[float, float]:
        """
        Train model locally for specified epochs.
        
        Args:
            num_epochs: Number of local epochs (uses config default if None)
            
        Returns:
            Tuple of (average_loss, training_time)
        """
        if num_epochs is None:
            num_epochs = self.local_epochs
        
        start_time = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (num_epochs * len(self.train_loader))
        training_time = time.time() - start_time
        self.training_time = training_time
        
        return avg_loss, training_time
    
    def evaluate_local(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_client_info(self) -> Dict:
        """Get client information."""
        return {
            'client_id': self.client_id,
            'edge_server_id': self.edge_server_id,
            'dataset_name': self.dataset_name,
            'num_samples': len(self.train_labels),
            'training_time': self.training_time
        }


class HierFedAVGEdgeServer:
    """
    Edge server for synchronous hierarchical federated learning.
    
    Manages synchronous aggregation of client models at the edge level.
    All clients must complete training before aggregation occurs.
    """
    
    def __init__(
        self,
        edge_server_id: int,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create edge model
        self.model = self._create_model()
        self.model.to(device)
        
        # Client management
        self.clients: Dict[int, HierFedAVGClient] = {}
        self.client_weights: Dict[int, Dict[str, np.ndarray]] = {}
        self.client_sample_counts: Dict[int, int] = {}
        
        # Aggregation statistics
        self.aggregation_time = 0.0
        self.num_aggregations = 0
        
    def _create_model(self) -> nn.Module:
        """Create CNN model based on dataset."""
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return CNNFashionMNIST(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'cifar10':
            return CNNCIFAR10(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'svhn':
            return CNNSVHN(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def add_client(self, client: HierFedAVGClient):
        """Add client to this edge server."""
        self.clients[client.client_id] = client
    
    def remove_client(self, client_id: int):
        """Remove client from this edge server."""
        if client_id in self.clients:
            del self.clients[client_id]
    
    def initialize_edge_model(self, global_weights: Dict[str, np.ndarray]):
        """Initialize edge model with global weights."""
        self.set_edge_model_weights(global_weights)
    
    def collect_client_weights(self, client_ids: Optional[List[int]] = None):
        """
        Collect weights from all clients (synchronous - wait for all).
        
        Args:
            client_ids: List of client IDs to collect from (None = all clients)
        """
        if client_ids is None:
            client_ids = list(self.clients.keys())
        
        self.client_weights = {}
        self.client_sample_counts = {}
        
        for client_id in client_ids:
            if client_id in self.clients:
                client = self.clients[client_id]
                self.client_weights[client_id] = client.get_model_weights()
                self.client_sample_counts[client_id] = len(client.train_labels)
    
    def perform_edge_aggregation(self) -> Dict[str, np.ndarray]:
        """
        Perform synchronous aggregation of client models.
        
        Uses weighted average based on client data sizes.
        
        Returns:
            Aggregated edge model weights
        """
        start_time = time.time()
        
        if not self.client_weights:
            # No client weights, return current edge model
            return self.get_edge_model_weights()
        
        # Calculate total samples
        total_samples = sum(self.client_sample_counts.values())
        
        # Aggregate weights
        aggregated_weights = {}
        for weight_name in list(self.client_weights.values())[0].keys():
            aggregated_weights[weight_name] = np.zeros_like(
                self.client_weights[list(self.client_weights.keys())[0]][weight_name]
            )
            
            for client_id, weights in self.client_weights.items():
                sample_count = self.client_sample_counts[client_id]
                weight_factor = sample_count / total_samples
                aggregated_weights[weight_name] += weights[weight_name] * weight_factor
        
        # Update edge model
        self.set_edge_model_weights(aggregated_weights)
        
        self.aggregation_time = time.time() - start_time
        self.num_aggregations += 1
        
        return aggregated_weights
    
    def get_edge_model_weights(self) -> Dict[str, np.ndarray]:
        """Get edge model weights."""
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_edge_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set edge model weights."""
        state_dict = {}
        for name, param in self.model.state_dict().items():
            if name in weights:
                state_dict[name] = torch.FloatTensor(weights[name])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
    
    def get_cluster_info(self) -> Dict:
        """Get edge server cluster information."""
        return {
            'edge_server_id': self.edge_server_id,
            'num_clients': len(self.clients),
            'client_ids': list(self.clients.keys()),
            'aggregation_time': self.aggregation_time,
            'num_aggregations': self.num_aggregations
        }


class HierFedAVGCloudServer:
    """
    Cloud server for synchronous hierarchical federated learning.
    
    Manages global model aggregation from edge servers and distributes
    updated global model back to edge servers.
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create global model
        self.model = self._create_model()
        self.model.to(device)
        
        # Edge server management
        self.edge_servers: Dict[int, HierFedAVGEdgeServer] = {}
        self.edge_weights: Dict[int, Dict[str, np.ndarray]] = {}
        self.edge_sample_counts: Dict[int, int] = {}
        
        # Aggregation statistics
        self.aggregation_time = 0.0
        self.num_aggregations = 0
        
    def _create_model(self) -> nn.Module:
        """Create CNN model based on dataset."""
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return CNNFashionMNIST(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'cifar10':
            return CNNCIFAR10(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'svhn':
            return CNNSVHN(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def register_edge_server(self, edge_server: HierFedAVGEdgeServer):
        """Register edge server with cloud."""
        self.edge_servers[edge_server.edge_server_id] = edge_server
    
    def receive_edge_model(
        self,
        edge_server_id: int,
        weights: Dict[str, np.ndarray],
        num_samples: int
    ):
        """
        Receive aggregated model from edge server.
        
        Args:
            edge_server_id: ID of edge server
            weights: Aggregated edge model weights
            num_samples: Total samples at edge server
        """
        self.edge_weights[edge_server_id] = weights
        self.edge_sample_counts[edge_server_id] = num_samples
    
    def perform_global_aggregation(self) -> Dict[str, np.ndarray]:
        """
        Perform synchronous global aggregation from all edge servers.
        
        Uses weighted average based on edge server data sizes.
        
        Returns:
            Aggregated global model weights
        """
        start_time = time.time()
        
        if not self.edge_weights:
            # No edge weights, return current global model
            return self.get_global_model_weights()
        
        # Calculate total samples
        total_samples = sum(self.edge_sample_counts.values())
        
        # Aggregate weights
        aggregated_weights = {}
        for weight_name in list(self.edge_weights.values())[0].keys():
            aggregated_weights[weight_name] = np.zeros_like(
                self.edge_weights[list(self.edge_weights.keys())[0]][weight_name]
            )
            
            for edge_id, weights in self.edge_weights.items():
                sample_count = self.edge_sample_counts[edge_id]
                weight_factor = sample_count / total_samples
                aggregated_weights[weight_name] += weights[weight_name] * weight_factor
        
        # Update global model
        self.set_global_model_weights(aggregated_weights)
        
        self.aggregation_time = time.time() - start_time
        self.num_aggregations += 1
        
        return aggregated_weights
    
    def distribute_global_model(self):
        """Distribute global model to all edge servers."""
        global_weights = self.get_global_model_weights()
        
        for edge_server in self.edge_servers.values():
            edge_server.initialize_edge_model(global_weights)
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_global_model_weights(self) -> Dict[str, np.ndarray]:
        """Get global model weights."""
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_global_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set global model weights."""
        state_dict = {}
        for name, param in self.model.state_dict().items():
            if name in weights:
                state_dict[name] = torch.FloatTensor(weights[name])
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
    
    def get_aggregation_statistics(self) -> Dict:
        """Get aggregation statistics."""
        return {
            'num_edge_servers': len(self.edge_servers),
            'aggregation_time': self.aggregation_time,
            'num_aggregations': self.num_aggregations
        }
    
    def reset_for_new_experiment(self):
        """Reset statistics for new experiment."""
        self.edge_weights = {}
        self.edge_sample_counts = {}
        self.aggregation_time = 0.0
        self.num_aggregations = 0


class HierFedAVGTrainer:
    """
    Complete training orchestrator for synchronous hierarchical federated learning.
    
    Coordinates the training process across clients, edge servers, and cloud server
    with synchronous aggregation at both levels.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('system', {}).get('device', 'cpu')
        self.dataset_name = config.get('experiment', {}).get('dataset', 'fashion_mnist')
        
        # Training configuration
        self.total_cloud_rounds = config.get('experiment', {}).get('total_cloud_rounds', 50)
        self.edge_rounds_per_cloud = config.get('experiment', {}).get('edge_rounds_per_cloud', 3)
        
        # Components (initialized externally)
        self.cloud_server: Optional[HierFedAVGCloudServer] = None
        self.edge_servers: Dict[int, HierFedAVGEdgeServer] = {}
        self.clients: Dict[int, HierFedAVGClient] = {}
        
        # Metrics
        self.accuracy_history = []
        self.loss_history = []
        self.time_history = []
        self.total_training_time = 0.0
        
    def register_clients(self, clients: Dict[int, HierFedAVGClient]):
        """Register all clients."""
        self.clients = clients
        
        # Assign clients to edge servers
        for client in clients.values():
            edge_id = client.edge_server_id
            if edge_id not in self.edge_servers:
                raise ValueError(f"Edge server {edge_id} not registered")
            self.edge_servers[edge_id].add_client(client)
    
    def register_edge_servers(self, edge_servers: Dict[int, HierFedAVGEdgeServer]):
        """Register all edge servers with cloud server."""
        self.edge_servers = edge_servers
        
        if self.cloud_server is not None:
            for edge_server in edge_servers.values():
                self.cloud_server.register_edge_server(edge_server)
    
    def register_cloud_server(self, cloud_server: HierFedAVGCloudServer):
        """Register cloud server."""
        self.cloud_server = cloud_server
        
        # Register edge servers with cloud
        for edge_server in self.edge_servers.values():
            self.cloud_server.register_edge_server(edge_server)
    
    def train(self, test_loader: DataLoader, verbose: bool = True) -> Dict[str, List]:
        """
        Run complete HierFedAVG training.
        
        Args:
            test_loader: DataLoader for test/evaluation data
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history (accuracy, loss, time)
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting HierFedAVG training for {self.total_cloud_rounds} cloud rounds")
            print(f"Edge rounds per cloud round: {self.edge_rounds_per_cloud}")
            print(f"Number of clients: {len(self.clients)}")
            print(f"Number of edge servers: {len(self.edge_servers)}")
        
        for cloud_round in range(self.total_cloud_rounds):
            round_start = time.time()
            
            # Perform edge-level training and aggregation
            for edge_round in range(self.edge_rounds_per_cloud):
                # Each client trains locally
                for client in self.clients.values():
                    client.train_local()
                
                # Each edge server aggregates client models synchronously
                for edge_server in self.edge_servers.values():
                    edge_server.collect_client_weights()
                    edge_server.perform_edge_aggregation()
            
            # Cloud-level aggregation
            for edge_server in self.edge_servers.values():
                edge_weights = edge_server.get_edge_model_weights()
                num_samples = sum(len(c.train_labels) for c in edge_server.clients.values())
                self.cloud_server.receive_edge_model(
                    edge_server.edge_server_id,
                    edge_weights,
                    num_samples
                )
            
            # Global aggregation and distribution
            self.cloud_server.perform_global_aggregation()
            self.cloud_server.distribute_global_model()
            
            # Evaluate global model
            metrics = self.cloud_server.evaluate_global_model(test_loader)
            
            round_time = time.time() - round_start
            self.accuracy_history.append(metrics['accuracy'])
            self.loss_history.append(metrics['loss'])
            self.time_history.append(round_time)
            
            if verbose and (cloud_round + 1) % 10 == 0:
                print(f"Cloud Round {cloud_round + 1}/{self.total_cloud_rounds} | "
                      f"Accuracy: {metrics['accuracy']:.2f}% | "
                      f"Round Time: {round_time:.2f}s")
        
        self.total_training_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining completed in {self.total_training_time:.2f}s")
            print(f"Final accuracy: {self.accuracy_history[-1]:.2f}%")
        
        return {
            'accuracy': self.accuracy_history,
            'loss': self.loss_history,
            'time': self.time_history,
            'total_time': self.total_training_time
        }
    
    def get_training_statistics(self) -> Dict:
        """Get comprehensive training statistics."""
        return {
            'total_cloud_rounds': self.total_cloud_rounds,
            'edge_rounds_per_cloud': self.edge_rounds_per_cloud,
            'total_training_time': self.total_training_time,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'best_accuracy': max(self.accuracy_history) if self.accuracy_history else 0.0,
            'num_clients': len(self.clients),
            'num_edge_servers': len(self.edge_servers)
        }


# Factory functions for easy instantiation

def create_hierfedavg_client(
    client_id: int,
    edge_server_id: int,
    dataset_name: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    config: Dict,
    device: str = 'cpu'
) -> HierFedAVGClient:
    """
    Factory function to create HierFedAVG client.
    
    Args:
        client_id: Unique client identifier
        edge_server_id: ID of associated edge server
        dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
        train_data: Client training data (numpy array)
        train_labels: Client training labels (numpy array)
        config: Configuration dictionary
        device: Training device ('cpu' or 'cuda')
        
    Returns:
        HierFedAVGClient: Configured client instance
    """
    return HierFedAVGClient(
        client_id=client_id,
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_hierfedavg_edge_server(
    edge_server_id: int,
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> HierFedAVGEdgeServer:
    """
    Factory function to create HierFedAVG edge server.
    
    Args:
        edge_server_id: Unique edge server identifier
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Training device
        
    Returns:
        HierFedAVGEdgeServer: Configured edge server instance
    """
    return HierFedAVGEdgeServer(
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_hierfedavg_cloud_server(
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> HierFedAVGCloudServer:
    """
    Factory function to create HierFedAVG cloud server.
    
    Args:
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Training device
        
    Returns:
        HierFedAVGCloudServer: Configured cloud server instance
    """
    return HierFedAVGCloudServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_hierfedavg_trainer(config: Dict) -> HierFedAVGTrainer:
    """
    Factory function to create HierFedAVG trainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HierFedAVGTrainer: Configured trainer instance
    """
    return HierFedAVGTrainer(config=config)
