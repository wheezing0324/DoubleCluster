"""
FedAT: Synchronous Intra-Tier, Asynchronous Cross-Tier Federated Learning

This module implements FedAT baseline for CSAHFL comparison experiments.
FedAT uses synchronous aggregation within each tier (client-edge, edge-cloud)
but allows asynchronous communication across tiers to reduce straggler impact.

Reference:
    "FedAT: A High-Performance and Communication-Efficient Federated Learning
     System with Asynchronous Tiers" - IEEE/ACM Supercomputing 2021
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Import model creation functions
from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class FedATClient:
    """
    Client for FedAT federated learning with local training.
    
    Clients perform local training synchronously within their edge server group,
    then send updates to edge server asynchronously.
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
        """
        Initialize FedAT client.
        
        Args:
            client_id: Unique client identifier
            edge_server_id: ID of associated edge server
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            train_data: Training data (numpy array)
            train_labels: Training labels (numpy array)
            config: Configuration dictionary
            device: Training device ('cuda' or 'cpu')
        """
        self.client_id = client_id
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Create data loader
        self.train_loader = self._create_data_loader(train_data, train_labels)
        self.num_samples = len(train_labels)
        
        # Training parameters
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.01)
        self.local_epochs = config.get('intra_cluster', {}).get('max_epochs', 10)
        self.batch_size = config.get('training', {}).get('batch_size', 10)
        
        # Timing
        self.last_update_time = None
        self.training_time = 0.0
        
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset."""
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'svhn':
            return create_svhn_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _create_data_loader(self, data: np.ndarray, labels: np.ndarray) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        # Convert to tensors
        data_tensor = torch.FloatTensor(data).permute(0, 3, 1, 2)  # NHWC to NCHW
        labels_tensor = torch.LongTensor(labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from state dict."""
        self.model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def train_local(self, global_weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Train model locally for specified epochs.
        
        Args:
            global_weights: Optional global model weights for initialization
            
        Returns:
            Dictionary with training metrics (loss, accuracy, time)
        """
        start_time = time.time()
        
        # Initialize from global weights if provided
        if global_weights is not None:
            self.set_model_weights(global_weights)
        
        # Set up training
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
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
                
                # Track metrics
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_labels).sum().item()
                total_samples += batch_labels.size(0)
        
        end_time = time.time()
        self.training_time = end_time - start_time
        self.last_update_time = time.time()
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': 100.0 * total_correct / total_samples,
            'time': self.training_time,
            'num_samples': self.num_samples
        }
    
    def evaluate_local(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on local test data.
        
        Args:
            test_data: Test data (numpy array)
            test_labels: Test labels (numpy array)
            
        Returns:
            Dictionary with evaluation metrics (loss, accuracy)
        """
        self.model.eval()
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(test_labels)
        
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_labels).sum().item()
                total_samples += batch_labels.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': 100.0 * total_correct / total_samples
        }
    
    def get_client_info(self) -> Dict:
        """Get client information."""
        return {
            'client_id': self.client_id,
            'edge_server_id': self.edge_server_id,
            'num_samples': self.num_samples,
            'last_update_time': self.last_update_time,
            'training_time': self.training_time
        }


class FedATEdgeServer:
    """
    Edge server for FedAT with synchronous intra-tier aggregation.
    
    Aggregates client updates synchronously (waits for all clients),
    then sends to cloud server asynchronously.
    """
    
    def __init__(
        self,
        edge_server_id: int,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedAT edge server.
        
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
        
        # Create edge model
        self.edge_model = self._create_model()
        self.edge_model.to(device)
        
        # Client management
        self.clients: Dict[int, FedATClient] = {}
        self.client_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.client_metrics: Dict[int, Dict] = {}
        
        # Aggregation tracking
        self.total_samples = 0
        self.last_aggregation_time = None
        self.aggregation_count = 0
        
        # Staleness tracking for async cross-tier
        self.staleness = 0  # Number of delayed rounds
        
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset."""
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'svhn':
            return create_svhn_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def add_client(self, client: FedATClient):
        """Add client to edge server."""
        self.clients[client.client_id] = client
        self.total_samples += client.num_samples
    
    def remove_client(self, client_id: int):
        """Remove client from edge server."""
        if client_id in self.clients:
            self.total_samples -= self.clients[client_id].num_samples
            del self.clients[client_id]
    
    def get_edge_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current edge model weights."""
        return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
    
    def set_edge_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set edge model weights from global model."""
        self.edge_model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def collect_client_weights(
        self,
        global_weights: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict]]:
        """
        Collect weights from all clients synchronously.
        
        Args:
            global_weights: Global model weights to initialize clients
            
        Returns:
            Tuple of (client_weights dict, client_metrics dict)
        """
        self.client_weights = {}
        self.client_metrics = {}
        
        # Train all clients synchronously
        for client_id, client in self.clients.items():
            metrics = client.train_local(global_weights)
            self.client_weights[client_id] = client.get_model_weights()
            self.client_metrics[client_id] = metrics
        
        self.last_aggregation_time = time.time()
        return self.client_weights, self.client_metrics
    
    def perform_edge_aggregation(self) -> Dict[str, float]:
        """
        Perform synchronous aggregation of client weights.
        
        Returns:
            Dictionary with aggregation statistics
        """
        if not self.client_weights:
            return {'success': False, 'error': 'No client weights to aggregate'}
        
        start_time = time.time()
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in self.client_weights[list(self.client_weights.keys())[0]].keys():
            aggregated_weights[key] = torch.zeros_like(
                self.client_weights[list(self.client_weights.keys())[0]][key]
            )
        
        # Weighted average based on sample counts
        total_samples = sum(
            self.client_metrics[client_id]['num_samples']
            for client_id in self.client_weights.keys()
        )
        
        for client_id, weights in self.client_weights.items():
            weight_factor = self.client_metrics[client_id]['num_samples'] / total_samples
            for key in weights.keys():
                aggregated_weights[key] += weights[key] * weight_factor
        
        # Update edge model
        self.edge_model.load_state_dict(aggregated_weights)
        
        end_time = time.time()
        self.aggregation_count += 1
        
        return {
            'success': True,
            'edge_server_id': self.edge_server_id,
            'num_clients': len(self.client_weights),
            'total_samples': total_samples,
            'aggregation_time': end_time - start_time,
            'staleness': self.staleness
        }
    
    def update_staleness(self, delayed: bool = True):
        """Update staleness counter for asynchronous cross-tier."""
        if delayed:
            self.staleness += 1
        else:
            self.staleness = 0
    
    def get_edge_info(self) -> Dict:
        """Get edge server information."""
        return {
            'edge_server_id': self.edge_server_id,
            'num_clients': len(self.clients),
            'total_samples': self.total_samples,
            'aggregation_count': self.aggregation_count,
            'staleness': self.staleness,
            'last_aggregation_time': self.last_aggregation_time
        }


class FedATCloudServer:
    """
    Cloud server for FedAT with asynchronous cross-tier aggregation.
    
    Aggregates edge server updates asynchronously (does not wait for all edges),
    using staleness-aware weighting.
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedAT cloud server.
        
        Args:
            dataset_name: Name of dataset
            config: Configuration dictionary
            device: Training device
        """
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create global model
        self.global_model = self._create_model()
        self.global_model.to(device)
        
        # Edge server management
        self.edge_models: Dict[int, Dict[str, torch.Tensor]] = {}
        self.edge_metrics: Dict[int, Dict] = {}
        self.edge_staleness: Dict[int, int] = {}
        
        # Aggregation tracking
        self.aggregation_count = 0
        self.last_aggregation_time = None
        
        # Asynchronous aggregation parameters
        self.staleness_threshold = config.get('fedat', {}).get('staleness_threshold', 3)
        self.staleness_discount = config.get('fedat', {}).get('staleness_discount', 0.5)
        
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset."""
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        elif self.dataset_name == 'svhn':
            return create_svhn_model(
                num_classes=10,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.5)
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def register_edge_server(self, edge_server_id: int):
        """Register edge server with cloud."""
        self.edge_staleness[edge_server_id] = 0
    
    def receive_edge_model(
        self,
        edge_server_id: int,
        edge_weights: Dict[str, torch.Tensor],
        edge_metrics: Dict,
        staleness: int = 0
    ):
        """
        Receive edge model update asynchronously.
        
        Args:
            edge_server_id: ID of edge server
            edge_weights: Edge model weights
            edge_metrics: Edge aggregation metrics
            staleness: Staleness of this update
        """
        self.edge_models[edge_server_id] = edge_weights
        self.edge_metrics[edge_server_id] = edge_metrics
        self.edge_staleness[edge_server_id] = staleness
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def set_global_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set global model weights."""
        self.global_model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def perform_global_aggregation(self, wait_for_all: bool = False) -> Dict[str, float]:
        """
        Perform asynchronous aggregation of edge models.
        
        Args:
            wait_for_all: If True, wait for all edge servers (synchronous)
                         If False, aggregate available updates (asynchronous)
        
        Returns:
            Dictionary with aggregation statistics
        """
        if not self.edge_models:
            return {'success': False, 'error': 'No edge models to aggregate'}
        
        start_time = time.time()
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in list(self.edge_models.values())[0].keys():
            aggregated_weights[key] = torch.zeros_like(
                list(self.edge_models.values())[0][key]
            )
        
        # Calculate staleness-aware weights
        total_weight = 0.0
        edge_weights = {}
        
        for edge_id, edge_weight_dict in self.edge_models.items():
            # Get sample count from metrics
            num_samples = self.edge_metrics.get(edge_id, {}).get('total_samples', 100)
            staleness = self.edge_staleness.get(edge_id, 0)
            
            # Apply staleness discount
            discount_factor = self.staleness_discount ** staleness
            weight = num_samples * discount_factor
            
            edge_weights[edge_id] = weight
            total_weight += weight
        
        # Normalize and aggregate
        for edge_id, edge_weight_dict in self.edge_models.items():
            weight_factor = edge_weights[edge_id] / total_weight
            for key in edge_weight_dict.keys():
                aggregated_weights[key] += edge_weight_dict[key] * weight_factor
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        end_time = time.time()
        self.aggregation_count += 1
        self.last_aggregation_time = end_time
        
        # Reset staleness for next round
        for edge_id in self.edge_staleness.keys():
            self.edge_staleness[edge_id] = 0
        
        return {
            'success': True,
            'num_edges': len(self.edge_models),
            'aggregation_time': end_time - start_time,
            'aggregation_count': self.aggregation_count,
            'asynchronous': not wait_for_all
        }
    
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
            Dictionary with evaluation metrics
        """
        self.global_model.eval()
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(test_labels)
        
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 64),
            shuffle=False,
            num_workers=0
        )
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.global_model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_labels).sum().item()
                total_samples += batch_labels.size(0)
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': 100.0 * total_correct / total_samples
        }
    
    def get_statistics(self) -> Dict:
        """Get cloud server statistics."""
        return {
            'aggregation_count': self.aggregation_count,
            'num_edge_servers': len(self.edge_models),
            'last_aggregation_time': self.last_aggregation_time,
            'staleness_threshold': self.staleness_threshold,
            'staleness_discount': self.staleness_discount
        }


class FedATTrainer:
    """
    Complete FedAT training orchestrator.
    
    Coordinates clients, edge servers, and cloud server for FedAT training.
    Implements synchronous intra-tier and asynchronous cross-tier aggregation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FedAT trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('system', {}).get('device', 'cpu')
        self.dataset_name = config.get('dataset', {}).get('name', 'fashion_mnist')
        
        # Training parameters
        self.total_rounds = config.get('experiment', {}).get('total_cloud_rounds', 50)
        self.edge_rounds_per_cloud = config.get('experiment', {}).get('edge_rounds_per_cloud', 3)
        
        # Components (initialized later)
        self.clients: Dict[int, FedATClient] = {}
        self.edge_servers: Dict[int, FedATEdgeServer] = {}
        self.cloud_server: Optional[FedATCloudServer] = None
        
        # Client-edge assignment
        self.client_edge_assignment: Dict[int, int] = {}
        
        # Metrics tracking
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        self.time_history: List[float] = []
        self.round_times: List[float] = []
        
        # Test data (set during registration)
        self.test_data: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        
    def register_clients(self, clients: Dict[int, FedATClient], assignment: Dict[int, int]):
        """
        Register clients and their edge server assignments.
        
        Args:
            clients: Dictionary of client_id -> FedATClient
            assignment: Dictionary of client_id -> edge_server_id
        """
        self.clients = clients
        self.client_edge_assignment = assignment
        
    def register_edge_servers(self, edge_servers: Dict[int, FedATEdgeServer]):
        """
        Register edge servers.
        
        Args:
            edge_servers: Dictionary of edge_server_id -> FedATEdgeServer
        """
        self.edge_servers = edge_servers
        
        # Add clients to their edge servers
        for client_id, edge_server_id in self.client_edge_assignment.items():
            if client_id in self.clients and edge_server_id in self.edge_servers:
                self.edge_servers[edge_server_id].add_client(self.clients[client_id])
    
    def register_cloud_server(self, cloud_server: FedATCloudServer):
        """
        Register cloud server.
        
        Args:
            cloud_server: FedATCloudServer instance
        """
        self.cloud_server = cloud_server
        
        # Register edge servers with cloud
        for edge_server_id in self.edge_servers.keys():
            self.cloud_server.register_edge_server(edge_server_id)
    
    def register_test_data(self, test_data: np.ndarray, test_labels: np.ndarray):
        """
        Register test data for evaluation.
        
        Args:
            test_data: Test data (numpy array)
            test_labels: Test labels (numpy array)
        """
        self.test_data = test_data
        self.test_labels = test_labels
    
    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Run FedAT training.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history (accuracy, loss, time)
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting FedAT training for {self.total_rounds} cloud rounds...")
            print(f"Dataset: {self.dataset_name}, Device: {self.device}")
        
        for cloud_round in range(self.total_rounds):
            round_start = time.time()
            
            # Get global model weights
            global_weights = self.cloud_server.get_global_model_weights()
            
            # Run edge rounds
            for edge_round in range(self.edge_rounds_per_cloud):
                # Each edge server performs synchronous intra-tier aggregation
                for edge_server_id, edge_server in self.edge_servers.items():
                    # Collect client weights synchronously
                    edge_server.collect_client_weights(global_weights)
                    
                    # Perform edge aggregation
                    edge_server.perform_edge_aggregation()
                    
                    # Send to cloud asynchronously (receive immediately)
                    self.cloud_server.receive_edge_model(
                        edge_server_id=edge_server_id,
                        edge_weights=edge_server.get_edge_model_weights(),
                        edge_metrics=edge_server.get_edge_info(),
                        staleness=edge_server.staleness
                    )
                    
                    # Update staleness for next round
                    edge_server.update_staleness(delayed=False)
            
            # Perform cloud aggregation (asynchronous - doesn't wait for all)
            agg_stats = self.cloud_server.perform_global_aggregation(wait_for_all=False)
            
            # Evaluate global model
            if self.test_data is not None and self.test_labels is not None:
                eval_metrics = self.cloud_server.evaluate_global_model(
                    self.test_data,
                    self.test_labels
                )
                self.accuracy_history.append(eval_metrics['accuracy'])
                self.loss_history.append(eval_metrics['loss'])
            else:
                self.accuracy_history.append(0.0)
                self.loss_history.append(0.0)
            
            round_time = time.time() - round_start
            self.time_history.append(time.time() - start_time)
            self.round_times.append(round_time)
            
            if verbose and (cloud_round + 1) % 10 == 0:
                print(f"  Cloud Round {cloud_round + 1}/{self.total_rounds}: "
                      f"Accuracy = {self.accuracy_history[-1]:.2f}%, "
                      f"Time = {round_time:.2f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nFedAT training completed in {total_time:.2f}s")
            print(f"Final accuracy: {self.accuracy_history[-1]:.2f}%")
        
        return {
            'accuracy': self.accuracy_history,
            'loss': self.loss_history,
            'time': self.time_history,
            'round_times': self.round_times,
            'total_time': total_time
        }
    
    def get_training_statistics(self) -> Dict:
        """Get comprehensive training statistics."""
        return {
            'total_rounds': self.total_rounds,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'best_accuracy': max(self.accuracy_history) if self.accuracy_history else 0.0,
            'total_time': self.time_history[-1] if self.time_history else 0.0,
            'avg_round_time': np.mean(self.round_times) if self.round_times else 0.0,
            'cloud_stats': self.cloud_server.get_statistics() if self.cloud_server else {},
            'edge_stats': {
                edge_id: edge.get_edge_info()
                for edge_id, edge in self.edge_servers.items()
            }
        }


def create_fedat_client(
    client_id: int,
    edge_server_id: int,
    dataset_name: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    config: Dict,
    device: str = 'cpu'
) -> FedATClient:
    """
    Factory function to create FedAT client.
    
    Args:
        client_id: Unique client identifier
        edge_server_id: ID of associated edge server
        dataset_name: Name of dataset
        train_data: Training data
        train_labels: Training labels
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedATClient: Configured client instance
    """
    return FedATClient(
        client_id=client_id,
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_fedat_edge_server(
    edge_server_id: int,
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> FedATEdgeServer:
    """
    Factory function to create FedAT edge server.
    
    Args:
        edge_server_id: Unique edge server identifier
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedATEdgeServer: Configured edge server instance
    """
    return FedATEdgeServer(
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_fedat_cloud_server(
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> FedATCloudServer:
    """
    Factory function to create FedAT cloud server.
    
    Args:
        dataset_name: Name of dataset
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedATCloudServer: Configured cloud server instance
    """
    return FedATCloudServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_fedat_trainer(config: Dict) -> FedATTrainer:
    """
    Factory function to create FedAT trainer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FedATTrainer: Configured trainer instance
    """
    return FedATTrainer(config=config)
