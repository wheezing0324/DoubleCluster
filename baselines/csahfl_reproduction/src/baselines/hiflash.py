"""
HiFlash: Adaptive Staleness Control + Heterogeneity-Aware Association Baseline
Implements the HiFlash federated learning method for comparison with CSAHFL.

HiFlash features:
- Adaptive staleness control for asynchronous aggregation
- Heterogeneity-aware client-edge association
- Dynamic adjustment of aggregation weights based on client performance
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
from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class HiFlashClient:
    """
    Client for HiFlash federated learning with adaptive staleness tracking.
    
    Implements local training with performance metrics for heterogeneity-aware association.
    """
    
    def __init__(self, client_id: int, edge_server_id: int, dataset_name: str,
                 train_data: np.ndarray, train_labels: np.ndarray,
                 config: Dict, device: str = 'cpu'):
        """
        Initialize HiFlash client.
        
        Args:
            client_id: Unique client identifier
            edge_server_id: Associated edge server ID
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            train_data: Training data (numpy array)
            train_labels: Training labels (numpy array)
            config: Configuration dictionary
            device: Training device ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Store training data
        self.train_data = train_data
        self.train_labels = train_labels
        
        # Performance tracking for heterogeneity awareness
        self.training_loss_history = []
        self.training_accuracy_history = []
        self.completion_times = []
        self.staleness = 0  # Number of delayed rounds
        
        # Heterogeneity metrics
        self.data_size = len(train_data)
        self.label_distribution = self._compute_label_distribution()
        
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
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def _compute_label_distribution(self) -> np.ndarray:
        """Compute label distribution for heterogeneity awareness."""
        num_classes = 10
        label_counts = np.bincount(self.train_labels, minlength=num_classes)
        return label_counts / len(self.train_labels)
    
    def _create_data_loader(self, batch_size: int = 10, shuffle: bool = True) -> DataLoader:
        """Create DataLoader for training data."""
        dataset = TensorDataset(
            torch.FloatTensor(self.train_data).permute(0, 3, 1, 2),
            torch.LongTensor(self.train_labels)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from global/edge model."""
        self.model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
        self.staleness = 0  # Reset staleness on weight update
    
    def increment_staleness(self):
        """Increment staleness counter for delayed rounds."""
        self.staleness += 1
    
    def train_local(self, epochs: int = 1, batch_size: int = 10,
                    learning_rate: float = 0.01) -> Tuple[float, float]:
        """
        Train model locally for specified epochs.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        start_time = time.time()
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        data_loader = self._create_data_loader(batch_size=batch_size)
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_data.size(0)
                total_samples += batch_data.size(0)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / total_samples
        self.training_loss_history.append(avg_loss)
        
        # Calculate training accuracy
        accuracy = self.evaluate_local()['accuracy']
        self.training_accuracy_history.append(accuracy)
        
        completion_time = time.time() - start_time
        self.completion_times.append(completion_time)
        
        return avg_loss, accuracy
    
    def evaluate_local(self, test_data: Optional[np.ndarray] = None,
                       test_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model on local data or provided test data.
        
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        if test_data is None:
            # Use portion of training data for evaluation
            eval_size = min(100, len(self.train_data) // 5)
            eval_data = self.train_data[:eval_size]
            eval_labels = self.train_labels[:eval_size]
        else:
            eval_data = test_data
            eval_labels = test_labels
        
        eval_dataset = TensorDataset(
            torch.FloatTensor(eval_data).permute(0, 3, 1, 2),
            torch.LongTensor(eval_labels)
        )
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in eval_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return {
            'loss': total_loss / total,
            'accuracy': 100.0 * correct / total
        }
    
    def get_client_info(self) -> Dict:
        """Get client information for heterogeneity-aware association."""
        return {
            'client_id': self.client_id,
            'edge_server_id': self.edge_server_id,
            'data_size': self.data_size,
            'label_distribution': self.label_distribution,
            'avg_completion_time': np.mean(self.completion_times) if self.completion_times else 0,
            'avg_training_loss': np.mean(self.training_loss_history) if self.training_loss_history else 0,
            'avg_accuracy': np.mean(self.training_accuracy_history) if self.training_accuracy_history else 0,
            'staleness': self.staleness
        }


class HiFlashEdgeServer:
    """
    Edge server for HiFlash with adaptive staleness control.
    
    Implements heterogeneity-aware client association and staleness-based aggregation.
    """
    
    def __init__(self, edge_server_id: int, dataset_name: str,
                 config: Dict, device: str = 'cpu'):
        """
        Initialize HiFlash edge server.
        
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
        self.clients: Dict[int, HiFlashClient] = {}
        
        # Edge model
        self.edge_model = self._create_model()
        self.edge_model.to(device)
        
        # Aggregation tracking
        self.client_weights_history = []
        self.staleness_weights = {}
        
        # HiFlash parameters
        self.staleness_threshold = config.get('hiflash', {}).get('staleness_threshold', 3)
        self.staleness_discount = config.get('hiflash', {}).get('staleness_discount', 0.5)
        self.heterogeneity_weight = config.get('hiflash', {}).get('heterogeneity_weight', 0.5)
        
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
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def add_client(self, client: HiFlashClient):
        """Add client to edge server."""
        self.clients[client.client_id] = client
    
    def remove_client(self, client_id: int):
        """Remove client from edge server."""
        if client_id in self.clients:
            del self.clients[client_id]
    
    def get_edge_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get edge model weights."""
        return {k: v.cpu().clone() for k, v in self.edge_model.state_dict().items()}
    
    def set_edge_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set edge model weights from global model."""
        self.edge_model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def calculate_heterogeneity_score(self, client: HiFlashClient) -> float:
        """
        Calculate heterogeneity score for client-edge association.
        
        Higher score indicates better match between client and edge server.
        """
        # Based on label distribution similarity and performance metrics
        client_info = client.get_client_info()
        
        # Performance component (normalized accuracy)
        perf_score = client_info['avg_accuracy'] / 100.0 if client_info['avg_accuracy'] > 0 else 0.5
        
        # Latency component (inverse of completion time)
        if client_info['avg_completion_time'] > 0:
            latency_score = 1.0 / (1.0 + client_info['avg_completion_time'])
        else:
            latency_score = 1.0
        
        # Staleness penalty
        staleness_penalty = 1.0 / (1.0 + client_info['staleness'])
        
        # Combined heterogeneity score
        hetero_score = (
            self.heterogeneity_weight * perf_score +
            (1 - self.heterogeneity_weight) * latency_score
        ) * staleness_penalty
        
        return hetero_score
    
    def calculate_staleness_weight(self, client: HiFlashClient) -> float:
        """
        Calculate staleness-based weight for aggregation.
        
        Implements adaptive staleness control from HiFlash.
        """
        staleness = client.staleness
        
        if staleness <= self.staleness_threshold:
            # Within threshold: use exponential decay
            weight = self.staleness_discount ** staleness
        else:
            # Beyond threshold: stronger penalty
            weight = (self.staleness_discount ** self.staleness_threshold) * \
                     (0.5 ** (staleness - self.staleness_threshold))
        
        return weight
    
    def perform_edge_aggregation(self) -> Dict[str, torch.Tensor]:
        """
        Perform edge-level aggregation with staleness-aware weighting.
        
        Returns:
            Aggregated edge model weights
        """
        if not self.clients:
            return self.get_edge_model_weights()
        
        # Collect client weights and calculate aggregation weights
        client_weights = []
        aggregation_weights = []
        
        for client_id, client in self.clients.items():
            weights = client.get_model_weights()
            client_weights.append(weights)
            
            # Calculate combined weight: data size × staleness weight × heterogeneity score
            data_weight = client.data_size / sum(c.data_size for c in self.clients.values())
            staleness_weight = self.calculate_staleness_weight(client)
            hetero_score = self.calculate_heterogeneity_score(client)
            
            combined_weight = data_weight * staleness_weight * hetero_score
            aggregation_weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(aggregation_weights)
        aggregation_weights = [w / total_weight for w in aggregation_weights]
        
        # Perform weighted aggregation
        aggregated_weights = {}
        for key in client_weights[0].keys():
            aggregated_weights[key] = sum(
                weight * weights[key] 
                for weight, weights in zip(aggregation_weights, client_weights)
            )
        
        # Update edge model
        self.edge_model.load_state_dict(aggregated_weights)
        
        # Increment staleness for all clients
        for client in self.clients.values():
            client.increment_staleness()
        
        return aggregated_weights
    
    def get_server_info(self) -> Dict:
        """Get edge server information."""
        return {
            'edge_server_id': self.edge_server_id,
            'num_clients': len(self.clients),
            'total_data_size': sum(c.data_size for c in self.clients.values()),
            'avg_staleness': np.mean([c.staleness for c in self.clients.values()]) if self.clients else 0
        }


class HiFlashCloudServer:
    """
    Cloud server for HiFlash with global aggregation.
    
    Implements global model aggregation with edge server weighting.
    """
    
    def __init__(self, dataset_name: str, config: Dict, device: str = 'cpu'):
        """
        Initialize HiFlash cloud server.
        
        Args:
            dataset_name: Name of dataset
            config: Configuration dictionary
            device: Training device
        """
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Edge server management
        self.edge_servers: Dict[int, HiFlashEdgeServer] = {}
        
        # Global model
        self.global_model = self._create_model()
        self.global_model.to(device)
        
        # Aggregation tracking
        self.edge_models: Dict[int, Dict[str, torch.Tensor]] = {}
        self.aggregation_history = []
        
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
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def register_edge_server(self, edge_server: HiFlashEdgeServer):
        """Register edge server with cloud."""
        self.edge_servers[edge_server.edge_server_id] = edge_server
    
    def receive_edge_model(self, edge_server_id: int, weights: Dict[str, torch.Tensor]):
        """Receive aggregated model from edge server."""
        self.edge_models[edge_server_id] = weights
    
    def perform_global_aggregation(self) -> Dict[str, torch.Tensor]:
        """
        Perform global aggregation across edge servers.
        
        Returns:
            Aggregated global model weights
        """
        if not self.edge_models:
            return self.get_global_model_weights()
        
        # Calculate edge server weights based on data size and performance
        edge_weights = []
        edge_model_weights = []
        
        for edge_id, edge_model in self.edge_models.items():
            edge_server = self.edge_servers[edge_id]
            edge_info = edge_server.get_server_info()
            
            # Weight based on total data size at edge
            weight = edge_info['total_data_size']
            edge_weights.append(weight)
            edge_model_weights.append(edge_model)
        
        # Normalize weights
        total_weight = sum(edge_weights)
        edge_weights = [w / total_weight for w in edge_weights]
        
        # Perform weighted aggregation
        aggregated_weights = {}
        for key in edge_model_weights[0].keys():
            aggregated_weights[key] = sum(
                weight * weights[key]
                for weight, weights in zip(edge_weights, edge_model_weights)
            )
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        # Track aggregation
        self.aggregation_history.append({
            'num_edges': len(self.edge_models),
            'weights': edge_weights
        })
        
        # Clear edge models for next round
        self.edge_models.clear()
        
        return aggregated_weights
    
    def distribute_global_model(self):
        """Distribute global model to all edge servers."""
        global_weights = self.get_global_model_weights()
        for edge_server in self.edge_servers.values():
            edge_server.set_edge_model_weights(global_weights)
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get global model weights."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def set_global_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set global model weights."""
        self.global_model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate global model on test data."""
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
                
                total_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        return {
            'loss': total_loss / total,
            'accuracy': 100.0 * correct / total
        }
    
    def get_statistics(self) -> Dict:
        """Get cloud server statistics."""
        return {
            'num_edge_servers': len(self.edge_servers),
            'aggregation_count': len(self.aggregation_history)
        }


class HiFlashTrainer:
    """
    Complete HiFlash training orchestrator.
    
    Coordinates clients, edge servers, and cloud server for HiFlash training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize HiFlash trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('system', {}).get('device', 'cpu')
        self.dataset_name = config.get('experiment', {}).get('dataset', 'fashion_mnist')
        
        # Components
        self.clients: Dict[int, HiFlashClient] = {}
        self.edge_servers: Dict[int, HiFlashEdgeServer] = {}
        self.cloud_server: Optional[HiFlashCloudServer] = None
        
        # Training statistics
        self.accuracy_history = []
        self.loss_history = []
        self.time_history = []
        self.training_start_time = None
        
    def register_clients(self, clients: Dict[int, HiFlashClient]):
        """Register clients with trainer."""
        self.clients = clients
        
        # Assign clients to edge servers
        for client in clients.values():
            if client.edge_server_id in self.edge_servers:
                self.edge_servers[client.edge_server_id].add_client(client)
    
    def register_edge_servers(self, edge_servers: Dict[int, HiFlashEdgeServer]):
        """Register edge servers with trainer."""
        self.edge_servers = edge_servers
    
    def register_cloud_server(self, cloud_server: HiFlashCloudServer):
        """Register cloud server with trainer."""
        self.cloud_server = cloud_server
    
    def train(self, total_rounds: int = 50, edge_rounds: int = 3,
              test_loader: Optional[DataLoader] = None) -> Dict[str, List]:
        """
        Run HiFlash training.
        
        Args:
            total_rounds: Number of global aggregation rounds
            edge_rounds: Number of edge aggregation rounds per global round
            test_loader: Test data loader for evaluation
            
        Returns:
            Dictionary with training history
        """
        self.training_start_time = time.time()
        
        learning_rate = self.config.get('training', {}).get('learning_rate', 0.01)
        local_epochs = self.config.get('intra_cluster', {}).get('max_epochs', 10)
        
        for global_round in range(total_rounds):
            round_start = time.time()
            
            # Edge aggregation rounds
            for edge_round in range(edge_rounds):
                # Local training at clients
                for client in self.clients.values():
                    client.train_local(epochs=local_epochs, learning_rate=learning_rate)
                
                # Edge aggregation
                for edge_server in self.edge_servers.values():
                    edge_weights = edge_server.perform_edge_aggregation()
                    self.cloud_server.receive_edge_model(edge_server.edge_server_id, edge_weights)
            
            # Global aggregation
            self.cloud_server.perform_global_aggregation()
            
            # Distribute global model
            self.cloud_server.distribute_global_model()
            
            # Set client weights from edge servers
            for client in self.clients.values():
                edge_server = self.edge_servers[client.edge_server_id]
                client.set_model_weights(edge_server.get_edge_model_weights())
            
            # Evaluate
            if test_loader is not None:
                eval_result = self.cloud_server.evaluate_global_model(test_loader)
                self.accuracy_history.append(eval_result['accuracy'])
                self.loss_history.append(eval_result['loss'])
            else:
                self.accuracy_history.append(0.0)
                self.loss_history.append(0.0)
            
            round_time = time.time() - round_start
            self.time_history.append(round_time)
        
        return {
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'time_history': self.time_history,
            'total_time': time.time() - self.training_start_time
        }
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            'num_clients': len(self.clients),
            'num_edge_servers': len(self.edge_servers),
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0,
            'avg_accuracy': np.mean(self.accuracy_history) if self.accuracy_history else 0,
            'total_time': sum(self.time_history)
        }


# Factory functions
def create_hiflash_client(client_id: int, edge_server_id: int, dataset_name: str,
                          train_data: np.ndarray, train_labels: np.ndarray,
                          config: Dict, device: str = 'cpu') -> HiFlashClient:
    """Factory function to create HiFlash client."""
    return HiFlashClient(
        client_id=client_id,
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_hiflash_edge_server(edge_server_id: int, dataset_name: str,
                               config: Dict, device: str = 'cpu') -> HiFlashEdgeServer:
    """Factory function to create HiFlash edge server."""
    return HiFlashEdgeServer(
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_hiflash_cloud_server(dataset_name: str, config: Dict,
                                device: str = 'cpu') -> HiFlashCloudServer:
    """Factory function to create HiFlash cloud server."""
    return HiFlashCloudServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_hiflash_trainer(config: Dict) -> HiFlashTrainer:
    """Factory function to create HiFlash trainer."""
    return HiFlashTrainer(config=config)
