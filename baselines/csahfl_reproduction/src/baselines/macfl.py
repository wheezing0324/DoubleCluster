"""
MACFL: Mobility-Aware Clustered Federated Learning
Baseline implementation for CSAHFL comparison experiments

This implements a mobility-aware clustering approach with redesigned aggregation
that considers client movement patterns between edge servers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class MACFLClient:
    """
    Client for MACFL with mobility-aware local training
    
    Args:
        client_id: Unique client identifier
        edge_server_id: Current edge server ID
        dataset_name: Name of dataset to use
        train_data: Training data (numpy array)
        train_labels: Training labels (numpy array)
        config: Configuration dictionary
        device: Training device ('cuda' or 'cpu')
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
        self.train_data = train_data
        self.train_labels = train_labels
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Training parameters
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.01)
        self.batch_size = config.get('training', {}).get('batch_size', 10)
        self.max_epochs = config.get('intra_cluster', {}).get('max_epochs', 10)
        
        # Mobility tracking
        self.mobility_history = []
        self.last_edge_server = edge_server_id
        
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset"""
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
    
    def _create_data_loader(self, batch_size: int) -> DataLoader:
        """Create DataLoader for training data"""
        tensor_data = torch.FloatTensor(self.train_data).permute(0, 3, 1, 2)
        tensor_labels = torch.LongTensor(self.train_labels)
        dataset = TensorDataset(tensor_data, tensor_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current model weights as numpy arrays"""
        weights = {}
        for name, param in self.model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights from numpy arrays"""
        state_dict = {}
        for name, param in self.model.state_dict().items():
            if name in weights:
                state_dict[name] = torch.FloatTensor(weights[name])
        self.model.load_state_dict(state_dict)
    
    def train_local(self, epochs: int = None) -> Tuple[float, float]:
        """
        Train model locally for specified epochs
        
        Args:
            epochs: Number of epochs (uses max_epochs if None)
            
        Returns:
            Tuple of (average_loss, training_time)
        """
        if epochs is None:
            epochs = self.max_epochs
            
        start_time = time.time()
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = self._create_data_loader(self.batch_size)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (epochs * num_batches) if num_batches > 0 else 0.0
        training_time = time.time() - start_time
        
        return avg_loss, training_time
    
    def evaluate_local(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data (numpy array)
            test_labels: Test labels (numpy array)
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        
        tensor_data = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        tensor_labels = torch.LongTensor(test_labels)
        dataset = TensorDataset(tensor_data, tensor_labels)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
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
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def update_edge_server(self, new_edge_server_id: int):
        """Update client's edge server assignment (mobility event)"""
        if new_edge_server_id != self.edge_server_id:
            self.mobility_history.append({
                'from': self.edge_server_id,
                'to': new_edge_server_id,
                'timestamp': time.time()
            })
            self.edge_server_id = new_edge_server_id
    
    def get_client_info(self) -> Dict:
        """Get client information"""
        return {
            'client_id': self.client_id,
            'edge_server_id': self.edge_server_id,
            'dataset_name': self.dataset_name,
            'num_samples': len(self.train_labels),
            'mobility_events': len(self.mobility_history)
        }


class MACFLEdgeServer:
    """
    Edge server for MACFL with mobility-aware client management
    
    Args:
        edge_server_id: Unique edge server identifier
        dataset_name: Name of dataset to use
        config: Configuration dictionary
        device: Training device
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
        
        # Client management
        self.clients: Dict[int, MACFLClient] = {}
        self.client_data_sizes: Dict[int, int] = {}
        
        # Edge model
        self.edge_model = self._create_model()
        self.edge_model.to(device)
        
        # Mobility-aware clustering
        self.client_mobility_scores: Dict[int, float] = {}
        self.cluster_stability = 1.0
        
        # Aggregation tracking
        self.aggregation_count = 0
        self.total_aggregation_time = 0.0
        
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset"""
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def add_client(self, client: MACFLClient):
        """Add client to this edge server"""
        self.clients[client.client_id] = client
        self.client_data_sizes[client.client_id] = len(client.train_labels)
        self.client_mobility_scores[client.client_id] = 1.0  # Initial stability score
    
    def remove_client(self, client_id: int):
        """Remove client from this edge server"""
        if client_id in self.clients:
            del self.clients[client_id]
        if client_id in self.client_data_sizes:
            del self.client_data_sizes[client_id]
        if client_id in self.client_mobility_scores:
            del self.client_mobility_scores[client_id]
    
    def update_mobility_scores(self, mobility_history: Dict[int, int]):
        """
        Update client mobility scores based on movement history
        
        Args:
            mobility_history: Dict mapping client_id to number of moves
        """
        for client_id, num_moves in mobility_history.items():
            if client_id in self.clients:
                # Decrease stability score with more moves
                self.client_mobility_scores[client_id] = 1.0 / (1.0 + num_moves)
        
        # Update cluster stability (average of all client scores)
        if self.client_mobility_scores:
            self.cluster_stability = np.mean(list(self.client_mobility_scores.values()))
    
    def collect_client_weights(self) -> Tuple[List[Dict[str, np.ndarray]], List[int], List[float]]:
        """
        Collect weights from all clients with mobility-aware weighting
        
        Returns:
            Tuple of (weights_list, data_sizes, mobility_weights)
        """
        weights_list = []
        data_sizes = []
        mobility_weights = []
        
        for client_id, client in self.clients.items():
            weights_list.append(client.get_model_weights())
            data_sizes.append(self.client_data_sizes[client_id])
            mobility_weights.append(self.client_mobility_scores[client_id])
        
        return weights_list, data_sizes, mobility_weights
    
    def perform_edge_aggregation(self) -> Dict[str, np.ndarray]:
        """
        Perform mobility-aware edge aggregation
        
        Returns:
            Aggregated edge model weights
        """
        start_time = time.time()
        
        weights_list, data_sizes, mobility_weights = self.collect_client_weights()
        
        if not weights_list:
            return self.get_edge_model_weights()
        
        # Calculate combined weights (data size × mobility stability)
        total_weight = sum(d * m for d, m in zip(data_sizes, mobility_weights))
        
        # Aggregate with mobility-aware weighting
        aggregated_weights = {}
        for name in weights_list[0].keys():
            aggregated = np.zeros_like(weights_list[0][name])
            for weights, data_size, mobility_weight in zip(weights_list, data_sizes, mobility_weights):
                combined_weight = (data_size * mobility_weight) / total_weight
                aggregated += combined_weight * weights[name]
            aggregated_weights[name] = aggregated
        
        # Update edge model
        state_dict = {name: torch.FloatTensor(weights) for name, weights in aggregated_weights.items()}
        self.edge_model.load_state_dict(state_dict)
        
        aggregation_time = time.time() - start_time
        self.aggregation_count += 1
        self.total_aggregation_time += aggregation_time
        
        return aggregated_weights
    
    def get_edge_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current edge model weights"""
        weights = {}
        for name, param in self.edge_model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_edge_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set edge model weights"""
        state_dict = {name: torch.FloatTensor(weights) for name, weights in weights.items()}
        self.edge_model.load_state_dict(state_dict)
    
    def get_cluster_info(self) -> Dict:
        """Get cluster information"""
        return {
            'edge_server_id': self.edge_server_id,
            'num_clients': len(self.clients),
            'cluster_stability': self.cluster_stability,
            'avg_mobility_score': np.mean(list(self.client_mobility_scores.values())) if self.client_mobility_scores else 0.0
        }


class MACFLCloudServer:
    """
    Cloud server for MACFL with global model aggregation
    
    Args:
        dataset_name: Name of dataset to use
        config: Configuration dictionary
        device: Training device
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
        
        # Edge server management
        self.edge_servers: Dict[int, MACFLEdgeServer] = {}
        self.edge_data_sizes: Dict[int, int] = {}
        
        # Global model
        self.global_model = self._create_model()
        self.global_model.to(device)
        
        # Aggregation tracking
        self.aggregation_history = []
        
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset"""
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def register_edge_server(self, edge_server: MACFLEdgeServer):
        """Register edge server with cloud"""
        self.edge_servers[edge_server.edge_server_id] = edge_server
        self.edge_data_sizes[edge_server.edge_server_id] = sum(
            edge_server.client_data_sizes.values()
        )
    
    def receive_edge_model(self, edge_server_id: int, weights: Dict[str, np.ndarray]):
        """Receive aggregated model from edge server"""
        if edge_server_id in self.edge_servers:
            self.edge_data_sizes[edge_server_id] = sum(
                self.edge_servers[edge_server_id].client_data_sizes.values()
            )
    
    def perform_global_aggregation(self) -> Dict[str, np.ndarray]:
        """
        Perform global aggregation across all edge servers
        
        Returns:
            Aggregated global model weights
        """
        edge_weights = []
        edge_sizes = []
        
        for edge_id, edge_server in self.edge_servers.items():
            edge_weights.append(edge_server.get_edge_model_weights())
            edge_sizes.append(self.edge_data_sizes[edge_id])
        
        if not edge_weights:
            return self.get_global_model_weights()
        
        # Weighted aggregation based on edge server data sizes
        total_size = sum(edge_sizes)
        
        aggregated_weights = {}
        for name in edge_weights[0].keys():
            aggregated = np.zeros_like(edge_weights[0][name])
            for weights, size in zip(edge_weights, edge_sizes):
                aggregated += (size / total_size) * weights[name]
            aggregated_weights[name] = aggregated
        
        # Update global model
        state_dict = {name: torch.FloatTensor(weights) for name, weights in aggregated_weights.items()}
        self.global_model.load_state_dict(state_dict)
        
        self.aggregation_history.append({
            'timestamp': time.time(),
            'num_edge_servers': len(edge_servers),
            'total_clients': sum(edge_sizes)
        })
        
        return aggregated_weights
    
    def distribute_global_model(self):
        """Distribute global model to all edge servers"""
        global_weights = self.get_global_model_weights()
        for edge_server in self.edge_servers.values():
            edge_server.set_edge_model_weights(global_weights)
    
    def evaluate_global_model(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate global model on test data
        
        Args:
            test_data: Test data (numpy array)
            test_labels: Test labels (numpy array)
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.global_model.eval()
        
        tensor_data = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        tensor_labels = torch.LongTensor(test_labels)
        dataset = TensorDataset(tensor_data, tensor_labels)
        batch_size = self.config.get('training', {}).get('batch_size', 100)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
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
        
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_global_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current global model weights"""
        weights = {}
        for name, param in self.global_model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_global_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set global model weights"""
        state_dict = {name: torch.FloatTensor(weights) for name, weights in weights.items()}
        self.global_model.load_state_dict(state_dict)
    
    def get_aggregation_statistics(self) -> Dict:
        """Get aggregation statistics"""
        return {
            'num_aggregations': len(self.aggregation_history),
            'num_edge_servers': len(self.edge_servers),
            'total_clients': sum(self.edge_data_sizes.values())
        }


class MACFLTrainer:
    """
    Complete MACFL training orchestrator
    
    Args:
        config: Configuration dictionary
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('system', {}).get('device', 'cpu')
        self.dataset_name = config.get('experiment', {}).get('dataset', 'fashion_mnist')
        
        # Components
        self.clients: Dict[int, MACFLClient] = {}
        self.edge_servers: Dict[int, MACFLEdgeServer] = {}
        self.cloud_server: Optional[MACFLCloudServer] = None
        
        # Training tracking
        self.accuracy_history = []
        self.loss_history = []
        self.time_history = []
        self.mobility_events = 0
        
    def register_clients(self, clients: Dict[int, MACFLClient]):
        """Register all clients"""
        self.clients = clients
    
    def register_edge_servers(self, edge_servers: Dict[int, MACFLEdgeServer]):
        """Register all edge servers"""
        self.edge_servers = edge_servers
    
    def register_cloud_server(self, cloud_server: MACFLCloudServer):
        """Register cloud server"""
        self.cloud_server = cloud_server
    
    def train(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        total_cloud_rounds: int = 50,
        edge_rounds_per_cloud: int = 3,
        clients_per_round: float = 1.0
    ) -> Dict[str, List]:
        """
        Run complete MACFL training
        
        Args:
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            total_cloud_rounds: Number of global aggregation rounds
            edge_rounds_per_cloud: Number of edge aggregation rounds per cloud round
            clients_per_round: Fraction of clients to sample per round
            
        Returns:
            Dictionary with training history
        """
        start_time = time.time()
        
        num_clients = len(self.clients)
        num_clients_per_round = max(1, int(num_clients * clients_per_round))
        
        for cloud_round in range(total_cloud_rounds):
            round_start = time.time()
            
            # Distribute global model to edge servers
            if self.cloud_server and cloud_round > 0:
                self.cloud_server.distribute_global_model()
            
            # Perform edge aggregation rounds
            for edge_round in range(edge_rounds_per_cloud):
                # Sample clients for this round
                client_ids = np.random.choice(
                    list(self.clients.keys()),
                    size=num_clients_per_round,
                    replace=False
                )
                
                # Train clients and aggregate at edge servers
                for client_id in client_ids:
                    client = self.clients[client_id]
                    client.train_local()
                
                # Aggregate at each edge server
                for edge_server in self.edge_servers.values():
                    edge_server.perform_edge_aggregation()
            
            # Global aggregation at cloud
            if self.cloud_server:
                self.cloud_server.perform_global_aggregation()
                
                # Evaluate global model
                metrics = self.cloud_server.evaluate_global_model(test_data, test_labels)
                self.accuracy_history.append(metrics['accuracy'])
                self.loss_history.append(metrics['loss'])
            
            round_time = time.time() - round_start
            self.time_history.append(round_time)
        
        total_time = time.time() - start_time
        
        return {
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'time_history': self.time_history,
            'total_time': total_time,
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0
        }
    
    def get_training_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'num_clients': len(self.clients),
            'num_edge_servers': len(self.edge_servers),
            'final_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'total_time': sum(self.time_history)
        }


# Factory functions for easy instantiation

def create_macfl_client(
    client_id: int,
    edge_server_id: int,
    dataset_name: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    config: Dict,
    device: str = 'cpu'
) -> MACFLClient:
    """
    Factory function to create MACFL client
    
    Returns:
        MACFLClient: Configured client instance
    """
    return MACFLClient(
        client_id=client_id,
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_macfl_edge_server(
    edge_server_id: int,
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> MACFLEdgeServer:
    """
    Factory function to create MACFL edge server
    
    Returns:
        MACFLEdgeServer: Configured edge server instance
    """
    return MACFLEdgeServer(
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_macfl_cloud_server(
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> MACFLCloudServer:
    """
    Factory function to create MACFL cloud server
    
    Returns:
        MACFLCloudServer: Configured cloud server instance
    """
    return MACFLCloudServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )


def create_macfl_trainer(config: Dict) -> MACFLTrainer:
    """
    Factory function to create MACFL trainer
    
    Returns:
        MACFLTrainer: Configured trainer instance
    """
    return MACFLTrainer(config=config)
