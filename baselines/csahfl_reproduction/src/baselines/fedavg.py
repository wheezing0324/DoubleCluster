"""
FedAVG Baseline Implementation for CSAHFL Framework

Implements standard Federated Averaging (FedAVG) as described in:
McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from 
Decentralized Data." AISTATS 2017.

This is a cloud-only federated learning baseline without hierarchical structure.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import time

from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class FedAVGClient:
    """
    Client for standard FedAVG federated learning.
    
    Performs local training on private data and returns model updates.
    """
    
    def __init__(self, client_id: int, dataset_name: str, train_data: np.ndarray,
                 train_labels: np.ndarray, config: Dict, device: str = 'cpu'):
        """
        Initialize FedAVG client.
        
        Args:
            client_id: Unique client identifier
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            train_data: Training data for this client (numpy array)
            train_labels: Training labels for this client
            config: Configuration dictionary
            device: Training device ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.train_labels = train_labels
        self.config = config
        self.device = device
        
        # Create local model
        self.model = self._create_model()
        self.model.to(device)
        
        # Training parameters
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.01)
        self.batch_size = config.get('training', {}).get('batch_size', 10)
        self.local_epochs = config.get('intra_cluster', {}).get('max_epochs', 10)
        
        # Create data loader
        self._create_data_loader()
        
        # Timing
        self.training_time = 0.0
    
    def _create_model(self) -> nn.Module:
        """Create model based on dataset name."""
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
    
    def _create_data_loader(self):
        """Create PyTorch DataLoader for local data."""
        # Convert numpy to torch tensors
        data_tensor = torch.FloatTensor(self.train_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(self.train_labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == 'cuda')
        )
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from global model."""
        self.model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def train_local(self, global_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Perform local training with global model initialization.
        
        Args:
            global_weights: Global model weights to initialize with
            
        Returns:
            Tuple of (local_model_weights, training_time)
        """
        # Set global weights
        self.set_model_weights(global_weights)
        
        # Training setup
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train for local epochs
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for batch_data, batch_labels in self.data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        training_time = time.time() - start_time
        self.training_time = training_time
        
        # Return updated weights
        return self.get_model_weights(), training_time
    
    def evaluate_local(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(test_labels)
        
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
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


class FedAVGServer:
    """
    Central server for standard FedAVG federated learning.
    
    Coordinates global model aggregation from client updates.
    """
    
    def __init__(self, dataset_name: str, config: Dict, device: str = 'cpu'):
        """
        Initialize FedAVG server.
        
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
        
        # Client management
        self.clients: Dict[int, FedAVGClient] = {}
        self.client_data_sizes: Dict[int, int] = {}
        
        # Statistics
        self.aggregation_history = []
        self.total_communication_rounds = 0
    
    def _create_model(self) -> nn.Module:
        """Create global model based on dataset name."""
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
    
    def register_client(self, client: FedAVGClient):
        """Register a client with the server."""
        self.clients[client.client_id] = client
        self.client_data_sizes[client.client_id] = len(client.train_data)
    
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
    
    def set_global_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set global model weights."""
        self.global_model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def perform_aggregation(self, client_weights: Dict[int, Dict[str, torch.Tensor]],
                           client_data_sizes: Dict[int, int]) -> Dict[str, torch.Tensor]:
        """
        Perform FedAVG aggregation (weighted average of client models).
        
        Args:
            client_weights: Dictionary mapping client_id to their model weights
            client_data_sizes: Dictionary mapping client_id to their data size
            
        Returns:
            Aggregated global model weights
        """
        total_samples = sum(client_data_sizes.values())
        
        # Initialize aggregated weights
        aggregated_weights = {}
        first_client_weights = list(client_weights.values())[0]
        
        for key in first_client_weights.keys():
            aggregated_weights[key] = torch.zeros_like(first_client_weights[key])
        
        # Weighted average
        for client_id, weights in client_weights.items():
            weight_factor = client_data_sizes[client_id] / total_samples
            for key in weights.keys():
                aggregated_weights[key] += weight_factor * weights[key]
        
        return aggregated_weights
    
    def train_round(self, selected_clients: Optional[List[int]] = None) -> Dict:
        """
        Perform one round of FedAVG training.
        
        Args:
            selected_clients: List of client IDs to participate (None = all)
            
        Returns:
            Dictionary with round statistics
        """
        if selected_clients is None:
            selected_clients = list(self.clients.keys())
        
        # Get current global weights
        global_weights = self.get_global_model_weights()
        
        # Collect client updates
        client_updates = {}
        client_sizes = {}
        total_training_time = 0.0
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Client trains locally
            local_weights, training_time = client.train_local(global_weights)
            client_updates[client_id] = local_weights
            client_sizes[client_id] = client_data_sizes[client_id]
            total_training_time = max(total_training_time, training_time)
        
        # Aggregate updates
        aggregated_weights = self.perform_aggregation(client_updates, client_sizes)
        
        # Update global model
        self.set_global_model_weights(aggregated_weights)
        
        # Update statistics
        self.total_communication_rounds += 1
        round_stats = {
            'round': self.total_communication_rounds,
            'num_clients': len(selected_clients),
            'training_time': total_training_time,
            'client_ids': selected_clients
        }
        self.aggregation_history.append(round_stats)
        
        return round_stats
    
    def evaluate_global_model(self, test_data: np.ndarray,
                             test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.global_model.eval()
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(test_labels)
        
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 10),
            shuffle=False
        )
        
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
    
    def get_statistics(self) -> Dict:
        """Get server statistics."""
        return {
            'total_rounds': self.total_communication_rounds,
            'num_clients': len(self.clients),
            'aggregation_history': self.aggregation_history
        }


def create_fedavg_client(client_id: int, dataset_name: str, train_data: np.ndarray,
                        train_labels: np.ndarray, config: Dict,
                        device: str = 'cpu') -> FedAVGClient:
    """
    Factory function to create FedAVG client.
    
    Args:
        client_id: Client identifier
        dataset_name: Dataset name
        train_data: Training data
        train_labels: Training labels
        config: Configuration
        device: Training device
        
    Returns:
        FedAVGClient instance
    """
    return FedAVGClient(
        client_id=client_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_fedavg_server(dataset_name: str, config: Dict,
                        device: str = 'cpu') -> FedAVGServer:
    """
    Factory function to create FedAVG server.
    
    Args:
        dataset_name: Dataset name
        config: Configuration
        device: Training device
        
    Returns:
        FedAVGServer instance
    """
    return FedAVGServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )


class FedAVGTrainer:
    """
    Complete FedAVG training orchestrator.
    
    Manages the full FedAVG training loop for experiments.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FedAVG trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_name = config.get('experiment', {}).get('dataset', 'fashion_mnist')
        self.device = config.get('system', {}).get('device', 'cpu')
        self.total_rounds = config.get('experiment', {}).get('total_cloud_rounds', 50)
        
        # Create server
        self.server = create_fedavg_server(
            dataset_name=self.dataset_name,
            config=config,
            device=self.device
        )
        
        # Training history
        self.accuracy_history = []
        self.loss_history = []
        self.time_history = []
    
    def register_clients(self, clients: List[FedAVGClient]):
        """Register clients with the server."""
        for client in clients:
            self.server.register_client(client)
    
    def train(self, test_data: np.ndarray, test_labels: np.ndarray,
             clients_per_round: Optional[int] = None) -> Dict:
        """
        Run complete FedAVG training.
        
        Args:
            test_data: Test data for evaluation
            test_labels: Test labels
            clients_per_round: Number of clients per round (None = all)
            
        Returns:
            Training results dictionary
        """
        import time as time_module
        
        start_time = time_module.time()
        
        for round_num in range(self.total_rounds):
            # Select clients for this round
            if clients_per_round is not None and clients_per_round < len(self.server.clients):
                selected_clients = np.random.choice(
                    list(self.server.clients.keys()),
                    size=clients_per_round,
                    replace=False
                ).tolist()
            else:
                selected_clients = None
            
            # Perform training round
            round_stats = self.server.train_round(selected_clients)
            
            # Evaluate global model
            eval_results = self.server.evaluate_global_model(test_data, test_labels)
            
            # Record metrics
            self.accuracy_history.append(eval_results['accuracy'])
            self.loss_history.append(eval_results['loss'])
            elapsed_time = time_module.time() - start_time
            self.time_history.append(elapsed_time)
            
            # Log progress
            if (round_num + 1) % 10 == 0 or round_num == 0:
                print(f"Round {round_num + 1}/{self.total_rounds} - "
                      f"Accuracy: {eval_results['accuracy']:.2f}%, "
                      f"Time: {elapsed_time:.1f}s")
        
        total_time = time_module.time() - start_time
        
        return {
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'time_history': self.time_history,
            'final_accuracy': self.accuracy_history[-1],
            'total_time': total_time,
            'server_stats': self.server.get_statistics()
        }
