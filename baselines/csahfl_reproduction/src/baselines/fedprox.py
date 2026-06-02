"""
FedProx: Federated Learning with Proximal Term
Implementation of FedProx baseline for CSAHFL comparison experiments.

FedProx adds a proximal term to the local objective function to handle
statistical heterogeneity in federated learning settings.

Reference: Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"FedProx: Analytical and Practical Improvements to Federated Learning."
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import time

from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class FedProxClient:
    """
    Client for FedProx federated learning with proximal term.
    
    Implements local training with proximal regularization to handle
    statistical heterogeneity (non-IID data).
    """
    
    def __init__(
        self,
        client_id: int,
        dataset_name: str,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedProx client.
        
        Args:
            client_id: Unique client identifier
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            train_data: Training data for this client
            train_labels: Training labels for this client
            config: Configuration dictionary
            device: Training device ('cuda' or 'cpu')
        """
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Create data loader
        self.train_loader = self._create_data_loader(train_data, train_labels)
        self.num_samples = len(train_labels)
        
        # FedProx specific: proximal term parameter (mu)
        self.mu = config.get('fedprox', {}).get('mu', 0.01)
        
        # Store global model weights for proximal term calculation
        self.global_weights = None
    
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset."""
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _create_data_loader(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        batch_size: int = None
    ) -> DataLoader:
        """Create PyTorch DataLoader for client data."""
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 10)
        
        # Convert numpy arrays to tensors
        data_tensor = torch.FloatTensor(data).permute(0, 3, 1, 2)  # NHWC to NCHW
        labels_tensor = torch.LongTensor(labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        return loader
    
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
                state_dict[name] = torch.FloatTensor(weights[name]).to(self.device)
        self.model.load_state_dict(state_dict)
        
        # Store global weights for proximal term
        self.global_weights = {k: v.copy() for k, v in weights.items()}
    
    def _compute_proximal_term(self) -> torch.Tensor:
        """
        Compute proximal term: (mu/2) * ||w - w_global||^2
        
        Returns:
            Scalar tensor representing proximal penalty
        """
        if self.global_weights is None:
            return torch.tensor(0.0, device=self.device)
        
        proximal_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.global_weights:
                global_param = torch.FloatTensor(self.global_weights[name]).to(self.device)
                proximal_loss += torch.sum((param - global_param) ** 2)
        
        return (self.mu / 2.0) * proximal_loss
    
    def train_local(
        self,
        epochs: int = None,
        learning_rate: float = None,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Train model locally with FedProx objective.
        
        Local objective: L(w) + (mu/2) * ||w - w_global||^2
        
        Args:
            epochs: Number of local training epochs
            learning_rate: Learning rate for SGD
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (average_loss, training_time)
        """
        if epochs is None:
            epochs = self.config.get('training', {}).get('local_epochs', 10)
        if learning_rate is None:
            learning_rate = self.config.get('training', {}).get('learning_rate', 0.01)
        
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                
                # Standard cross-entropy loss
                ce_loss = criterion(outputs, targets)
                
                # Add proximal term
                proximal_loss = self._compute_proximal_term()
                
                # Total loss = CE loss + proximal term
                loss = ce_loss + proximal_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += ce_loss.item()  # Track only CE loss for reporting
                epoch_batches += 1
            
            total_loss += epoch_loss
            num_batches += epoch_batches
            
            if verbose and (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / epoch_batches
                print(f"Client {self.client_id}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        average_loss = total_loss / max(num_batches, 1)
        
        return average_loss, training_time
    
    def evaluate_local(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate model on local test data.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        
        # Create test loader
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(test_labels)
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_client_info(self) -> Dict:
        """Get client information."""
        return {
            'client_id': self.client_id,
            'dataset_name': self.dataset_name,
            'num_samples': self.num_samples,
            'device': self.device,
            'proximal_mu': self.mu
        }


class FedProxServer:
    """
    Central server for FedProx federated learning.
    
    Implements standard federated averaging aggregation.
    """
    
    def __init__(
        self,
        dataset_name: str,
        config: Dict,
        device: str = 'cpu'
    ):
        """
        Initialize FedProx server.
        
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
        
        # Registered clients
        self.clients: Dict[int, FedProxClient] = {}
        
        # Statistics tracking
        self.aggregation_history = []
        self.total_training_time = 0.0
    
    def _create_model(self) -> nn.Module:
        """Create appropriate model for dataset."""
        dropout_rate = self.config.get('model', {}).get('dropout_rate', 0.5)
        
        if self.dataset_name == 'fashion_mnist':
            return create_fashion_mnist_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'cifar10':
            return create_cifar10_model(num_classes=10, dropout_rate=dropout_rate)
        elif self.dataset_name == 'svhn':
            return create_svhn_model(num_classes=10, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def register_client(self, client: FedProxClient):
        """Register a client with the server."""
        self.clients[client.client_id] = client
    
    def get_global_model_weights(self) -> Dict[str, np.ndarray]:
        """Get global model weights."""
        weights = {}
        for name, param in self.global_model.state_dict().items():
            weights[name] = param.cpu().numpy().copy()
        return weights
    
    def set_global_model_weights(self, weights: Dict[str, np.ndarray]):
        """Set global model weights."""
        state_dict = {}
        for name, param in self.global_model.state_dict().items():
            if name in weights:
                state_dict[name] = torch.FloatTensor(weights[name]).to(self.device)
        self.global_model.load_state_dict(state_dict)
    
    def perform_aggregation(
        self,
        client_weights: Dict[int, Dict[str, np.ndarray]],
        client_samples: Dict[int, int]
    ) -> Dict[str, np.ndarray]:
        """
        Perform federated averaging aggregation.
        
        w_global = sum(n_i * w_i) / sum(n_i)
        
        Args:
            client_weights: Dictionary mapping client_id to model weights
            client_samples: Dictionary mapping client_id to number of samples
            
        Returns:
            Aggregated global model weights
        """
        total_samples = sum(client_samples.values())
        
        # Initialize aggregated weights
        aggregated_weights = {}
        first_client_id = list(client_weights.keys())[0]
        first_weights = client_weights[first_client_id]
        
        for name in first_weights.keys():
            aggregated_weights[name] = np.zeros_like(first_weights[name])
        
        # Weighted average
        for client_id, weights in client_weights.items():
            n_i = client_samples[client_id]
            weight_factor = n_i / total_samples
            
            for name in weights.keys():
                aggregated_weights[name] += weight_factor * weights[name]
        
        return aggregated_weights
    
    def train_round(
        self,
        selected_clients: List[int] = None,
        epochs: int = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Execute one round of FedProx training.
        
        Args:
            selected_clients: List of client IDs to participate (None = all)
            epochs: Number of local epochs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with round statistics
        """
        if selected_clients is None:
            selected_clients = list(self.clients.keys())
        
        start_time = time.time()
        
        # Get current global weights
        global_weights = self.get_global_model_weights()
        
        # Distribute global model to clients
        client_weights = {}
        client_samples = {}
        total_client_time = 0.0
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Set global weights (also stores for proximal term)
            client.set_model_weights(global_weights)
            
            # Local training with FedProx objective
            _, client_time = client.train_local(epochs=epochs, verbose=verbose)
            total_client_time += client_time
            
            # Collect updated weights
            client_weights[client_id] = client.get_model_weights()
            client_samples[client_id] = client.num_samples
        
        # Aggregate
        aggregated_weights = self.perform_aggregation(client_weights, client_samples)
        self.set_global_model_weights(aggregated_weights)
        
        round_time = time.time() - start_time
        self.total_training_time += round_time
        
        # Track statistics
        stats = {
            'round_time': round_time,
            'client_time': total_client_time,
            'num_clients': len(selected_clients),
            'total_samples': sum(client_samples.values())
        }
        self.aggregation_history.append(stats)
        
        return stats
    
    def evaluate_global_model(
        self,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_data: Test data
            test_labels: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.global_model.eval()
        
        # Create test loader
        data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(test_labels)
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.global_model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_statistics(self) -> Dict:
        """Get server statistics."""
        return {
            'num_clients': len(self.clients),
            'total_training_time': self.total_training_time,
            'num_rounds': len(self.aggregation_history),
            'aggregation_history': self.aggregation_history
        }


class FedProxTrainer:
    """
    Complete FedProx training orchestrator.
    
    Manages the full training loop for FedProx experiments.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize FedProx trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('system', {}).get('device', 'cpu')
        self.dataset_name = config.get('experiment', {}).get('dataset', 'fashion_mnist')
        
        # Create server
        self.server = FedProxServer(
            dataset_name=self.dataset_name,
            config=config,
            device=self.device
        )
        
        # Training history
        self.accuracy_history = []
        self.loss_history = []
        self.time_history = []
    
    def register_clients(
        self,
        clients_data: List[Tuple[int, np.ndarray, np.ndarray]]
    ):
        """
        Register clients with the server.
        
        Args:
            clients_data: List of (client_id, train_data, train_labels) tuples
        """
        for client_id, train_data, train_labels in clients_data:
            client = FedProxClient(
                client_id=client_id,
                dataset_name=self.dataset_name,
                train_data=train_data,
                train_labels=train_labels,
                config=self.config,
                device=self.device
            )
            self.server.register_client(client)
    
    def train(
        self,
        total_rounds: int = 50,
        clients_per_round: int = None,
        local_epochs: int = None,
        test_data: np.ndarray = None,
        test_labels: np.ndarray = None,
        eval_frequency: int = 1,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Run complete FedProx training.
        
        Args:
            total_rounds: Total number of communication rounds
            clients_per_round: Number of clients per round (None = all)
            local_epochs: Number of local epochs per round
            test_data: Test data for evaluation
            test_labels: Test labels for evaluation
            eval_frequency: Evaluate every N rounds
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        if verbose:
            print(f"Starting FedProx training for {total_rounds} rounds...")
        
        start_time = time.time()
        
        for round_num in range(total_rounds):
            # Select clients
            all_clients = list(self.server.clients.keys())
            if clients_per_round is not None and clients_per_round < len(all_clients):
                selected_clients = np.random.choice(
                    all_clients,
                    size=clients_per_round,
                    replace=False
                ).tolist()
            else:
                selected_clients = all_clients
            
            # Train round
            round_stats = self.server.train_round(
                selected_clients=selected_clients,
                epochs=local_epochs,
                verbose=(verbose and round_num % 10 == 0)
            )
            
            # Evaluate
            if test_data is not None and (round_num + 1) % eval_frequency == 0:
                eval_results = self.server.evaluate_global_model(test_data, test_labels)
                self.accuracy_history.append(eval_results['accuracy'])
                self.loss_history.append(eval_results['loss'])
            else:
                self.accuracy_history.append(None)
                self.loss_history.append(None)
            
            self.time_history.append(time.time() - start_time)
            
            if verbose and (round_num + 1) % 10 == 0:
                if test_data is not None and self.accuracy_history[-1] is not None:
                    print(f"Round {round_num+1}/{total_rounds}, "
                          f"Accuracy: {self.accuracy_history[-1]:.2f}%, "
                          f"Time: {self.time_history[-1]:.1f}s")
                else:
                    print(f"Round {round_num+1}/{total_rounds}, "
                          f"Time: {self.time_history[-1]:.1f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"FedProx training completed in {total_time:.1f}s")
        
        return {
            'accuracy_history': self.accuracy_history,
            'loss_history': self.loss_history,
            'time_history': self.time_history,
            'total_time': total_time,
            'server_stats': self.server.get_statistics()
        }


def create_fedprox_client(
    client_id: int,
    dataset_name: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    config: Dict,
    device: str = 'cpu'
) -> FedProxClient:
    """
    Factory function to create FedProx client.
    
    Args:
        client_id: Client identifier
        dataset_name: Dataset name
        train_data: Training data
        train_labels: Training labels
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedProxClient instance
    """
    return FedProxClient(
        client_id=client_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )


def create_fedprox_server(
    dataset_name: str,
    config: Dict,
    device: str = 'cpu'
) -> FedProxServer:
    """
    Factory function to create FedProx server.
    
    Args:
        dataset_name: Dataset name
        config: Configuration dictionary
        device: Training device
        
    Returns:
        FedProxServer instance
    """
    return FedProxServer(
        dataset_name=dataset_name,
        config=config,
        device=device
    )
