"""
Client module for CSAHFL framework.

Implements the base client class that performs local training with adaptive
workload adjustment based on semi-asynchronous aggregation requirements.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import time

from src.models.cnn_fashion_mnist import CNNFashionMNIST, create_model as create_fashion_mnist_model
from src.models.cnn_cifar10 import CNNCIFAR10, create_model as create_cifar10_model
from src.models.cnn_svhn import CNNSVHN, create_model as create_svhn_model


class Client:
    """
    Base client class for federated learning.
    
    Handles local training with configurable epochs, supports workload adjustment
    for semi-asynchronous aggregation, and manages model weight exchange.
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
        Initialize a client with local data and configuration.
        
        Args:
            client_id: Unique client identifier
            edge_server_id: ID of associated edge server
            dataset_name: Name of dataset ('fashion_mnist', 'cifar10', 'svhn')
            train_data: Local training data (numpy array)
            train_labels: Local training labels (numpy array)
            config: Configuration dictionary with training parameters
            device: Training device ('cuda' or 'cpu')
        """
        self.client_id = client_id
        self.edge_server_id = edge_server_id
        self.dataset_name = dataset_name
        self.config = config
        self.device = device
        
        # Local data
        self.train_data = train_data
        self.train_labels = train_labels
        self.num_samples = len(train_labels)
        
        # Training parameters
        self.batch_size = config.get('training', {}).get('batch_size', 10)
        self.learning_rate = config.get('training', {}).get('learning_rate', 0.01)
        self.max_epochs = config.get('intra_cluster', {}).get('max_epochs', 10)
        
        # Current epoch assignment (for workload adjustment)
        self.current_epochs = self.max_epochs
        
        # Staleness tracking (for semi-asynchronous aggregation)
        self.staleness = 0  # Number of delayed rounds
        self.last_active_round = 0
        
        # Timing information
        self.t_unit = 0.0  # Time per batch-epoch (will be estimated)
        self.t_comp = 0.0  # Computation time
        self.t_com = 0.0   # Communication time
        
        # Create model
        self.model = self._create_model()
        self.model.to(device)
        
        # Create data loader
        self._create_data_loader()
        
        # Estimate t_unit (time per batch-epoch)
        self._estimate_t_unit()
    
    def _create_model(self) -> torch.nn.Module:
        """Create appropriate model based on dataset."""
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
        # Convert numpy arrays to tensors
        # Handle different data formats
        if len(self.train_data.shape) == 3:
            # Fashion-MNIST: (N, 28, 28) -> add channel dimension
            data_tensor = torch.FloatTensor(self.train_data).unsqueeze(1)
        else:
            # CIFAR10/SVHN: (N, H, W, C) -> (N, C, H, W)
            data_tensor = torch.FloatTensor(self.train_data).permute(0, 3, 1, 2)
        
        labels_tensor = torch.LongTensor(self.train_labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        self.data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        # Calculate number of batches
        self.num_batches = len(self.data_loader)
    
    def _estimate_t_unit(self):
        """
        Estimate time per batch-epoch (t_unit) for workload adjustment.
        
        Runs a single batch through the model to measure computation time.
        """
        self.model.train()
        
        # Get one batch
        for batch_data, batch_labels in self.data_loader:
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Measure computation time
            start_time = time.time()
            
            # Forward pass
            outputs = self.model(batch_data)
            loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Optimizer step
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            optimizer.step()
            
            end_time = time.time()
            
            self.t_unit = (end_time - start_time) / self.batch_size
            break
        
        # Ensure minimum t_unit to avoid division issues
        self.t_unit = max(self.t_unit, 1e-6)
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current model weights.
        
        Returns:
            Dictionary of model state dict
        """
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set model weights from global/edge model.
        
        Args:
            weights: Dictionary of model state dict
        """
        self.model.load_state_dict({
            k: v.to(self.device) for k, v in weights.items()
        })
    
    def get_model_size(self) -> int:
        """
        Get model size in bytes for communication time estimation.
        
        Returns:
            Model size in bytes
        """
        if self.dataset_name == 'fashion_mnist':
            model = create_fashion_mnist_model()
        elif self.dataset_name == 'cifar10':
            model = create_cifar10_model()
        else:
            model = create_svhn_model()
        
        return model.get_model_size()
    
    def adjust_workload(self, T_theta: float, t_com: float):
        """
        Adjust local training epochs based on aggregation interval (Eq. 9).
        
        E_i = max(min((T_θ - t_com) / (N_BS × t_unit), E_max), 1)
        
        Args:
            T_theta: Aggregation interval time
            t_com: Communication time
        """
        N_BS = self.num_batches  # Number of batches
        
        # Calculate adjusted epochs
        if N_BS * self.t_unit > 0:
            adjusted_epochs = (T_theta - t_com) / (N_BS * self.t_unit)
            self.current_epochs = int(max(min(adjusted_epochs, self.max_epochs), 1))
        else:
            self.current_epochs = self.max_epochs
        
        # Ensure at least 1 epoch
        self.current_epochs = max(1, self.current_epochs)
    
    def train_local(
        self,
        global_round: int,
        edge_round: int,
        epochs: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """
        Perform local training for specified epochs.
        
        Args:
            global_round: Current global round number
            edge_round: Current edge round number
            epochs: Number of epochs to train (uses current_epochs if None)
        
        Returns:
            Tuple of (model_weights, training_loss, training_time)
        """
        if epochs is None:
            epochs = self.current_epochs
        
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        start_time = time.time()
        total_loss = 0.0
        total_samples = 0
        
        # Train for specified epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in self.data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_data.size(0)
                total_samples += batch_data.size(0)
            
            total_loss += epoch_loss
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Update timing information
        self.t_comp = training_time
        self.staleness = max(0, global_round - self.last_active_round)
        self.last_active_round = global_round
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return self.get_model_weights(), avg_loss, training_time
    
    def evaluate_local(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on local test data.
        
        Args:
            test_data: Test data
            test_labels: Test labels
        
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        
        # Convert to tensors
        if len(test_data.shape) == 3:
            data_tensor = torch.FloatTensor(test_data).unsqueeze(1)
        else:
            data_tensor = torch.FloatTensor(test_data).permute(0, 3, 1, 2)
        
        labels_tensor = torch.LongTensor(test_labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item() * batch_data.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def get_client_info(self) -> Dict:
        """
        Get client information for logging and monitoring.
        
        Returns:
            Dictionary with client statistics
        """
        return {
            'client_id': self.client_id,
            'edge_server_id': self.edge_server_id,
            'num_samples': self.num_samples,
            'num_batches': self.num_batches,
            'max_epochs': self.max_epochs,
            'current_epochs': self.current_epochs,
            't_unit': self.t_unit,
            'staleness': self.staleness,
            'label_distribution': self._get_label_distribution()
        }
    
    def _get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of labels in local data."""
        unique, counts = np.unique(self.train_labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}
    
    def __repr__(self):
        return f"Client(id={self.client_id}, edge={self.edge_server_id}, samples={self.num_samples})"


def create_client(
    client_id: int,
    edge_server_id: int,
    dataset_name: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    config: Dict,
    device: str = 'cpu'
) -> Client:
    """
    Factory function to create a Client instance.
    
    Args:
        client_id: Unique client identifier
        edge_server_id: ID of associated edge server
        dataset_name: Name of dataset
        train_data: Local training data
        train_labels: Local training labels
        config: Configuration dictionary
        device: Training device
    
    Returns:
        Initialized Client instance
    """
    return Client(
        client_id=client_id,
        edge_server_id=edge_server_id,
        dataset_name=dataset_name,
        train_data=train_data,
        train_labels=train_labels,
        config=config,
        device=device
    )
