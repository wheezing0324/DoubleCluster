"""
CNN model for CIFAR10 dataset.

Implements standard 2-convolutional-layer architecture for CIFAR10 classification.
Architecture:
- Input: 32x32x3 RGB images
- Conv2D(32, 3x3) + ReLU + MaxPool(2x2)
- Conv2D(64, 3x3) + ReLU + MaxPool(2x2)
- Flatten + Dense(128) + ReLU + Dropout(0.5)
- Dense(10) + Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import io


class CNNCIFAR10(nn.Module):
    """
    CNN architecture for CIFAR10 classification.
    
    Standard 2-convolutional-layer network suitable for federated learning
    experiments with moderate computational requirements.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Initialize CNN for CIFAR10.
        
        Args:
            num_classes: Number of output classes (default: 10 for CIFAR10)
            dropout_rate: Dropout rate for regularization (default: 0.5)
        """
        super(CNNCIFAR10, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 2 pooling layers: 32x32 -> 16x16 -> 8x8
        # Feature map size: 64 channels x 8 x 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block: Conv + ReLU + Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block: Conv + ReLU + Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_size(self) -> int:
        """
        Calculate model size in bytes for communication time estimation.
        
        Returns:
            Model size in bytes (serialized)
        """
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell()
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes: int = 10, dropout_rate: float = 0.5) -> CNNCIFAR10:
    """
    Factory function to create CNNCIFAR10 model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Initialized CNNCIFAR10 model instance
    """
    return CNNCIFAR10(num_classes=num_classes, dropout_rate=dropout_rate)


def get_optimizer(model: nn.Module, learning_rate: float = 0.01) -> optim.Optimizer:
    """
    Create SGD optimizer for the model.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for SGD
        
    Returns:
        Configured SGD optimizer with momentum
    """
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def get_criterion() -> nn.Module:
    """
    Get loss function for classification.
    
    Returns:
        CrossEntropyLoss criterion
    """
    return nn.CrossEntropyLoss()


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 1
) -> float:
    """
    Train model for specified number of epochs.
    
    Args:
        model: The neural network model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for gradient updates
        criterion: Loss function
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
        
    Returns:
        Average training loss across all epochs
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test/validation data.
    
    Args:
        model: The neural network model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary with 'loss' and 'accuracy' metrics
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total_samples += target.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }
