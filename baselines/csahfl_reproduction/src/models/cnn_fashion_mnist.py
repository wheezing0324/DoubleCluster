"""
CNN model for Fashion-MNIST dataset.

Implements a standard 2-convolutional-layer CNN architecture as specified in the CSAHFL paper.
Architecture:
- Input: 28x28 grayscale images
- Conv2D(32, 3x3) + ReLU + MaxPool(2x2)
- Conv2D(64, 3x3) + ReLU + MaxPool(2x2)
- Flatten + Dense(128) + ReLU + Dropout(0.5)
- Dense(10) + Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CNNFashionMNIST(nn.Module):
    """
    CNN architecture for Fashion-MNIST classification.
    
    Attributes:
        conv1: First convolutional layer (1 -> 32 channels)
        conv2: Second convolutional layer (32 -> 64 channels)
        fc1: First fully connected layer (1600 -> 128)
        fc2: Second fully connected layer (128 -> 10)
        dropout: Dropout layer with rate 0.5
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Initialize the CNN model for Fashion-MNIST.
        
        Args:
            num_classes: Number of output classes (default: 10 for Fashion-MNIST)
            dropout_rate: Dropout probability (default: 0.5)
        """
        super(CNNFashionMNIST, self).__init__()
        
        # First convolutional block
        # Input: (N, 1, 28, 28)
        # Output: (N, 32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        # Input: (N, 32, 14, 14)
        # Output: (N, 64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After pooling: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (N, 1, 28, 28)
            
        Returns:
            Output logits of shape (N, num_classes)
        """
        # First conv block: Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block: Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_size(self) -> int:
        """
        Get the serialized model size in bytes.
        
        Returns:
            Model size in bytes
        """
        import io
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell()
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes: int = 10, dropout_rate: float = 0.5) -> CNNFashionMNIST:
    """
    Factory function to create a CNNFashionMNIST model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        
    Returns:
        Initialized CNNFashionMNIST model
    """
    return CNNFashionMNIST(num_classes=num_classes, dropout_rate=dropout_rate)


def get_optimizer(model: nn.Module, learning_rate: float = 0.01) -> torch.optim.Optimizer:
    """
    Create SGD optimizer for the model.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for SGD
        
    Returns:
        Configured SGD optimizer
    """
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def get_criterion() -> nn.Module:
    """
    Get the loss function (cross-entropy).
    
    Returns:
        CrossEntropyLoss criterion
    """
    return nn.CrossEntropyLoss()


def train_epoch(
    model: CNNFashionMNIST,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 1
) -> float:
    """
    Train the model for specified epochs.
    
    Args:
        model: The CNN model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        
    Returns:
        Average training loss over all epochs
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
    
    return total_loss / total_samples


def evaluate(
    model: CNNFashionMNIST,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Args:
        model: The CNN model to evaluate
        data_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Dictionary containing 'loss' and 'accuracy'
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
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': 100.0 * correct / total_samples
    }
