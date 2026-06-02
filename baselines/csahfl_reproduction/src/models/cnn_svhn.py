"""
CNN Model for SVHN Dataset

Implements a standard 2-convolutional-layer CNN architecture for SVHN (Street View House Numbers)
classification. This model is used in the CSAHFL framework for federated learning experiments.

Architecture (standard CNN for SVHN):
- Input: 32x32x3 (RGB image)
- Conv2D(32, 3x3) + ReLU + MaxPool(2x2)
- Conv2D(64, 3x3) + ReLU + MaxPool(2x2)
- Flatten + Dense(128) + ReLU + Dropout(0.5)
- Dense(10) + Softmax

Reference: CSAHFL Paper Section 4 (Experimental Setup)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from io import BytesIO


class CNNSVHN(nn.Module):
    """
    CNN architecture for SVHN classification.
    
    Standard 2-convolutional-layer network suitable for federated learning
    experiments with heterogeneous clients.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Initialize the CNN model for SVHN.
        
        Args:
            num_classes: Number of output classes (default: 10 for SVHN)
            dropout_rate: Dropout rate for regularization (default: 0.5)
        """
        super(CNNSVHN, self).__init__()
        
        # First convolutional block
        # Input: 32x32x3, Output: 32x32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # After MaxPool: 16x16x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        # Input: 16x16x32, Output: 16x16x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # After MaxPool: 8x8x64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Input: 64 * 8 * 8 = 4096
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
        # First conv block: Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block: Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten: (batch_size, 64, 8, 8) -> (batch_size, 4096)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_size(self) -> int:
        """
        Calculate the serialized model size in bytes.
        
        This is used for communication time estimation (Eq. 7):
        t_com = M / B_w, where M is model file size in bytes
        
        Returns:
            Model size in bytes
        """
        buffer = BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.tell()
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes: int = 10, dropout_rate: float = 0.5) -> CNNSVHN:
    """
    Factory function to create a CNNSVHN model.
    
    Args:
        num_classes: Number of output classes (default: 10)
        dropout_rate: Dropout rate for regularization (default: 0.5)
        
    Returns:
        Initialized CNNSVHN model instance
    """
    return CNNSVHN(num_classes=num_classes, dropout_rate=dropout_rate)


def get_optimizer(model: nn.Module, learning_rate: float = 0.01) -> optim.Optimizer:
    """
    Create SGD optimizer with momentum for the model.
    
    Args:
        model: The neural network model
        learning_rate: Learning rate for SGD (default: 0.01)
        
    Returns:
        Configured SGD optimizer with momentum=0.9
    """
    return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


def get_criterion() -> nn.Module:
    """
    Get the loss function for classification.
    
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
    Train the model for specified number of epochs.
    
    Args:
        model: The neural network model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer for gradient updates
        criterion: Loss function
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs (default: 1)
        
    Returns:
        Average training loss across all epochs
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            batch_size = data.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
        
        total_loss += epoch_loss
        total_samples += epoch_samples
    
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model on test/validation data.
    
    Args:
        model: The neural network model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        Dictionary containing:
            - 'loss': Average loss on evaluation data
            - 'accuracy': Classification accuracy (0-1)
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Get predicted class
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return {'loss': avg_loss, 'accuracy': accuracy}
