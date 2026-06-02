"""
Models package initialization for CSAHFL framework.

This module provides CNN architectures for Fashion-MNIST, CIFAR10, and SVHN datasets
with standardized training and evaluation utilities.
"""

from .cnn_fashion_mnist import (
    CNNFashionMNIST,
    create_model as create_fashion_mnist_model,
    get_optimizer as get_fashion_mnist_optimizer,
    get_criterion as get_fashion_mnist_criterion,
    train_epoch as train_fashion_mnist_epoch,
    evaluate as evaluate_fashion_mnist,
)

from .cnn_cifar10 import (
    CNNCIFAR10,
    create_model as create_cifar10_model,
    get_optimizer as get_cifar10_optimizer,
    get_criterion as get_cifar10_criterion,
    train_epoch as train_cifar10_epoch,
    evaluate as evaluate_cifar10,
)

from .cnn_svhn import (
    CNNSVHN,
    create_model as create_svhn_model,
    get_optimizer as get_svhn_optimizer,
    get_criterion as get_svhn_criterion,
    train_epoch as train_svhn_epoch,
    evaluate as evaluate_svhn,
)

# Model registry for dataset-specific model creation
MODEL_REGISTRY = {
    'fashion_mnist': {
        'model_class': CNNFashionMNIST,
        'create_model': create_fashion_mnist_model,
        'get_optimizer': get_fashion_mnist_optimizer,
        'get_criterion': get_fashion_mnist_criterion,
        'train_epoch': train_fashion_mnist_epoch,
        'evaluate': evaluate_fashion_mnist,
        'input_channels': 1,
        'input_size': 28,
    },
    'cifar10': {
        'model_class': CNNCIFAR10,
        'create_model': create_cifar10_model,
        'get_optimizer': get_cifar10_optimizer,
        'get_criterion': get_cifar10_criterion,
        'train_epoch': train_cifar10_epoch,
        'evaluate': evaluate_cifar10,
        'input_channels': 3,
        'input_size': 32,
    },
    'svhn': {
        'model_class': CNNSVHN,
        'create_model': create_svhn_model,
        'get_optimizer': get_svhn_optimizer,
        'get_criterion': get_svhn_criterion,
        'train_epoch': train_svhn_epoch,
        'evaluate': evaluate_svhn,
        'input_channels': 3,
        'input_size': 32,
    },
}


def get_model_info(dataset_name: str) -> dict:
    """
    Get model information for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('fashion_mnist', 'cifar10', 'svhn')
    
    Returns:
        Dictionary containing model class, input channels, and input size
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[dataset_name]


def create_model_for_dataset(dataset_name: str, num_classes: int = 10, dropout_rate: float = 0.5):
    """
    Create a model instance for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('fashion_mnist', 'cifar10', 'svhn')
        num_classes: Number of output classes (default: 10)
        dropout_rate: Dropout rate for regularization (default: 0.5)
    
    Returns:
        Initialized model instance for the specified dataset
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[dataset_name]['create_model'](num_classes=num_classes, dropout_rate=dropout_rate)


def get_optimizer_for_dataset(model, dataset_name: str, learning_rate: float = 0.01):
    """
    Get optimizer for the specified dataset's model.
    
    Args:
        model: Model instance to optimize
        dataset_name: Name of the dataset ('fashion_mnist', 'cifar10', 'svhn')
        learning_rate: Learning rate for optimizer (default: 0.01)
    
    Returns:
        Configured optimizer instance
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[dataset_name]['get_optimizer'](model, learning_rate=learning_rate)


def get_criterion_for_dataset(dataset_name: str):
    """
    Get loss criterion for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('fashion_mnist', 'cifar10', 'svhn')
    
    Returns:
        Loss criterion instance (CrossEntropyLoss)
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[dataset_name]['get_criterion']()


__all__ = [
    # Model classes
    'CNNFashionMNIST',
    'CNNCIFAR10',
    'CNNSVHN',
    
    # Factory functions for each dataset
    'create_fashion_mnist_model',
    'create_cifar10_model',
    'create_svhn_model',
    
    # Optimizer and criterion functions
    'get_fashion_mnist_optimizer',
    'get_cifar10_optimizer',
    'get_svhn_optimizer',
    'get_fashion_mnist_criterion',
    'get_cifar10_criterion',
    'get_svhn_criterion',
    
    # Training and evaluation functions
    'train_fashion_mnist_epoch',
    'train_cifar10_epoch',
    'train_svhn_epoch',
    'evaluate_fashion_mnist',
    'evaluate_cifar10',
    'evaluate_svhn',
    
    # Utility functions
    'MODEL_REGISTRY',
    'get_model_info',
    'create_model_for_dataset',
    'get_optimizer_for_dataset',
    'get_criterion_for_dataset',
]
