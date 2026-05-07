import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataset(name, root):
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        n_classes = 10
    elif name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        n_classes = 10
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train_dataset, test_dataset, n_classes

def dirichlet_partition(dataset, n_clients, alpha, seed=42):
    np.random.seed(seed)
    
    # Get labels
    if isinstance(dataset, datasets.CIFAR10):
        labels = np.array(dataset.targets)
        n_classes = 10
    elif isinstance(dataset, datasets.CIFAR100):
        labels = np.array(dataset.targets)
        n_classes = 100
    elif isinstance(dataset, datasets.MNIST):
        labels = dataset.targets.numpy()
        n_classes = 10
    else:
        labels = np.array(dataset.targets)
        n_classes = len(np.unique(labels))

    # Dictionary to store indices per class
    class_indices = {k: np.where(labels == k)[0] for k in range(n_classes)}
    
    # Client dict to store indices
    client_indices = {i: [] for i in range(n_clients)}
    
    # Distribute per class
    client_data_map = {i: [] for i in range(n_clients)}
    
    for k in range(n_classes):
        indices = class_indices[k]
        np.random.shuffle(indices)
        
        # Dirichlet distribution for class k
        # Sample proportions from Dirichlet distribution
        # alpha is concentration parameter.
        # size=n_clients means we get a vector of length n_clients that sums to 1.
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        
        # Calculate number of samples of class k for each client
        # We use floor to ensure integer counts, and handle remainder
        counts = (proportions * len(indices)).astype(int)
        
        # Fix rounding errors (some samples might be left over)
        diff = len(indices) - counts.sum()
        if diff > 0:
            # Add remaining samples to random clients
            for i in np.random.choice(n_clients, diff, replace=False):
                counts[i] += 1
                
        # Assign indices to clients
        current_idx = 0
        for client_id in range(n_clients):
            n_samples = counts[client_id]
            if n_samples > 0:
                samples = indices[current_idx : current_idx + n_samples]
                client_data_map[client_id].extend(samples)
                current_idx += n_samples

    # Convert to list of indices
    final_client_indices = [client_data_map[i] for i in range(n_clients)]
                
    return final_client_indices

def get_client_loaders(dataset_name, root, n_clients, batch_size, alpha):
    print(f"Loading {dataset_name} and partitioning...")
    train_dataset, test_dataset, _ = get_dataset(dataset_name, root)
    
    client_indices = dirichlet_partition(train_dataset, n_clients, alpha)
    
    loaders = []
    for i in range(n_clients):
        indices = client_indices[i]
        if len(indices) == 0:
            print(f"Warning: Client {i} has no data")
            # Create dummy loader or handle gracefully?
            # Ideally Dirichlet with alpha=0.5 gives data to most
            # But let's handle empty
            loaders.append(None)
            continue
            
        subset = Subset(train_dataset, indices)
        # drop_last=True to avoid BatchNorm error on batch_size=1 (or single sample batches)
        # If len < batch_size, this results in 0 batches, which is handled by training loop.
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        loaders.append(loader)
        
    # Global test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data loaded.")
    
    return loaders, test_loader
