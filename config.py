
import os

# Create data directory if not exists
if not os.path.exists('./data'):
    os.makedirs('./data')

# Force CPU for now to avoid CUDA errors
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

CONFIG = {
    'n_clients': 100,
    'n_topics': 3,          # Topic A, B, C
    'global_rounds': 300,   # T
    
    # 1. Data Heterogeneity
    'dataset': 'cifar10', # 'mnist' or 'cifar10'
    'input_channels': 3,  # 1 for MNIST, 3 for CIFAR-10
    'data_root': './data',
    'batch_size': 32,
    'dirichlet_alpha': 0.1, # < 0.5 for Non-IID
    
    # 2. System Heterogeneity
    # Latency Profiles (ms)
    'profiles': {
        'fast': {'mean': 100, 'std': 10},
        'medium': {'mean': 500, 'std': 100},
        'slow': {'mean': 2000, 'std': 500}
    },
    # Population distribution
    'profile_ratio': [0.6, 0.3, 0.1], # 60% Fast, 30% Med, 10% Slow
    'network_volatility': 0.2, # Multiplier noise
    
    # Tier Thresholds (ms)
    'tier_thresholds': {
        'T1': 200,   # < 200ms -> Fast
        'T2': 1000   # < 1000ms -> Medium (and >= 200ms)
        # > 1000ms -> Slow
    },
    
    # Training Config
    'local_epochs':5,
    'model_name': 'resnet18',
    'num_classes': 10,
    'lr': 0.01,
    'lr_decay_step': 200,      # Decay LR after this many rounds
    'lr_decay_gamma': 0.1,     # Decay factor (lr = lr * gamma)
    'momentum': 0.9,
    'device': 'cuda', # Forced CPU for debugging
    
    # Aggregation Config
    'buffer_size': 5,          # Tier 2 Buffer
    'async_alpha': 0.6,        # Tier 3 Learning Rate Correction
    'tier2_alpha': 0.8,        # Tier 2 Soft Update Rate
    
    # Clustering Config
    'clustering_interval': 50, # Dual-Frequency Profiling (Low Freq)
    'warmup_rounds': 20,       # Warm-up phase rounds (Global Training only)
    
    # Sampling Config
    'sample_fraction': 0.2,    # Sample 10% clients per round (Fair Comparison)
}
