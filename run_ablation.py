
import numpy as np
import torch
from config import CONFIG
from client import Client
from server import Server
from data_utils import get_client_loaders
import copy

def run_experiment(variant_name, custom_config):
    print(f"\n{'='*20} Running Variant: {variant_name} {'='*20}")
    
    # Merge custom config
    EXP_CONFIG = copy.deepcopy(CONFIG)
    EXP_CONFIG.update(custom_config)
    
    print(f"Config Override: {custom_config}")
    
    # 1. Load Data
    client_loaders, test_loader = get_client_loaders(
        EXP_CONFIG['dataset'], 
        EXP_CONFIG['data_root'], 
        EXP_CONFIG['n_clients'], 
        EXP_CONFIG['batch_size'], 
        EXP_CONFIG['dirichlet_alpha']
    )
    
    # 2. Initialize Clients
    n_clients = EXP_CONFIG['n_clients']
    ratios = EXP_CONFIG['profile_ratio']
    n_fast = int(n_clients * ratios[0])
    n_medium = int(n_clients * ratios[1])
    n_slow = n_clients - n_fast - n_medium
    
    profiles = ['fast'] * n_fast + ['medium'] * n_medium + ['slow'] * n_slow
    np.random.seed(42)
    np.random.shuffle(profiles)
    
    clients = {}
    for i in range(n_clients):
        clients[i] = Client(i, client_loaders[i], profile_type=profiles[i], device=EXP_CONFIG['device'])
        
    # 3. Initialize Server
    # We need to monkey-patch or subclass Server to support specific variants if Config isn't enough
    # Fortunately, our Server class is data-driven by CONFIG.
    # But we need to make sure Server reads from EXP_CONFIG, not the global CONFIG module.
    
    # Hack: Temporarily overwrite the global CONFIG in server module?
    # Better: Pass config to Server init. But Server imports CONFIG directly.
    # Let's use a trick: Modify the config module's dict in place, then restore it.
    
    original_config_dict = copy.deepcopy(CONFIG)
    CONFIG.update(custom_config)
    
    try:
        server = Server(clients, device=EXP_CONFIG['device'])
        server.test_loader = test_loader
        server.run()
    finally:
        # Restore Config
        CONFIG.clear()
        CONFIG.update(original_config_dict)

def main():
    # 1. Variant: DC w/o Tiering (Cluster Only)
    # Logic: Keep Clusters (n_topics=3), but disable Tiering (Force everyone to T1)
    # We can do this by setting T1 threshold to Infinity
    config_cluster_only = {
        'n_topics': 3,
        'tier_thresholds': {
            'T1': 999999, # Everyone is Fast/Sync
            'T2': 999999
        },
        'global_rounds': 300 # Set to full rounds for real experiment
    }
    
    # 2. Variant: DC w/o Clustering (Tier Only)
    # Logic: Disable Clusters (n_topics=1), but keep Tiering thresholds
    config_tier_only = {
        'n_topics': 1,
        'tier_thresholds': {
            'T1': 200,   
            'T2': 1000   
        },
        'global_rounds': 300 # Set to full rounds for real experiment
    }
    
    # Run
    # run_experiment("DC_Cluster_Only", config_cluster_only)
    run_experiment("DC_Tier_Only", config_tier_only)

if __name__ == '__main__':
    main()
