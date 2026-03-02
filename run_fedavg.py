
import numpy as np
import torch
from config import CONFIG
from client import Client
from fedavg_server import FedAvgServer
from data_utils import get_client_loaders

def main():
    print(f"Initializing FedAvg Baseline with {CONFIG['dataset']}...")
    
    # Force FedAvg specific configs if needed (or rely on config.py)
    # Usually FedAvg samples 10% clients
    CONFIG['fedavg_fraction'] = 0.1 
    
    # 1. Load Data
    client_loaders, test_loader = get_client_loaders(
        CONFIG['dataset'], 
        CONFIG['data_root'], 
        CONFIG['n_clients'], 
        CONFIG['batch_size'], 
        CONFIG['dirichlet_alpha']
    )
    
    # 2. Initialize Clients (Same Profiles as Main Sim)
    n_clients = CONFIG['n_clients']
    ratios = CONFIG['profile_ratio']
    n_fast = int(n_clients * ratios[0])
    n_medium = int(n_clients * ratios[1])
    n_slow = n_clients - n_fast - n_medium
    
    profiles = ['fast'] * n_fast + ['medium'] * n_medium + ['slow'] * n_slow
    np.random.seed(42) # Fixed seed for reproducible profile assignment
    np.random.shuffle(profiles)
    
    clients = {}
    for i in range(n_clients):
        clients[i] = Client(i, client_loaders[i], profile_type=profiles[i], device=CONFIG['device'])
        
    # 3. Initialize FedAvg Server
    server = FedAvgServer(clients, device=CONFIG['device'])
    server.test_loader = test_loader
    
    # 4. Run
    server.run()

if __name__ == '__main__':
    main()
