
import numpy as np
import torch
from config import CONFIG
from client import Client
from server import Server
from models import get_model
from data_utils import get_client_loaders

import sys
import datetime
import os

# Helper to redirect stdout to file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Ensure log directory exists
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log = open(filename, "w", buffering=1) # buffering=1 means line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure file is flushed on every write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./DoubleCluster/log"
    log_file = f"{log_dir}/simulation_{CONFIG['dataset']}_alpha{CONFIG['dirichlet_alpha']}_{timestamp}.log"
    sys.stdout = Logger(log_file)
    print(f"Logging to {log_file}")

    print(f"Initializing Simulation with {CONFIG['dataset']}...")
    
    # 1. 初始化数据加载器
    client_loaders, test_loader = get_client_loaders(
        CONFIG['dataset'], 
        CONFIG['data_root'], 
        CONFIG['n_clients'], 
        CONFIG['batch_size'], 
        CONFIG['dirichlet_alpha']
    )
    
    # 2. 初始化客户端 (分配 Profile)
    n_clients = CONFIG['n_clients']
    ratios = CONFIG['profile_ratio']
    n_fast = int(n_clients * ratios[0])
    n_medium = int(n_clients * ratios[1])
    n_slow = n_clients - n_fast - n_medium
    
    profiles = ['fast'] * n_fast + ['medium'] * n_medium + ['slow'] * n_slow
    np.random.shuffle(profiles)
    
    clients = {}
    for i in range(n_clients):
        clients[i] = Client(i, client_loaders[i], profile_type=profiles[i], device=CONFIG['device'])
        
    # 3. 初始化服务器
    server = Server(clients, device=CONFIG['device'])
    server.test_loader = test_loader # 手动注入 test_loader
    
    # 4. 运行仿真
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        raise e
    finally:
        # Restore stdout
        sys.stdout = sys.stdout.terminal


if __name__ == '__main__':
    main()
