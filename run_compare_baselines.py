import numpy as np
import torch
import sys
import os
import copy
import matplotlib.pyplot as plt
import datetime
from config import CONFIG
from client import Client
from server import Server
from eafl_server import EAFLServer
from csahfl_server import CSAHFLServer
from data_utils import get_client_loaders

# Helper to redirect stdout to file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_experiment(name, ServerClass, custom_config):
    original_config = copy.deepcopy(CONFIG)
    CONFIG.update(custom_config)
    
    try:
        # Load Data
        client_loaders, test_loader = get_client_loaders(
            CONFIG['dataset'], 
            CONFIG['data_root'], 
            CONFIG['n_clients'], 
            CONFIG['batch_size'], 
            CONFIG['dirichlet_alpha']
        )
        
        # Initialize Clients
        n_clients = CONFIG['n_clients']
        ratios = CONFIG['profile_ratio']
        n_fast = int(n_clients * ratios[0])
        n_medium = int(n_clients * ratios[1])
        n_slow = n_clients - n_fast - n_medium
        
        profiles = ['fast'] * n_fast + ['medium'] * n_medium + ['slow'] * n_slow
        # Use a fixed seed for profile distribution to ensure fair comparison across algorithms
        profile_seed = CONFIG.get('seed', 42)
        np.random.seed(profile_seed)
        np.random.shuffle(profiles)
        
        clients = {}
        for i in range(n_clients):
            clients[i] = Client(i, client_loaders[i], profile_type=profiles[i], device=CONFIG['device'])
            
        # Initialize Server
        server = ServerClass(clients, device=CONFIG['device'])
        server.test_loader = test_loader
        
        # Inject configs to server if it's our method
        if isinstance(server, Server) and not isinstance(server, (EAFLServer, CSAHFLServer)):
            server.beta = custom_config.get('beta', 0.1)
            server.clustering_mode = custom_config.get('clustering_mode', 'soft')
            server.aggregation_strategy = custom_config.get('aggregation_strategy', 'hide_adf')
        
        # Run
        server.run()
        
        # Return accuracy history
        if hasattr(server, 'metrics'):
            return server.metrics['accuracy']
        else:
            return []
            
    except Exception as e:
        print(f"Experiment {name} Failed: {e}")
        return []
    finally:
        CONFIG.clear()
        CONFIG.update(original_config)

def plot_results(results, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_dir = "./DoubleCluster/log/baseline_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    full_path = os.path.join(plot_dir, filename)
    
    for label, accuracies in results.items():
        rounds = range(1, len(accuracies) + 1)
        ax.plot(rounds, accuracies, label=label)
        
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(full_path)
    print(f"Saved plot to {full_path}")

def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./DoubleCluster/log/baselines"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{log_dir}/compare_baselines_{timestamp}.log"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(">>> Running Baseline Comparisons across Multiple Datasets")
    
    # Pull default parameters from config.py to respect user settings
    global_rounds = CONFIG.get('global_rounds', 200)
    device = 'cpu' # Force CPU as requested
    
    # We will test all three datasets, overriding the config's single dataset choice
    datasets_to_run = [
        {'dataset': 'cifar100', 'input_channels': 3, 'num_classes': 100},
        {'dataset': 'mnist', 'input_channels': 1, 'num_classes': 10},
        {'dataset': 'cifar10', 'input_channels': 3, 'num_classes': 10}
       
    ]

    for ds_config in datasets_to_run:
        ds_name = ds_config['dataset']
        print(f"\n{'='*40}")
        print(f"Starting Experiments for Dataset: {ds_name.upper()}")
        print(f"{'='*40}")
        
        base_config = {
            'global_rounds': global_rounds,
            'dataset': ds_name,
            'input_channels': ds_config['input_channels'],
            'num_classes': ds_config['num_classes'],
            'device': device
        }
        
        results = {}
        
        # 1. Run EAFL
        print(f"\n--- Running EAFL on {ds_name.upper()} ---")
        eafl_config = copy.deepcopy(base_config)
        results['EAFL'] = run_experiment(f"EAFL_{ds_name}", EAFLServer, eafl_config)
        
        # 2. Run CSAHFL
        print(f"\n--- Running CSAHFL on {ds_name.upper()} ---")
        csahfl_config = copy.deepcopy(base_config)
        results['CSAHFL'] = run_experiment(f"CSAHFL_{ds_name}", CSAHFLServer, csahfl_config)
        
        # 3. Run PoiFL (Ours)
        print(f"\n--- Running PoiFL (Ours) on {ds_name.upper()} ---")
        poifl_config = copy.deepcopy(base_config)
        poifl_config['clustering_mode'] = 'soft'
        poifl_config['aggregation_strategy'] = 'hide_adf'
        results['PoiFL (Ours)'] = run_experiment(f"PoiFL_{ds_name}", Server, poifl_config)
        
        # Plotting for this dataset
        plot_title = f"PoiFL vs EAFL vs CSAHFL - {ds_name.upper()}"
        plot_filename = f"poifl_vs_baselines_{ds_name}_{timestamp}.png"
        plot_results(results, plot_title, plot_filename)
        
    print("\n>>> All experiments finished successfully!")
    sys.stdout = logger.terminal
    logger.log.close()

if __name__ == '__main__':
    main()
