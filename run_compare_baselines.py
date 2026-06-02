import torch
import sys
import os
import copy
import matplotlib.pyplot as plt
import datetime
from config import CONFIG
from server import Server
from eafl_server import EAFLServer
from csahfl_server import CSAHFLServer
from experiment_environment import create_environment

# Helper to redirect stdout to file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_experiment(name, ServerClass, custom_config, environment=None):
    original_config = copy.deepcopy(CONFIG)
    CONFIG.update(custom_config)
    server = None
    
    try:
        if environment is None:
            environment = create_environment(CONFIG, seed=CONFIG.get('seed', 42), device=CONFIG['device'])

        clients = environment.clone_clients(device=CONFIG['device'])
            
        # Initialize Server
        server = ServerClass(clients, device=CONFIG['device'], environment=environment)
        server.test_loader = environment.test_loader
        server.configure_tracking(name, checkpoint_interval=CONFIG.get('checkpoint_interval', 10))
        
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
    except KeyboardInterrupt:
        print(f"Experiment {name} interrupted. Saving latest checkpoint before exit...")
        if server is not None:
            server.save_checkpoint(keep_round=False)
        raise
            
    except Exception as e:
        print(f"Experiment {name} Failed: {e}")
        return []
    finally:
        CONFIG.clear()
        CONFIG.update(original_config)

def plot_results(results, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "baseline_plots")
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
    device = CONFIG.get('device', 'cpu')
    print(f"Using device: {device}")
    
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
        env_config = copy.deepcopy(CONFIG)
        env_config.update(base_config)
        environment = create_environment(env_config, seed=env_config.get('seed', 42), device=device)
        
        results = {}
        
        # 1. Run EAFL
        print(f"\n--- Running EAFL on {ds_name.upper()} ---")
        eafl_config = copy.deepcopy(base_config)
        results['EAFL'] = run_experiment(f"EAFL_{ds_name}", EAFLServer, eafl_config, environment=environment)
        
        # 2. Run CSAHFL
        print(f"\n--- Running CSAHFL on {ds_name.upper()} ---")
        csahfl_config = copy.deepcopy(base_config)
        results['CSAHFL'] = run_experiment(f"CSAHFL_{ds_name}", CSAHFLServer, csahfl_config, environment=environment)
        
        # 3. Run Ours
        print(f"\n--- Running Ours on {ds_name.upper()} ---")
        ours_config = copy.deepcopy(base_config)
        ours_config['clustering_mode'] = 'soft'
        ours_config['aggregation_strategy'] = 'hide_adf'
        results['Ours'] = run_experiment(f"Ours_{ds_name}", Server, ours_config, environment=environment)
        
        # Plotting for this dataset
        plot_title = f"Ours vs EAFL vs CSAHFL - {ds_name.upper()}"
        plot_filename = f"ours_vs_baselines_{ds_name}_{timestamp}.png"
        plot_results(results, plot_title, plot_filename)
        
    print("\n>>> All experiments finished successfully!")
    sys.stdout = logger.terminal
    logger.log.close()

if __name__ == '__main__':
    main()
