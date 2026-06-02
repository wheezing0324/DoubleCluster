
import torch
import sys
import os
import copy
import matplotlib.pyplot as plt
import datetime
from config import CONFIG
from server import Server
from experiment_environment import create_environment

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

# Helper to capture metrics
class MetricsLogger:
    def __init__(self):
        self.metrics = {'loss': [], 'accuracy': []}

def run_experiment(variant_name, custom_config):
    # Setup Log for this specific experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./DoubleCluster/log/ablation"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{log_dir}/{variant_name}_{timestamp}.log"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print(f"\n{'='*20} Running Variant: {variant_name} {'='*20}")
    print(f"Logging to {log_filename}")
    
    # 1. Config Injection
    original_config_dict = copy.deepcopy(CONFIG)
    CONFIG.update(custom_config)
    
    try:
        # 2. Initialize shared environment
        environment = create_environment(CONFIG, seed=CONFIG.get('seed', 42), device=CONFIG['device'])
        clients = environment.clone_clients(device=CONFIG['device'])
            
        # 3. Initialize Server with custom behavior flags
        server = Server(clients, device=CONFIG['device'], environment=environment)
        server.test_loader = environment.test_loader
        
        # Inject Ablation Flags
        server.beta = custom_config.get('beta', 0.1)
        server.clustering_mode = custom_config.get('clustering_mode', 'soft')
        server.aggregation_strategy = custom_config.get('aggregation_strategy', 'hide_adf')
        
        # Run
        server.run()
        
        return server.metrics
        
    except Exception as e:
        print(f"Experiment {variant_name} Failed: {e}")
        return {'loss': [], 'accuracy': []}
    finally:
        # Restore Config and Stdout
        CONFIG.clear()
        CONFIG.update(original_config_dict)
        sys.stdout = logger.terminal
        logger.log.close()

def plot_ablation_results(results, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure log dir exists for plots too
    plot_dir = "./DoubleCluster/log/ablation_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    full_path = os.path.join(plot_dir, filename)
    
    for label, metrics in results.items():
        rounds = range(1, len(metrics['accuracy']) + 1)
        ax.plot(rounds, metrics['accuracy'], label=label)
        
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(full_path)
    print(f"Saved plot to {full_path}")

def main():
    # 仅运行：聚类模式消融（Soft/Hard/Random），除聚类类别外其余参数取自 config.py
    print("\n>>> Experiment: Clustering Mode Only")
    print("\n>>> Experiment 2: Clustering Mode Analysis")
    modes = ['soft', 'hard', 'random']
    exp2_results = {}
    for m in modes:
        # 仅更改聚类模式；其他超参数完全使用 config.py 中的默认值
        cfg = {'clustering_mode': m}
        name = f"Cluster_{m}"
        exp2_results[name] = run_experiment(name, cfg)
    plot_ablation_results(exp2_results, "Soft vs Hard Clustering", "ablation_exp2_cluster.png")
    
if __name__ == '__main__':
    main()
