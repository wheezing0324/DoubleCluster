
from config import CONFIG
from server import Server
from experiment_environment import create_environment

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

import matplotlib.pyplot as plt

def plot_metrics(metrics, filename):
    rounds = range(1, len(metrics['loss']) + 1)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(rounds, metrics['loss'], color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(rounds, metrics['accuracy'], color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Training Metrics')
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Metrics plot saved to {filename}")

def main():
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f"{log_dir}/simulation_{CONFIG['dataset']}_alpha{CONFIG['dirichlet_alpha']}_{timestamp}.log"
    sys.stdout = Logger(log_file)
    print(f"Logging to {log_file}")

    print(f"Initializing Simulation with {CONFIG['dataset']}...")
    
    # 1. 初始化公共实验环境：数据划分、profile 分配、采样与 latency trace
    environment = create_environment(CONFIG, seed=CONFIG.get('seed', 42), device=CONFIG['device'])
    clients = environment.clone_clients(device=CONFIG['device'])
        
    # 2. 初始化服务器
    server = Server(clients, device=CONFIG['device'], environment=environment)
    server.test_loader = environment.test_loader # 手动注入 test_loader
    
    # 4. 运行仿真
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        raise e
    finally:
        # Plot metrics
        plot_file = f"{log_dir}/simulation_{CONFIG['dataset']}_alpha{CONFIG['dirichlet_alpha']}_{timestamp}.png"
        plot_metrics(server.metrics, plot_file)
        
        # Restore stdout
        sys.stdout = sys.stdout.terminal


if __name__ == '__main__':
    main()
