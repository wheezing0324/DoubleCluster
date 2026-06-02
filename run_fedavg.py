
from config import CONFIG
from fedavg_server import FedAvgServer
from experiment_environment import create_environment

def main():
    print(f"Initializing FedAvg Baseline with {CONFIG['dataset']}...")
    
    # Force FedAvg specific configs if needed (or rely on config.py)
    # Usually FedAvg samples 10% clients
    CONFIG['fedavg_fraction'] = 0.1 
    
    # 1. Initialize shared environment
    environment = create_environment(CONFIG, seed=CONFIG.get('seed', 42), device=CONFIG['device'])
    clients = environment.clone_clients(device=CONFIG['device'])
        
    # 2. Initialize FedAvg Server
    server = FedAvgServer(clients, device=CONFIG['device'], environment=environment)
    server.test_loader = environment.test_loader
    
    # 4. Run
    server.run()

if __name__ == '__main__':
    main()
