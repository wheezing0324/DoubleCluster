
import copy
import torch
from config import CONFIG
from server import Server
from models import get_model

class FedAvgServer(Server):
    def __init__(self, clients, device='cpu', environment=None):
        super().__init__(clients, device, environment=environment)
        # FedAvg only needs one global model, but we can reuse self.global_model
        # We don't need topic_models, buffers, or async queues for standard FedAvg
        # But to minimize code changes, we'll just ignore them.
        
    def run(self):
        print(f"FedAvg Server started. Training for {CONFIG['global_rounds']} rounds.")
        
        for r in range(1, CONFIG['global_rounds'] + 1):
            self.current_round = r
            print(f"\n=== FedAvg Round {r} ===")
            
            # FedAvg: No Profiling, No Clustering
            
            # Step 1: Select Clients (Random Sampling)
            # Standard FedAvg selects a fraction of clients (C=0.1 usually)
            # Here we select 10 clients per round (0.1 * 100)
            m = max(1, int(CONFIG.get('fedavg_fraction', 0.1) * CONFIG['n_clients']))
            if self.environment is not None:
                selected_indices = self.environment.select_clients(
                    r,
                    population=range(CONFIG['n_clients']),
                    sample_fraction=CONFIG.get('fedavg_fraction', 0.1),
                )
            else:
                selected_indices = np.random.choice(range(CONFIG['n_clients']), m, replace=False)
            
            # Step 2: Training & Aggregation (Synchronous)
            # In standard FedAvg, we wait for ALL selected clients to finish (Straggler effect)
            # We simulate this by calculating the max latency among selected clients
            
            max_latency = 0
            updates = []
            total_samples = 0
            
            print(f"  Selected {m} clients: {selected_indices}")
            
            for cid in selected_indices:
                client = self.clients[cid]
                
                # Simulate latency
                latency = client.probe_latency(r)
                if latency > max_latency:
                    max_latency = latency
                    
                # Train
                # FedAvg uses global model weights
                model_weights = copy.deepcopy(self.global_model.state_dict())
                model = get_model(CONFIG['model_name'], CONFIG['num_classes'])
                model.load_state_dict(model_weights)
                
                # Standard FedAvg usually does E=1 or E=5. We use config.
                # Important: LR Decay logic should also apply here for fair comparison
                res = client.train(model, epochs=CONFIG.get('local_epochs', 5), return_prototype=False, current_round=self.current_round)
                
                updates.append((res['weights'], len(client.data_loader.dataset) if client.data_loader else 0))
                total_samples += (len(client.data_loader.dataset) if client.data_loader else 0)
                
            # Simulate waiting for the slowest client
            # In a real system, this adds to the Wall-clock time.
            # We can log this "Virtual Time".
            print(f"  [Time] Round finished. Slowest client took {max_latency:.2f}ms")
            
            # Step 3: Aggregation (Weighted Averaging)
            if total_samples > 0:
                avg_weights = copy.deepcopy(updates[0][0])
                
                # Initialize with weighted first update
                first_weight = updates[0][1] / total_samples
                for key in avg_weights:
                    # Ensure device
                    if avg_weights[key].device != self.device:
                        avg_weights[key] = avg_weights[key].to(self.device)
                    # Use non-inplace mul for type safety
                    avg_weights[key] = avg_weights[key] * first_weight
                    
                # Add others
                for i in range(1, len(updates)):
                    w_dict, n_samples = updates[i]
                    weight = n_samples / total_samples
                    for key in avg_weights:
                        val = w_dict[key]
                        if val.device != self.device:
                            val = val.to(self.device)
                        avg_weights[key] += val * weight
                        
                self.global_model.load_state_dict(avg_weights)
            
            # Step 4: Evaluation
            acc = self.evaluate_global()
            if acc is not None:
                print(f"FedAvg Global Accuracy: {acc:.2f}%")

    def evaluate_global(self):
        # Same as Server.evaluate_global
        return super().evaluate_global()

import numpy as np
