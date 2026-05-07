import copy
import numpy as np
import torch
from config import CONFIG
from server import Server
from models import get_model

class CSAHFLServer(Server):
    def __init__(self, clients, device='cpu'):
        super().__init__(clients, device)
        self.num_edge_servers = 5
        self.edge_rounds_per_cloud = 5
        
        # Assign clients to edge servers (Hierarchical structure)
        self.edge_clients = {i: [] for i in range(self.num_edge_servers)}
        cids = list(self.clients.keys())
        np.random.seed(42)
        np.random.shuffle(cids)
        
        chunk_size = len(cids) // self.num_edge_servers
        for i in range(self.num_edge_servers):
            self.edge_clients[i] = cids[i*chunk_size : (i+1)*chunk_size]
            
        # Initialize Edge Models
        self.edge_models = {i: copy.deepcopy(self.global_model.state_dict()) for i in range(self.num_edge_servers)}
        
    def run(self):
        print(f"CSAHFL Server started. Training for {CONFIG['global_rounds']} rounds.")
        
        for r in range(1, CONFIG['global_rounds'] + 1):
            self.current_round = r
            print(f"\n=== CSAHFL Round {r} ===")
            
            # Step 1: Intra-cluster (Edge) Training & Aggregation
            # CSAHFL selects clients within each edge server
            m_per_edge = max(1, int(CONFIG.get('sample_fraction', 0.1) * (CONFIG['n_clients'] / self.num_edge_servers)))
            
            for edge_id in range(self.num_edge_servers):
                edge_model_state = self.edge_models[edge_id]
                
                # Select clients
                available_clients = self.edge_clients[edge_id]
                selected_cids = np.random.choice(available_clients, min(m_per_edge, len(available_clients)), replace=False)
                
                updates = []
                total_data = 0
                max_latency = 0
                
                for cid in selected_cids:
                    client = self.clients[cid]
                    latency = client.probe_latency()
                    if latency > max_latency: max_latency = latency
                    
                    # Train
                    model = get_model(CONFIG['model_name'], CONFIG['num_classes']).to(self.device)
                    model.load_state_dict(copy.deepcopy(edge_model_state))
                    res = client.train(model, epochs=CONFIG.get('local_epochs', 1), return_prototype=False, current_round=r)
                    
                    data_size = len(client.data_loader.dataset) if client.data_loader else 0
                    
                    # CSAHFL Adaptive staleness discount (Eq 10 simplified)
                    # discount = exp(-staleness / T_theta)
                    staleness = latency / 100.0
                    discount = np.exp(-staleness / 10.0) 
                    
                    updates.append((res['weights'], discount, data_size))
                    total_data += (discount * data_size)
                
                # Edge Aggregation
                if updates and total_data > 0:
                    new_edge = copy.deepcopy(updates[0][0])
                    for key in new_edge:
                        if new_edge[key].device != self.device:
                            new_edge[key] = new_edge[key].to(self.device)
                            
                        if not new_edge[key].is_floating_point(): continue
                        new_edge[key] = torch.zeros_like(new_edge[key])
                        for w_dict, disc, d_size in updates:
                            client_w = w_dict[key].to(self.device)
                            if not torch.isfinite(client_w).all():
                                print(f"Warning: NaN/Inf detected in client update for key {key}. Skipping this client's contribution to edge {edge_id}.")
                                continue
                            new_edge[key] += client_w * (disc * d_size / total_data)
                    self.edge_models[edge_id] = new_edge
                    
            # Step 2: Inter-cluster (Cloud) Aggregation every E rounds
            if r % self.edge_rounds_per_cloud == 0:
                print(f"  [CSAHFL] Performing Cloud-Level Aggregation at round {r}")
                # Cloud Server aggregates all Edge Models
                new_global = copy.deepcopy(self.edge_models[0])
                for key in new_global:
                    if new_global[key].device != self.device:
                        new_global[key] = new_global[key].to(self.device)
                        
                    if not new_global[key].is_floating_point(): continue
                    new_global[key] = torch.zeros_like(new_global[key])
                    for edge_id in range(self.num_edge_servers):
                        edge_w = self.edge_models[edge_id][key].to(self.device)
                        if not torch.isfinite(edge_w).all():
                            print(f"Warning: NaN/Inf detected in edge {edge_id} model for key {key}. Skipping this edge server.")
                            continue
                        # Simple average across edge servers for cloud aggregation
                        new_global[key] += edge_w * (1.0 / self.num_edge_servers)
                        
                self.global_model.load_state_dict(new_global)
                
                # Distribute back to Edge Servers
                for edge_id in range(self.num_edge_servers):
                    self.edge_models[edge_id] = copy.deepcopy(new_global)
                    
            # Evaluate Global Model (For plotting purposes, we evaluate the global model or average edge model)
            acc = self.evaluate_global()
            if acc is not None:
                print(f"CSAHFL Global Accuracy: {acc:.2f}%")