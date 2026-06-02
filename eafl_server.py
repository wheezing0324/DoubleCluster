import copy
import numpy as np
import torch
from sklearn.cluster import KMeans
from config import CONFIG
from server import Server
from models import get_model

class EAFLServer(Server):
    def __init__(self, clients, device='cpu', environment=None):
        super().__init__(clients, device, environment=environment)
        self.num_clusters = 5
        self.clusters = {} # cluster_id -> list of client_ids

    def get_checkpoint_state(self):
        state = super().get_checkpoint_state()
        state.update({
            "num_clusters": self.num_clusters,
            "clusters": self.clusters,
        })
        return state
        
    def perform_clustering(self):
        print(">>> [EAFL] Performing Gradient-based Clustering...")
        # Get gradients from all clients for 1 epoch
        gradients = []
        cids = list(self.clients.keys())
        max_batches = CONFIG.get('eafl_clustering_max_batches')
        
        for idx, cid in enumerate(cids, start=1):
            if idx == 1 or idx % 10 == 0 or idx == len(cids):
                print(f"  [EAFL] Clustering signature {idx}/{len(cids)} clients...")
            client = self.clients[cid]
            model = get_model(CONFIG['model_name'], CONFIG['num_classes']).to(self.device)
            model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
            
            # Quick train to get gradient
            res = client.train(
                model,
                epochs=1,
                return_prototype=False,
                current_round=0,
                max_batches=max_batches,
            )
            
            # Compute gradient vector (flattened)
            grad_vec = []
            for k in res['weights']:
                if res['weights'][k].is_floating_point():
                    diff = res['weights'][k].to(self.device) - self.global_model.state_dict()[k].to(self.device)
                    grad_vec.append(diff.flatten())
            
            if grad_vec:
                flat_grad = torch.cat(grad_vec).cpu().numpy()
                gradients.append(flat_grad)
            else:
                gradients.append(np.zeros(1))
                
        # K-Means
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = kmeans.fit_predict(gradients)
        
        self.clusters = {i: [] for i in range(self.num_clusters)}
        for idx, label in enumerate(labels):
            self.clusters[label].append(cids[idx])
            
        print(f"  [EAFL] Clustering results: {[len(self.clusters[i]) for i in range(self.num_clusters)]} clients per cluster")

    def run(self):
        print(f"EAFL Server started. Training for {CONFIG['global_rounds']} rounds.")
        
        # Initial clustering
        self.perform_clustering()
        
        for r in range(1, CONFIG['global_rounds'] + 1):
            self.current_round = r
            print(f"\n=== EAFL Round {r} ===")
            
            # EAFL selects a fraction of clients per round
            m = max(1, int(CONFIG.get('sample_fraction', 0.1) * CONFIG['n_clients']))
            if self.environment is not None:
                selected_indices = self.environment.select_clients(r, population=range(CONFIG['n_clients']))
            else:
                selected_indices = np.random.choice(range(CONFIG['n_clients']), m, replace=False)
            
            cluster_updates = {i: [] for i in range(self.num_clusters)}
            
            max_latency = 0
            
            for cid in selected_indices:
                client = self.clients[cid]
                latency = client.probe_latency(r)
                if latency > max_latency: max_latency = latency
                
                # Train
                model = get_model(CONFIG['model_name'], CONFIG['num_classes']).to(self.device)
                model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
                res = client.train(model, epochs=CONFIG.get('local_epochs', 1), return_prototype=False, current_round=r)
                
                # Find which cluster this client belongs to
                c_id = 0
                for k, v in self.clusters.items():
                    if cid in v:
                        c_id = k
                        break
                
                staleness = max(1, int(latency / 100)) # Simple staleness based on latency
                weight = 1.0 / staleness # EAFL inverse staleness
                data_size = len(client.data_loader.dataset) if client.data_loader else 0
                
                cluster_updates[c_id].append((res['weights'], weight, data_size))
                
            print(f"  [Time] Round finished. Max latency: {max_latency:.2f}ms")
            
            # Two-Stage Aggregation
            cluster_models = {}
            cluster_data_sizes = {}
            
            # Stage 1: Intra-cluster
            for c_id, updates in cluster_updates.items():
                if not updates: continue
                
                total_weight = sum([u[1] * u[2] for u in updates]) # weight * data_size
                if total_weight == 0: continue
                
                agg_model = copy.deepcopy(updates[0][0])
                for key in agg_model:
                    if agg_model[key].device != self.device:
                        agg_model[key] = agg_model[key].to(self.device)
                        
                    if not agg_model[key].is_floating_point(): continue
                    agg_model[key] = torch.zeros_like(agg_model[key])
                    for w_dict, st_weight, d_size in updates:
                        # Defensive check for NaNs/Infs in client updates
                        client_w = w_dict[key].to(self.device)
                        if not torch.isfinite(client_w).all():
                            print(f"Warning: NaN/Inf detected in client update for key {key}. Skipping this client's contribution to cluster {c_id}.")
                            continue
                        agg_model[key] += client_w * (st_weight * d_size / total_weight)
                        
                cluster_models[c_id] = agg_model
                cluster_data_sizes[c_id] = sum([u[2] for u in updates])
                
            # Stage 2: Inter-cluster
            if cluster_models:
                total_data = sum(cluster_data_sizes.values())
                new_global = copy.deepcopy(list(cluster_models.values())[0])
                
                for key in new_global:
                    if new_global[key].device != self.device:
                        new_global[key] = new_global[key].to(self.device)
                        
                    if not new_global[key].is_floating_point(): continue
                    new_global[key] = torch.zeros_like(new_global[key])
                    for c_id, c_model in cluster_models.items():
                        cluster_w = c_model[key].to(self.device)
                        # Defensive check for NaNs/Infs in cluster models
                        if not torch.isfinite(cluster_w).all():
                            print(f"Warning: NaN/Inf detected in cluster {c_id} model for key {key}. Skipping this cluster.")
                            continue
                        new_global[key] += cluster_w * (cluster_data_sizes[c_id] / total_data)
                        
                self.global_model.load_state_dict(new_global)
                
            # Evaluate
            acc = self.evaluate_global()
            if acc is not None:
                print(f"EAFL Global Accuracy: {acc:.2f}%")
