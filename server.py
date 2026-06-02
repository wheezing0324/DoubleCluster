
import heapq
import numpy as np
import copy
import torch
import torch.nn as nn
import csv
import os
from config import CONFIG
from models import get_model

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, priority, item):
        # priority: typically arrival_round
        # item: any data
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)

    def empty(self):
        return len(self._queue) == 0
    
    def peek_priority(self):
        if self.empty(): return None
        return self._queue[0][0]

class Server:
    def __init__(self, clients, device='cpu', environment=None):
        self.clients = clients
        self.device = device
        self.environment = environment
        self.n_topics = CONFIG['n_topics']
        
        # 1. Models
        self.global_model = get_model(CONFIG['model_name'], CONFIG['num_classes']).to(self.device)
        self.topic_models = {}
        self.topic_versions = {}
        
        # Initialize Topic Models with Noise (to break symmetry)
        print("Initializing Topic Models with noise...")
        base_state = self.global_model.state_dict()
        for i in range(self.n_topics):
            model_i = copy.deepcopy(base_state)
            # Add noise
            for key in model_i:
                if ('weight' in key or 'bias' in key) and model_i[key].is_floating_point():
                    noise = torch.randn_like(model_i[key]) * 0.01
                    model_i[key] += noise
            self.topic_models[i] = model_i
            self.topic_versions[i] = 0
            
        # 2. Clustering Metadata
        # self.clients[cid].topic_weights is managed by Client, but Server needs to know/update it.
        # We also need prototypes for clustering.
        self.client_prototypes = {} 
        self.topic_prototypes = {}
        
        # 3. Buffers & Queues
        self.tier2_buffer = {i: [] for i in range(self.n_topics)} # {topic: [weights]}
        self.async_queue = PriorityQueue() # For Tier 3
        
        # State
        self.current_round = 0
        self.test_loader = None # Will be injected
        self.metrics = {'loss': [], 'accuracy': []}
        self.tracking_name = None
        self.metrics_path = None
        self.checkpoint_dir = None
        self.checkpoint_interval = CONFIG.get('checkpoint_interval', 10)
        
        # Ablation flags (default values)
        self.beta = CONFIG.get('beta', 0)
        self.clustering_mode = CONFIG.get('clustering_mode', 'soft')
        self.aggregation_strategy = CONFIG.get('aggregation_strategy', 'hide_adf')

    def configure_tracking(self, name, root=None, checkpoint_interval=None):
        self.tracking_name = name
        base_root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "runs")
        run_dir = os.path.join(base_root, name)
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.csv")
        self.checkpoint_interval = checkpoint_interval or CONFIG.get('checkpoint_interval', 10)
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "loss", "accuracy"])
        print(f"  [Tracking] metrics={self.metrics_path}")
        print(f"  [Tracking] checkpoints={self.checkpoint_dir}, interval={self.checkpoint_interval}")

    def get_checkpoint_state(self):
        return {
            "server_class": self.__class__.__name__,
            "round": self.current_round,
            "metrics": self.metrics,
            "global_model": self.global_model.state_dict(),
            "topic_models": self.topic_models,
            "topic_versions": self.topic_versions,
            "client_prototypes": self.client_prototypes,
            "topic_prototypes": self.topic_prototypes,
            "tier2_buffer": self.tier2_buffer,
            "async_queue": {
                "queue": self.async_queue._queue,
                "index": self.async_queue._index,
            },
            "config": copy.deepcopy(CONFIG),
        }

    def _to_cpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_cpu(v) for v in value)
        return value

    def save_checkpoint(self, keep_round=True):
        if not self.checkpoint_dir or not self.tracking_name:
            return
        state = self._to_cpu(self.get_checkpoint_state())
        round_path = os.path.join(self.checkpoint_dir, f"round_{self.current_round:04d}.pt")
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(state, latest_path)
        if keep_round:
            torch.save(state, round_path)
            print(f"  [Checkpoint] Saved {round_path}")
        else:
            print(f"  [Checkpoint] Saved {latest_path}")

    def record_metrics(self, loss, accuracy):
        if self.metrics_path:
            with open(self.metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.current_round, loss, accuracy])
        if self.checkpoint_dir and self.current_round > 0:
            keep_round = (
                self.current_round % self.checkpoint_interval == 0
                or self.current_round == CONFIG.get('global_rounds', -1)
            )
            self.save_checkpoint(keep_round=keep_round)
        
    def run(self):
        print(f"Server started. Training for {CONFIG['global_rounds']} rounds.")
        
        for r in range(1, CONFIG['global_rounds'] + 1):
            self.current_round = r
            print(f"\n=== Round {r} ===")
            
            # Step 1: Profiling & Clustering
            # If warmup, skip clustering update
            if self.current_round == CONFIG.get('lr_decay_step', 9999) + 1:
                old_lr = CONFIG['lr']
                new_lr = old_lr * CONFIG.get('lr_decay_gamma', 0.1)
                print(f"\n!!! LR Decay Triggered: {old_lr} -> {new_lr} !!!\n")

            if self.current_round > CONFIG.get('warmup_rounds', 0):
                self.dual_frequency_profiling()
            else:
                print(f"  [Warm-up] Round {r}/{CONFIG.get('warmup_rounds', 0)}. Training Global Model only.")
                # Force Topic Models to be identical to Global Model during Warmup
                # This ensures that when we assign Topic 0, it is actually the Global Model.
                # And other topics stay synced.
                global_state = self.global_model.state_dict()
                for k in range(self.n_topics):
                    # Deepcopy to avoid reference issues
                    self.topic_models[k] = copy.deepcopy(global_state)
            
            # Step 2: Distribution & Training (Simulation)
            # Returns updates for T1, buffers T2, queues T3
            t1_updates = self.distribute_and_train()
            
            # Step 3: Differentiated Aggregation
            
            # 3.1 Tier 1 (Sync) - Aggregated immediately
            self.aggregate_tier1(t1_updates)
            
            # 3.2 Tier 2 (Buffer) - Handled during distribution (append to buffer)
            # Check buffer size and aggregate if full
            self.check_tier2_buffers()
            
            # 3.3 Tier 3 (Async) - Process arrived events
            self.process_async_queue()
            
            # Step 4: Global Fusion & Evaluation
            self.global_fusion()
            self.evaluate_global()
            
    def dual_frequency_profiling(self):
        # 1. Data Profiling (Low Freq) - every 50 rounds
        if self.current_round % CONFIG['clustering_interval'] == 0:
            self.update_clustering()
            
        # 2. System Profiling (High Freq) - every round
        # In this sim, we do it per client in distribute step
        pass

    def update_clustering(self):
        print(">>> [Data Profiling] Updating Soft Clustering...")
        # Simple K-Means like update using Prototypes
        
        # 1. Compute Topic Centroids
        valid_topics = []
        for k in range(self.n_topics):
            # Find clients primarily assigned to k (or weighted?)
            # Let's use primary assignment for centroid calc
            cids = [cid for cid, c in self.clients.items() if c.assigned_topic == k]
            protos = [self.client_prototypes[cid] for cid in cids if cid in self.client_prototypes]
            
            if protos:
                mean_proto = np.mean(np.stack(protos), axis=0)
                self.topic_prototypes[k] = mean_proto
                valid_topics.append(k)
            # Else keep old or init?
            
        # 2. Update Client Weights based on distance to Centroids
        for cid, client in self.clients.items():
            if cid not in self.client_prototypes: continue
            
            p_i = self.client_prototypes[cid]
            dists = []
            for k in range(self.n_topics):
                if k in self.topic_prototypes:
                    # Use Squared Euclidean Distance for Gaussian Kernel similarity
                    # Formula 1.2: - || phi(Di) - mu_k ||^2 / tau
                    d = np.linalg.norm(p_i - self.topic_prototypes[k]) ** 2
                    dists.append(d)
                else:
                    dists.append(1e9) # Far away if no prototype
            
            # Softmax to get weights
            dists = np.array(dists)
            
            # Adaptive Temperature Scaling to ensure Soft Clustering
            # If raw distances are used, exp(-d) might decay too fast (Hard Clustering).
            # We scale logits so they have a standard deviation of ~1.0, preserving relative order but ensuring softness.
            
            # Ablation: Clustering Mode
            if self.clustering_mode == 'random':
                weights = np.random.dirichlet(np.ones(self.n_topics))
            elif self.clustering_mode == 'hard':
                # Hard Assignment: 1 for min distance, 0 for others
                weights = np.zeros(self.n_topics)
                min_idx = np.argmin(dists)
                weights[min_idx] = 1.0
            else:
                # Soft Assignment (Default)
                # Normalize distances by their standard deviation to serve as temperature tau
                dist_std = np.std(dists)
                temperature = dist_std if dist_std > 1e-6 else 1.0
                
                # Formula: exp( -dist / tau )
                logits = -dists / temperature
                logits -= np.max(logits) # Numerical stability
                weights = np.exp(logits) / np.sum(np.exp(logits))
            
            client.topic_weights = weights
            
    def distribute_and_train(self):
        t1_updates = {i: [] for i in range(self.n_topics)}
        
        # Determine if we need prototypes this round
        request_proto = (self.current_round % CONFIG['clustering_interval'] == 0)
        
        # Sample active clients (C=0.1)
        # Strategy B: Force full sampling during clustering rounds to get fresh prototypes from everyone
        if request_proto:
            print(">>> [Clustering Round] Force Full Sampling to update prototypes...")
            m = len(self.clients)
        else:
            m = max(1, int(CONFIG.get('sample_fraction', 0.1) * len(self.clients)))

        if self.environment is not None:
            active_cids = self.environment.select_clients(
                self.current_round,
                population=list(self.clients.keys()),
                force_all=request_proto,
            )
        else:
            active_cids = np.random.choice(list(self.clients.keys()), m, replace=False)
        
        # Base round time (for async calc) -> Mean of Fast profile
        base_time = CONFIG['profiles']['fast']['mean']
        
        # === Parallel Execution Helper ===
        def process_client(cid):
            client = self.clients[cid]
            
            # 1. Probe Latency
            latency = client.probe_latency(self.current_round)
            
            # 2. Determine Tier
            if latency < CONFIG['tier_thresholds']['T1']:
                tier = 'T1'
            elif latency < CONFIG['tier_thresholds']['T2']:
                tier = 'T2'
            else:
                tier = 'T3'
            
            # Determine Training Mode based on Tier
            # Tier 1 & 2: Update Backbone & Head (Full Training)
            # Tier 3: Update Head (Async)
            # Apply Ablation Strategy: 'drop_slow' -> Skip Tier 3
            if self.aggregation_strategy == 'drop_slow' and tier == 'T3':
                return None
            
            # Apply Ablation Strategy: 'all_sync' -> Treat T3 as T1 (Sync)
            if self.aggregation_strategy == 'all_sync':
                 tier = 'T1' # Force sync
            
            # Apply Ablation Strategy: 'all_async' -> Treat T1/T2 as T3 (Async)
            if self.aggregation_strategy == 'all_async':
                 tier = 'T3' # Force async
            
            mode = 'all'
            if tier in ['T1', 'T2']:
                mode = 'all'
            elif tier == 'T3':
                mode = 'head_only'
            
            # 3. Soft Mixing of Models (Soft-EM Downlink)
            is_warmup = (self.current_round <= CONFIG.get('warmup_rounds', 0))
            if is_warmup:
                # Warmup: Force Global (Topic 0)
                mixed_weights = copy.deepcopy(self.topic_models[0])
                client_topic_weights = np.zeros(self.n_topics)
                client_topic_weights[0] = 1.0
                client.assigned_topic = 0 
            else:
                # Soft-EM Mixing
                client_topic_weights = client.topic_weights
                mixed_weights = None
                
                # Ablation Strategy: 'all_async' (No Tiering, All Async)
                # But here we are simulating mixed_weights construction.
                # If all_async, we still need an initial model.
                # Soft-EM is fine for model initialization.
                
                # Assign primary topic for tracking/logging
                client.assigned_topic = int(np.argmax(client_topic_weights))
                
                # Weighted Sum of Topic Models
                # Optimization: Find first non-zero weight to init
                first_k = -1
                for k in range(self.n_topics):
                    if client_topic_weights[k] > 1e-4:
                        first_k = k
                        break
                
                if first_k == -1:
                    mixed_weights = copy.deepcopy(self.topic_models[0])
                else:
                    mixed_weights = copy.deepcopy(self.topic_models[first_k])
                    w_first = client_topic_weights[first_k]
                    for key in mixed_weights:
                        # Only scale floating point weights
                        if mixed_weights[key].is_floating_point():
                            mixed_weights[key] *= w_first
                        
                    # Add others
                    for k in range(first_k + 1, self.n_topics):
                        w_k = client_topic_weights[k]
                        if w_k > 1e-4:
                            model_k = self.topic_models[k]
                            for key in mixed_weights:
                                # Only mix floating point weights
                                if mixed_weights[key].is_floating_point():
                                    mixed_weights[key] += model_k[key] * w_k
                                    
                # Check mixed_weights for NaN/Inf before sending to client
                for key in mixed_weights:
                    if mixed_weights[key].is_floating_point():
                        if not torch.isfinite(mixed_weights[key]).all():
                            print(f"Warning: NaN/Inf detected in mixed_weights (key: {key}) before sending to client. Replacing with zeros to recover.")
                            mixed_weights[key] = torch.zeros_like(mixed_weights[key])
            
            # 4. Train (Simulation)
            model = get_model(CONFIG['model_name'], CONFIG['num_classes'])
            
            # Ensure mixed_weights are compatible with model structure (device/type)
            # load_state_dict might fail if types don't match exactly (e.g. Float vs Long)
            # We already protected against this by only mixing floats.
            try:
                model.load_state_dict(mixed_weights)
            except Exception as e:
                print(f"Error loading state_dict: {e}")
                return None
            
            res = client.train(model, epochs=CONFIG['local_epochs'], return_prototype=request_proto, current_round=self.current_round)
            
            # 5. Compute Delta (Update)
            # Optimization: Compute Delta on GPU to utilize parallelism and reduce CPU overhead
            new_weights = res['weights']
            delta = {}
            for k in new_weights:
                mw_val = mixed_weights[k]
                nw_val = new_weights[k]
                
                # Skip LongTensor (e.g. num_batches_tracked) for delta calculation
                
                # Check if it is a Tensor first
                if not isinstance(nw_val, torch.Tensor) or not isinstance(mw_val, torch.Tensor):
                    continue

                # IMPORTANT: Use is_floating_point() check instead of manual dtype list
                # This handles all float types (16, 32, 64, bfloat16) correctly
                if not nw_val.is_floating_point():
                    continue
                
                # Create a copy to avoid modifying the original model in place or causing view issues
                # Use .detach().clone() to be absolutely safe from autograd graph issues
                # And force to float32
                nw_val_f = nw_val.detach().clone().float()
                
                # Handle mw_val potentially not being a float tensor (e.g. if mixed_weights was shallow copied from a model with LongTensors)
                if isinstance(mw_val, torch.Tensor) and mw_val.is_floating_point():
                     mw_val_f = mw_val.detach().clone().float()
                else:
                     # If mixed weights is somehow not float (unlikely), cast it
                     mw_val_f = mw_val.detach().clone().float()

                # Ensure both are on the same device before subtraction
                # Prefer GPU for subtraction if available
                # Use device from nw_val_f as anchor
                target_device = nw_val_f.device if nw_val_f.device.type != 'cpu' else 'cpu'
                
                if nw_val_f.device != target_device:
                    nw_val_f = nw_val_f.to(target_device)
                if mw_val_f.device != target_device:
                    mw_val_f = mw_val_f.to(target_device)
                
                diff = nw_val_f - mw_val_f
                delta[k] = diff.cpu()
                
            # Prepare result packet
            result = {
                'cid': cid,
                'tier': tier,
                'delta': delta,
                'client_topic_weights': client_topic_weights,
                'prototype': res.get('prototype'),
                'latency': latency,
                'start_version': self.topic_versions[client.assigned_topic] # Approx version tracking
            }
            return result

        # Execute Parallel Training
        # Use ThreadPoolExecutor for I/O bound or GPU-released GIL tasks
        # RTX 4090 has 24GB VRAM. ResNet18 is ~45MB.
        # With batch_size=32, peak activation memory is small.
        # We can safely increase concurrency.
        # Let's try 10 workers to saturate the GPU better without OOM.
        from concurrent.futures import ThreadPoolExecutor
        results = []
        max_workers = 10 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_client, cid) for cid in active_cids]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error in client processing: {e}")

        # Process Results sequentially to update server state safely
        for res in results:
            if res is None: continue # Skip failed clients
            
            cid = res['cid']
            tier = res['tier']
            delta_cpu = res['delta']
            client_topic_weights = res['client_topic_weights']
            
            # Update Prototype
            if res['prototype'] is not None:
                self.client_prototypes[cid] = res['prototype']
                
            # Distribute based on Tier
            if tier == 'T1':
                for k in range(self.n_topics):
                    w_k = client_topic_weights[k]
                    if w_k > 1e-4:
                        t1_updates[k].append((delta_cpu, w_k))
                        
            elif tier == 'T2':
                for k in range(self.n_topics):
                    w_k = client_topic_weights[k]
                    if w_k > 1e-4:
                        self.tier2_buffer[k].append((delta_cpu, w_k))
                        
            elif tier == 'T3':
                # Async
                latency = res['latency']
                rounds_delay = int(latency / base_time)
                arrival_round = self.current_round + rounds_delay
                
                event = {
                    'cid': cid,
                    'start_round': self.current_round,
                    'delta': delta_cpu,
                    'weights': client_topic_weights
                }
                self.async_queue.push(arrival_round, event)
                
        return t1_updates

    def aggregate_tier1(self, t1_updates):
        # Weighted Aggregation for Soft Clustering
        for k in range(self.n_topics):
            updates = t1_updates[k] # List of (delta, w_k)
            if not updates: continue
            
            print(f"  [Tier 1] Aggregating {len(updates)} weighted updates for Topic {k}")
            
            total_weight = 0.0
            aggregated_delta = None
            
            for delta, w_k in updates:
                total_weight += w_k
                if aggregated_delta is None:
                    # Filter only float tensors for aggregation to avoid LongTensor multiplication errors
                    aggregated_delta = {}
                    for key, val in delta.items():
                        if val.dtype in [torch.float32, torch.float16, torch.float64]:
                            aggregated_delta[key] = val * w_k
                else:
                    for key in aggregated_delta:
                        if key in delta:
                            aggregated_delta[key] += delta[key] * w_k
            
            if total_weight > 0:
                # Apply Average Delta
                current_model = self.topic_models[k]
                
                for key in current_model:
                    # Only update if we have an aggregated delta for this key
                    if key not in aggregated_delta:
                        continue
                        
                    if current_model[key].device != self.device:
                        current_model[key] = current_model[key].to(self.device)
                    
                    update = aggregated_delta[key].to(self.device)
                    current_model[key] += update / total_weight
                    
                self.topic_versions[k] += 1
            
    def check_tier2_buffers(self):
        # Ablation: Buffer Size Override
        threshold = CONFIG.get('tier2_buffer_threshold', CONFIG['buffer_size'])
        
        for k in range(self.n_topics):
            buffer = self.tier2_buffer[k] # List of (delta, w_k)
            if len(buffer) >= threshold:
                print(f"  [Tier 2] Buffer full for Topic {k} ({len(buffer)} >= {threshold}). Aggregating...")
                
                total_weight = 0.0
                aggregated_delta = None
                
                for delta, w_k in buffer:
                    total_weight += w_k
                    if aggregated_delta is None:
                        # Filter only float tensors
                        aggregated_delta = {}
                        for key, val in delta.items():
                            if val.dtype in [torch.float32, torch.float16, torch.float64]:
                                aggregated_delta[key] = val * w_k
                    else:
                        for key in aggregated_delta:
                            if key in delta:
                                aggregated_delta[key] += delta[key] * w_k
                            
                if total_weight > 0:
                    alpha = CONFIG['tier2_alpha']
                    current_model = self.topic_models[k]
                    
                    for key in current_model:
                        # Only update if we have an aggregated delta for this key
                        if key not in aggregated_delta:
                            continue

                        if current_model[key].device != self.device:
                            current_model[key] = current_model[key].to(self.device)
                        
                        update = aggregated_delta[key].to(self.device) / total_weight
                        current_model[key] += alpha * update
                        
                    self.topic_versions[k] += 1
                
                self.tier2_buffer[k] = [] # Clear

    def process_async_queue(self):
        # Process events where arrival_round <= current_round
        while not self.async_queue.empty():
            arrival_round = self.async_queue.peek_priority()
            if arrival_round > self.current_round:
                break
            
            # Pop and process
            priority, _, event = self.async_queue.pop()
            self.aggregate_tier3(event)
            
    def aggregate_tier3(self, event):
        cid = event['cid']
        delta = event['delta']
        client_weights = event['weights']
        start_round = event['start_round']
        
        # Staleness tau
        staleness = self.current_round - start_round
        
        # Polynomial Decay (Formula: alpha(tau) = 1 / (1 + tau)^gamma)
        gamma = CONFIG.get('async_gamma', 1.0) # Decay hyperparameter
        decay_factor = 1.0 / ((1 + staleness) ** gamma)
        
        # Gradient Clipping on Delta
        total_norm = 0.0
        for k in delta:
            if delta[k].dtype in [torch.float32, torch.float16, torch.float64]:
                total_norm += delta[k].norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        scale = 1.0
        if total_norm > 10.0:
            scale = 10.0 / total_norm
            
        # Apply to all relevant topics
        for k in range(self.n_topics):
            w_k = client_weights[k] # pi_{i,k}
            if w_k < 1e-4: continue
            
            print(f"  [Tier 3] Client {cid} update (w={w_k:.2f}) arrived for Topic {k}. Staleness: {staleness}, Factor: {decay_factor:.4f}")
            
            current_model = self.topic_models[k]
            lr = CONFIG.get('async_lr', 0.01) # eta_g
            
            # W_{t+1} = W_t + eta_g * alpha(tau) * pi_{i,k} * Delta
            for key in current_model:
                if current_model[key].dtype not in [torch.float32, torch.float16, torch.float64]:
                    continue
                if key not in delta:
                    continue
                    
                if current_model[key].device != self.device:
                    current_model[key] = current_model[key].to(self.device)
                
                # Apply update
                # update = lr * decay_factor * w_k * delta[key].to(self.device) * scale
                # scale is for gradient clipping
                
                d_val = delta[key].to(self.device)
                update = lr * decay_factor * w_k * d_val * scale
                
                current_model[key] += update
            
            self.topic_versions[k] += 1

    def global_fusion(self):
        # Merge A, B, C based on data counts or just average?
        # User said: "Weighted by data count".
        # We need data counts for each topic.
        # Estimate from assignments.
        
        counts = {k: 0 for k in range(self.n_topics)}
        for c in self.clients.values():
            counts[c.assigned_topic] += len(c.data_loader.dataset) if c.data_loader else 0
            
        total_data = sum(counts.values())
        if total_data == 0: return
        
        global_weights = None
        
        for k in range(self.n_topics):
            weight = counts[k] / total_data
            model = self.topic_models[k]
            
            if global_weights is None:
                global_weights = copy.deepcopy(model)
                for key in global_weights:
                    # Ensure global_weights are on device
                    if global_weights[key].device != self.device:
                         global_weights[key] = global_weights[key].to(self.device)
                    # Avoid in-place multiplication to allow type promotion (Long -> Float) for buffers like num_batches_tracked
                    global_weights[key] = global_weights[key] * weight
            else:
                for key in global_weights:
                    # Ensure model[key] is on device
                    val = model[key]
                    if val.device != self.device:
                        val = val.to(self.device)
                    global_weights[key] += val * weight
                    
        self.global_model.load_state_dict(global_weights)

        # === Global Model Re-injection (Knowledge Distillation) ===
        # Inject Global Knowledge back to Topic Models to prevent isolation
        # Oscillation Fix: Beta = 0.0 disables re-injection, leading to drift.
        # But Beta > 0 (e.g., 0.5) caused homogenization.
        # Oscillation usually comes from:
        # 1. Tier 3 Stale Updates overwriting good models.
        # 2. Clustering Updates shifting centroids drastically.
        # 3. Global Fusion overwriting specialized topics too aggressively if Beta is high.
        
        # Current Beta is 0. Let's keep it 0 or small (0.05) to test.
        # But to fix oscillation, we should check if Tier 3 updates are too aggressive.
        # And if Global Fusion is "resetting" the models too much.
        # Actually, if Beta=0, Global Fusion DOES NOT affect Topic Models.
        # So oscillation is purely from Topic Model training (T1/T2/T3).
        
        # Check T3 Aggregation LR. It's 0.01. Maybe too high for async?
        # Check T1 Aggregation. It's simple averaging.
        
        # Let's try to Smooth Global Fusion for evaluation stability.
        # Instead of replacing Global Model, we can use EMA.
        # self.global_model = alpha * self.global_model + (1-alpha) * new_global
        
        alpha_global = 0.2 # Smooth update for global model to reduce eval noise
        current_global_state = self.global_model.state_dict()
        
        for key in global_weights:
            if key in current_global_state:
                # Ensure devices
                w_curr = current_global_state[key]
                w_new = global_weights[key]
                if w_curr.device != self.device: w_curr = w_curr.to(self.device)
                if w_new.device != self.device: w_new = w_new.to(self.device)
                
                # EMA Update
                global_weights[key] = (1 - alpha_global) * w_curr + alpha_global * w_new
        
        self.global_model.load_state_dict(global_weights)
        
        # Re-injection
        beta = self.beta # Use ablation flag
        print(f"  [Fusion] Re-injecting global knowledge into Topic Models (beta={beta})...")
        
        for k in range(self.n_topics):
            topic_model = self.topic_models[k]
            for key in topic_model:
                 # Skip non-floating point tensors (like LongTensors)
                if not topic_model[key].is_floating_point():
                    continue
                    
                w_topic = topic_model[key]
                w_global = global_weights[key]
                
                if w_topic.device != self.device: w_topic = w_topic.to(self.device)
                if w_global.device != self.device: w_global = w_global.to(self.device)
                
                # Check for NaNs before applying to prevent CUDA invalid value errors
                if torch.isnan(w_topic).any() or torch.isnan(w_global).any():
                    print(f"Warning: NaN detected in topic {k} key {key}. Skipping re-injection.")
                    continue
                    
                topic_model[key] = (1 - beta) * w_topic + beta * w_global

    def evaluate_global(self):
        if not self.test_loader: return
        
        self.global_model.eval()
        self.global_model.to(self.device)
        
        correct = 0
        total = 0
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                
                loss = criterion(output, target)
                test_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
                
        acc = 100. * correct / total
        avg_loss = test_loss / total
        
        self.metrics['loss'].append(avg_loss)
        self.metrics['accuracy'].append(acc)
        self.record_metrics(avg_loss, acc)
        
        print(f"Global Accuracy: {acc:.2f}%, Global Loss: {avg_loss:.4f}")
        return acc
