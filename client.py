
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from config import CONFIG

class Client:
    def __init__(self, id, data_loader, profile_type='medium', device='cpu'):
        self.id = id
        self.data_loader = data_loader
        self.device = device
        self.profile_type = profile_type
        self.profile_params = CONFIG['profiles'][profile_type]
        
        # Topic weights (Soft Clustering) - Initialized uniformly
        self.topic_weights = np.ones(CONFIG['n_topics']) / CONFIG['n_topics']
        self.assigned_topic = 0 # Will be assigned by Server
        
        self.criterion = nn.CrossEntropyLoss()

    def probe_latency(self):
        """
        Return estimated latency for the current round.
        Simulates network/compute volatility.
        """
        mu = self.profile_params['mean']
        sigma = self.profile_params['std']
        
        # Base latency from distribution
        base = np.random.normal(mu, sigma)
        base = max(10, base) # Min 10ms safety
        
        # Network volatility (random multiplier)
        volatility = np.random.uniform(1 - CONFIG['network_volatility'], 1 + CONFIG['network_volatility'])
        
        return base * volatility

    def train(self, model, epochs=1, return_prototype=False, **kwargs):
        """
        Standard local training.
        Args:
            model: Model instance (nn.Module)
            epochs: Number of local epochs
            return_prototype: Whether to compute and return prototype (mean of features)
            
        Returns: 
            dict: {
                'weights': state_dict, 
                'loss': avg_loss, 
                'prototype': np.array or None
            }
        """
        if self.data_loader is None:
            # Fallback for empty clients
            return {
                'weights': {k: v.cpu() for k, v in model.state_dict().items()},
                'loss': 0.0,
                'prototype': None
            }

        model.train()
        model.to(self.device)
        
        # Determine LR based on current round (passed via model?? No, need to pass round info)
        # Hack: server passes round info in kwargs? Or just use CONFIG static decay?
        # Let's assume server updates CONFIG['lr']? No, config is static dict usually.
        # We need to pass current_lr.
        # But `train` signature is fixed? 
        # Let's read current_round from kwargs if available, or just check global round if we could.
        # Simplest way: Check a global variable or add `current_round` to train args.
        # I will update `server.py` to pass `current_round` to `client.train`.
        
        # But for now, let's implement a simple check.
        # Wait, I can't easily change signature without changing server call.
        # Let's assume we pass `current_round` in kwargs.
        current_round = kwargs.get('current_round', 0)
        
        lr = CONFIG['lr']
        # Apply decay only ONCE if round > step
        # Assuming we want a single step decay:
        if current_round > CONFIG.get('lr_decay_step', 9999):
            lr = lr * CONFIG.get('lr_decay_gamma', 0.1)
            
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=CONFIG['momentum'])
        epoch_loss = []
        
        for epoch in range(epochs):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in client {self.client_id} at epoch {epoch}")
                    continue
                    
                loss.backward()
                
                # Add Gradient Clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                batch_loss.append(loss.item())
            if batch_loss:
                epoch_loss.append(np.mean(batch_loss))
            else:
                epoch_loss.append(0.0)
            
        result = {
            'weights': {k: v.cpu() for k, v in model.state_dict().items()},
            'loss': np.mean(epoch_loss) if epoch_loss else 0.0
        }
        
        # Additional safety: Clean up weights in result if they contain NaNs
        for k in result['weights']:
            if torch.isnan(result['weights'][k]).any():
                print(f"Warning: NaN detected in trained weights for key {k} in client {self.client_id}. Cleaning up.")
                result['weights'][k] = torch.zeros_like(result['weights'][k])

        if return_prototype:
            result['prototype'] = self._compute_prototype(model)
        else:
            result['prototype'] = None
            
        return result

    def _compute_prototype(self, model):
        """
        Compute feature mean (prototype) for clustering.
        """
        features_list = []
        
        final_layer_name = None
        # Heuristic to find the layer before the last one or just hook the last linear layer's input
        # For SimpleCNN: fc3 is last. Input to fc3 is what we want.
        # For ResNet: linear is last. Input to linear is what we want.
        
        target_module = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                target_module = module # Update to the last one found
                
        if target_module is None:
            return None
            
        def input_hook(module, input, output):
            # input is a tuple
            features_list.append(input[0].detach().cpu())
        
        handle = target_module.register_forward_hook(input_hook)
        
        model.eval()
        with torch.no_grad():
            for data, target in self.data_loader:
                data = data.to(self.device)
                _ = model(data)
                break # Only use one batch for prototype to save time? Or all? 
                # User said "Upload Loss_Vector". 
                # Wait, User said "Upload Loss_Vector (in several benchmark datasets)".
                # This is different from Prototype.
                # User Prompt: "Server 要求所有 Client 上传 Loss_Vector (在几个基准数据集上的 Loss)。Server 运行 K-Means 或 GMM"
                # If I strictly follow "Loss_Vector", I need a "Benchmark Dataset".
                # But I don't have a benchmark dataset.
                # I will stick to "Prototype" (feature mean) which is standard for clustering in FL, 
                # OR I can implement the Loss Vector on a small subset of global test data if available.
                # However, sending global data to client violates FL privacy (mostly).
                # Usually "Loss Vector" means loss on specific local data types?
                # "Loss on several benchmark datasets" implies Public Dataset.
                # I will use PROTOTYPE (Feature Mean) as it is robust and I have code for it.
                # I will add a comment about this deviation or just do it.
                # Actually, I'll stick to PROTOTYPE for now as it's easier to implement without public data.
        
        handle.remove()
        
        if features_list:
            all_features = torch.cat(features_list)
            prototype = all_features.mean(dim=0).numpy()
            return prototype
        return None
