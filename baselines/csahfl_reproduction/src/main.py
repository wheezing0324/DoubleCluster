"""
CSAHFL Main Training Loop (Algorithm 1)
Implements the complete CSAHFL framework with:
- Privacy-preserving one-shot clustering
- Adaptive semi-asynchronous intra-cluster aggregation
- Dynamic distribution-aware inter-cluster aggregation
"""

import os
import sys
import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from data.non_iid_partition import NonIIDPartitioner
from data.mobility_simulator import MobilitySimulator
from clustering.hierarchical_clusterer import HierarchicalClusterer
from clients.client import Client
from edge_servers.edge_server import EdgeServer
from cloud.cloud_server import CloudServer
from models.cnn_fashion_mnist import CNNFashionMNIST
from models.cnn_cifar10 import CNNCIFAR10
from models.cnn_svhn import CNNSVHN
from utils.logger import ExperimentLogger
from utils.metrics import MetricsCalculator


class CSAHFLTrainer:
    """
    Main trainer class implementing Algorithm 1 from the paper.
    Coordinates all components of the CSAHFL framework.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize CSAHFL trainer with configuration.
        
        Args:
            config: Configuration dictionary with all hyperparameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract configuration parameters
        self.num_clients = config['system']['num_clients']
        self.num_edge_servers = config['system']['num_edge_servers']
        self.num_classes = config['system']['num_classes']
        self.total_cloud_rounds = config['experiment']['total_cloud_rounds']
        self.num_rounds_edge = config['experiment']['edge_rounds_per_cloud']
        self.batch_size = config['intra_cluster']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.emax = config['hyperparameter_analysis']['emax_sweep']
        self.alpha = config['hyperparameter_analysis']['alpha_sweep']
        self.beta = config['hyperparameter_analysis']['beta_sweep']
        self.svd_components = config['clustering']['svd_components']
        self.staying_prob = config['mobility']['staying_prob']
        
        # Initialize components
        self._initialize_components()
        
        # Metrics tracking
        self.metrics = MetricsCalculator()
        self.logger = ExperimentLogger(config['experiment']['log_dir'])
        
    def _initialize_components(self):
        """Initialize all CSAHFL components."""
        # 1. Data loading and partitioning
        self.data_loader = DataLoader(self.config)
        self.partition = NonIIDPartitioner(self.config)
        
        # Load dataset
        dataset_name = self.config['dataset']['name']
        self.train_data, self.test_data = self.data_loader.load_dataset(dataset_name)
        
        # Partition data with dual-layer non-IID
        self.client_data, self.es_classes = self.partition.partition_data(
            self.train_data, 
            self.num_clients, 
            self.num_edge_servers
        )
        
        # 2. Mobility simulator
        self.mobility = MobilitySimulator(
            self.num_clients,
            self.num_edge_servers,
            self.staying_prob
        )
        
        # 3. Initialize clients
        self.clients = self._create_clients()
        
        # 4. Initialize edge servers
        self.edge_servers = self._create_edge_servers()
        
        # 5. Initialize cloud server
        self.cloud_server = CloudServer(self.config, self.device)
        
        # 6. Perform one-shot clustering
        self.clusterer = HierarchicalClusterer(self.config)
        self.client_clusters = self.clusterer.perform_clustering(
            self.clients,
            self.client_data
        )
        
        # Assign clients to edge servers based on clustering
        self._assign_clients_to_edge_servers()
        
    def _create_clients(self) -> List[Client]:
        """Create client instances with appropriate models."""
        clients = []
        dataset_name = self.config['dataset']['name']
        
        for client_id in range(self.num_clients):
            # Select model based on dataset
            if dataset_name == 'fashion_mnist':
                model = CNNFashionMNIST(self.num_classes)
            elif dataset_name == 'cifar10':
                model = CNNCIFAR10(self.num_classes)
            elif dataset_name == 'svhn':
                model = CNNSVHN(self.num_classes)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            client = Client(
                client_id=client_id,
                model=model,
                config=self.config,
                device=self.device
            )
            clients.append(client)
        
        return clients
    
    def _create_edge_servers(self) -> List[EdgeServer]:
        """Create edge server instances."""
        edge_servers = []
        dataset_name = self.config['dataset']['name']
        
        for es_id in range(self.num_edge_servers):
            # Select model based on dataset
            if dataset_name == 'fashion_mnist':
                model = CNNFashionMNIST(self.num_classes)
            elif dataset_name == 'cifar10':
                model = CNNCIFAR10(self.num_classes)
            elif dataset_name == 'svhn':
                model = CNNSVHN(self.num_classes)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            edge_server = EdgeServer(
                es_id=es_id,
                model=model,
                config=self.config,
                device=self.device
            )
            edge_servers.append(edge_server)
        
        return edge_servers
    
    def _assign_clients_to_edge_servers(self):
        """Assign clients to edge servers based on clustering results."""
        # Group clients by cluster
        clusters = {}
        for client_id, cluster_id in self.client_clusters.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_id)
        
        # Assign each cluster to an edge server
        cluster_ids = sorted(clusters.keys())
        for idx, cluster_id in enumerate(cluster_ids):
            es_id = idx % self.num_edge_servers
            for client_id in clusters[cluster_id]:
                self.clients[client_id].edge_server_id = es_id
                self.edge_servers[es_id].add_client(client_id)
    
    def train(self) -> Dict:
        """
        Execute the main CSAHFL training loop (Algorithm 1).
        
        Returns:
            Dictionary containing training metrics and results
        """
        print(f"Starting CSAHFL training for {self.total_cloud_rounds} cloud rounds")
        print(f"Device: {self.device}")
        
        training_history = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'training_time': []
        }
        
        start_time = datetime.now()
        
        for cloud_round in range(self.total_cloud_rounds):
            round_start = datetime.now()
            
            print(f"\n{'='*60}")
            print(f"Cloud Round {cloud_round + 1}/{self.total_cloud_rounds}")
            print(f"{'='*60}")
            
            # Step 1: Handle client mobility (if not static)
            if self.staying_prob < 1.0:
                self._handle_mobility(cloud_round)
            
            # Step 2: Edge-level aggregation (re rounds)
            for edge_round in range(self.num_rounds_edge):
                self._perform_edge_aggregation(cloud_round, edge_round)
            
            # Step 3: Cloud-level aggregation
            self._perform_cloud_aggregation(cloud_round)
            
            # Step 4: Evaluate global model
            accuracy, loss = self._evaluate_global_model(cloud_round)
            
            round_time = (datetime.now() - round_start).total_seconds()
            
            # Record metrics
            training_history['round'].append(cloud_round + 1)
            training_history['accuracy'].append(accuracy)
            training_history['loss'].append(loss)
            training_history['training_time'].append(round_time)
            
            print(f"Round {cloud_round + 1}: Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={round_time:.2f}s")
            
            # Log results
            self.logger.log_round(cloud_round, {
                'accuracy': accuracy,
                'loss': loss,
                'time': round_time
            })
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Final evaluation
        final_metrics = self._final_evaluation()
        
        results = {
            'training_history': training_history,
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': final_metrics['loss'],
            'total_time': total_time,
            'config': self.config
        }
        
        # Save results
        self.logger.save_results(results)
        
        return results
    
    def _handle_mobility(self, cloud_round: int):
        """
        Handle client mobility between edge servers.
        
        Args:
            cloud_round: Current cloud aggregation round
        """
        # Simulate client movement
        mobility_events = self.mobility.simulate_movement()
        
        # Update client-edge server assignments
        for client_id, new_es_id in mobility_events.items():
            old_es_id = self.clients[client_id].edge_server_id
            if old_es_id != new_es_id:
                # Remove from old edge server
                self.edge_servers[old_es_id].remove_client(client_id)
                # Add to new edge server
                self.clients[client_id].edge_server_id = new_es_id
                self.edge_servers[new_es_id].add_client(client_id)
                
                # Update cluster assignment if needed
                # (In high mobility, may need to re-cluster periodically)
    
    def _perform_edge_aggregation(self, cloud_round: int, edge_round: int):
        """
        Perform semi-asynchronous intra-cluster aggregation at edge servers.
        
        Args:
            cloud_round: Current cloud round
            edge_round: Current edge round within cloud round
        """
        for es_id, edge_server in enumerate(self.edge_servers):
            # Get clients assigned to this edge server
            client_ids = edge_server.get_client_ids()
            
            if len(client_ids) == 0:
                continue
            
            # Collect updates from clients (semi-asynchronous)
            client_updates = []
            client_data_sizes = []
            
            for client_id in client_ids:
                client = self.clients[client_id]
                data = self.client_data[client_id]
                
                # Perform local training
                update, data_size, training_time = client.local_train(
                    data, 
                    edge_round,
                    cloud_round
                )
                
                client_updates.append({
                    'client_id': client_id,
                    'update': update,
                    'data_size': data_size,
                    'training_time': training_time
                })
                client_data_sizes.append(data_size)
            
            # Perform semi-asynchronous aggregation
            edge_server.aggregate_updates(
                client_updates,
                cloud_round,
                edge_round
            )
    
    def _perform_cloud_aggregation(self, cloud_round: int):
        """
        Perform distribution-aware inter-cluster aggregation at cloud server.
        
        Args:
            cloud_round: Current cloud round
        """
        # Collect edge server models and metadata
        edge_models = []
        edge_data_sizes = []
        edge_distributions = []
        
        for es_id, edge_server in enumerate(self.edge_servers):
            model_state = edge_server.get_model_state()
            data_size = edge_server.get_total_data_size()
            data_dist = edge_server.get_label_distribution()
            
            edge_models.append(model_state)
            edge_data_sizes.append(data_size)
            edge_distributions.append(data_dist)
        
        # Perform distribution-aware aggregation
        self.cloud_server.aggregate_edge_models(
            edge_models,
            edge_data_sizes,
            edge_distributions,
            cloud_round
        )
        
        # Distribute global model back to edge servers and clients
        global_model = self.cloud_server.get_global_model()
        
        for edge_server in self.edge_servers:
            edge_server.update_model(global_model)
        
        for client in self.clients:
            client.update_model(global_model)
    
    def _evaluate_global_model(self, cloud_round: int) -> Tuple[float, float]:
        """
        Evaluate global model on test data.
        
        Args:
            cloud_round: Current cloud round
            
        Returns:
            Tuple of (accuracy, loss)
        """
        global_model = self.cloud_server.get_global_model()
        
        # Create temporary model for evaluation
        dataset_name = self.config['dataset']['name']
        if dataset_name == 'fashion_mnist':
            eval_model = CNNFashionMNIST(self.num_classes)
        elif dataset_name == 'cifar10':
            eval_model = CNNCIFAR10(self.num_classes)
        elif dataset_name == 'svhn':
            eval_model = CNNSVHN(self.num_classes)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        eval_model.load_state_dict(global_model)
        eval_model.to(self.device)
        eval_model.eval()
        
        # Evaluate on test data
        accuracy, loss = self.metrics.evaluate_model(
            eval_model,
            self.test_data,
            self.device
        )
        
        return accuracy, loss
    
    def _final_evaluation(self) -> Dict:
        """
        Perform final evaluation and generate comprehensive metrics.
        
        Returns:
            Dictionary with final evaluation metrics
        """
        global_model = self.cloud_server.get_global_model()
        
        dataset_name = self.config['dataset']['name']
        if dataset_name == 'fashion_mnist':
            eval_model = CNNFashionMNIST(self.num_classes)
        elif dataset_name == 'cifar10':
            eval_model = CNNCIFAR10(self.num_classes)
        elif dataset_name == 'svhn':
            eval_model = CNNSVHN(self.num_classes)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        eval_model.load_state_dict(global_model)
        eval_model.to(self.device)
        eval_model.eval()
        
        accuracy, loss = self.metrics.evaluate_model(
            eval_model,
            self.test_data,
            self.device
        )
        
        # Additional metrics
        per_class_accuracy = self.metrics.per_class_accuracy(
            eval_model,
            self.test_data,
            self.device
        )
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'per_class_accuracy': per_class_accuracy
        }


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML files.
    
    Args:
        config_path: Path to main configuration file
        
    Returns:
        Merged configuration dictionary
    """
    # Load default config
    with open('config/default_config.yaml', 'r') as f:
        default_config = yaml.safe_load(f)
    
    # Load dataset config
    with open('config/dataset_config.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Load experiment config
    with open('config/experiment_config.yaml', 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    # Merge configurations
    config = default_config
    config.update(dataset_config)
    config.update(experiment_config)
    
    # Override with command-line config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
            config.update(custom_config)
    
    return config


def main():
    """Main entry point for CSAHFL training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CSAHFL Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom configuration file')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['fashion_mnist', 'cifar10', 'svhn'],
                        help='Dataset to use')
    parser.add_argument('--eniid', type=str, default='EIID',
                        choices=['EIID', 'ENIID50%', 'ENIID30%'],
                        help='Edge-layer non-IID setting')
    parser.add_argument('--mobility', type=float, default=1.0,
                        help='Client staying probability (1.0=static, 0.5=high mobility)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command-line arguments
    config['dataset']['name'] = args.dataset
    config['system']['eniid_setting'] = args.eniid
    config['mobility']['staying_prob'] = args.mobility
    config['experiment']['seed'] = args.seed
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['experiment']['log_dir'] = f"experiments/results/{args.dataset}_{args.eniid}_{timestamp}"
    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['experiment']['log_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize and run trainer
    trainer = CSAHFLTrainer(config)
    results = trainer.train()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Final Loss: {results['final_loss']:.4f}")
    print(f"Total Training Time: {results['total_time']:.2f}s")
    print(f"Results saved to: {config['experiment']['log_dir']}")
    
    return results


if __name__ == '__main__':
    main()
