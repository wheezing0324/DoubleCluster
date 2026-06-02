#!/usr/bin/env python3
"""
Hyperparameter Analysis Experiment Runner for CSAHFL

This script reproduces Figure 4 from the CSAHFL paper, analyzing the impact of:
- α (tolerance parameter): Controls straggler tolerance in semi-asynchronous aggregation
- E_max (maximum epochs): Controls local training computation
- β (clustering threshold): Controls cluster granularity in hierarchical clustering

Expected Results (Figure 4):
- Optimal α = 1.5 (balances straggler tolerance vs. staleness)
- Optimal E_max = 10 (balances local training vs. communication)
- Optimal β = 20 (balances cluster count vs. homogeneity)
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import CSAHFLTrainer, load_config
from src.utils.logger import ExperimentLogger
from src.utils.metrics import MetricsCalculator


class HyperparameterAnalysisRunner:
    """
    Orchestrates hyperparameter analysis experiments for CSAHFL.
    
    Performs systematic sweeps over α, E_max, and β to identify optimal values
    and reproduce Figure 4 from the paper.
    """
    
    def __init__(self, config: Dict, results_dir: str):
        """
        Initialize hyperparameter analysis runner.
        
        Args:
            config: Experiment configuration dictionary
            results_dir: Directory to save results and figures
        """
        self.config = config
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
        
        # Initialize logger
        self.logger = ExperimentLogger(
            log_dir=os.path.join(results_dir, "logs"),
            experiment_name="hyperparameter_analysis",
            config=config
        )
        
        # Storage for results
        self.alpha_results = {}
        self.emax_results = {}
        self.beta_results = {}
        self.all_results = []
        
        # Default sweep ranges from paper
        self.alpha_values = [0.5, 1.0, 1.5, 2.0]
        self.emax_values = [5, 10, 20, 30]
        self.beta_values = [0, 5, 10, 15, 20, 25]
        
        # Expected optimal values
        self.expected_optimal = {
            'alpha': 1.5,
            'emax': 10,
            'beta': 20
        }
        
    def run_alpha_sweep(self, dataset: str = 'fashion_mnist', 
                       eniid: str = 'ENIID50%',
                       mobility: float = 1.0,
                       seed: int = 42) -> Dict:
        """
        Sweep over α (tolerance parameter) values.
        
        Args:
            dataset: Dataset name
            eniid: Edge-layer non-IID setting
            mobility: Staying probability
            seed: Random seed
            
        Returns:
            Dictionary mapping α values to results
        """
        print(f"\n{'='*60}")
        print(f"ALPHA SWEEP (α ∈ {self.alpha_values})")
        print(f"Dataset: {dataset}, ENIID: {eniid}, Mobility: {mobility}")
        print(f"{'='*60}")
        
        results = {}
        
        for alpha in self.alpha_values:
            print(f"\n--- Testing α = {alpha} ---")
            
            # Create modified config
            exp_config = self.config.copy()
            exp_config['intra_cluster']['tolerance_alpha'] = alpha
            
            # Update experiment config
            if 'experiment' not in exp_config:
                exp_config['experiment'] = {}
            exp_config['experiment']['dataset'] = dataset
            exp_config['experiment']['eniid_level'] = eniid
            exp_config['experiment']['mobility_staying_prob'] = mobility
            exp_config['experiment']['seed'] = seed
            
            try:
                # Run experiment
                trainer = CSAHFLTrainer(exp_config)
                final_accuracy, training_time, history = trainer.train()
                
                results[alpha] = {
                    'accuracy': final_accuracy,
                    'training_time': training_time,
                    'convergence_round': self._find_convergence_round(history),
                    'history': history
                }
                
                print(f"  α={alpha}: Accuracy={final_accuracy:.2f}%, Time={training_time:.1f}s")
                
                # Log result
                self.logger.log_metric(f'alpha_sweep/accuracy_alpha_{alpha}', final_accuracy)
                self.logger.log_metric(f'alpha_sweep/time_alpha_{alpha}', training_time)
                
            except Exception as e:
                print(f"  Error with α={alpha}: {e}")
                results[alpha] = {
                    'accuracy': 0.0,
                    'training_time': 0.0,
                    'error': str(e)
                }
        
        self.alpha_results = results
        return results
    
    def run_emax_sweep(self, dataset: str = 'fashion_mnist',
                      eniid: str = 'ENIID50%',
                      mobility: float = 1.0,
                      seed: int = 42) -> Dict:
        """
        Sweep over E_max (maximum local epochs) values.
        
        Args:
            dataset: Dataset name
            eniid: Edge-layer non-IID setting
            mobility: Staying probability
            seed: Random seed
            
        Returns:
            Dictionary mapping E_max values to results
        """
        print(f"\n{'='*60}")
        print(f"EMAX SWEEP (E_max ∈ {self.emax_values})")
        print(f"Dataset: {dataset}, ENIID: {eniid}, Mobility: {mobility}")
        print(f"{'='*60}")
        
        results = {}
        
        for emax in self.emax_values:
            print(f"\n--- Testing E_max = {emax} ---")
            
            # Create modified config
            exp_config = self.config.copy()
            exp_config['intra_cluster']['max_epochs'] = emax
            
            # Update experiment config
            if 'experiment' not in exp_config:
                exp_config['experiment'] = {}
            exp_config['experiment']['dataset'] = dataset
            exp_config['experiment']['eniid_level'] = eniid
            exp_config['experiment']['mobility_staying_prob'] = mobility
            exp_config['experiment']['seed'] = seed
            
            try:
                # Run experiment
                trainer = CSAHFLTrainer(exp_config)
                final_accuracy, training_time, history = trainer.train()
                
                results[emax] = {
                    'accuracy': final_accuracy,
                    'training_time': training_time,
                    'convergence_round': self._find_convergence_round(history),
                    'history': history
                }
                
                print(f"  E_max={emax}: Accuracy={final_accuracy:.2f}%, Time={training_time:.1f}s")
                
                # Log result
                self.logger.log_metric(f'emax_sweep/accuracy_emax_{emax}', final_accuracy)
                self.logger.log_metric(f'emax_sweep/time_emax_{emax}', training_time)
                
            except Exception as e:
                print(f"  Error with E_max={emax}: {e}")
                results[emax] = {
                    'accuracy': 0.0,
                    'training_time': 0.0,
                    'error': str(e)
                }
        
        self.emax_results = results
        return results
    
    def run_beta_sweep(self, dataset: str = 'fashion_mnist',
                      eniid: str = 'ENIID50%',
                      mobility: float = 1.0,
                      seed: int = 42) -> Dict:
        """
        Sweep over β (clustering threshold) values.
        
        Args:
            dataset: Dataset name
            eniid: Edge-layer non-IID setting
            mobility: Staying probability
            seed: Random seed
            
        Returns:
            Dictionary mapping β values to results
        """
        print(f"\n{'='*60}")
        print(f"BETA SWEEP (β ∈ {self.beta_values})")
        print(f"Dataset: {dataset}, ENIID: {eniid}, Mobility: {mobility}")
        print(f"{'='*60}")
        
        results = {}
        
        for beta in self.beta_values:
            print(f"\n--- Testing β = {beta} ---")
            
            # Create modified config
            exp_config = self.config.copy()
            exp_config['clustering']['clustering_threshold'] = beta
            
            # Update experiment config
            if 'experiment' not in exp_config:
                exp_config['experiment'] = {}
            exp_config['experiment']['dataset'] = dataset
            exp_config['experiment']['eniid_level'] = eniid
            exp_config['experiment']['mobility_staying_prob'] = mobility
            exp_config['experiment']['seed'] = seed
            
            try:
                # Run experiment
                trainer = CSAHFLTrainer(exp_config)
                final_accuracy, training_time, history = trainer.train()
                
                # Count clusters formed
                num_clusters = len(set(trainer.cluster_assignments.values())) if hasattr(trainer, 'cluster_assignments') else 0
                
                results[beta] = {
                    'accuracy': final_accuracy,
                    'training_time': training_time,
                    'num_clusters': num_clusters,
                    'convergence_round': self._find_convergence_round(history),
                    'history': history
                }
                
                print(f"  β={beta}: Accuracy={final_accuracy:.2f}%, Clusters={num_clusters}, Time={training_time:.1f}s")
                
                # Log result
                self.logger.log_metric(f'beta_sweep/accuracy_beta_{beta}', final_accuracy)
                self.logger.log_metric(f'beta_sweep/time_beta_{beta}', training_time)
                self.logger.log_metric(f'beta_sweep/clusters_beta_{beta}', num_clusters)
                
            except Exception as e:
                print(f"  Error with β={beta}: {e}")
                results[beta] = {
                    'accuracy': 0.0,
                    'training_time': 0.0,
                    'num_clusters': 0,
                    'error': str(e)
                }
        
        self.beta_results = results
        return results
    
    def run_full_analysis(self, dataset: str = 'fashion_mnist',
                         eniid: str = 'ENIID50%',
                         mobility: float = 1.0,
                         seed: int = 42) -> Dict:
        """
        Run complete hyperparameter analysis (all three sweeps).
        
        Args:
            dataset: Dataset name
            eniid: Edge-layer non-IID setting
            mobility: Staying probability
            seed: Random seed
            
        Returns:
            Complete analysis results
        """
        print(f"\n{'#'*60}")
        print(f"COMPLETE HYPERPARAMETER ANALYSIS")
        print(f"Dataset: {dataset}, ENIID: {eniid}, Mobility: {mobility}, Seed: {seed}")
        print(f"{'#'*60}")
        
        # Run all sweeps
        self.run_alpha_sweep(dataset, eniid, mobility, seed)
        self.run_emax_sweep(dataset, eniid, mobility, seed)
        self.run_beta_sweep(dataset, eniid, mobility, seed)
        
        # Compile all results
        all_results = {
            'alpha_sweep': self.alpha_results,
            'emax_sweep': self.emax_results,
            'beta_sweep': self.beta_results,
            'optimal_values': self.find_optimal_values(),
            'expected_values': self.expected_optimal,
            'metadata': {
                'dataset': dataset,
                'eniid': eniid,
                'mobility': mobility,
                'seed': seed,
                'timestamp': self.timestamp
            }
        }
        
        self.all_results = all_results
        return all_results
    
    def find_optimal_values(self) -> Dict[str, float]:
        """
        Find optimal hyperparameter values from sweep results.
        
        Returns:
            Dictionary with optimal α, E_max, and β values
        """
        optimal = {}
        
        # Find optimal α (highest accuracy)
        if self.alpha_results:
            best_alpha = max(self.alpha_results.keys(),
                           key=lambda x: self.alpha_results[x].get('accuracy', 0))
            optimal['alpha'] = best_alpha
        
        # Find optimal E_max (best accuracy-time tradeoff)
        if self.emax_results:
            # Use accuracy as primary criterion
            best_emax = max(self.emax_results.keys(),
                          key=lambda x: self.emax_results[x].get('accuracy', 0))
            optimal['emax'] = best_emax
        
        # Find optimal β (highest accuracy)
        if self.beta_results:
            best_beta = max(self.beta_results.keys(),
                          key=lambda x: self.beta_results[x].get('accuracy', 0))
            optimal['beta'] = best_beta
        
        return optimal
    
    def _find_convergence_round(self, history: Dict, threshold: float = 0.6) -> int:
        """
        Find the round where model first reaches threshold accuracy.
        
        Args:
            history: Training history dictionary
            threshold: Accuracy threshold for convergence
            
        Returns:
            Round number where convergence occurred
        """
        if 'accuracy_history' not in history:
            return -1
        
        for i, acc in enumerate(history['accuracy_history']):
            if acc >= threshold:
                return i + 1
        return len(history.get('accuracy_history', []))
    
    def generate_figure4(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Figure 4: Hyperparameter analysis plots.
        
        Creates 4 subplots showing:
        (a) Accuracy vs α
        (b) Accuracy vs E_max
        (c) Accuracy vs β
        (d) Summary table of optimal values
        
        Args:
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        print("\nGenerating Figure 4: Hyperparameter Analysis...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot (a): Accuracy vs α
        ax1 = axes[0, 0]
        if self.alpha_results:
            alphas = list(self.alpha_results.keys())
            accuracies = [self.alpha_results[a].get('accuracy', 0) for a in alphas]
            times = [self.alpha_results[a].get('training_time', 0) for a in alphas]
            
            ax1_twin = ax1.twinx()
            
            # Accuracy line
            line1 = ax1.plot(alphas, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
            ax1.set_xlabel('α (Tolerance Parameter)', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_ylim([min(accuracies) - 2, max(accuracies) + 2])
            
            # Time line
            line2 = ax1_twin.plot(alphas, times, 'rs--', linewidth=2, markersize=8, label='Training Time')
            ax1_twin.set_ylabel('Training Time (s)', color='r', fontsize=12)
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
            # Mark optimal
            optimal_alpha = self.expected_optimal['alpha']
            ax1.axvline(x=optimal_alpha, color='g', linestyle=':', linewidth=2, 
                       label=f'Optimal α={optimal_alpha}')
            
            ax1.set_title('(a) Impact of α on Accuracy and Training Time', fontsize=14)
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
        
        # Plot (b): Accuracy vs E_max
        ax2 = axes[0, 1]
        if self.emax_results:
            emaxes = list(self.emax_results.keys())
            accuracies = [self.emax_results[e].get('accuracy', 0) for e in emaxes]
            times = [self.emax_results[e].get('training_time', 0) for e in emaxes]
            
            ax2_twin = ax2.twinx()
            
            # Accuracy line
            line1 = ax2.plot(emaxes, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
            ax2.set_xlabel('E_max (Maximum Local Epochs)', fontsize=12)
            ax2.set_ylabel('Accuracy (%)', color='b', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Time line
            line2 = ax2_twin.plot(emaxes, times, 'rs--', linewidth=2, markersize=8, label='Training Time')
            ax2_twin.set_ylabel('Training Time (s)', color='r', fontsize=12)
            ax2_twin.tick_params(axis='y', labelcolor='r')
            
            # Mark optimal
            optimal_emax = self.expected_optimal['emax']
            ax2.axvline(x=optimal_emax, color='g', linestyle=':', linewidth=2,
                       label=f'Optimal E_max={optimal_emax}')
            
            ax2.set_title('(b) Impact of E_max on Accuracy and Training Time', fontsize=14)
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        
        # Plot (c): Accuracy vs β
        ax3 = axes[1, 0]
        if self.beta_results:
            betas = list(self.beta_results.keys())
            accuracies = [self.beta_results[b].get('accuracy', 0) for b in betas]
            num_clusters = [self.beta_results[b].get('num_clusters', 0) for b in betas]
            
            ax3_twin = ax3.twinx()
            
            # Accuracy line
            line1 = ax3.plot(betas, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
            ax3.set_xlabel('β (Clustering Threshold)', fontsize=12)
            ax3.set_ylabel('Accuracy (%)', color='b', fontsize=12)
            ax3.tick_params(axis='y', labelcolor='b')
            
            # Number of clusters line
            line2 = ax3_twin.plot(betas, num_clusters, 'gs--', linewidth=2, markersize=8, label='# Clusters')
            ax3_twin.set_ylabel('Number of Clusters', color='g', fontsize=12)
            ax3_twin.tick_params(axis='y', labelcolor='g')
            
            # Mark optimal
            optimal_beta = self.expected_optimal['beta']
            ax3.axvline(x=optimal_beta, color='r', linestyle=':', linewidth=2,
                       label=f'Optimal β={optimal_beta}')
            
            ax3.set_title('(c) Impact of β on Accuracy and Cluster Count', fontsize=14)
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        
        # Plot (d): Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table
        optimal = self.find_optimal_values()
        table_data = [
            ['Hyperparameter', 'Tested Values', 'Optimal Value', 'Expected Value'],
            ['α (tolerance)', str(self.alpha_values), str(optimal.get('alpha', 'N/A')), str(self.expected_optimal['alpha'])],
            ['E_max (epochs)', str(self.emax_values), str(optimal.get('emax', 'N/A')), str(self.expected_optimal['emax'])],
            ['β (threshold)', str(self.beta_values), str(optimal.get('beta', 'N/A')), str(self.expected_optimal['beta'])]
        ]
        
        table = ax4.table(cellText=table_data[1:],
                         colLabels=table_data[0],
                         loc='center',
                         cellLoc='center',
                         colColours=['lightblue'] * 4)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)
        
        ax4.set_title('(d) Optimal Hyperparameter Summary', fontsize=14, pad=20)
        
        plt.suptitle('Figure 4: Hyperparameter Analysis for CSAHFL', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'figures', f'figure4_hyperparameter_{self.timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure 4 saved to: {save_path}")
        
        return fig
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of hyperparameter analysis results.
        
        Returns:
            Pandas DataFrame with summary results
        """
        print("\nGenerating summary table...")
        
        # Compile results into table format
        table_data = []
        
        # Alpha sweep results
        for alpha, result in self.alpha_results.items():
            table_data.append({
                'Hyperparameter': 'α',
                'Value': alpha,
                'Accuracy (%)': result.get('accuracy', 0),
                'Training Time (s)': result.get('training_time', 0),
                'Convergence Round': result.get('convergence_round', -1)
            })
        
        # Emax sweep results
        for emax, result in self.emax_results.items():
            table_data.append({
                'Hyperparameter': 'E_max',
                'Value': emax,
                'Accuracy (%)': result.get('accuracy', 0),
                'Training Time (s)': result.get('training_time', 0),
                'Convergence Round': result.get('convergence_round', -1)
            })
        
        # Beta sweep results
        for beta, result in self.beta_results.items():
            table_data.append({
                'Hyperparameter': 'β',
                'Value': beta,
                'Accuracy (%)': result.get('accuracy', 0),
                'Training Time (s)': result.get('training_time', 0),
                'Num Clusters': result.get('num_clusters', 0),
                'Convergence Round': result.get('convergence_round', -1)
            })
        
        df = pd.DataFrame(table_data)
        
        # Save to CSV
        csv_path = os.path.join(self.results_dir, 'tables', f'hyperparameter_summary_{self.timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Summary table saved to: {csv_path}")
        
        return df
    
    def save_results(self):
        """Save all results to JSON file."""
        results_path = os.path.join(self.results_dir, f'hyperparameter_results_{self.timestamp}.json')
        
        # Convert results to JSON-serializable format
        serializable_results = {
            'alpha_sweep': {str(k): v for k, v in self.alpha_results.items()},
            'emax_sweep': {str(k): v for k, v in self.emax_results.items()},
            'beta_sweep': {str(k): v for k, v in self.beta_results.items()},
            'optimal_values': self.find_optimal_values(),
            'expected_values': self.expected_optimal,
            'metadata': {
                'timestamp': self.timestamp,
                'alpha_values': self.alpha_values,
                'emax_values': self.emax_values,
                'beta_values': self.beta_values
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_path}")
    
    def validate_results(self) -> Dict[str, bool]:
        """
        Validate results against expected optimal values.
        
        Returns:
            Dictionary with validation results
        """
        print("\nValidating results against expected values...")
        
        optimal = self.find_optimal_values()
        validation = {}
        
        # Validate α
        if 'alpha' in optimal:
            validation['alpha_correct'] = (optimal['alpha'] == self.expected_optimal['alpha'])
            print(f"  α optimal: {optimal['alpha']} (expected: {self.expected_optimal['alpha']}) - "
                  f"{'✓' if validation['alpha_correct'] else '✗'}")
        
        # Validate E_max
        if 'emax' in optimal:
            validation['emax_correct'] = (optimal['emax'] == self.expected_optimal['emax'])
            print(f"  E_max optimal: {optimal['emax']} (expected: {self.expected_optimal['emax']}) - "
                  f"{'✓' if validation['emax_correct'] else '✗'}")
        
        # Validate β
        if 'beta' in optimal:
            validation['beta_correct'] = (optimal['beta'] == self.expected_optimal['beta'])
            print(f"  β optimal: {optimal['beta']} (expected: {self.expected_optimal['beta']}) - "
                  f"{'✓' if validation['beta_correct'] else '✗'}")
        
        # Overall validation
        validation['all_correct'] = all(validation.values())
        print(f"\nOverall validation: {'✓ PASSED' if validation['all_correct'] else '✗ FAILED'}")
        
        return validation
    
    def close(self):
        """Clean up resources."""
        self.logger.close()


def main():
    """Main entry point for hyperparameter analysis experiments."""
    parser = argparse.ArgumentParser(description='CSAHFL Hyperparameter Analysis')
    
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                       choices=['fashion_mnist', 'cifar10', 'svhn'],
                       help='Dataset to use')
    parser.add_argument('--eniid', type=str, default='ENIID50%',
                       choices=['EIID', 'ENIID50%', 'ENIID30%'],
                       help='Edge-layer non-IID setting')
    parser.add_argument('--mobility', type=float, default=1.0,
                       help='Staying probability (1.0=static, 0.5=high-mobility)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--config', type=str, 
                       default='config/experiment_config.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--results-dir', type=str, 
                       default='experiments/results/hyperparameter',
                       help='Directory to save results')
    parser.add_argument('--sweep', type=str, default='all',
                       choices=['alpha', 'emax', 'beta', 'all'],
                       help='Which hyperparameter to sweep')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with fewer values')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               args.config)
    
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        # Use default config if file not found
        config = {
            'system': {'num_clients': 200, 'num_edge_servers': 5, 'seed': args.seed},
            'clustering': {'svd_components': 10, 'clustering_threshold': 20},
            'intra_cluster': {'max_epochs': 10, 'tolerance_alpha': 1.5},
            'inter_cluster': {'kl_epsilon': 1e-10},
            'training': {'learning_rate': 0.01},
            'experiment': {
                'total_cloud_rounds': 10 if args.quick else 50,
                'edge_rounds_per_cloud': 3,
                'dataset': args.dataset,
                'eniid_level': args.eniid,
                'mobility_staying_prob': args.mobility
            },
            'model': {'dropout_rate': 0.5},
            'logging': {'level': 'INFO'}
        }
    
    # Quick mode: reduce sweep values
    if args.quick:
        print("\n[QUICK MODE] Using reduced sweep values for testing...")
        alpha_values = [1.0, 1.5]
        emax_values = [5, 10]
        beta_values = [10, 20]
    else:
        alpha_values = [0.5, 1.0, 1.5, 2.0]
        emax_values = [5, 10, 20, 30]
        beta_values = [0, 5, 10, 15, 20, 25]
    
    # Initialize runner
    runner = HyperparameterAnalysisRunner(config, args.results_dir)
    
    # Set sweep values
    runner.alpha_values = alpha_values
    runner.emax_values = emax_values
    runner.beta_values = beta_values
    
    try:
        # Run analysis
        if args.sweep == 'alpha':
            results = runner.run_alpha_sweep(args.dataset, args.eniid, args.mobility, args.seed)
        elif args.sweep == 'emax':
            results = runner.run_emax_sweep(args.dataset, args.eniid, args.mobility, args.seed)
        elif args.sweep == 'beta':
            results = runner.run_beta_sweep(args.dataset, args.eniid, args.mobility, args.seed)
        else:
            results = runner.run_full_analysis(args.dataset, args.eniid, args.mobility, args.seed)
        
        # Generate outputs
        runner.generate_figure4()
        runner.generate_summary_table()
        runner.save_results()
        
        # Validate
        validation = runner.validate_results()
        
        print(f"\n{'='*60}")
        print("HYPERPARAMETER ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Optimal α: {results['optimal_values'].get('alpha', 'N/A')}")
        print(f"Optimal E_max: {results['optimal_values'].get('emax', 'N/A')}")
        print(f"Optimal β: {results['optimal_values'].get('beta', 'N/A')}")
        print(f"Validation: {'PASSED' if validation.get('all_correct', False) else 'FAILED'}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.close()


if __name__ == '__main__':
    main()
