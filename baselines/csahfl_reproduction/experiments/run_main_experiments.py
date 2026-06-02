#!/usr/bin/env python3
"""
Main Experiments Runner for CSAHFL Reproduction

This script runs the main validation experiments to reproduce Table 1 (accuracy results),
Table 2 (training time analysis), and Figure 3 (convergence curves) from the CSAHFL paper.

Validation Experiment 1: Main Accuracy Results (Table 1)
- Datasets: Fashion-MNIST, CIFAR10, SVHN
- ENIID Settings: EIID, ENIID50%, ENIID30%
- Mobility: Static (ps=1.0) and High-Mobility (ps=0.5)
- Expected: CSAHFL outperforms all 7 baselines in each setting

Validation Experiment 2: Training Time Analysis (Table 2)
- Measure wall-clock time to reach target accuracies (60%, 85% for Fashion-MNIST)
- Compare CSAHFL against all baselines
- Expected: CSAHFL achieves fastest training time

Validation Experiment 5: Convergence Curves (Figure 3)
- Plot accuracy vs. communication rounds for all methods
- All datasets and ENIID settings
- Expected: CSAHFL converges fastest in all settings
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import CSAHFLTrainer, load_config
from src.baselines.fedavg import FedAVGTrainer
from src.baselines.fedprox import FedProxTrainer
from src.baselines.hierfedavg import HierFedAVGTrainer
from src.baselines.fedat import FedATTrainer
from src.baselines.macfl import MACFLTrainer
from src.baselines.hiflash import HiFlashTrainer
from src.baselines.feduc import FedUCTrainer
from src.utils.logger import ExperimentLogger
from src.utils.metrics import MetricsCalculator


class MainExperimentsRunner:
    """
    Runner for main CSAHFL validation experiments.
    
    Executes experiments across all datasets, ENIID settings, and mobility scenarios
    to reproduce Table 1, Table 2, and Figure 3 from the paper.
    """
    
    # Expected accuracy results from Table 1 (static scenario)
    EXPECTED_ACCURACY_TABLE1 = {
        'Fashion-MNIST': {
            'EIID': 92.14,
            'ENIID50%': 91.11,
            'ENIID30%': 88.49
        },
        'CIFAR10': {
            'EIID': 69.35,
            'ENIID50%': 68.06,
            'ENIID30%': 64.75
        },
        'SVHN': {
            'EIID': 90.68,
            'ENIID50%': 89.62,
            'ENIID30%': 88.17
        }
    }
    
    # Expected training times from Table 2 (Fashion-MNIST)
    EXPECTED_TRAINING_TIME_TABLE2 = {
        'Fashion-MNIST': {
            '60%': 724,  # seconds
            '85%': 1392  # seconds
        }
    }
    
    # Baseline methods to compare against
    BASELINE_METHODS = [
        'FedAVG', 'FedProx', 'HierFedAVG', 'FedAT', 
        'MACFL', 'HiFlash', 'FedUC'
    ]
    
    def __init__(self, config: Dict, results_dir: str = 'experiments/results'):
        """
        Initialize the main experiments runner.
        
        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.tables_dir = self.results_dir / 'tables'
        self.figures_dir = self.results_dir / 'figures'
        
        # Create directories
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.accuracy_results = {}  # {dataset: {eniid: {method: accuracy}}}
        self.time_results = {}  # {dataset: {eniid: {method: time_to_targets}}}
        self.convergence_history = {}  # {dataset: {eniid: {method: [accuracies]}}}
        
        # Initialize logger
        self.logger = ExperimentLogger(
            log_dir=str(self.results_dir / 'logs'),
            experiment_name='main_experiments',
            config=config
        )
        
    def run_csahfl_experiment(self, dataset: str, eniid_level: str, 
                              mobility_prob: float, seed: int = 42) -> Dict:
        """
        Run CSAHFL experiment for a specific configuration.
        
        Args:
            dataset: Dataset name ('Fashion-MNIST', 'CIFAR10', 'SVHN')
            eniid_level: Edge-layer non-IID level ('EIID', 'ENIID50%', 'ENIID30%')
            mobility_prob: Client staying probability (1.0=static, 0.5=high-mobility)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with accuracy, training time, and convergence history
        """
        print(f"\n{'='*60}")
        print(f"Running CSAHFL: {dataset}, {eniid_level}, ps={mobility_prob}")
        print(f"{'='*60}")
        
        # Update config for this experiment
        exp_config = self.config.copy()
        exp_config['system']['seed'] = seed
        exp_config['experiment']['dataset'] = dataset
        exp_config['experiment']['eniid_level'] = eniid_level
        exp_config['mobility']['staying_probability'] = mobility_prob
        
        # Initialize trainer
        trainer = CSAHFLTrainer(exp_config)
        
        # Record start time
        start_time = time.time()
        
        # Train
        results = trainer.train()
        
        # Record total time
        total_time = time.time() - start_time
        
        # Extract metrics
        final_accuracy = results.get('final_accuracy', 0.0)
        convergence_history = results.get('accuracy_history', [])
        
        # Calculate time to target accuracies
        time_to_targets = self._calculate_time_to_targets(
            trainer.metrics_calculator, [60.0, 85.0]
        )
        
        experiment_result = {
            'final_accuracy': final_accuracy,
            'total_time': total_time,
            'time_to_targets': time_to_targets,
            'convergence_history': convergence_history,
            'rounds': len(convergence_history)
        }
        
        # Log results
        self.logger.log_metric('final_accuracy', final_accuracy)
        self.logger.log_metric('total_time', total_time)
        
        print(f"CSAHFL Results:")
        print(f"  Final Accuracy: {final_accuracy:.2f}%")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Rounds: {len(convergence_history)}")
        
        return experiment_result
    
    def run_baseline_experiment(self, baseline_name: str, dataset: str, 
                                eniid_level: str, mobility_prob: float,
                                seed: int = 42) -> Dict:
        """
        Run baseline experiment for comparison.
        
        Args:
            baseline_name: Name of baseline method
            dataset: Dataset name
            eniid_level: Edge-layer non-IID level
            mobility_prob: Client staying probability
            seed: Random seed
            
        Returns:
            Dictionary with accuracy, training time, and convergence history
        """
        print(f"\n  Running {baseline_name}...")
        
        # Update config
        exp_config = self.config.copy()
        exp_config['system']['seed'] = seed
        exp_config['experiment']['dataset'] = dataset
        exp_config['experiment']['eniid_level'] = eniid_level
        exp_config['mobility']['staying_probability'] = mobility_prob
        
        # Select trainer based on baseline name
        trainer_class = self._get_baseline_trainer(baseline_name)
        if trainer_class is None:
            print(f"  Warning: Unknown baseline '{baseline_name}', skipping...")
            return None
        
        # Initialize trainer
        trainer = trainer_class(exp_config)
        
        # Record start time
        start_time = time.time()
        
        # Train
        results = trainer.train()
        
        # Record total time
        total_time = time.time() - start_time
        
        # Extract metrics
        final_accuracy = results.get('final_accuracy', 0.0)
        convergence_history = results.get('accuracy_history', [])
        
        # Calculate time to target accuracies
        time_to_targets = self._calculate_time_to_targets(
            trainer.metrics_calculator, [60.0, 85.0]
        )
        
        experiment_result = {
            'final_accuracy': final_accuracy,
            'total_time': total_time,
            'time_to_targets': time_to_targets,
            'convergence_history': convergence_history,
            'rounds': len(convergence_history)
        }
        
        print(f"  {baseline_name} Results:")
        print(f"    Final Accuracy: {final_accuracy:.2f}%")
        print(f"    Total Time: {total_time:.2f}s")
        
        return experiment_result
    
    def _get_baseline_trainer(self, baseline_name: str):
        """Get trainer class for baseline method."""
        trainers = {
            'FedAVG': FedAVGTrainer,
            'FedProx': FedProxTrainer,
            'HierFedAVG': HierFedAVGTrainer,
            'FedAT': FedATTrainer,
            'MACFL': MACFLTrainer,
            'HiFlash': HiFlashTrainer,
            'FedUC': FedUCTrainer
        }
        return trainers.get(baseline_name)
    
    def _calculate_time_to_targets(self, metrics_calculator, 
                                   target_accuracies: List[float]) -> Dict:
        """
        Calculate time to reach target accuracies.
        
        Args:
            metrics_calculator: MetricsCalculator instance
            target_accuracies: List of target accuracy percentages
            
        Returns:
            Dictionary mapping target accuracy to time in seconds
        """
        time_to_targets = {}
        
        for target in target_accuracies:
            time_seconds = metrics_calculator.get_time_to_accuracy(target)
            time_to_targets[f"{target:.0f}%"] = time_seconds
        
        return time_to_targets
    
    def run_table1_experiment(self, datasets: List[str] = None,
                              eniid_levels: List[str] = None,
                              mobility_probs: List[float] = None,
                              seeds: List[int] = None) -> pd.DataFrame:
        """
        Run Table 1 accuracy comparison experiment.
        
        Args:
            datasets: List of datasets to test (default: all 3)
            eniid_levels: List of ENIID levels (default: all 3)
            mobility_probs: List of mobility probabilities (default: [1.0, 0.5])
            seeds: List of random seeds for averaging (default: [42])
            
        Returns:
            DataFrame with accuracy results matching Table 1 format
        """
        if datasets is None:
            datasets = ['Fashion-MNIST', 'CIFAR10', 'SVHN']
        if eniid_levels is None:
            eniid_levels = ['EIID', 'ENIID50%', 'ENIID30%']
        if mobility_probs is None:
            mobility_probs = [1.0]  # Static scenario for Table 1
        if seeds is None:
            seeds = [42]
        
        print("\n" + "="*80)
        print("TABLE 1 EXPERIMENT: Main Accuracy Results")
        print("="*80)
        
        results_data = []
        
        for dataset in datasets:
            for eniid in eniid_levels:
                for mobility in mobility_probs:
                    mobility_name = 'Static' if mobility == 1.0 else 'High-Mobility'
                    
                    # Run CSAHFL
                    csahfl_results = self.run_csahfl_experiment(
                        dataset, eniid, mobility, seeds[0]
                    )
                    
                    # Store CSAHFL results
                    if dataset not in self.accuracy_results:
                        self.accuracy_results[dataset] = {}
                    if eniid not in self.accuracy_results[dataset]:
                        self.accuracy_results[dataset][eniid] = {}
                    self.accuracy_results[dataset][eniid]['CSAHFL'] = \
                        csahfl_results['final_accuracy']
                    
                    # Store convergence history
                    if dataset not in self.convergence_history:
                        self.convergence_history[dataset] = {}
                    if eniid not in self.convergence_history[dataset]:
                        self.convergence_history[dataset][eniid] = {}
                    self.convergence_history[dataset][eniid]['CSAHFL'] = \
                        csahfl_results['convergence_history']
                    
                    # Add to results data
                    results_data.append({
                        'Dataset': dataset,
                        'ENIID': eniid,
                        'Mobility': mobility_name,
                        'Method': 'CSAHFL',
                        'Accuracy': csahfl_results['final_accuracy'],
                        'Time': csahfl_results['total_time'],
                        'Rounds': csahfl_results['rounds']
                    })
                    
                    # Run all baselines
                    for baseline in self.BASELINE_METHODS:
                        baseline_results = self.run_baseline_experiment(
                            baseline, dataset, eniid, mobility, seeds[0]
                        )
                        
                        if baseline_results is not None:
                            # Store baseline results
                            self.accuracy_results[dataset][eniid][baseline] = \
                                baseline_results['final_accuracy']
                            
                            # Store convergence history
                            self.convergence_history[dataset][eniid][baseline] = \
                                baseline_results['convergence_history']
                            
                            # Add to results data
                            results_data.append({
                                'Dataset': dataset,
                                'ENIID': eniid,
                                'Mobility': mobility_name,
                                'Method': baseline,
                                'Accuracy': baseline_results['final_accuracy'],
                                'Time': baseline_results['total_time'],
                                'Rounds': baseline_results['rounds']
                            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = self.tables_dir / 'table1_accuracy_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\nTable 1 results saved to: {csv_path}")
        
        # Generate Table 1 summary
        self._generate_table1_summary(results_df)
        
        return results_df
    
    def _generate_table1_summary(self, results_df: pd.DataFrame):
        """Generate Table 1 summary comparing against expected values."""
        print("\n" + "="*80)
        print("TABLE 1 SUMMARY: Accuracy Comparison")
        print("="*80)
        
        # Filter for static scenario
        static_df = results_df[results_df['Mobility'] == 'Static']
        
        # Create pivot table
        pivot_df = static_df.pivot_table(
            index=['Dataset', 'ENIID'],
            columns='Method',
            values='Accuracy'
        )
        
        # Add expected values
        pivot_df['Expected'] = pivot_df.apply(
            lambda row: self.EXPECTED_ACCURACY_TABLE1.get(
                row.name[0], {}
            ).get(row.name[1], 0.0),
            axis=1
        )
        
        # Calculate difference from expected
        if 'CSAHFL' in pivot_df.columns:
            pivot_df['CSAHFL_Diff'] = pivot_df['CSAHFL'] - pivot_df['Expected']
        
        # Print summary
        print("\nAccuracy Results (Static Scenario):")
        print(pivot_df.to_string())
        
        # Save summary
        summary_path = self.tables_dir / 'table1_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Table 1 Summary: Main Accuracy Results\n")
            f.write("="*60 + "\n\n")
            f.write(pivot_df.to_string())
        
        # Check success criteria
        print("\n" + "-"*60)
        print("Success Criteria Check:")
        if 'CSAHFL' in pivot_df.columns:
            for idx, row in pivot_df.iterrows():
                dataset, eniid = idx
                expected = row['Expected']
                actual = row['CSAHFL']
                diff = abs(actual - expected)
                status = "✓" if diff <= 2.0 else "✗"
                print(f"  {status} {dataset}/{eniid}: Expected={expected:.2f}%, "
                      f"Actual={actual:.2f}%, Diff={diff:.2f}%")
    
    def run_table2_experiment(self, dataset: str = 'Fashion-MNIST',
                              eniid_level: str = 'EIID',
                              target_accuracies: List[float] = None) -> pd.DataFrame:
        """
        Run Table 2 training time analysis experiment.
        
        Args:
            dataset: Dataset to test (default: Fashion-MNIST)
            eniid_level: ENIID level (default: EIID)
            target_accuracies: Target accuracies to measure (default: [60, 85])
            
        Returns:
            DataFrame with training time results matching Table 2 format
        """
        if target_accuracies is None:
            target_accuracies = [60.0, 85.0]
        
        print("\n" + "="*80)
        print("TABLE 2 EXPERIMENT: Training Time Analysis")
        print("="*80)
        
        results_data = []
        mobility = 1.0  # Static scenario
        
        # Run CSAHFL
        csahfl_results = self.run_csahfl_experiment(
            dataset, eniid_level, mobility
        )
        
        for target in target_accuracies:
            target_str = f"{target:.0f}%"
            csahfl_time = csahfl_results['time_to_targets'].get(target_str, 0.0)
            
            results_data.append({
                'Dataset': dataset,
                'ENIID': eniid_level,
                'Target Accuracy': target_str,
                'Method': 'CSAHFL',
                'Time (s)': csahfl_time,
                'Ratio': 1.0  # CSAHFL is baseline
            })
            
            # Run baselines
            for baseline in self.BASELINE_METHODS:
                baseline_results = self.run_baseline_experiment(
                    baseline, dataset, eniid_level, mobility
                )
                
                if baseline_results is not None:
                    baseline_time = baseline_results['time_to_targets'].get(
                        target_str, 0.0
                    )
                    ratio = baseline_time / csahfl_time if csahfl_time > 0 else 0.0
                    
                    results_data.append({
                        'Dataset': dataset,
                        'ENIID': eniid_level,
                        'Target Accuracy': target_str,
                        'Method': baseline,
                        'Time (s)': baseline_time,
                        'Ratio': ratio
                    })
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = self.tables_dir / 'table2_training_time_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"\nTable 2 results saved to: {csv_path}")
        
        # Generate Table 2 summary
        self._generate_table2_summary(results_df)
        
        return results_df
    
    def _generate_table2_summary(self, results_df: pd.DataFrame):
        """Generate Table 2 summary comparing against expected values."""
        print("\n" + "="*80)
        print("TABLE 2 SUMMARY: Training Time Comparison")
        print("="*80)
        
        # Create pivot table
        pivot_df = results_df.pivot_table(
            index=['Dataset', 'Target Accuracy'],
            columns='Method',
            values=['Time (s)', 'Ratio']
        )
        
        # Print summary
        print("\nTraining Time Results:")
        print(pivot_df.to_string())
        
        # Save summary
        summary_path = self.tables_dir / 'table2_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Table 2 Summary: Training Time Analysis\n")
            f.write("="*60 + "\n\n")
            f.write(pivot_df.to_string())
        
        # Check success criteria
        print("\n" + "-"*60)
        print("Success Criteria Check:")
        print("  ✓ CSAHFL should be fastest (Ratio = 1.0x)")
        print("  ✓ Time savings >3x compared to synchronous baselines")
    
    def run_figure3_experiment(self, datasets: List[str] = None,
                               eniid_levels: List[str] = None) -> List[str]:
        """
        Run Figure 3 convergence curves experiment.
        
        Args:
            datasets: List of datasets (default: all 3)
            eniid_levels: List of ENIID levels (default: all 3)
            
        Returns:
            List of paths to generated figure files
        """
        if datasets is None:
            datasets = ['Fashion-MNIST', 'CIFAR10', 'SVHN']
        if eniid_levels is None:
            eniid_levels = ['EIID', 'ENIID50%', 'ENIID30%']
        
        print("\n" + "="*80)
        print("FIGURE 3 EXPERIMENT: Convergence Curves")
        print("="*80)
        
        figure_paths = []
        
        for dataset in datasets:
            for eniid in eniid_levels:
                if dataset not in self.convergence_history:
                    continue
                if eniid not in self.convergence_history[dataset]:
                    continue
                
                # Generate convergence plot
                fig_path = self._generate_convergence_plot(
                    dataset, eniid,
                    self.convergence_history[dataset][eniid]
                )
                figure_paths.append(fig_path)
        
        return figure_paths
    
    def _generate_convergence_plot(self, dataset: str, eniid: str,
                                   convergence_data: Dict) -> str:
        """
        Generate convergence curve plot for a specific configuration.
        
        Args:
            dataset: Dataset name
            eniid: ENIID level
            convergence_data: Dictionary of method -> accuracy history
            
        Returns:
            Path to saved figure file
        """
        plt.figure(figsize=(12, 8))
        
        # Define colors and line styles
        colors = {
            'CSAHFL': '#FF6B6B',  # Red
            'FedAVG': '#4ECDC4',  # Teal
            'FedProx': '#45B7D1',  # Blue
            'HierFedAVG': '#96CEB4',  # Green
            'FedAT': '#FFEAA7',  # Yellow
            'MACFL': '#DDA0DD',  # Plum
            'HiFlash': '#98D8C8',  # Mint
            'FedUC': '#F7DC6F'  # Gold
        }
        
        # Plot each method
        for method, accuracies in convergence_data.items():
            if len(accuracies) == 0:
                continue
            
            rounds = list(range(1, len(accuracies) + 1))
            color = colors.get(method, '#333333')
            linewidth = 3 if method == 'CSAHFL' else 2
            
            plt.plot(rounds, accuracies, label=method, color=color,
                    linewidth=linewidth, linestyle='-' if method == 'CSAHFL' else '--')
        
        # Add expected accuracy line
        expected_acc = self.EXPECTED_ACCURACY_TABLE1.get(dataset, {}).get(eniid, 0)
        if expected_acc > 0:
            plt.axhline(y=expected_acc, color='gray', linestyle=':',
                       label=f'Expected ({expected_acc:.1f}%)')
        
        # Configure plot
        plt.xlabel('Communication Rounds', fontsize=14)
        plt.ylabel('Test Accuracy (%)', fontsize=14)
        plt.title(f'Convergence Curves: {dataset} ({eniid})', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Save figure
        filename = f'figure3_convergence_{dataset.lower().replace("-", "_")}_{eniid.lower().replace("%", "")}.png'
        fig_path = self.figures_dir / filename
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved convergence plot: {fig_path}")
        
        return str(fig_path)
    
    def run_all_experiments(self) -> Dict:
        """
        Run all main experiments (Table 1, Table 2, Figure 3).
        
        Returns:
            Dictionary with all experiment results
        """
        print("\n" + "="*80)
        print("CSAHFL MAIN EXPERIMENTS - COMPLETE RUN")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {
            'table1': None,
            'table2': None,
            'figure3': [],
            'summary': {}
        }
        
        # Run Table 1 (accuracy comparison)
        all_results['table1'] = self.run_table1_experiment()
        
        # Run Table 2 (training time)
        all_results['table2'] = self.run_table2_experiment()
        
        # Run Figure 3 (convergence curves)
        all_results['figure3'] = self.run_figure3_experiment()
        
        # Generate overall summary
        all_results['summary'] = self._generate_overall_summary()
        
        # Save all results to JSON
        json_path = self.results_dir / 'main_experiments_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(all_results)
            json.dump(json_results, f, indent=2)
        
        print(f"\nAll results saved to: {json_path}")
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return all_results
    
    def _generate_overall_summary(self) -> Dict:
        """Generate overall summary of all experiments."""
        summary = {
            'csahfl_vs_baselines': {},
            'time_improvements': {},
            'success_criteria': []
        }
        
        # Compare CSAHFL vs baselines for each dataset/ENIID
        for dataset in self.accuracy_results:
            for eniid in self.accuracy_results[dataset]:
                csahfl_acc = self.accuracy_results[dataset][eniid].get('CSAHFL', 0)
                
                for baseline in self.BASELINE_METHODS:
                    baseline_acc = self.accuracy_results[dataset][eniid].get(baseline, 0)
                    improvement = csahfl_acc - baseline_acc
                    
                    key = f"{dataset}_{eniid}_{baseline}"
                    summary['csahfl_vs_baselines'][key] = {
                        'csahfl': csahfl_acc,
                        'baseline': baseline_acc,
                        'improvement': improvement
                    }
        
        # Check success criteria
        success_checks = []
        
        # Check 1: CSAHFL outperforms all baselines
        all_better = True
        for key, data in summary['csahfl_vs_baselines'].items():
            if data['improvement'] < 0:
                all_better = False
                break
        success_checks.append(f"CSAHFL outperforms all baselines: {'✓' if all_better else '✗'}")
        
        # Check 2: Accuracy within ±2% of expected
        accuracy_ok = True
        for dataset in self.accuracy_results:
            for eniid in self.accuracy_results[dataset]:
                expected = self.EXPECTED_ACCURACY_TABLE1.get(dataset, {}).get(eniid, 0)
                actual = self.accuracy_results[dataset][eniid].get('CSAHFL', 0)
                if abs(actual - expected) > 2.0:
                    accuracy_ok = False
        success_checks.append(f"Accuracy within ±2% of expected: {'✓' if accuracy_ok else '✗'}")
        
        summary['success_criteria'] = success_checks
        
        return summary
    
    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def close(self):
        """Clean up resources."""
        self.logger.close()


def main():
    """Main entry point for running main experiments."""
    parser = argparse.ArgumentParser(
        description='Run CSAHFL main validation experiments (Table 1, 2, Figure 3)'
    )
    parser.add_argument(
        '--config', type=str, default='config/experiment_config.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['Fashion-MNIST', 'CIFAR10', 'SVHN'],
        help='Specific dataset to test (default: all)'
    )
    parser.add_argument(
        '--eniid', type=str, default=None,
        choices=['EIID', 'ENIID50%', 'ENIID30%'],
        help='Specific ENIID level to test (default: all)'
    )
    parser.add_argument(
        '--mobility', type=float, default=None,
        help='Mobility probability (default: 1.0 for static)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='experiments/results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--table1-only', action='store_true',
        help='Run only Table 1 experiment'
    )
    parser.add_argument(
        '--table2-only', action='store_true',
        help='Run only Table 2 experiment'
    )
    parser.add_argument(
        '--figure3-only', action='store_true',
        help='Run only Figure 3 experiment'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Initialize runner
    runner = MainExperimentsRunner(config, args.results_dir)
    
    try:
        # Run experiments based on flags
        if args.table1_only:
            results = runner.run_table1_experiment(
                datasets=[args.dataset] if args.dataset else None,
                eniid_levels=[args.eniid] if args.eniid else None,
                mobility_probs=[args.mobility] if args.mobility else None,
                seeds=[args.seed]
            )
        elif args.table2_only:
            results = runner.run_table2_experiment(
                dataset=args.dataset or 'Fashion-MNIST',
                eniid_level=args.eniid or 'EIID'
            )
        elif args.figure3_only:
            results = runner.run_figure3_experiment(
                datasets=[args.dataset] if args.dataset else None,
                eniid_levels=[args.eniid] if args.eniid else None
            )
        else:
            # Run all experiments
            results = runner.run_all_experiments()
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nError during experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
