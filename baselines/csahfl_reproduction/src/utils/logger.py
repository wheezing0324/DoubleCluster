"""
Experiment logging utility for CSAHFL framework.

Provides comprehensive logging functionality for tracking training progress,
metrics, and experiment results across all CSAHFL components.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import yaml


class ExperimentLogger:
    """
    Logger for CSAHFL experiments.
    
    Handles logging of training metrics, configuration, and results
    to both console and file outputs.
    
    Attributes:
        log_dir: Directory path for log files
        logger: Python logging instance
        metrics_history: Dictionary storing metrics over time
        config: Experiment configuration
    """
    
    def __init__(
        self,
        log_dir: str = "experiments/results/logs",
        experiment_name: str = "csahfl_experiment",
        config: Optional[Dict] = None,
        level: str = "INFO"
    ):
        """
        Initialize the experiment logger.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name of the experiment for file naming
            config: Optional configuration dictionary to log
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = config or {}
        
        # Initialize metrics storage
        self.metrics_history: Dict[str, List[float]] = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_loss': [],
            'test_loss': [],
            'training_time': [],
            'communication_round': []
        }
        
        # Setup logging
        self._setup_logging(level)
        
        # Log configuration if provided
        if self.config:
            self._log_config()
    
    def _setup_logging(self, level: str):
        """
        Configure Python logging with console and file handlers.
        
        Args:
            level: Logging level string
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(f"CSAHFL_{self.experiment_name}")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def _log_config(self):
        """Log the experiment configuration to file."""
        config_file = self.log_dir / f"config_{self.timestamp}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_file}")
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        round_num: Optional[int] = None
    ):
        """
        Log a single metric value.
        
        Args:
            metric_name: Name of the metric (e.g., 'accuracy', 'loss')
            value: Metric value to log
            round_num: Optional communication round number
        """
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append(value)
        
        if round_num is not None:
            self.logger.info(
                f"Round {round_num}: {metric_name} = {value:.4f}"
            )
        else:
            self.logger.info(f"{metric_name} = {value:.4f}")
    
    def log_round_metrics(
        self,
        round_num: int,
        metrics: Dict[str, float],
        log_to_console: bool = True
    ):
        """
        Log multiple metrics for a single training round.
        
        Args:
            round_num: Communication round number
            metrics: Dictionary of metric names to values
            log_to_console: Whether to print to console
        """
        # Store metrics
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
        
        # Log summary
        if log_to_console:
            metric_str = ", ".join([
                f"{k}={v:.4f}" for k, v in metrics.items()
            ])
            self.logger.info(f"Round {round_num}: {metric_str}")
    
    def log_message(self, message: str, level: str = "INFO"):
        """
        Log a custom message.
        
        Args:
            message: Message to log
            level: Logging level
        """
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
    
    def log_cluster_info(
        self,
        cluster_assignments: Dict[int, List[int]],
        round_num: int
    ):
        """
        Log clustering information.
        
        Args:
            cluster_assignments: Dictionary mapping cluster IDs to client IDs
            round_num: Current round number
        """
        num_clusters = len(cluster_assignments)
        cluster_sizes = [len(clients) for clients in cluster_assignments.values()]
        
        self.logger.info(
            f"Round {round_num}: Clustering - {num_clusters} clusters, "
            f"sizes: {cluster_sizes}"
        )
    
    def log_aggregation_info(
        self,
        aggregation_type: str,
        num_participants: int,
        round_num: int,
        additional_info: Optional[Dict] = None
    ):
        """
        Log aggregation information.
        
        Args:
            aggregation_type: Type of aggregation (intra/inter cluster)
            num_participants: Number of clients/servers participating
            round_num: Current round number
            additional_info: Optional additional information
        """
        msg = (
            f"Round {round_num}: {aggregation_type} aggregation - "
            f"{num_participants} participants"
        )
        if additional_info:
            msg += f", info: {additional_info}"
        self.logger.info(msg)
    
    def log_mobility_event(
        self,
        num_moved_clients: int,
        staying_probability: float,
        round_num: int
    ):
        """
        Log client mobility events.
        
        Args:
            num_moved_clients: Number of clients that moved
            staying_probability: Probability clients stayed (ps)
            round_num: Current round number
        """
        self.logger.info(
            f"Round {round_num}: Mobility - {num_moved_clients} clients moved, "
            f"ps={staying_probability}"
        )
    
    def save_metrics(self, filename: Optional[str] = None):
        """
        Save all metrics to a JSON file.
        
        Args:
            filename: Optional custom filename (default: auto-generated)
        """
        if filename is None:
            filename = f"metrics_{self.timestamp}.json"
        
        metrics_file = self.log_dir / filename
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        return metrics_file
    
    def save_results_table(
        self,
        results: Dict[str, Any],
        table_name: str = "results"
    ):
        """
        Save results in table format (JSON/CSV compatible).
        
        Args:
            results: Dictionary of results to save
            table_name: Name for the results table
        """
        results_file = self.log_dir / f"{table_name}_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results table saved to {results_file}")
        return results_file
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """
        Get the complete metrics history.
        
        Returns:
            Dictionary mapping metric names to lists of values
        """
        return self.metrics_history.copy()
    
    def get_metric_average(self, metric_name: str) -> Optional[float]:
        """
        Calculate average of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Average value or None if metric doesn't exist
        """
        if metric_name not in self.metrics_history:
            return None
        values = self.metrics_history[metric_name]
        if len(values) == 0:
            return None
        return sum(values) / len(values)
    
    def get_metric_latest(self, metric_name: str) -> Optional[float]:
        """
        Get the latest value of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Latest value or None if metric doesn't exist
        """
        if metric_name not in self.metrics_history:
            return None
        values = self.metrics_history[metric_name]
        if len(values) == 0:
            return None
        return values[-1]
    
    def close(self):
        """Close all logging handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        self.logger.info("Logger closed")


def create_logger(
    experiment_name: str,
    config: Optional[Dict] = None,
    log_dir: str = "experiments/results/logs"
) -> ExperimentLogger:
    """
    Factory function to create an experiment logger.
    
    Args:
        experiment_name: Name of the experiment
        config: Optional configuration dictionary
        log_dir: Directory for log files
        
    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        config=config
    )
