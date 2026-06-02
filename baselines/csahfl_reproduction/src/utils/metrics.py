"""
Metrics calculation utilities for CSAHFL framework.

This module provides comprehensive metrics calculation for:
- Model accuracy and loss evaluation
- Training time tracking and analysis
- Communication round statistics
- Convergence monitoring
- Performance comparison across methods

References:
    Section 4.2: Evaluation Metrics
    Table 1, 2: Accuracy and Training Time Results
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import DataLoader


class MetricsCalculator:
    """
    Calculates and tracks various metrics for CSAHFL experiments.
    
    Handles accuracy computation, training time tracking, and performance
    statistics collection across all framework components.
    
    Attributes:
        metrics_history: Dict storing metric values over rounds
        start_time: Training start timestamp for time tracking
        round_times: List of per-round training times
    """
    
    def __init__(self):
        """Initialize metrics calculator with empty history."""
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.start_time: Optional[float] = None
        self.round_times: List[float] = []
        self.current_round: int = 0
        self.round_start_time: Optional[float] = None
        
        # Aggregation statistics
        self.edge_aggregation_times: List[float] = []
        self.cloud_aggregation_times: List[float] = []
        
        # Client statistics
        self.client_training_times: Dict[int, List[float]] = defaultdict(list)
        self.client_participation_rates: Dict[int, float] = {}
        
        # Convergence tracking
        self.accuracy_history: List[float] = []
        self.loss_history: List[float] = []
        
    def start_training(self) -> None:
        """Record training start time for total time calculation."""
        self.start_time = time.time()
        self.round_times = []
        
    def start_round(self, round_num: int) -> None:
        """
        Start timing for a new training round.
        
        Args:
            round_num: Current global round number
        """
        self.current_round = round_num
        self.round_start_time = time.time()
        
    def end_round(self) -> float:
        """
        End timing for current round and record duration.
        
        Returns:
            Round duration in seconds
        """
        if self.round_start_time is None:
            return 0.0
            
        round_time = time.time() - self.round_start_time
        self.round_times.append(round_time)
        self.round_start_time = None
        return round_time
    
    def record_accuracy(self, accuracy: float, round_num: Optional[int] = None) -> None:
        """
        Record accuracy metric for a round.
        
        Args:
            accuracy: Accuracy value (0.0 to 1.0)
            round_num: Optional round number (uses current_round if None)
        """
        if round_num is None:
            round_num = self.current_round
        self.metrics_history['accuracy'].append(accuracy)
        self.accuracy_history.append(accuracy)
        
    def record_loss(self, loss: float, round_num: Optional[int] = None) -> None:
        """
        Record loss metric for a round.
        
        Args:
            loss: Loss value
            round_num: Optional round number (uses current_round if None)
        """
        if round_num is None:
            round_num = self.current_round
        self.metrics_history['loss'].append(loss)
        self.loss_history.append(loss)
        
    def record_edge_aggregation_time(self, time_seconds: float) -> None:
        """
        Record edge aggregation time.
        
        Args:
            time_seconds: Aggregation duration in seconds
        """
        self.edge_aggregation_times.append(time_seconds)
        
    def record_cloud_aggregation_time(self, time_seconds: float) -> None:
        """
        Record cloud aggregation time.
        
        Args:
            time_seconds: Aggregation duration in seconds
        """
        self.cloud_aggregation_times.append(time_seconds)
        
    def record_client_training_time(self, client_id: int, time_seconds: float) -> None:
        """
        Record individual client training time.
        
        Args:
            client_id: Client identifier
            time_seconds: Training duration in seconds
        """
        self.client_training_times[client_id].append(time_seconds)
        
    def calculate_accuracy(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: str = 'cpu'
    ) -> Tuple[float, float]:
        """
        Calculate model accuracy and loss on dataset.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader with test/validation data
            device: Device for computation ('cpu' or 'cuda')
            
        Returns:
            Tuple of (accuracy, average_loss)
        """
        model.eval()
        model.to(device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data = data.to(device)
                targets = targets.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        average_loss = total_loss / total
        
        return accuracy, average_loss
    
    def calculate_accuracy_from_arrays(
        self,
        model: torch.nn.Module,
        data: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        device: str = 'cpu'
    ) -> Tuple[float, float]:
        """
        Calculate accuracy from numpy arrays (for client-side evaluation).
        
        Args:
            model: PyTorch model to evaluate
            data: Numpy array of input data (N, H, W, C)
            labels: Numpy array of labels (N,)
            batch_size: Batch size for evaluation
            device: Device for computation
            
        Returns:
            Tuple of (accuracy, average_loss)
        """
        model.eval()
        model.to(device)
        
        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        # Transpose from NHWC to NCHW if needed
        if len(data_tensor.shape) == 4 and data_tensor.shape[3] in [1, 3]:
            data_tensor = data_tensor.permute(0, 3, 1, 2)
        
        dataset = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return self.calculate_accuracy(model, data_loader, device)
    
    def get_total_training_time(self) -> float:
        """
        Get total training time since start.
        
        Returns:
            Total time in seconds, or 0.0 if not started
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_average_round_time(self) -> float:
        """
        Get average round duration.
        
        Returns:
            Average round time in seconds
        """
        if not self.round_times:
            return 0.0
        return np.mean(self.round_times)
    
    def get_average_edge_aggregation_time(self) -> float:
        """
        Get average edge aggregation time.
        
        Returns:
            Average time in seconds
        """
        if not self.edge_aggregation_times:
            return 0.0
        return np.mean(self.edge_aggregation_times)
    
    def get_average_cloud_aggregation_time(self) -> float:
        """
        Get average cloud aggregation time.
        
        Returns:
            Average time in seconds
        """
        if not self.cloud_aggregation_times:
            return 0.0
        return np.mean(self.cloud_aggregation_times)
    
    def get_average_client_training_time(self, client_id: Optional[int] = None) -> float:
        """
        Get average client training time.
        
        Args:
            client_id: Specific client ID, or None for all clients
            
        Returns:
            Average training time in seconds
        """
        if client_id is not None:
            times = self.client_training_times.get(client_id, [])
            if not times:
                return 0.0
            return np.mean(times)
        
        # Average across all clients
        all_times = []
        for times in self.client_training_times.values():
            all_times.extend(times)
        
        if not all_times:
            return 0.0
        return np.mean(all_times)
    
    def get_client_participation_rate(self, client_id: int, total_rounds: int) -> float:
        """
        Calculate client participation rate.
        
        Args:
            client_id: Client identifier
            total_rounds: Total number of rounds
            
        Returns:
            Participation rate (0.0 to 1.0)
        """
        if total_rounds == 0:
            return 0.0
        
        rounds_participated = len(self.client_training_times.get(client_id, []))
        return rounds_participated / total_rounds
    
    def get_best_accuracy(self) -> float:
        """
        Get best accuracy achieved during training.
        
        Returns:
            Maximum accuracy value
        """
        if not self.accuracy_history:
            return 0.0
        return max(self.accuracy_history)
    
    def get_latest_accuracy(self) -> float:
        """
        Get latest accuracy value.
        
        Returns:
            Most recent accuracy value
        """
        if not self.accuracy_history:
            return 0.0
        return self.accuracy_history[-1]
    
    def get_convergence_round(self, target_accuracy: float) -> Optional[int]:
        """
        Find the round where target accuracy was first achieved.
        
        Args:
            target_accuracy: Target accuracy threshold
            
        Returns:
            Round number (0-indexed), or None if not achieved
        """
        for i, acc in enumerate(self.accuracy_history):
            if acc >= target_accuracy:
                return i
        return None
    
    def get_time_to_accuracy(self, target_accuracy: float) -> Optional[float]:
        """
        Calculate time to reach target accuracy.
        
        Args:
            target_accuracy: Target accuracy threshold
            
        Returns:
            Time in seconds, or None if not achieved
        """
        convergence_round = self.get_convergence_round(target_accuracy)
        
        if convergence_round is None:
            return None
        
        if convergence_round >= len(self.round_times):
            return sum(self.round_times)
        
        return sum(self.round_times[:convergence_round + 1])
    
    def get_metrics_summary(self) -> Dict:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all key metrics
        """
        summary = {
            'total_training_time': self.get_total_training_time(),
            'average_round_time': self.get_average_round_time(),
            'total_rounds': len(self.round_times),
            'best_accuracy': self.get_best_accuracy(),
            'latest_accuracy': self.get_latest_accuracy(),
            'average_edge_aggregation_time': self.get_average_edge_aggregation_time(),
            'average_cloud_aggregation_time': self.get_average_cloud_aggregation_time(),
            'average_client_training_time': self.get_average_client_training_time(),
        }
        
        # Add accuracy history if available
        if self.accuracy_history:
            summary['accuracy_history'] = self.accuracy_history.copy()
            summary['accuracy_std'] = np.std(self.accuracy_history)
            summary['accuracy_improvement'] = (
                self.accuracy_history[-1] - self.accuracy_history[0]
                if len(self.accuracy_history) > 1 else 0.0
            )
        
        # Add loss history if available
        if self.loss_history:
            summary['loss_history'] = self.loss_history.copy()
            summary['final_loss'] = self.loss_history[-1]
        
        return summary
    
    def get_accuracy_at_rounds(self, round_numbers: List[int]) -> List[float]:
        """
        Get accuracy values at specific round numbers.
        
        Args:
            round_numbers: List of round numbers to query
            
        Returns:
            List of accuracy values (0.0 for out-of-range rounds)
        """
        accuracies = []
        for round_num in round_numbers:
            if 0 <= round_num < len(self.accuracy_history):
                accuracies.append(self.accuracy_history[round_num])
            else:
                accuracies.append(0.0)
        return accuracies
    
    def reset(self) -> None:
        """Reset all metrics for new experiment."""
        self.metrics_history.clear()
        self.start_time = None
        self.round_times = []
        self.current_round = 0
        self.round_start_time = None
        self.edge_aggregation_times = []
        self.cloud_aggregation_times = []
        self.client_training_times.clear()
        self.accuracy_history = []
        self.loss_history = []


def calculate_top_k_accuracy(
    model: torch.nn.Module,
    data_loader: DataLoader,
    k: int = 5,
    device: str = 'cpu'
) -> float:
    """
    Calculate top-k accuracy for model evaluation.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with test data
        k: Number of top predictions to consider
        device: Device for computation
        
    Returns:
        Top-k accuracy percentage
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            _, predicted = outputs.topk(k, 1, True, True)
            
            total += targets.size(0)
            correct += predicted.eq(targets.view(-1, 1).expand_as(predicted)).sum().item()
    
    return 100.0 * correct / total


def calculate_confusion_matrix(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Calculate confusion matrix for model evaluation.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader with test data
        num_classes: Number of classes
        device: Device for computation
        
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    model.eval()
    model.to(device)
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion[t.item(), p.item()] += 1
    
    return confusion


def calculate_per_class_accuracy(
    confusion_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate per-class accuracy from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (num_classes x num_classes)
        
    Returns:
        Array of per-class accuracies
    """
    class_counts = confusion_matrix.sum(axis=1)
    correct_predictions = np.diag(confusion_matrix)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.where(
            class_counts > 0,
            correct_predictions / class_counts,
            0.0
        )
    
    return per_class_acc


def calculate_fairness_metric(
    per_class_accuracies: np.ndarray
) -> float:
    """
    Calculate fairness metric (accuracy variance across classes).
    
    Lower variance indicates fairer model across classes.
    
    Args:
        per_class_accuracies: Array of per-class accuracies
        
    Returns:
        Standard deviation of accuracies (fairness metric)
    """
    return float(np.std(per_class_accuracies))


def create_metrics_calculator() -> MetricsCalculator:
    """
    Factory function to create MetricsCalculator instance.
    
    Returns:
        Initialized MetricsCalculator instance
    """
    return MetricsCalculator()
