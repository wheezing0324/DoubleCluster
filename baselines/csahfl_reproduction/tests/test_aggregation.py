"""
Test suite for CSAHFL aggregation components.

Validates:
- Intra-cluster aggregation (Section 3.4, Eq. 7-10):
  * Total time calculation (computation + communication)
  * Adaptive aggregation interval (median + α × IQR)
  * Workload adjustment (epoch bounding)
  * Semi-asynchronous aggregation with staleness discount

- Inter-cluster aggregation (Section 3.5, Eq. 11-14):
  * Quantity weight calculation
  * Distribution weight via KL divergence
  * Combined weight normalization
  * Global model aggregation
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.aggregation.intra_cluster import IntraClusterAggregator, create_intra_cluster_aggregator
from src.aggregation.inter_cluster import InterClusterAggregator, create_inter_cluster_aggregator
from src.utils.kl_divergence import calculate_kl_divergence, normalize_distribution, create_uniform_distribution


class TestIntraClusterAggregation(unittest.TestCase):
    """Tests for adaptive semi-asynchronous intra-cluster aggregation (Eq. 7-10)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'intra_cluster': {
                'max_epochs': 10,
                'tolerance_alpha': 1.5,
                'batch_size': 10
            },
            'training': {
                'learning_rate': 0.01
            },
            'model': {
                'dropout_rate': 0.5
            }
        }
        self.edge_server_id = 0
        self.device = 'cpu'
        self.aggregator = create_intra_cluster_aggregator(
            self.edge_server_id, 
            self.config, 
            self.device
        )
    
    def test_aggregator_creation(self):
        """Test IntraClusterAggregator initialization."""
        self.assertIsNotNone(self.aggregator)
        self.assertEqual(self.aggregator.edge_server_id, self.edge_server_id)
        self.assertEqual(self.aggregator.max_epochs, 10)
        self.assertEqual(self.aggregator.tolerance_alpha, 1.5)
    
    def test_computation_time_calculation(self):
        """Test computation time calculation (Eq. 7: t_cmp = E_max × N_BS × t_unit)."""
        # Test with known values
        num_batches = 5  # N_BS
        epochs = 10  # E_max
        t_unit = 0.01  # seconds per batch
        
        t_cmp = self.aggregator.calculate_computation_time(epochs, num_batches, t_unit)
        expected_t_cmp = epochs * num_batches * t_unit  # 10 * 5 * 0.01 = 0.5
        
        self.assertAlmostEqual(t_cmp, expected_t_cmp, places=6)
        self.assertGreater(t_cmp, 0)
    
    def test_communication_time_calculation(self):
        """Test communication time calculation (Eq. 7: t_com = M / B_w)."""
        model_size = 1000000  # 1MB in bytes
        bandwidth = 10000000  # 10MB/s
        
        t_com = self.aggregator.calculate_communication_time(model_size, bandwidth)
        expected_t_com = model_size / bandwidth  # 0.1 seconds
        
        self.assertAlmostEqual(t_com, expected_t_com, places=6)
        self.assertGreater(t_com, 0)
    
    def test_total_time_calculation(self):
        """Test total time calculation (Eq. 7: t_total = t_cmp + t_com)."""
        t_cmp = 0.5
        t_com = 0.1
        
        t_total = self.aggregator.calculate_total_time(t_cmp, t_com)
        expected_t_total = t_cmp + t_com  # 0.6 seconds
        
        self.assertAlmostEqual(t_total, expected_t_total, places=6)
    
    def test_aggregation_interval_calculation(self):
        """Test adaptive aggregation interval (Eq. 8: T_θ = median + α × IQR)."""
        # Simulate client times with known distribution
        client_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        for t in client_times:
            self.aggregator.update_client_time(len(client_times), t)
        
        T_theta = self.aggregator.calculate_aggregation_interval()
        
        # median = 5.5, Q25 = 3.25, Q75 = 7.75, IQR = 4.5
        # T_θ = 5.5 + 1.5 × 4.5 = 5.5 + 6.75 = 12.25
        expected_median = np.median(client_times)
        expected_q75 = np.percentile(client_times, 75)
        expected_q25 = np.percentile(client_times, 25)
        expected_iqr = expected_q75 - expected_q25
        expected_T_theta = expected_median + self.aggregator.tolerance_alpha * expected_iqr
        
        self.assertAlmostEqual(T_theta, expected_T_theta, places=6)
        self.assertGreater(T_theta, expected_median)  # T_θ should be > median
    
    def test_workload_adjustment_bounds(self):
        """Test workload adjustment bounds (Eq. 9: E_i bounded between 1 and E_max)."""
        T_theta = 5.0  # Aggregation interval
        t_com = 1.0  # Communication time
        N_BS = 10  # Number of batches
        t_unit = 0.1  # Time per batch
        
        # Test normal case
        E_i = self.aggregator.calculate_workload(T_theta, t_com, N_BS, t_unit)
        self.assertGreaterEqual(E_i, 1)
        self.assertLessEqual(E_i, self.aggregator.max_epochs)
        
        # Test weak client (should get minimum epochs)
        T_theta_weak = 0.5  # Very tight deadline
        E_i_weak = self.aggregator.calculate_workload(T_theta_weak, t_com, N_BS, t_unit)
        self.assertEqual(E_i_weak, 1)  # Should be bounded to minimum
        
        # Test strong client (should get maximum epochs)
        T_theta_strong = 100.0  # Very loose deadline
        E_i_strong = self.aggregator.calculate_workload(T_theta_strong, t_com, N_BS, t_unit)
        self.assertEqual(E_i_strong, self.aggregator.max_epochs)  # Should be bounded to maximum
    
    def test_discount_factor_calculation(self):
        """Test staleness discount factor (Eq. 10: τ_j = 1 / (1 + γ_j))."""
        # No staleness
        tau_0 = self.aggregator.calculate_discount_factor(0)
        self.assertEqual(tau_0, 1.0)
        
        # 1 round delayed
        tau_1 = self.aggregator.calculate_discount_factor(1)
        self.assertEqual(tau_1, 0.5)
        
        # 2 rounds delayed
        tau_2 = self.aggregator.calculate_discount_factor(2)
        self.assertEqual(tau_2, 1.0 / 3.0)
        
        # 5 rounds delayed
        tau_5 = self.aggregator.calculate_discount_factor(5)
        self.assertEqual(tau_5, 1.0 / 6.0)
        
        # Discount factor should decrease with staleness
        self.assertGreater(tau_0, tau_1)
        self.assertGreater(tau_1, tau_2)
        self.assertGreater(tau_2, tau_5)
    
    def test_staleness_tracking(self):
        """Test staleness tracking across rounds."""
        client_id = 42
        
        # Initial staleness should be 0
        self.assertEqual(self.aggregator.staleness[client_id], 0)
        
        # Increment staleness
        self.aggregator.increment_staleness(client_id)
        self.assertEqual(self.aggregator.staleness[client_id], 1)
        
        # Increment again
        self.aggregator.increment_staleness(client_id)
        self.assertEqual(self.aggregator.staleness[client_id], 2)
        
        # Reset staleness
        self.aggregator.reset_staleness(client_id)
        self.assertEqual(self.aggregator.staleness[client_id], 0)
    
    def test_client_classification(self):
        """Test classification of synchronous vs asynchronous clients."""
        T_theta = 5.0
        client_times = {
            0: 3.0,  # Synchronous (within T_θ)
            1: 4.0,  # Synchronous
            2: 5.0,  # Synchronous (at boundary)
            3: 6.0,  # Asynchronous (delayed)
            4: 7.0,  # Asynchronous
        }
        
        for client_id, time in client_times.items():
            self.aggregator.update_client_time(client_id, time)
        
        sync_clients, async_clients = self.aggregator.classify_clients(T_theta, client_times)
        
        # Clients 0, 1, 2 should be synchronous
        self.assertIn(0, sync_clients)
        self.assertIn(1, sync_clients)
        self.assertIn(2, sync_clients)
        
        # Clients 3, 4 should be asynchronous
        self.assertIn(3, async_clients)
        self.assertIn(4, async_clients)
    
    def test_semi_asynchronous_aggregation(self):
        """Test semi-asynchronous aggregation (Eq. 10)."""
        # Create simple model weights
        model_weights = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
        }
        
        # Simulate synchronous clients (reported within T_θ)
        sync_clients = [
            {'client_id': 0, 'weights': model_weights, 'num_samples': 100, 'staleness': 0},
            {'client_id': 1, 'weights': model_weights, 'num_samples': 200, 'staleness': 0},
        ]
        
        # Simulate asynchronous clients (delayed)
        async_clients = [
            {'client_id': 2, 'weights': model_weights, 'num_samples': 150, 'staleness': 1},
        ]
        
        aggregated_weights = self.aggregator.perform_aggregation(
            sync_clients, async_clients
        )
        
        # Check aggregation produced valid weights
        self.assertIsNotNone(aggregated_weights)
        self.assertIn('layer1.weight', aggregated_weights)
        self.assertIn('layer1.bias', aggregated_weights)
        
        # Check weight shapes are preserved
        self.assertEqual(aggregated_weights['layer1.weight'].shape, model_weights['layer1.weight'].shape)
        self.assertEqual(aggregated_weights['layer1.bias'].shape, model_weights['layer1.bias'].shape)
    
    def test_aggregation_statistics(self):
        """Test aggregation statistics tracking."""
        stats = self.aggregator.get_aggregation_statistics()
        
        self.assertIn('total_aggregations', stats)
        self.assertIn('synchronous_clients', stats)
        self.assertIn('asynchronous_clients', stats)
        self.assertIn('avg_staleness', stats)


class TestInterClusterAggregation(unittest.TestCase):
    """Tests for dynamic distribution-aware inter-cluster aggregation (Eq. 11-14)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'inter_cluster': {
                'kl_epsilon': 1e-10
            },
            'training': {
                'num_classes': 10
            }
        }
        self.device = 'cpu'
        self.aggregator = create_inter_cluster_aggregator(self.config, self.device)
    
    def test_aggregator_creation(self):
        """Test InterClusterAggregator initialization."""
        self.assertIsNotNone(self.aggregator)
        self.assertEqual(self.aggregator.kl_epsilon, 1e-10)
        self.assertEqual(self.aggregator.num_classes, 10)
    
    def test_quantity_weight_calculation(self):
        """Test quantity weight calculation (Eq. 12: ω_k^qua = N_edge^k / Σ N_edge^k)."""
        cluster_data_counts = {
            0: 1000,  # Cluster 0 has 1000 samples
            1: 2000,  # Cluster 1 has 2000 samples
            2: 3000,  # Cluster 2 has 3000 samples
            3: 4000,  # Cluster 3 has 4000 samples
        }
        
        quantity_weights = self.aggregator.calculate_quantity_weight(cluster_data_counts)
        
        # Total = 10000
        # Weights should be: 0.1, 0.2, 0.3, 0.4
        self.assertAlmostEqual(quantity_weights[0], 0.1, places=6)
        self.assertAlmostEqual(quantity_weights[1], 0.2, places=6)
        self.assertAlmostEqual(quantity_weights[2], 0.3, places=6)
        self.assertAlmostEqual(quantity_weights[3], 0.4, places=6)
        
        # Weights should sum to 1
        total_weight = sum(quantity_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_distribution_weight_calculation(self):
        """Test distribution weight calculation (Eq. 13: ω_k^dis = 1 / (1 + D_KL))."""
        # Uniform global distribution
        global_dist = create_uniform_distribution(10, 1e-10)
        self.aggregator.set_global_distribution(global_dist)
        
        # Cluster distributions with varying divergence from uniform
        cluster_distributions = {
            0: np.array([0.1] * 10),  # Perfect uniform (KL = 0, weight = 1.0)
            1: np.array([0.5] + [0.5/9] * 9),  # Skewed distribution
            2: np.array([0.9] + [0.1/9] * 9),  # Highly skewed
        }
        
        distribution_weights = {}
        for cluster_id, dist in cluster_distributions.items():
            dist = normalize_distribution(dist, 1e-10)
            kl_div = self.aggregator.calculate_kl_divergence(dist, global_dist)
            distribution_weights[cluster_id] = self.aggregator.calculate_distribution_weight(dist, global_dist)
        
        # Cluster 0 (uniform) should have highest weight
        self.assertEqual(distribution_weights[0], 1.0)
        
        # More skewed distributions should have lower weights
        self.assertGreater(distribution_weights[0], distribution_weights[1])
        self.assertGreater(distribution_weights[1], distribution_weights[2])
        
        # All weights should be in (0, 1]
        for weight in distribution_weights.values():
            self.assertGreater(weight, 0)
            self.assertLessEqual(weight, 1.0)
    
    def test_combined_weight_normalization(self):
        """Test combined weight normalization (Eq. 14: λ_k = (ω^qua × ω^dis) / Σ(ω^qua × ω^dis))."""
        quantity_weights = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        distribution_weights = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        
        combined_weights = self.aggregator.calculate_combined_weights(
            quantity_weights, distribution_weights
        )
        
        # Check weights sum to 1
        total_weight = sum(combined_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # Check relative ordering (higher quantity × distribution = higher combined)
        self.assertGreater(combined_weights[0], combined_weights[1])
        self.assertGreater(combined_weights[1], combined_weights[2])
        self.assertGreater(combined_weights[2], combined_weights[3])
    
    def test_global_aggregation(self):
        """Test global model aggregation (Eq. 11: w_g = Σ λ_k w_edge^k)."""
        # Create edge server models with different weights
        edge_models = {
            0: {'layer1.weight': torch.ones(10, 5) * 1.0},
            1: {'layer1.weight': torch.ones(10, 5) * 2.0},
            2: {'layer1.weight': torch.ones(10, 5) * 3.0},
        }
        
        # Equal weights for simplicity
        cluster_weights = {0: 1/3, 1: 1/3, 2: 1/3}
        
        global_model = self.aggregator.perform_global_aggregation(edge_models, cluster_weights)
        
        # Check aggregation produced valid model
        self.assertIsNotNone(global_model)
        self.assertIn('layer1.weight', global_model)
        
        # With equal weights, result should be average: (1 + 2 + 3) / 3 = 2.0
        expected_value = 2.0
        actual_value = global_model['layer1.weight'].mean().item()
        self.assertAlmostEqual(actual_value, expected_value, places=5)
    
    def test_kl_divergence_numerical_stability(self):
        """Test KL divergence numerical stability with epsilon."""
        # Test with zero probabilities (should not cause log(0))
        p = np.array([0.5, 0.5, 0.0, 0.0])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        kl_div = calculate_kl_divergence(p, q, epsilon=1e-10)
        
        # Should be finite and non-negative
        self.assertTrue(np.isfinite(kl_div))
        self.assertGreaterEqual(kl_div, 0)
    
    def test_distribution_normalization(self):
        """Test distribution normalization."""
        # Unnormalized distribution
        dist = np.array([1.0, 2.0, 3.0, 4.0])
        
        normalized = normalize_distribution(dist, epsilon=1e-10)
        
        # Should sum to 1
        self.assertAlmostEqual(normalized.sum(), 1.0, places=6)
        
        # Relative proportions should be preserved
        self.assertAlmostEqual(normalized[0] / normalized[1], 1.0 / 2.0, places=6)
    
    def test_weight_statistics(self):
        """Test weight statistics tracking."""
        quantity_weights = {0: 0.5, 1: 0.5}
        distribution_weights = {0: 1.0, 1: 0.5}
        
        combined_weights = self.aggregator.calculate_combined_weights(
            quantity_weights, distribution_weights
        )
        
        stats = self.aggregator.get_weight_statistics()
        
        self.assertIn('quantity_weights', stats)
        self.assertIn('distribution_weights', stats)
        self.assertIn('combined_weights', stats)
    
    def test_aggregator_reset(self):
        """Test aggregator reset for new experiment."""
        # Set some state
        global_dist = create_uniform_distribution(10, 1e-10)
        self.aggregator.set_global_distribution(global_dist)
        
        # Reset
        self.aggregator.reset()
        
        # Global distribution should be cleared
        self.assertIsNone(self.aggregator.global_distribution)


class TestAggregationIntegration(unittest.TestCase):
    """Integration tests for complete aggregation pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = {
            'intra_cluster': {
                'max_epochs': 10,
                'tolerance_alpha': 1.5,
                'batch_size': 10
            },
            'inter_cluster': {
                'kl_epsilon': 1e-10
            },
            'training': {
                'learning_rate': 0.01,
                'num_classes': 10
            },
            'model': {
                'dropout_rate': 0.5
            }
        }
        self.device = 'cpu'
    
    def test_intra_to_inter_aggregation_flow(self):
        """Test complete flow from intra-cluster to inter-cluster aggregation."""
        # Create aggregators
        intra_aggregator = create_intra_cluster_aggregator(0, self.config, self.device)
        inter_aggregator = create_inter_cluster_aggregator(self.config, self.device)
        
        # Simulate intra-cluster aggregation (edge level)
        model_weights = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
        }
        
        sync_clients = [
            {'client_id': 0, 'weights': model_weights, 'num_samples': 100, 'staleness': 0},
            {'client_id': 1, 'weights': model_weights, 'num_samples': 200, 'staleness': 0},
        ]
        async_clients = []
        
        edge_model = intra_aggregator.perform_aggregation(sync_clients, async_clients)
        
        # Simulate inter-cluster aggregation (cloud level)
        global_dist = create_uniform_distribution(10, 1e-10)
        inter_aggregator.set_global_distribution(global_dist)
        
        edge_models = {
            0: edge_model,
            1: edge_model,
            2: edge_model,
        }
        
        cluster_data_counts = {0: 1000, 1: 1000, 2: 1000}
        quantity_weights = inter_aggregator.calculate_quantity_weight(cluster_data_counts)
        distribution_weights = {0: 1.0, 1: 1.0, 2: 1.0}  # All uniform
        combined_weights = inter_aggregator.calculate_combined_weights(
            quantity_weights, distribution_weights
        )
        
        global_model = inter_aggregator.perform_global_aggregation(edge_models, combined_weights)
        
        # Verify global model is valid
        self.assertIsNotNone(global_model)
        self.assertIn('layer1.weight', global_model)
        self.assertIn('layer1.bias', global_model)
    
    def test_staleness_impact_on_aggregation(self):
        """Test that staleness correctly reduces client contribution."""
        intra_aggregator = create_intra_cluster_aggregator(0, self.config, self.device)
        
        # Create two different weight sets
        weights_a = {'layer1.weight': torch.ones(10, 5) * 10.0}
        weights_b = {'layer1.weight': torch.ones(10, 5) * 20.0}
        
        # Synchronous client with weights_a
        sync_clients = [
            {'client_id': 0, 'weights': weights_a, 'num_samples': 100, 'staleness': 0},
        ]
        
        # Asynchronous client with weights_b (delayed by 1 round, τ = 0.5)
        async_clients = [
            {'client_id': 1, 'weights': weights_b, 'num_samples': 100, 'staleness': 1},
        ]
        
        aggregated = intra_aggregator.perform_aggregation(sync_clients, async_clients)
        
        # Result should be weighted average: (100×10 + 0.5×100×20) / (100 + 0.5×100) = 13.33
        expected_value = (100 * 10.0 + 0.5 * 100 * 20.0) / (100 + 0.5 * 100)
        actual_value = aggregated['layer1.weight'].mean().item()
        
        self.assertAlmostEqual(actual_value, expected_value, places=4)
    
    def test_distribution_weight_impact(self):
        """Test that distribution-aware weighting favors uniform clusters."""
        inter_aggregator = create_inter_cluster_aggregator(self.config, self.device)
        
        # Set global uniform distribution
        global_dist = create_uniform_distribution(10, 1e-10)
        inter_aggregator.set_global_distribution(global_dist)
        
        # Create edge models
        edge_models = {
            0: {'layer1.weight': torch.ones(10, 5)},
            1: {'layer1.weight': torch.ones(10, 5) * 2},
        }
        
        # Equal quantity, different distribution quality
        cluster_data_counts = {0: 1000, 1: 1000}
        quantity_weights = inter_aggregator.calculate_quantity_weight(cluster_data_counts)
        
        # Cluster 0: uniform (high distribution weight)
        # Cluster 1: skewed (low distribution weight)
        dist_uniform = normalize_distribution(np.array([0.1] * 10), 1e-10)
        dist_skewed = normalize_distribution(np.array([0.9] + [0.1/9] * 9), 1e-10)
        
        distribution_weights = {
            0: inter_aggregator.calculate_distribution_weight(dist_uniform, global_dist),
            1: inter_aggregator.calculate_distribution_weight(dist_skewed, global_dist),
        }
        
        combined_weights = inter_aggregator.calculate_combined_weights(
            quantity_weights, distribution_weights
        )
        
        # Cluster 0 (uniform) should have higher combined weight
        self.assertGreater(combined_weights[0], combined_weights[1])


def run_tests():
    """Execute all aggregation tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestIntraClusterAggregation))
    suite.addTests(loader.loadTestsFromTestCase(TestInterClusterAggregation))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
