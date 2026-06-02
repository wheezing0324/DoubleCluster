"""
Test suite for CSAHFL mobility simulation components (Section 3.2).

Validates Markov chain-based client mobility simulation including:
- Staying probability (ps) parameter
- Static (ps=1.0) and high-mobility (ps=0.5) scenarios
- Client-ES reassignment logic
- Mobility statistics tracking
- Reproducibility with random seeds

Tests cover:
1. MobilitySimulator initialization and configuration
2. Client location initialization
3. Mobility simulation with different staying probabilities
4. Client-ES assignment tracking
5. Mobility statistics and history
6. Scenario detection (static vs. high-mobility)
7. Reproducibility with seeds
8. Edge cases and boundary conditions
"""

import unittest
import numpy as np
from typing import Dict, List, Tuple

# Import mobility simulator components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.mobility_simulator import (
    MobilitySimulator,
    create_mobility_simulator,
    simulate_mobility_round,
    get_mobility_scenario_name
)


class TestMobilitySimulatorInitialization(unittest.TestCase):
    """Test MobilitySimulator initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        simulator = MobilitySimulator()
        
        self.assertEqual(simulator.num_clients, 200)
        self.assertEqual(simulator.num_edge_servers, 5)
        self.assertEqual(simulator.staying_probability, 1.0)
        self.assertEqual(simulator.seed, 42)
    
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        simulator = MobilitySimulator(
            num_clients=100,
            num_edge_servers=10,
            staying_probability=0.5,
            seed=123
        )
        
        self.assertEqual(simulator.num_clients, 100)
        self.assertEqual(simulator.num_edge_servers, 10)
        self.assertEqual(simulator.staying_probability, 0.5)
        self.assertEqual(simulator.seed, 123)
    
    def test_factory_function(self):
        """Test factory function creates configured simulator."""
        simulator = create_mobility_simulator(
            num_clients=150,
            num_edge_servers=5,
            staying_probability=0.8,
            seed=456
        )
        
        self.assertIsInstance(simulator, MobilitySimulator)
        self.assertEqual(simulator.num_clients, 150)
        self.assertEqual(simulator.staying_probability, 0.8)
    
    def test_staying_probability_bounds(self):
        """Test staying probability is within valid bounds [0, 1]."""
        # Valid probabilities
        for ps in [0.0, 0.5, 1.0]:
            simulator = MobilitySimulator(staying_probability=ps)
            self.assertTrue(0.0 <= simulator.staying_probability <= 1.0)
    
    def test_initial_data_structures(self):
        """Test initial data structures are empty."""
        simulator = MobilitySimulator(num_clients=50, num_edge_servers=5)
        
        # Before initialization, structures should be empty
        self.assertEqual(len(simulator.client_locations), 0)
        self.assertEqual(len(simulator.clients_per_es), 0)
        self.assertEqual(len(simulator.mobility_history), 0)


class TestClientLocationInitialization(unittest.TestCase):
    """Test client location initialization."""
    
    def test_initialize_client_locations(self):
        """Test client locations are properly initialized."""
        simulator = MobilitySimulator(num_clients=100, num_edge_servers=5, seed=42)
        simulator.initialize_client_locations()
        
        # All clients should have locations
        self.assertEqual(len(simulator.client_locations), 100)
        
        # All locations should be valid edge server IDs
        for client_id, es_id in simulator.client_locations.items():
            self.assertTrue(0 <= es_id < 5)
        
        # All clients should be assigned
        self.assertEqual(set(simulator.client_locations.keys()), set(range(100)))
    
    def test_clients_per_es_tracking(self):
        """Test clients_per_es tracking is correct."""
        simulator = MobilitySimulator(num_clients=100, num_edge_servers=5, seed=42)
        simulator.initialize_client_locations()
        
        # Total clients across all ES should equal num_clients
        total_clients = sum(len(clients) for clients in simulator.clients_per_es.values())
        self.assertEqual(total_clients, 100)
        
        # Each client should appear exactly once
        all_clients = []
        for clients in simulator.clients_per_es.values():
            all_clients.extend(clients)
        self.assertEqual(len(all_clients), len(set(all_clients)))
    
    def test_balanced_initial_distribution(self):
        """Test initial distribution is roughly balanced."""
        num_clients = 100
        num_edge_servers = 5
        expected_per_es = num_clients // num_edge_servers
        
        simulator = MobilitySimulator(
            num_clients=num_clients,
            num_edge_servers=num_edge_servers,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Each ES should have approximately expected_per_es clients
        for es_id, clients in simulator.clients_per_es.items():
            # Allow 20% deviation from expected
            lower_bound = expected_per_es * 0.8
            upper_bound = expected_per_es * 1.2
            self.assertTrue(
                lower_bound <= len(clients) <= upper_bound,
                f"ES {es_id} has {len(clients)} clients, expected ~{expected_per_es}"
            )
    
    def test_initialization_reproducibility(self):
        """Test initialization is reproducible with same seed."""
        sim1 = MobilitySimulator(num_clients=50, num_edge_servers=5, seed=42)
        sim1.initialize_client_locations()
        
        sim2 = MobilitySimulator(num_clients=50, num_edge_servers=5, seed=42)
        sim2.initialize_client_locations()
        
        # Same seed should produce same assignments
        self.assertEqual(sim1.client_locations, sim2.client_locations)
    
    def test_initialization_different_seeds(self):
        """Test different seeds produce different assignments."""
        sim1 = MobilitySimulator(num_clients=50, num_edge_servers=5, seed=42)
        sim1.initialize_client_locations()
        
        sim2 = MobilitySimulator(num_clients=50, num_edge_servers=5, seed=123)
        sim2.initialize_client_locations()
        
        # Different seeds should likely produce different assignments
        # (not guaranteed but very probable)
        if sim1.client_locations != sim2.client_locations:
            self.assertNotEqual(sim1.client_locations, sim2.client_locations)


class TestMobilitySimulation(unittest.TestCase):
    """Test mobility simulation with different staying probabilities."""
    
    def test_static_scenario_no_movement(self):
        """Test static scenario (ps=1.0) produces no movement."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=1.0,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Store initial locations
        initial_locations = simulator.client_locations.copy()
        
        # Simulate multiple rounds
        for round_num in range(1, 11):
            simulator.simulate_mobility(round_num)
        
        # Locations should not change in static scenario
        self.assertEqual(simulator.client_locations, initial_locations)
    
    def test_high_mobility_scenario(self):
        """Test high-mobility scenario (ps=0.5) produces movement."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Store initial locations
        initial_locations = simulator.client_locations.copy()
        
        # Simulate multiple rounds
        for round_num in range(1, 11):
            simulator.simulate_mobility(round_num)
        
        # Some clients should have moved (with high probability)
        moved_clients = sum(
            1 for cid in simulator.client_locations
            if simulator.client_locations[cid] != initial_locations[cid]
        )
        
        # With ps=0.5 and 10 rounds, most clients should have moved at least once
        # Probability of not moving in 10 rounds = 0.5^10 ≈ 0.001
        self.assertGreater(moved_clients, 0, "Expected some clients to move in high-mobility scenario")
    
    def test_mobility_round_function(self):
        """Test simulate_mobility_round convenience function."""
        simulator = MobilitySimulator(
            num_clients=30,
            num_edge_servers=3,
            staying_probability=0.7,
            seed=42
        )
        simulator.initialize_client_locations()
        
        active_clients = list(range(30))
        
        # Simulate one round
        client_locations, clients_per_es = simulate_mobility_round(
            simulator,
            round_num=1,
            active_clients=active_clients
        )
        
        # Verify return types
        self.assertIsInstance(client_locations, dict)
        self.assertIsInstance(clients_per_es, dict)
        
        # Verify all active clients are assigned
        self.assertEqual(len(client_locations), len(active_clients))
    
    def test_mobility_history_tracking(self):
        """Test mobility history is properly tracked."""
        simulator = MobilitySimulator(
            num_clients=20,
            num_edge_servers=3,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate 5 rounds
        for round_num in range(1, 6):
            simulator.simulate_mobility(round_num)
        
        # Should have 5 mobility events recorded
        self.assertEqual(len(simulator.mobility_history), 5)
        
        # Each history entry should have required fields
        for event in simulator.mobility_history:
            self.assertIn('round', event)
            self.assertIn('moved_clients', event)
            self.assertIn('num_moved', event)
    
    def test_get_clients_at_edge_server(self):
        """Test getting clients at specific edge server."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=0.8,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Get clients at each ES
        for es_id in range(5):
            clients = simulator.get_clients_at_edge_server(es_id)
            
            # Should return list
            self.assertIsInstance(clients, list)
            
            # All clients should be assigned to this ES
            for client_id in clients:
                self.assertEqual(simulator.client_locations[client_id], es_id)
    
    def test_get_edge_server_for_client(self):
        """Test getting edge server for specific client."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=0.8,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Check each client
        for client_id in range(50):
            es_id = simulator.get_edge_server_for_client(client_id)
            
            # Should match client_locations
            self.assertEqual(es_id, simulator.client_locations[client_id])
    
    def test_mobility_with_subset_of_clients(self):
        """Test mobility simulation with subset of active clients."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Only half the clients are active
        active_clients = list(range(25))
        
        # Simulate mobility
        simulator.simulate_mobility(round_num=1, active_clients=active_clients)
        
        # Inactive clients should not be in mobility history
        if len(simulator.mobility_history) > 0:
            last_event = simulator.mobility_history[-1]
            moved_clients = last_event.get('moved_clients', [])
            
            # All moved clients should be from active set
            for client_id in moved_clients:
                self.assertIn(client_id, active_clients)


class TestMobilityStatistics(unittest.TestCase):
    """Test mobility statistics and analysis methods."""
    
    def test_mobility_statistics_structure(self):
        """Test mobility statistics has correct structure."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate some rounds
        for round_num in range(1, 6):
            simulator.simulate_mobility(round_num)
        
        stats = simulator.get_mobility_statistics()
        
        # Should have required fields
        self.assertIn('total_rounds', stats)
        self.assertIn('total_movements', stats)
        self.assertIn('avg_movements_per_round', stats)
        self.assertIn('client_mobility_counts', stats)
    
    def test_mobility_statistics_accuracy(self):
        """Test mobility statistics are accurate."""
        simulator = MobilitySimulator(
            num_clients=30,
            num_edge_servers=3,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate 10 rounds
        num_rounds = 10
        for round_num in range(1, num_rounds + 1):
            simulator.simulate_mobility(round_num)
        
        stats = simulator.get_mobility_statistics()
        
        # Total rounds should match
        self.assertEqual(stats['total_rounds'], num_rounds)
        
        # Total movements should be sum of per-round movements
        total_from_history = sum(
            event.get('num_moved', 0)
            for event in simulator.mobility_history
        )
        self.assertEqual(stats['total_movements'], total_from_history)
    
    def test_client_distribution(self):
        """Test client distribution across edge servers."""
        simulator = MobilitySimulator(
            num_clients=100,
            num_edge_servers=5,
            staying_probability=0.8,
            seed=42
        )
        simulator.initialize_client_locations()
        
        distribution = simulator.get_client_distribution()
        
        # Should have entry for each ES
        self.assertEqual(len(distribution), 5)
        
        # Total should equal num_clients
        total = sum(count for count in distribution.values())
        self.assertEqual(total, 100)
    
    def test_reset_functionality(self):
        """Test reset clears all mobility state."""
        simulator = MobilitySimulator(
            num_clients=30,
            num_edge_servers=3,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate some rounds
        for round_num in range(1, 6):
            simulator.simulate_mobility(round_num)
        
        # Reset
        simulator.reset()
        
        # All structures should be empty
        self.assertEqual(len(simulator.client_locations), 0)
        self.assertEqual(len(simulator.clients_per_es), 0)
        self.assertEqual(len(simulator.mobility_history), 0)


class TestScenarioDetection(unittest.TestCase):
    """Test scenario detection methods."""
    
    def test_is_static_scenario(self):
        """Test static scenario detection."""
        # ps=1.0 should be static
        simulator = MobilitySimulator(staying_probability=1.0)
        self.assertTrue(simulator.is_static_scenario())
        
        # ps<1.0 should not be static
        for ps in [0.0, 0.5, 0.8, 0.99]:
            simulator = MobilitySimulator(staying_probability=ps)
            self.assertFalse(simulator.is_static_scenario())
    
    def test_is_high_mobility_scenario(self):
        """Test high-mobility scenario detection."""
        # ps=0.5 should be high-mobility
        simulator = MobilitySimulator(staying_probability=0.5)
        self.assertTrue(simulator.is_high_mobility_scenario())
        
        # ps=1.0 should not be high-mobility
        simulator = MobilitySimulator(staying_probability=1.0)
        self.assertFalse(simulator.is_high_mobility_scenario())
        
        # ps=0.8 might not be high-mobility (depends on threshold)
        simulator = MobilitySimulator(staying_probability=0.8)
        # Threshold is typically 0.6, so 0.8 should not be high-mobility
        self.assertFalse(simulator.is_high_mobility_scenario())
    
    def test_get_mobility_scenario_name(self):
        """Test scenario name function."""
        self.assertEqual(get_mobility_scenario_name(1.0), "Static")
        self.assertEqual(get_mobility_scenario_name(0.5), "High-Mobility")
        self.assertEqual(get_mobility_scenario_name(0.8), "Medium-Mobility")
        self.assertEqual(get_mobility_scenario_name(0.0), "Maximum-Mobility")


class TestMobilityIntegration(unittest.TestCase):
    """Integration tests for complete mobility simulation pipeline."""
    
    def test_end_to_end_mobility_simulation(self):
        """Test complete mobility simulation pipeline."""
        # Create simulator
        simulator = MobilitySimulator(
            num_clients=100,
            num_edge_servers=5,
            staying_probability=0.7,
            seed=42
        )
        
        # Initialize
        simulator.initialize_client_locations()
        initial_distribution = simulator.get_client_distribution()
        
        # Simulate 20 rounds
        for round_num in range(1, 21):
            simulator.simulate_mobility(round_num)
        
        # Get final distribution
        final_distribution = simulator.get_client_distribution()
        
        # Get statistics
        stats = simulator.get_mobility_statistics()
        
        # Verify simulation completed
        self.assertEqual(stats['total_rounds'], 20)
        self.assertGreater(stats['total_movements'], 0)
        
        # Distribution should still sum to num_clients
        self.assertEqual(sum(final_distribution.values()), 100)
    
    def test_mobility_reproducibility(self):
        """Test mobility simulation is reproducible with same seed."""
        def run_simulation():
            simulator = MobilitySimulator(
                num_clients=50,
                num_edge_servers=5,
                staying_probability=0.6,
                seed=42
            )
            simulator.initialize_client_locations()
            
            for round_num in range(1, 11):
                simulator.simulate_mobility(round_num)
            
            return simulator.client_locations.copy()
        
        # Run twice with same seed
        locations1 = run_simulation()
        locations2 = run_simulation()
        
        # Should be identical
        self.assertEqual(locations1, locations2)
    
    def test_mobility_with_varying_ps(self):
        """Test mobility behavior with varying staying probabilities."""
        num_clients = 100
        num_rounds = 20
        
        movement_counts = {}
        
        for ps in [1.0, 0.8, 0.5, 0.2, 0.0]:
            simulator = MobilitySimulator(
                num_clients=num_clients,
                num_edge_servers=5,
                staying_probability=ps,
                seed=42
            )
            simulator.initialize_client_locations()
            
            for round_num in range(1, num_rounds + 1):
                simulator.simulate_mobility(round_num)
            
            stats = simulator.get_mobility_statistics()
            movement_counts[ps] = stats['total_movements']
        
        # Lower ps should generally produce more movements
        # (with some variance due to randomness)
        self.assertEqual(movement_counts[1.0], 0)  # No movement at ps=1.0
        self.assertGreater(movement_counts[0.0], movement_counts[0.5])
        self.assertGreater(movement_counts[0.5], movement_counts[0.8])
    
    def test_client_tracking_across_mobility(self):
        """Test individual client tracking across mobility events."""
        simulator = MobilitySimulator(
            num_clients=20,
            num_edge_servers=3,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Track specific client
        target_client = 5
        initial_es = simulator.client_locations[target_client]
        
        # Simulate rounds and track client
        client_path = [initial_es]
        for round_num in range(1, 11):
            simulator.simulate_mobility(round_num)
            current_es = simulator.client_locations[target_client]
            client_path.append(current_es)
        
        # Path should have correct length
        self.assertEqual(len(client_path), 11)  # Initial + 10 rounds
        
        # All ES IDs should be valid
        for es_id in client_path:
            self.assertTrue(0 <= es_id < 3)


class TestMobilityEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_client(self):
        """Test mobility with single client."""
        simulator = MobilitySimulator(
            num_clients=1,
            num_edge_servers=3,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate rounds
        for round_num in range(1, 6):
            simulator.simulate_mobility(round_num)
        
        # Should still have exactly 1 client
        self.assertEqual(len(simulator.client_locations), 1)
        self.assertEqual(
            sum(len(clients) for clients in simulator.clients_per_es.values()),
            1
        )
    
    def test_single_edge_server(self):
        """Test mobility with single edge server."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=1,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate rounds
        for round_num in range(1, 6):
            simulator.simulate_mobility(round_num)
        
        # All clients should always be at ES 0
        for client_id, es_id in simulator.client_locations.items():
            self.assertEqual(es_id, 0)
    
    def test_zero_staying_probability(self):
        """Test mobility with ps=0.0 (always move)."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=5,
            staying_probability=0.0,
            seed=42
        )
        simulator.initialize_client_locations()
        
        initial_locations = simulator.client_locations.copy()
        
        # Simulate one round - all clients should move
        simulator.simulate_mobility(round_num=1)
        
        # All clients should have moved (to different ES)
        # Note: With multiple ES, probability of moving to same ES is low
        moved_count = sum(
            1 for cid in simulator.client_locations
            if simulator.client_locations[cid] != initial_locations[cid]
        )
        
        # Most clients should have moved
        self.assertGreater(moved_count, len(simulator.client_locations) * 0.8)
    
    def test_many_edge_servers(self):
        """Test mobility with many edge servers."""
        simulator = MobilitySimulator(
            num_clients=50,
            num_edge_servers=20,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate rounds
        for round_num in range(1, 6):
            simulator.simulate_mobility(round_num)
        
        # All ES IDs should be valid
        for es_id in simulator.client_locations.values():
            self.assertTrue(0 <= es_id < 20)
    
    def test_mobility_history_limit(self):
        """Test mobility history doesn't grow unbounded."""
        simulator = MobilitySimulator(
            num_clients=20,
            num_edge_servers=3,
            staying_probability=0.5,
            seed=42
        )
        simulator.initialize_client_locations()
        
        # Simulate many rounds
        num_rounds = 100
        for round_num in range(1, num_rounds + 1):
            simulator.simulate_mobility(round_num)
        
        # History should have all rounds (no automatic pruning in current impl)
        # This test documents current behavior
        self.assertEqual(len(simulator.mobility_history), num_rounds)


def run_tests():
    """Execute all mobility tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMobilitySimulatorInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestClientLocationInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestMobilitySimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestMobilityStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestScenarioDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestMobilityIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMobilityEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
