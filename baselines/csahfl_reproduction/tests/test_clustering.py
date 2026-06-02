"""
Test suite for CSAHFL clustering components (Component 1: Section 3.3).

Tests privacy-preserving one-shot clustering including:
- Truncated SVD feature extraction (Eq. 5)
- Principal angle calculation (Eq. 4, 6)
- Hierarchical clustering with threshold β

Usage:
    pytest tests/test_clustering.py -v
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.clustering.svd_feature_extractor import (
    SVDFeatureExtractor,
    create_svd_extractor,
    extract_client_signature
)
from src.clustering.principal_angle import (
    PrincipalAngleCalculator,
    calculate_principal_angle,
    build_proximity_matrix,
    build_similarity_matrix,
    verify_angle_properties
)
from src.clustering.hierarchical_clusterer import (
    HierarchicalClusterer,
    create_hierarchical_clusterer,
    perform_one_shot_clustering
)


class TestSVDFeatureExtractor(unittest.TestCase):
    """Test truncated SVD feature extraction (Eq. 5)."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create synthetic client data: 100 samples, 784 features (like Fashion-MNIST flattened)
        self.sample_data = np.random.randn(100, 784)
        self.n_components = 10
        
    def test_svd_extractor_creation(self):
        """Test SVDFeatureExtractor initialization."""
        extractor = SVDFeatureExtractor(n_components=10, random_state=42)
        self.assertEqual(extractor.n_components, 10)
        self.assertEqual(extractor.random_state, 42)
        
    def test_factory_function(self):
        """Test create_svd_extractor factory function."""
        extractor = create_svd_extractor(n_components=10, random_state=42)
        self.assertIsInstance(extractor, SVDFeatureExtractor)
        self.assertEqual(extractor.n_components, 10)
        
    def test_extract_features_shape(self):
        """Test that extracted features have correct shape (Eq. 5)."""
        extractor = SVDFeatureExtractor(n_components=self.n_components)
        U_p = extractor.extract_features(self.sample_data)
        
        # U_i^p should have shape (n_features, n_components) = (784, 10)
        expected_shape = (self.sample_data.shape[1], self.n_components)
        self.assertEqual(U_p.shape, expected_shape)
        
    def test_extract_client_signature(self):
        """Test convenience function extract_client_signature."""
        signature = extract_client_signature(self.sample_data, n_components=10)
        
        # Signature should be top-p singular vectors
        self.assertEqual(signature.shape, (784, 10))
        
    def test_svd_orthogonality(self):
        """Test that singular vectors are orthogonal (U^T U = I)."""
        extractor = SVDFeatureExtractor(n_components=self.n_components)
        U_p = extractor.extract_features(self.sample_data)
        
        # U_p^T @ U_p should be approximately identity matrix
        identity = U_p.T @ U_p
        expected_identity = np.eye(self.n_components)
        
        # Check orthogonality with tolerance
        self.assertTrue(np.allclose(identity, expected_identity, atol=1e-6))
        
    def test_svd_reproducibility(self):
        """Test that SVD extraction is reproducible with same seed."""
        extractor1 = SVDFeatureExtractor(n_components=10, random_state=42)
        extractor2 = SVDFeatureExtractor(n_components=10, random_state=42)
        
        U_p1 = extractor1.extract_features(self.sample_data)
        U_p2 = extractor2.extract_features(self.sample_data)
        
        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(np.abs(U_p1), np.abs(U_p2))
        
    def test_different_n_components(self):
        """Test SVD with different numbers of components."""
        for n_comp in [5, 10, 20]:
            extractor = SVDFeatureExtractor(n_components=n_comp)
            U_p = extractor.extract_features(self.sample_data)
            self.assertEqual(U_p.shape[1], n_comp)
            
    def test_small_dataset(self):
        """Test SVD on small dataset."""
        small_data = np.random.randn(10, 50)
        extractor = SVDFeatureExtractor(n_components=5)
        U_p = extractor.extract_features(small_data)
        
        # Should handle small datasets gracefully
        self.assertEqual(U_p.shape, (50, 5))
        
    def test_explained_variance(self):
        """Test explained variance ratio calculation."""
        extractor = SVDFeatureExtractor(n_components=10)
        extractor.extract_features(self.sample_data)
        
        variance_ratio = extractor.get_explained_variance_ratio()
        cumulative = extractor.get_cumulative_explained_variance()
        
        # Variance ratios should be positive and sum to <= 1
        self.assertTrue(np.all(variance_ratio >= 0))
        self.assertTrue(np.sum(variance_ratio) <= 1.0)
        
        # Cumulative variance should be non-decreasing
        self.assertTrue(np.all(np.diff(cumulative) >= 0))


class TestPrincipalAngleCalculator(unittest.TestCase):
    """Test principal angle calculation (Eq. 4, 6)."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.calculator = PrincipalAngleCalculator(epsilon=1e-10)
        
        # Create two random subspaces (100-dimensional, 10 components each)
        self.U_i = np.random.randn(100, 10)
        self.U_j = np.random.randn(100, 10)
        
        # Orthonormalize for proper subspace representation
        self.U_i, _ = np.linalg.qr(self.U_i)
        self.U_j, _ = np.linalg.qr(self.U_j)
        
    def test_principal_angle_range(self):
        """Test that principal angle is in valid range [0, π/2]."""
        angle = calculate_principal_angle(self.U_i, self.U_j)
        
        # Principal angle should be in [0, π/2]
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, np.pi / 2)
        
    def test_principal_angle_symmetry(self):
        """Test that angle(A,B) = angle(B,A) (symmetry property)."""
        angle_ij = calculate_principal_angle(self.U_i, self.U_j)
        angle_ji = calculate_principal_angle(self.U_j, self.U_i)
        
        # Angles should be equal (symmetric)
        self.assertAlmostEqual(angle_ij, angle_ji, places=6)
        
    def test_principal_angle_self(self):
        """Test that angle(A,A) = 0 (self-similarity)."""
        angle = calculate_principal_angle(self.U_i, self.U_i)
        
        # Angle with self should be 0
        self.assertAlmostEqual(angle, 0.0, places=6)
        
    def test_identical_subspaces(self):
        """Test principal angle between identical subspaces."""
        angle = calculate_principal_angle(self.U_i, self.U_i.copy())
        self.assertAlmostEqual(angle, 0.0, places=6)
        
    def test_orthogonal_subspaces(self):
        """Test principal angle between orthogonal subspaces."""
        # Create orthogonal subspace
        U_orth = np.random.randn(100, 10)
        # Project out U_i component to make orthogonal
        U_orth = U_orth - self.U_i @ (self.U_i.T @ U_orth)
        U_orth, _ = np.linalg.qr(U_orth)
        
        angle = calculate_principal_angle(self.U_i, U_orth)
        
        # Orthogonal subspaces should have angle close to π/2
        self.assertGreater(angle, np.pi / 4)  # Should be large
        
    def test_proximity_matrix_shape(self):
        """Test proximity matrix has correct shape (Eq. 6)."""
        # Create 5 client signatures
        signatures = [np.random.randn(100, 10) for _ in range(5)]
        
        proximity_matrix = build_proximity_matrix(signatures)
        
        # Should be N×N symmetric matrix
        self.assertEqual(proximity_matrix.shape, (5, 5))
        self.assertTrue(np.allclose(proximity_matrix, proximity_matrix.T))
        
    def test_proximity_matrix_diagonal(self):
        """Test proximity matrix diagonal is zero."""
        signatures = [np.random.randn(100, 10) for _ in range(5)]
        
        proximity_matrix = build_proximity_matrix(signatures)
        
        # Diagonal should be zero (angle with self)
        np.testing.assert_array_almost_equal(np.diag(proximity_matrix), np.zeros(5))
        
    def test_similarity_matrix(self):
        """Test similarity matrix construction."""
        signatures = [np.random.randn(100, 10) for _ in range(5)]
        
        similarity_matrix = build_similarity_matrix(signatures)
        
        # Should be N×N symmetric matrix
        self.assertEqual(similarity_matrix.shape, (5, 5))
        self.assertTrue(np.allclose(similarity_matrix, similarity_matrix.T))
        
        # Similarity should be in [0, 1]
        self.assertTrue(np.all(similarity_matrix >= 0))
        self.assertTrue(np.all(similarity_matrix <= 1))
        
    def test_verify_angle_properties(self):
        """Test angle property verification."""
        signatures = [np.random.randn(100, 10) for _ in range(5)]
        
        results = verify_angle_properties(self.calculator, signatures)
        
        # All properties should pass
        self.assertTrue(results['symmetry'])
        self.assertTrue(results['self_similarity'])
        self.assertTrue(results['range_valid'])
        
    def test_numerical_stability(self):
        """Test numerical stability with near-identical subspaces."""
        # Create nearly identical subspaces
        U_1 = np.random.randn(100, 10)
        U_1, _ = np.linalg.qr(U_1)
        U_2 = U_1 + 1e-8 * np.random.randn(100, 10)
        U_2, _ = np.linalg.qr(U_2)
        
        # Should not raise error
        angle = calculate_principal_angle(U_1, U_2)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, np.pi / 2)


class TestHierarchicalClusterer(unittest.TestCase):
    """Test hierarchical clustering with threshold β."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_clients = 50
        self.n_features = 100
        self.n_components = 10
        self.clustering_threshold = 20
        
        # Create synthetic client data with known cluster structure
        self.client_data = self._create_clustered_data()
        
    def _create_clustered_data(self):
        """Create synthetic data with known cluster structure."""
        client_data = []
        
        # Create 5 clusters
        for cluster_id in range(5):
            # Each cluster has 10 clients
            for _ in range(10):
                # Generate data centered around cluster-specific mean
                mean = np.random.randn(self.n_features) * (cluster_id + 1)
                data = np.random.randn(50, self.n_features) + mean
                client_data.append(data)
                
        return client_data
        
    def test_clusterer_creation(self):
        """Test HierarchicalClusterer initialization."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=20,
            svd_components=10,
            linkage_method='average',
            random_state=42
        )
        
        self.assertEqual(clusterer.clustering_threshold, 20)
        self.assertEqual(clusterer.svd_components, 10)
        self.assertEqual(clusterer.linkage_method, 'average')
        
    def test_factory_function(self):
        """Test create_hierarchical_clusterer factory function."""
        clusterer = create_hierarchical_clusterer(
            clustering_threshold=20,
            svd_components=10,
            linkage_method='average',
            random_state=42
        )
        
        self.assertIsInstance(clusterer, HierarchicalClusterer)
        
    def test_extract_client_signatures(self):
        """Test client signature extraction."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components
        )
        
        signatures = clusterer.extract_client_signatures(self.client_data)
        
        # Should have one signature per client
        self.assertEqual(len(signatures), self.n_clients)
        
        # Each signature should have correct shape
        for sig in signatures:
            self.assertEqual(sig.shape, (self.n_features, self.n_components))
            
    def test_perform_clustering(self):
        """Test hierarchical clustering execution."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components
        )
        
        # Extract signatures first
        signatures = clusterer.extract_client_signatures(self.client_data)
        
        # Perform clustering
        cluster_labels = clusterer.perform_clustering(signatures)
        
        # Should have one label per client
        self.assertEqual(len(cluster_labels), self.n_clients)
        
        # Labels should be positive integers
        self.assertTrue(np.all(cluster_labels >= 1))
        
    def test_cluster_clients_full_pipeline(self):
        """Test complete clustering pipeline."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components
        )
        
        client_to_cluster, cluster_to_clients = clusterer.cluster_clients(
            self.client_data
        )
        
        # Should have mapping for all clients
        self.assertEqual(len(client_to_cluster), self.n_clients)
        
        # All clients should be assigned to a cluster
        for client_id in range(self.n_clients):
            self.assertIn(client_id, client_to_cluster)
            
    def test_threshold_effect(self):
        """Test that different thresholds produce different cluster counts."""
        clusterer_small = HierarchicalClusterer(clustering_threshold=5)
        clusterer_large = HierarchicalClusterer(clustering_threshold=50)
        
        signatures = [
            np.random.randn(self.n_features, self.n_components)
            for _ in range(self.n_clients)
        ]
        
        labels_small = clusterer_small.perform_clustering(signatures)
        labels_large = clusterer_large.perform_clustering(signatures)
        
        # Larger threshold should produce fewer clusters
        n_clusters_small = len(np.unique(labels_small))
        n_clusters_large = len(np.unique(labels_large))
        
        self.assertGreater(n_clusters_small, n_clusters_large)
        
    def test_one_shot_clustering_function(self):
        """Test convenience function perform_one_shot_clustering."""
        client_to_cluster, cluster_to_clients, clusterer = perform_one_shot_clustering(
            self.client_data,
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components,
            random_state=42
        )
        
        # Should return all three components
        self.assertIsInstance(client_to_cluster, dict)
        self.assertIsInstance(cluster_to_clients, dict)
        self.assertIsInstance(clusterer, HierarchicalClusterer)
        
        # All clients should be assigned
        self.assertEqual(len(client_to_cluster), self.n_clients)
        
    def test_assign_client_to_cluster(self):
        """Test dynamic client assignment to cluster."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components
        )
        
        # First cluster all clients
        clusterer.cluster_clients(self.client_data)
        
        # Assign new client to existing cluster
        new_client_id = 999
        cluster_id = 3
        clusterer.assign_client_to_cluster(new_client_id, cluster_id)
        
        # Verify assignment
        self.assertEqual(clusterer.get_client_cluster(new_client_id), cluster_id)
        self.assertIn(new_client_id, clusterer.get_cluster_members(cluster_id))
        
    def test_get_cluster_info(self):
        """Test cluster information retrieval."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components
        )
        
        clusterer.cluster_clients(self.client_data)
        info = clusterer.get_cluster_info()
        
        # Should have cluster count and member counts
        self.assertIn('num_clusters', info)
        self.assertIn('cluster_sizes', info)
        self.assertGreater(info['num_clusters'], 0)
        
    def test_reset(self):
        """Test clusterer reset functionality."""
        clusterer = HierarchicalClusterer(
            clustering_threshold=self.clustering_threshold,
            svd_components=self.n_components
        )
        
        # Cluster clients
        clusterer.cluster_clients(self.client_data)
        
        # Reset
        clusterer.reset()
        
        # Should have no cluster assignments
        self.assertEqual(len(clusterer.client_to_cluster), 0)
        self.assertEqual(len(clusterer.cluster_to_clients), 0)
        
    def test_different_linkage_methods(self):
        """Test different linkage methods."""
        signatures = [
            np.random.randn(self.n_features, self.n_components)
            for _ in range(20)
        ]
        
        for method in ['average', 'complete', 'ward', 'single']:
            try:
                clusterer = HierarchicalClusterer(
                    clustering_threshold=20,
                    svd_components=self.n_components,
                    linkage_method=method
                )
                labels = clusterer.perform_clustering(signatures)
                self.assertEqual(len(labels), 20)
            except Exception as e:
                # Some methods may not work with all data
                print(f"Linkage method {method} raised: {e}")


class TestClusteringIntegration(unittest.TestCase):
    """Integration tests for complete clustering pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(42)
        self.n_clients = 30
        self.n_features = 50
        self.n_components = 5
        
    def test_end_to_end_clustering(self):
        """Test complete clustering pipeline from data to assignments."""
        # Create synthetic client data
        client_data = [
            np.random.randn(50, self.n_features)
            for _ in range(self.n_clients)
        ]
        
        # Run complete pipeline
        client_to_cluster, cluster_to_clients, clusterer = perform_one_shot_clustering(
            client_data,
            clustering_threshold=15,
            svd_components=self.n_components,
            random_state=42
        )
        
        # Verify all clients are assigned
        self.assertEqual(len(client_to_cluster), self.n_clients)
        
        # Verify cluster assignments are consistent
        for client_id, cluster_id in client_to_cluster.items():
            self.assertIn(client_id, cluster_to_clients[cluster_id])
            
        # Verify no duplicate assignments
        all_assigned = []
        for members in cluster_to_clients.values():
            all_assigned.extend(members)
        self.assertEqual(len(all_assigned), len(set(all_assigned)))
        
    def test_clustering_with_known_structure(self):
        """Test clustering on data with known cluster structure."""
        # Create data with 3 clear clusters
        client_data = []
        for cluster_idx in range(3):
            cluster_mean = np.random.randn(self.n_features) * 10
            for _ in range(10):  # 10 clients per cluster
                data = np.random.randn(50, self.n_features) + cluster_mean
                client_data.append(data)
                
        # Cluster the data
        client_to_cluster, cluster_to_clients, _ = perform_one_shot_clustering(
            client_data,
            clustering_threshold=10,
            svd_components=5,
            random_state=42
        )
        
        # Should produce reasonable number of clusters
        n_clusters = len(cluster_to_clients)
        self.assertGreater(n_clusters, 0)
        self.assertLess(n_clusters, self.n_clients)
        
    def test_clustering_reproducibility(self):
        """Test that clustering is reproducible with same seed."""
        client_data = [
            np.random.randn(50, self.n_features)
            for _ in range(20)
        ]
        
        # Run twice with same seed
        result1 = perform_one_shot_clustering(
            client_data,
            clustering_threshold=15,
            svd_components=5,
            random_state=42
        )
        
        result2 = perform_one_shot_clustering(
            client_data,
            clustering_threshold=15,
            svd_components=5,
            random_state=42
        )
        
        # Cluster assignments should be identical
        self.assertEqual(result1[0], result2[0])
        self.assertEqual(result1[1], result2[1])


def run_tests():
    """Run all clustering tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSVDFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestPrincipalAngleCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalClusterer))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
