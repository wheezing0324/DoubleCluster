"""
Hierarchical Clusterer for CSAHFL
Implements privacy-preserving one-shot clustering using hierarchical clustering with threshold β (Section 3.3, Eq. 6)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings

from .svd_feature_extractor import SVDFeatureExtractor, extract_client_signature
from .principal_angle import PrincipalAngleCalculator, build_proximity_matrix


class HierarchicalClusterer:
    """
    Performs hierarchical clustering on clients based on their data distribution signatures.
    Implements one-shot clustering at the beginning of training (Section 3.3).
    
    Args:
        clustering_threshold: β parameter for cutting dendrogram (default=20, optimal per paper)
        svd_components: p parameter for truncated SVD (default=10)
        linkage_method: Method for hierarchical clustering ('average', 'complete', 'single', 'ward')
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        clustering_threshold: float = 20.0,
        svd_components: int = 10,
        linkage_method: str = 'average',
        random_state: Optional[int] = None
    ):
        self.clustering_threshold = clustering_threshold  # β parameter
        self.svd_components = svd_components  # p parameter
        self.linkage_method = linkage_method
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize components
        self.svd_extractor = SVDFeatureExtractor(
            n_components=svd_components,
            random_state=random_state
        )
        self.angle_calculator = PrincipalAngleCalculator()
        
        # Clustering results
        self.client_clusters: Dict[int, int] = {}  # client_id -> cluster_id
        self.cluster_clients: Dict[int, List[int]] = {}  # cluster_id -> [client_ids]
        self.proximity_matrix: Optional[np.ndarray] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.num_clusters: int = 0
        
    def extract_client_signatures(
        self,
        client_data: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Extract privacy-preserving SVD signatures from all clients.
        Each client computes SVD on local data and retains top-p singular vectors.
        
        Args:
            client_data: Dictionary mapping client_id to data array (N, H, W, C) or (N, H, W)
            
        Returns:
            Dictionary mapping client_id to SVD signature (p, d) where d is flattened feature dim
        """
        signatures = {}
        
        for client_id, data in client_data.items():
            try:
                # Extract signature using truncated SVD (Eq. 5)
                signature = extract_client_signature(data, self.svd_components)
                signatures[client_id] = signature
            except Exception as e:
                warnings.warn(f"Failed to extract signature for client {client_id}: {e}")
                # Fallback: random signature with correct dimensions
                if len(data.shape) == 4:
                    feat_dim = data.shape[1] * data.shape[2] * data.shape[3]
                else:
                    feat_dim = data.shape[1] * data.shape[2]
                signatures[client_id] = np.random.randn(self.svd_components, min(feat_dim, self.svd_components))
        
        return signatures
    
    def build_proximity_matrix(
        self,
        signatures: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Build N×N proximity matrix using principal angles (Eq. 6).
        A[i,j] = Θ_1(U_i^p, U_j^p) where smaller values = higher similarity.
        
        Args:
            signatures: Dictionary mapping client_id to SVD signature
            
        Returns:
            N×N symmetric proximity matrix
        """
        client_ids = sorted(signatures.keys())
        n_clients = len(client_ids)
        
        # Build ordered list of signatures
        sig_list = [signatures[cid] for cid in client_ids]
        
        # Build proximity matrix using principal angles
        proximity = build_proximity_matrix(sig_list)
        
        # Store mapping for later use
        self.client_id_to_idx = {cid: idx for idx, cid in enumerate(client_ids)}
        self.idx_to_client_id = {idx: cid for cid, idx in self.client_id_to_idx.items()}
        
        return proximity
    
    def perform_clustering(
        self,
        proximity_matrix: np.ndarray
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Perform hierarchical clustering using scipy with threshold β.
        
        Args:
            proximity_matrix: N×N symmetric matrix of pairwise distances
            
        Returns:
            Tuple of (client_clusters, cluster_clients) dictionaries
        """
        n_clients = proximity_matrix.shape[0]
        
        # Convert proximity matrix to condensed distance matrix for scipy
        # scipy.linkage expects condensed form (upper triangle, flattened)
        condensed_dist = squareform(proximity_matrix)
        
        # Perform hierarchical clustering with specified linkage method
        # 'average' is recommended for this use case
        self.linkage_matrix = linkage(condensed_dist, method=self.linkage_method)
        
        # Cut dendrogram at threshold β to get flat clusters
        # criterion='distance' uses the threshold directly
        cluster_labels = fcluster(
            self.linkage_matrix,
            t=self.clustering_threshold,
            criterion='distance'
        )
        
        # Build client-to-cluster mapping
        client_clusters = {}
        cluster_clients = {}
        
        for idx, cluster_id in enumerate(cluster_labels):
            client_id = self.idx_to_client_id[idx]
            client_clusters[client_id] = int(cluster_id)
            
            if cluster_id not in cluster_clients:
                cluster_clients[cluster_id] = []
            cluster_clients[cluster_id].append(client_id)
        
        self.client_clusters = client_clusters
        self.cluster_clients = cluster_clients
        self.num_clusters = len(cluster_clients)
        
        return client_clusters, cluster_clients
    
    def cluster_clients(
        self,
        client_data: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Complete clustering pipeline: extract signatures → build proximity → cluster.
        This is the one-shot clustering performed at the beginning of training.
        
        Args:
            client_data: Dictionary mapping client_id to data array
            
        Returns:
            Tuple of (client_clusters, cluster_clients) dictionaries
        """
        # Step 1: Extract privacy-preserving signatures (Eq. 5)
        signatures = self.extract_client_signatures(client_data)
        
        # Step 2: Build proximity matrix using principal angles (Eq. 6)
        self.proximity_matrix = self.build_proximity_matrix(signatures)
        
        # Step 3: Perform hierarchical clustering with threshold β
        client_clusters, cluster_clients = self.perform_clustering(self.proximity_matrix)
        
        return client_clusters, cluster_clients
    
    def assign_client_to_cluster(
        self,
        client_id: int,
        client_data: np.ndarray,
        existing_signatures: Dict[int, np.ndarray]
    ) -> int:
        """
        Assign a new client to an existing cluster (for mobility scenarios).
        Finds the cluster with most similar clients based on principal angles.
        
        Args:
            client_id: ID of the new/moving client
            client_data: Client's local data
            existing_signatures: Signatures of existing clients
            
        Returns:
            Assigned cluster ID
        """
        # Extract signature for new client
        new_signature = extract_client_signature(client_data, self.svd_components)
        
        # Calculate average distance to each cluster
        cluster_distances = {}
        
        for cluster_id, members in self.cluster_clients.items():
            distances = []
            for member_id in members:
                if member_id in existing_signatures:
                    angle = self.angle_calculator.calculate_principal_angle(
                        new_signature,
                        existing_signatures[member_id]
                    )
                    distances.append(angle)
            
            if distances:
                cluster_distances[cluster_id] = np.mean(distances)
        
        # Assign to closest cluster
        if cluster_distances:
            assigned_cluster = min(cluster_distances, key=cluster_distances.get)
        else:
            # Fallback: assign to smallest cluster for balance
            assigned_cluster = min(self.cluster_clients, key=lambda k: len(self.cluster_clients[k]))
        
        # Update cluster assignments
        self.client_clusters[client_id] = assigned_cluster
        if client_id not in self.cluster_clients[assigned_cluster]:
            self.cluster_clients[assigned_cluster].append(client_id)
        
        return assigned_cluster
    
    def get_cluster_info(self) -> Dict:
        """
        Get information about current clustering.
        
        Returns:
            Dictionary with clustering statistics
        """
        cluster_sizes = [len(members) for members in self.cluster_clients.values()]
        
        return {
            'num_clusters': self.num_clusters,
            'num_clients': len(self.client_clusters),
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'std_cluster_size': np.std(cluster_sizes) if cluster_sizes else 0,
            'clustering_threshold': self.clustering_threshold,
            'svd_components': self.svd_components,
            'linkage_method': self.linkage_method
        }
    
    def get_client_cluster(self, client_id: int) -> Optional[int]:
        """
        Get cluster ID for a specific client.
        
        Args:
            client_id: Client ID to look up
            
        Returns:
            Cluster ID or None if client not found
        """
        return self.client_clusters.get(client_id)
    
    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """
        Get list of client IDs in a specific cluster.
        
        Args:
            cluster_id: Cluster ID to query
            
        Returns:
            List of client IDs in the cluster
        """
        return self.cluster_clients.get(cluster_id, [])
    
    def reset(self):
        """Reset clustering state for new experiment."""
        self.client_clusters = {}
        self.cluster_clients = {}
        self.proximity_matrix = None
        self.linkage_matrix = None
        self.num_clusters = 0


def create_hierarchical_clusterer(
    clustering_threshold: float = 20.0,
    svd_components: int = 10,
    linkage_method: str = 'average',
    random_state: Optional[int] = None
) -> HierarchicalClusterer:
    """
    Factory function to create HierarchicalClusterer instance.
    
    Args:
        clustering_threshold: β parameter (default=20, optimal per paper)
        svd_components: p parameter for SVD (default=10)
        linkage_method: Hierarchical clustering method
        random_state: Random seed
        
    Returns:
        Configured HierarchicalClusterer instance
    """
    return HierarchicalClusterer(
        clustering_threshold=clustering_threshold,
        svd_components=svd_components,
        linkage_method=linkage_method,
        random_state=random_state
    )


def perform_one_shot_clustering(
    client_data: Dict[int, np.ndarray],
    clustering_threshold: float = 20.0,
    svd_components: int = 10,
    random_state: Optional[int] = None
) -> Tuple[Dict[int, int], Dict[int, List[int]], HierarchicalClusterer]:
    """
    Convenience function to perform one-shot clustering in a single call.
    
    Args:
        client_data: Dictionary mapping client_id to data array
        clustering_threshold: β parameter
        svd_components: p parameter for SVD
        random_state: Random seed
        
    Returns:
        Tuple of (client_clusters, cluster_clients, clusterer)
    """
    clusterer = create_hierarchical_clusterer(
        clustering_threshold=clustering_threshold,
        svd_components=svd_components,
        random_state=random_state
    )
    
    client_clusters, cluster_clients = clusterer.cluster_clients(client_data)
    
    return client_clusters, cluster_clients, clusterer
