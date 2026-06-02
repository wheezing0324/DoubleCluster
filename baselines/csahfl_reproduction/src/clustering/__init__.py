"""
Clustering module for CSAHFL - Privacy-preserving one-shot clustering.

This module implements the privacy-preserving clustering component (Section 3.3)
including truncated SVD feature extraction, principal angle calculation, and
hierarchical clustering with threshold β.
"""

from .svd_feature_extractor import (
    SVDFeatureExtractor,
    create_svd_extractor,
    extract_client_signature
)

from .principal_angle import (
    PrincipalAngleCalculator,
    calculate_principal_angle,
    build_proximity_matrix,
    build_similarity_matrix,
    verify_angle_properties
)

from .hierarchical_clusterer import (
    HierarchicalClusterer,
    create_hierarchical_clusterer,
    perform_one_shot_clustering
)

__all__ = [
    # SVD Feature Extractor
    'SVDFeatureExtractor',
    'create_svd_extractor',
    'extract_client_signature',
    
    # Principal Angle Calculator
    'PrincipalAngleCalculator',
    'calculate_principal_angle',
    'build_proximity_matrix',
    'build_similarity_matrix',
    'verify_angle_properties',
    
    # Hierarchical Clusterer
    'HierarchicalClusterer',
    'create_hierarchical_clusterer',
    'perform_one_shot_clustering'
]
