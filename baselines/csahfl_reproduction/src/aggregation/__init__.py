"""
CSAHFL Aggregation Module

This module provides aggregation mechanisms for the CSAHFL framework:
- Intra-cluster aggregation: Adaptive semi-asynchronous aggregation at edge server level (Section 3.4)
- Inter-cluster aggregation: Distribution-aware aggregation at cloud server level (Section 3.5)
"""

from .intra_cluster import IntraClusterAggregator, create_intra_cluster_aggregator
from .inter_cluster import InterClusterAggregator, create_inter_cluster_aggregator

__all__ = [
    'IntraClusterAggregator',
    'create_intra_cluster_aggregator',
    'InterClusterAggregator',
    'create_inter_cluster_aggregator',
]
