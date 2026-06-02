"""
CSAHFL: Clustered Semi-Asynchronous Hierarchical Federated Learning
for Dual-layer Non-IID in Heterogeneous Edge Computing Networks

This package implements the complete CSAHFL framework with:
- Privacy-preserving one-shot clustering (Section 3.3)
- Adaptive semi-asynchronous intra-cluster aggregation (Section 3.4)
- Dynamic distribution-aware inter-cluster aggregation (Section 3.5)
"""

__version__ = '1.0.0'
__author__ = 'CSAHFL Reproduction'

# Main trainer
from .main import CSAHFLTrainer, load_config, main

__all__ = [
    'CSAHFLTrainer',
    'load_config',
    'main',
]
