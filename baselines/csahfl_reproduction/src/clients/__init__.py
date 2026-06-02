"""
Client module for CSAHFL framework.

This module provides client-side functionality for federated learning,
including local training, workload adjustment, and model weight management.
"""

from .client import Client, create_client
from .workload_adapter import WorkloadAdapter, create_workload_adapter

__all__ = [
    'Client',
    'create_client',
    'WorkloadAdapter',
    'create_workload_adapter',
]
