"""
Cloud server module for CSAHFL framework.

This module provides the cloud-level server that coordinates global model
aggregation across all edge servers using distribution-aware inter-cluster
aggregation (Section 3.5, Eq. 11-14).
"""

from .cloud_server import CloudServer, create_cloud_server

__all__ = ['CloudServer', 'create_cloud_server']
