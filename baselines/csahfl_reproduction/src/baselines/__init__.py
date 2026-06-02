"""
Baselines module for CSAHFL reproduction.

This module contains implementations of baseline federated learning methods
for comparison with CSAHFL:
- FedAVG: Standard federated averaging
- FedProx: Federated learning with proximal term
- HierFedAVG: Synchronous hierarchical federated averaging
- FedAT: Asynchronous cross-tier federated learning
- MACFL: Mobility-aware clustered federated learning
- HiFlash: Adaptive staleness control with heterogeneity-aware association
- FedUC: Time-sharing scheduling for intra-cluster latency
"""

from .fedavg import (
    FedAVGClient,
    FedAVGServer,
    FedAVGTrainer,
    create_fedavg_client,
    create_fedavg_server
)

from .fedprox import (
    FedProxClient,
    FedProxServer,
    FedProxTrainer,
    create_fedprox_client,
    create_fedprox_server
)

from .hierfedavg import (
    HierFedAVGClient,
    HierFedAVGEdgeServer,
    HierFedAVGCloudServer,
    HierFedAVGTrainer,
    create_hierfedavg_client,
    create_hierfedavg_edge_server,
    create_hierfedavg_cloud_server
)

from .fedat import (
    FedATClient,
    FedATEdgeServer,
    FedATCloudServer,
    FedATTrainer,
    create_fedat_client,
    create_fedat_edge_server,
    create_fedat_cloud_server
)

from .macfl import (
    MACFLClient,
    MACFLEdgeServer,
    MACFLCloudServer,
    MACFLTrainer,
    create_macfl_client,
    create_macfl_edge_server,
    create_macfl_cloud_server
)

from .hiflash import (
    HiFlashClient,
    HiFlashEdgeServer,
    HiFlashCloudServer,
    HiFlashTrainer,
    create_hiflash_client,
    create_hiflash_edge_server,
    create_hiflash_cloud_server
)

from .feduc import (
    FedUCClient,
    FUCEdgeServer,
    FedUCCloudServer,
    FedUCTrainer,
    create_feduc_client,
    create_feduc_edge_server,
    create_feduc_cloud_server
)

__all__ = [
    # FedAVG
    'FedAVGClient',
    'FedAVGServer',
    'FedAVGTrainer',
    'create_fedavg_client',
    'create_fedavg_server',
    
    # FedProx
    'FedProxClient',
    'FedProxServer',
    'FedProxTrainer',
    'create_fedprox_client',
    'create_fedprox_server',
    
    # HierFedAVG
    'HierFedAVGClient',
    'HierFedAVGEdgeServer',
    'HierFedAVGCloudServer',
    'HierFedAVGTrainer',
    'create_hierfedavg_client',
    'create_hierfedavg_edge_server',
    'create_hierfedavg_cloud_server',
    
    # FedAT
    'FedATClient',
    'FedATEdgeServer',
    'FedATCloudServer',
    'FedATTrainer',
    'create_fedat_client',
    'create_fedat_edge_server',
    'create_fedat_cloud_server',
    
    # MACFL
    'MACFLClient',
    'MACFLEdgeServer',
    'MACFLCloudServer',
    'MACFLTrainer',
    'create_macfl_client',
    'create_macfl_edge_server',
    'create_macfl_cloud_server',
    
    # HiFlash
    'HiFlashClient',
    'HiFlashEdgeServer',
    'HiFlashCloudServer',
    'HiFlashTrainer',
    'create_hiflash_client',
    'create_hiflash_edge_server',
    'create_hiflash_cloud_server',
    
    # FedUC
    'FedUCClient',
    'FUCEdgeServer',
    'FedUCCloudServer',
    'FedUCTrainer',
    'create_feduc_client',
    'create_feduc_edge_server',
    'create_feduc_cloud_server',
]
