# CSAHFL: Clustered Semi-Asynchronous Hierarchical Federated Learning

## Overview

This repository contains a complete reproduction of **CSAHFL** (Clustered Semi-Asynchronous Hierarchical Federated Learning), a three-component HFL framework addressing dual-layer non-IID data distributions in heterogeneous edge computing networks.

**Paper**: CSAHFL: Clustered Semi-Asynchronous Hierarchical Federated Learning for Dual-layer Non-IID in Heterogeneous Edge Computing Networks

**Core Contribution**: A hierarchical federated learning framework that addresses dual-layer non-IID through:
1. Privacy-preserving one-shot clustering based on data distribution signatures
2. Adaptive semi-asynchronous intra-cluster aggregation with workload adjustment
3. Dynamic distribution-aware inter-cluster aggregation

## 📁 Project Structure

```
csahfl_reproduction/
├── README.md                          # This documentation file
├── requirements.txt                   # Python dependencies
├── config/
│   ├── default_config.yaml           # System hyperparameters (α=1.5, β=20, E_max=10)
│   ├── dataset_config.yaml           # Dataset-specific settings
│   └── experiment_config.yaml        # Experiment configurations
├── src/
│   ├── main.py                       # Main CSAHFL training loop (Algorithm 1)
│   ├── models/
│   │   ├── cnn_fashion_mnist.py      # CNN for Fashion-MNIST
│   │   ├── cnn_cifar10.py            # CNN for CIFAR10
│   │   └── cnn_svhn.py               # CNN for SVHN
│   ├── clustering/
│   │   ├── svd_feature_extractor.py  # Truncated SVD (Eq. 5)
│   │   ├── principal_angle.py        # Principal angles (Eq. 4, 6)
│   │   └── hierarchical_clusterer.py # Clustering with threshold β
│   ├── aggregation/
│   │   ├── intra_cluster.py          # Semi-async aggregation (Eq. 7-10)
│   │   └── inter_cluster.py          # Distribution-aware (Eq. 11-14)
│   ├── data/
│   │   ├── data_loader.py            # Dataset loading
│   │   ├── non_iid_partition.py      # Dual-layer non-IID generation
│   │   └── mobility_simulator.py     # Markov chain mobility (ps)
│   ├── clients/
│   │   ├── client.py                 # Base client class
│   │   └── workload_adapter.py       # Workload adjustment (Eq. 9)
│   ├── edge_servers/
│   │   └── edge_server.py            # Edge aggregation management
│   ├── cloud/
│   │   └── cloud_server.py           # Global model aggregation
│   ├── utils/
│   │   ├── metrics.py                # Accuracy, training time
│   │   ├── kl_divergence.py          # KL divergence calculation
│   │   └── logger.py                 # Experiment logging
│   └── baselines/
│       ├── fedavg.py                 # Standard federated averaging
│       ├── fedprox.py                # FedProx with proximal term
│       ├── hierfedavg.py             # Synchronous HFL
│       ├── fedat.py                  # Async cross-tier aggregation
│       ├── macfl.py                  # Mobility-aware clustering
│       ├── hiflash.py                # Adaptive staleness control
│       └── feduc.py                  # Time-sharing scheduling
├── experiments/
│   ├── run_main_experiments.py       # Table 1, 2 experiments
│   ├── run_ablation_study.py         # Table 3 ablation
│   ├── run_hyperparameter_analysis.py # Figure 4 analysis
│   └── results/
│       ├── tables/
│       └── figures/
├── tests/
│   ├── test_clustering.py
│   ├── test_aggregation.py
│   └── test_mobility.py
└── scripts/
    ├── download_datasets.sh
    └── run_all_experiments.sh
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv csahfl_env
source csahfl_env/bin/activate  # Linux/Mac
# or
csahfl_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download datasets
bash scripts/download_datasets.sh
```

### 2. Run CSAHFL Training

```bash
# Basic training on Fashion-MNIST with ENIID50%
python src/main.py --dataset fashion_mnist --eniid ENIID50% --mobility 1.0

# Training with custom parameters
python src/main.py --dataset cifar10 --eniid ENIID30% --mobility 0.5 --seed 42
```

### 3. Run Experiments

```bash
# Main experiments (Table 1, 2, Figure 3)
python experiments/run_main_experiments.py --all

# Ablation study (Table 3)
python experiments/run_ablation_study.py --dataset fashion_mnist

# Hyperparameter analysis (Figure 4)
python experiments/run_hyperparameter_analysis.py --all
```

## 📊 Key Components

### Component 1: Privacy-Preserving One-Shot Clustering (Section 3.3)

**Location**: `src/clustering/`

Clients compute truncated SVD on local data and share only top-p singular vectors as privacy-preserving signatures:

```python
# Eq. 5: Truncated SVD
U_i^p = [u_1, u_2, ..., u_p] where p << rank(D_i)

# Eq. 4: Principal Angle
Θ_1(X, Y) = min arccos(|x^T y| / (||x|| ||y||))

# Eq. 6: Proximity Matrix
A[i,j] = Θ_1(U_i^p, U_j^p) for i,j = 1,...,N
```

**Key Parameters**:
- `svd_components` (p): 10 (number of singular vectors)
- `clustering_threshold` (β): 20 (optimal threshold for hierarchical clustering)
- `linkage_method`: 'average'

### Component 2: Adaptive Semi-Asynchronous Intra-Cluster Aggregation (Section 3.4)

**Location**: `src/aggregation/intra_cluster.py`, `src/clients/workload_adapter.py`

Adaptive aggregation interval and workload adjustment to handle stragglers:

```python
# Eq. 7: Total Time
t_total = t_cmp + t_com
t_cmp = E_max × N_BS × t_unit
t_com = M / B_w

# Eq. 8: Adaptive Aggregation Interval
T_θ = median(T_total) + α × IQR(T_total)

# Eq. 9: Workload Adjustment
E_i = max(min((T_θ - t_com) / (N_BS × t_unit), E_max), 1)

# Eq. 10: Semi-Asynchronous Aggregation
w_edge^{k,t} = (1/N_edge^k) × (Σ_{i∈S_k} n_i w_i^t + Σ_{j∈A_k} τ_j n_j w_j^t)
```

**Key Parameters**:
- `max_epochs` (E_max): 10
- `tolerance_alpha` (α): 1.5
- `batch_size` (B): 10

### Component 3: Dynamic Distribution-Aware Inter-Cluster Aggregation (Section 3.5)

**Location**: `src/aggregation/inter_cluster.py`, `src/cloud/cloud_server.py`

Combines quantity-based and distribution-based weighting for global aggregation:

```python
# Eq. 12: Quantity Weight
ω_k^qua = N_edge^k / Σ_{k=1}^K N_edge^k

# Eq. 13: Distribution Weight
ω_k^dis = 1 / (1 + D_KL(P_k || P_g))

# Eq. 14: Combined Weight
λ_k = (ω_k^qua × ω_k^dis) / Σ_{j=1}^K (ω_j^qua × ω_j^dis)

# Eq. 11: Global Aggregation
w_g = Σ_{k=1}^K λ_k w_edge^k
```

**Key Parameters**:
- `kl_epsilon`: 1e-10 (numerical stability)

## 📈 Expected Results

### Table 1: Main Accuracy Results (Static Scenario, ps=1.0)

| Dataset       | ENIID Setting | Target Accuracy | CSAHFL (Expected) |
|---------------|---------------|-----------------|-------------------|
| Fashion-MNIST | EIID          | 92.14%          | ±2% tolerance     |
| Fashion-MNIST | ENIID50%      | 91.11%          | ±2% tolerance     |
| Fashion-MNIST | ENIID30%      | 88.49%          | ±2% tolerance     |
| CIFAR10       | EIID          | 69.35%          | ±2% tolerance     |
| CIFAR10       | ENIID50%      | 68.06%          | ±2% tolerance     |
| CIFAR10       | ENIID30%      | 64.75%          | ±2% tolerance     |
| SVHN          | EIID          | 90.68%          | ±2% tolerance     |
| SVHN          | ENIID50%      | 89.62%          | ±2% tolerance     |
| SVHN          | ENIID30%      | 88.17%          | ±2% tolerance     |

### Table 2: Training Time Analysis (Fashion-MNIST)

| Target Accuracy | CSAHFL Time | Baseline Ratio |
|-----------------|-------------|----------------|
| 60%             | 724s        | 1.0x (fastest) |
| 85%             | 1392s       | 1.0x (fastest) |

### Table 3: Ablation Study

| Variant   | Accuracy Drop | Time Increase |
|-----------|---------------|---------------|
| w/o C     | >5%           | Minimal       |
| w/o SA    | Minimal       | >3x           |
| w/o D     | >1%           | Minimal       |

### Figure 4: Optimal Hyperparameters

- **Optimal α**: 1.5 (balances straggler tolerance vs. staleness)
- **Optimal E_max**: 10 (balances local training vs. communication)
- **Optimal β**: 20 (balances cluster count vs. homogeneity)

## 🔧 Configuration

### System Configuration (`config/default_config.yaml`)

```yaml
system:
  num_clients: 200        # N = 200 clients
  num_edge_servers: 5     # M = 5 edge servers
  seed: 42                # Random seed for reproducibility
  device: "cuda"          # Training device

clustering:
  svd_components: 10      # p = 10 singular vectors
  clustering_threshold: 20 # β = 20 optimal threshold

intra_cluster:
  max_epochs: 10          # E_max = 10
  tolerance_alpha: 1.5    # α = 1.5

training:
  learning_rate: 0.01     # SGD learning rate
  batch_size: 64

experiment:
  total_cloud_rounds: 50  # rc = 50
  edge_rounds_per_cloud: 3 # re = 3
```

### Dataset Configuration (`config/dataset_config.yaml`)

Supports three datasets with automatic download:
- **Fashion-MNIST**: 28×28 grayscale, 10 classes
- **CIFAR10**: 32×32 RGB, 10 classes
- **SVHN**: 32×32 RGB, 10 classes

### Non-IID Settings

- **EIID**: Edge-layer IID (all ESs get all 10 classes)
- **ENIID50%**: Each ES gets 5 of 10 classes
- **ENIID30%**: Each ES gets 3 of 10 classes
- **Client-layer**: Always 20% (2 classes per client)

### Mobility Scenarios

- **Static**: `staying_probability = 1.0` (no movement)
- **High-Mobility**: `staying_probability = 0.5` (50% stay)

## 🧪 Running Experiments

### Main Experiments (Table 1, 2, Figure 3)

```bash
# Run all main experiments
python experiments/run_main_experiments.py --all

# Run specific dataset
python experiments/run_main_experiments.py --dataset fashion_mnist

# Run specific ENIID setting
python experiments/run_main_experiments.py --eniid ENIID50%

# Run with mobility
python experiments/run_main_experiments.py --mobility 0.5
```

### Ablation Study (Table 3)

```bash
# Full ablation study
python experiments/run_ablation_study.py --all

# Specific variant
python experiments/run_ablation_study.py --variant without_clustering
```

### Hyperparameter Analysis (Figure 4)

```bash
# Full hyperparameter sweep
python experiments/run_hyperparameter_analysis.py --all

# Specific parameter
python experiments/run_hyperparameter_analysis.py --param alpha
python experiments/run_hyperparameter_analysis.py --param emax
python experiments/run_hyperparameter_analysis.py --param beta
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_clustering.py -v
python -m pytest tests/test_aggregation.py -v
python -m pytest tests/test_mobility.py -v
```

## 📦 Dependencies

### Core Dependencies

- `numpy>=1.21.0` - Array operations, SVD computation
- `scipy>=1.7.0` - Hierarchical clustering, statistics
- `torch>=1.9.0` - Neural network models
- `torchvision>=0.10.0` - Dataset loading
- `scikit-learn>=0.24.0` - ML utilities
- `pyyaml>=5.4.0` - Configuration parsing
- `tqdm>=4.62.0` - Progress bars
- `matplotlib>=3.4.0` - Plotting
- `pandas>=1.3.0` - Results dataframes
- `seaborn>=0.11.0` - Enhanced visualization

### Hardware Requirements

- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum
- **GPU**: Optional (NVIDIA with 4GB+ VRAM, CUDA 11.0+)
- **Storage**: 10GB for datasets and checkpoints

## 🔬 Baseline Methods

The following 7 baseline methods are implemented for comparison:

1. **FedAVG**: Standard federated averaging (cloud-based only)
2. **FedProx**: Adds proximal term to local objective
3. **HierFedAVG**: Synchronous HFL with client-edge-cloud hierarchy
4. **FedAT**: Synchronous intra-tier, asynchronous cross-tier
5. **MACFL**: Mobility-aware clustering with redesigned aggregation
6. **HiFlash**: Adaptive staleness control + heterogeneity-aware association
7. **FedUC**: Time-sharing scheduling for intra-cluster latency

## 📝 Algorithm Summary

### Algorithm 1: CSAHFL Training Loop

```
Input: N clients, M edge servers, total_cloud_rounds rc, edge_rounds_per_cloud re
Output: Global model w_g

1. Initialize global model w_g
2. Perform one-shot clustering (Section 3.3)
   - Each client computes SVD signature U_i^p
   - Build proximity matrix using principal angles
   - Apply hierarchical clustering with threshold β
3. For each cloud round r = 1 to rc:
   a. Handle client mobility (if ps < 1.0)
   b. For each edge round e = 1 to re:
      - Each client trains locally with adjusted epochs E_i
      - Edge servers perform semi-asynchronous aggregation (Eq. 10)
   c. Cloud server performs distribution-aware aggregation (Eq. 11)
   d. Evaluate global model on test data
4. Return final global model w_g
```

## 🎯 Validation Criteria

### Success Criteria

1. **Accuracy**: CSAHFL accuracy within ±2% of target values (Table 1)
2. **Performance**: CSAHFL outperforms all 7 baselines in each setting
3. **Training Time**: CSAHFL achieves target accuracy fastest (Table 2)
4. **Ablation**: Each component contributes as expected (Table 3)
   - Clustering: >5% accuracy improvement
   - Semi-asynchronous: >3x time reduction
   - Distribution-aware: >1% accuracy improvement
5. **Hyperparameters**: Optimal values match paper (α=1.5, E_max=10, β=20)
6. **Mobility**: Accuracy drop <5% from static to high-mobility

## 📚 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{csahfl2024,
  title={CSAHFL: Clustered Semi-Asynchronous Hierarchical Federated Learning for Dual-layer Non-IID in Heterogeneous Edge Computing Networks},
  author={CSAHFL Authors},
  journal={IEEE Transactions on Mobile Computing},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original CSAHFL paper authors for the research
- PyTorch team for the deep learning framework
- torchvision team for dataset utilities

## 📞 Support

For issues and questions:
1. Check existing issues in the repository
2. Review the documentation above
3. Open a new issue with detailed description

---

**Last Updated**: 2024
**Version**: 1.0.0
