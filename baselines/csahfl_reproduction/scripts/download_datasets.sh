#!/bin/bash
# =============================================================================
# CSAHFL Dataset Download Script
# =============================================================================
# Purpose: Download and prepare all datasets required for CSAHFL experiments
# Datasets: Fashion-MNIST, CIFAR10, SVHN
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"
LOG_FILE="${SCRIPT_DIR}/download_log.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO" "$1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "SUCCESS" "$1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING" "$1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR" "$1"
}

# Create data directory
create_data_directory() {
    log_info "Creating data directory: ${DATA_DIR}"
    mkdir -p "$DATA_DIR"
    if [ $? -eq 0 ]; then
        log_success "Data directory created successfully"
    else
        log_error "Failed to create data directory"
        exit 1
    fi
}

# Check Python and required packages
check_dependencies() {
    log_info "Checking Python dependencies..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: ${PYTHON_VERSION}"
    
    # Check required packages
    python3 -c "
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    import torchvision
    print(f'TorchVision version: {torchvision.__version__}')
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except ImportError as e:
    print(f'Missing package: {e}', file=sys.stderr)
    sys.exit(1)
" || {
        log_error "Required Python packages are not installed."
        log_info "Please run: pip install -r ${PROJECT_DIR}/requirements.txt"
        exit 1
    }
    
    log_success "All dependencies are available"
}

# Download Fashion-MNIST
download_fashion_mnist() {
    log_info "Downloading Fashion-MNIST dataset..."
    
    python3 << EOF
import os
import torchvision
from torchvision import datasets

data_dir = "${DATA_DIR}"

try:
    # Download training set
    print("  Downloading Fashion-MNIST training set...")
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True
    )
    
    # Download test set
    print("  Downloading Fashion-MNIST test set...")
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print("  Fashion-MNIST download complete!")
    
except Exception as e:
    print(f"Error downloading Fashion-MNIST: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "Fashion-MNIST downloaded successfully"
    else
        log_error "Failed to download Fashion-MNIST"
        exit 1
    fi
}

# Download CIFAR10
download_cifar10() {
    log_info "Downloading CIFAR10 dataset..."
    
    python3 << EOF
import os
import torchvision
from torchvision import datasets

data_dir = "${DATA_DIR}"

try:
    # Download training set
    print("  Downloading CIFAR10 training set...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True
    )
    
    # Download test set
    print("  Downloading CIFAR10 test set...")
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print("  CIFAR10 download complete!")
    
except Exception as e:
    print(f"Error downloading CIFAR10: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "CIFAR10 downloaded successfully"
    else
        log_error "Failed to download CIFAR10"
        exit 1
    fi
}

# Download SVHN
download_svhn() {
    log_info "Downloading SVHN dataset..."
    
    python3 << EOF
import os
import torchvision
from torchvision import datasets

data_dir = "${DATA_DIR}"

try:
    # Download training set (SVHN has separate train and test splits)
    print("  Downloading SVHN training set...")
    train_dataset = datasets.SVHN(
        root=data_dir,
        split='train',
        download=True
    )
    
    # Download test set
    print("  Downloading SVHN test set...")
    test_dataset = datasets.SVHN(
        root=data_dir,
        split='test',
        download=True
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print("  SVHN download complete!")
    
except Exception as e:
    print(f"Error downloading SVHN: {e}")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "SVHN downloaded successfully"
    else
        log_error "Failed to download SVHN"
        exit 1
    fi
}

# Verify downloaded datasets
verify_datasets() {
    log_info "Verifying downloaded datasets..."
    
    python3 << EOF
import os
import sys

data_dir = "${DATA_DIR}"

required_files = [
    # Fashion-MNIST
    os.path.join(data_dir, 'FashionMNIST', 'raw', 'train-images-idx3-ubyte.gz'),
    os.path.join(data_dir, 'FashionMNIST', 'raw', 'train-labels-idx1-ubyte.gz'),
    os.path.join(data_dir, 'FashionMNIST', 'raw', 't10k-images-idx3-ubyte.gz'),
    os.path.join(data_dir, 'FashionMNIST', 'raw', 't10k-labels-idx1-ubyte.gz'),
    
    # CIFAR10
    os.path.join(data_dir, 'CIFAR10', 'raw', 'data_batch_1.bin'),
    os.path.join(data_dir, 'CIFAR10', 'raw', 'data_batch_2.bin'),
    os.path.join(data_dir, 'CIFAR10', 'raw', 'data_batch_3.bin'),
    os.path.join(data_dir, 'CIFAR10', 'raw', 'data_batch_4.bin'),
    os.path.join(data_dir, 'CIFAR10', 'raw', 'data_batch_5.bin'),
    os.path.join(data_dir, 'CIFAR10', 'raw', 'test_batch.bin'),
    
    # SVHN
    os.path.join(data_dir, 'SVHN', 'train_32x32.mat'),
    os.path.join(data_dir, 'SVHN', 'test_32x32.mat'),
]

all_exist = True
for file_path in required_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  ✓ {os.path.basename(file_path)} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {os.path.basename(file_path)} - MISSING")
        all_exist = False

if all_exist:
    print("\nAll dataset files verified successfully!")
    sys.exit(0)
else:
    print("\nSome dataset files are missing!")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        log_success "All datasets verified successfully"
    else
        log_error "Dataset verification failed"
        exit 1
    fi
}

# Display dataset summary
display_summary() {
    log_info "Dataset Download Summary"
    echo "============================================"
    echo "Data Directory: ${DATA_DIR}"
    echo ""
    echo "Downloaded Datasets:"
    echo "  - Fashion-MNIST: 28x28 grayscale images (10 classes)"
    echo "  - CIFAR10: 32x32 color images (10 classes)"
    echo "  - SVHN: 32x32 color images (10 digits)"
    echo ""
    echo "Total samples:"
    echo "  - Fashion-MNIST: 60,000 train + 10,000 test"
    echo "  - CIFAR10: 50,000 train + 10,000 test"
    echo "  - SVHN: 73,257 train + 26,032 test"
    echo ""
    echo "Log file: ${LOG_FILE}"
    echo "============================================"
}

# Main execution
main() {
    echo "============================================"
    echo "  CSAHFL Dataset Download Script"
    echo "============================================"
    echo ""
    
    # Initialize log file
    echo "CSAHFL Dataset Download Log" > "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
    
    create_data_directory
    check_dependencies
    
    echo ""
    log_info "Starting dataset downloads..."
    echo ""
    
    download_fashion_mnist
    echo ""
    
    download_cifar10
    echo ""
    
    download_svhn
    echo ""
    
    verify_datasets
    echo ""
    
    display_summary
    
    echo ""
    log_success "All datasets downloaded and verified successfully!"
    echo ""
    echo "You can now run CSAHFL experiments:"
    echo "  python src/main.py --dataset fashion_mnist"
    echo "  python experiments/run_main_experiments.py"
    echo ""
}

# Run main function
main "$@"
