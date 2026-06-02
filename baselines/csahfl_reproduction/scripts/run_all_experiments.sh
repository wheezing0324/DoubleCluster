#!/bin/bash
# =============================================================================
# CSAHFL: Complete Experiment Runner
# =============================================================================
# This script orchestrates all validation experiments for the CSAHFL framework:
# - Main Experiments (Table 1: Accuracy, Table 2: Training Time, Figure 3: Convergence)
# - Ablation Study (Table 3: Component Contributions)
# - Hyperparameter Analysis (Figure 4: α, E_max, β sweeps)
# - Mobility Experiments (Static vs High-Mobility scenarios)
#
# Usage:
#   bash scripts/run_all_experiments.sh [--quick] [--dataset DATASET] [--eniid ENIID]
#
# Options:
#   --quick       Run with reduced rounds for testing (default: full experiments)
#   --dataset     Specific dataset to test (fashion_mnist, cifar10, svhn)
#   --eniid       Specific ENIID setting (EIID, ENIID50%, ENIID30%)
#   --help        Display this help message
#
# Prerequisites:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Download datasets: bash scripts/download_datasets.sh
#   3. Activate virtual environment (if using one)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging
LOG_DIR="${PROJECT_DIR}/experiments/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/all_experiments_${TIMESTAMP}.log"

# Experiment flags
RUN_MAIN=true
RUN_ABLATION=true
RUN_HYPERPARAM=true
RUN_MOBILITY=true
QUICK_MODE=false
SPECIFIC_DATASET=""
SPECIFIC_ENIID=""

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $1" >> "${LOG_FILE}"
}

log_section() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
    echo "" >> "${LOG_FILE}"
    echo "========================================" >> "${LOG_FILE}"
    echo "$1" >> "${LOG_FILE}"
    echo "========================================" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed. Please install Python 3.8+."
        exit 1
    fi
    
    # Check required packages
    python3 -c "import torch" 2>/dev/null || { log_error "torch not installed"; exit 1; }
    python3 -c "import scipy" 2>/dev/null || { log_error "scipy not installed"; exit 1; }
    python3 -c "import numpy" 2>/dev/null || { log_error "numpy not installed"; exit 1; }
    python3 -c "import yaml" 2>/dev/null || { log_error "pyyaml not installed"; exit 1; }
    python3 -c "import pandas" 2>/dev/null || { log_error "pandas not installed"; exit 1; }
    python3 -c "import matplotlib" 2>/dev/null || { log_error "matplotlib not installed"; exit 1; }
    
    log_success "All dependencies are installed."
}

check_datasets() {
    log_info "Checking dataset availability..."
    
    DATA_DIR="${PROJECT_DIR}/data"
    
    # Check if data directory exists
    if [ ! -d "${DATA_DIR}" ]; then
        log_warning "Data directory not found. Attempting to download datasets..."
        bash "${SCRIPT_DIR}/download_datasets.sh"
    fi
    
    # Verify datasets exist (torchvision downloads to ~/.torch or specified location)
    log_success "Dataset check complete."
}

setup_environment() {
    log_info "Setting up experiment environment..."
    
    # Create log directory
    mkdir -p "${LOG_DIR}"
    
    # Create results directories
    mkdir -p "${PROJECT_DIR}/experiments/results/tables"
    mkdir -p "${PROJECT_DIR}/experiments/results/figures"
    
    # Set environment variables for reproducibility
    export PYTHONHASHSEED=42
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    
    log_success "Environment setup complete."
}

run_main_experiments() {
    log_section "Running Main Experiments (Table 1, Table 2, Figure 3)"
    
    cd "${PROJECT_DIR}"
    
    local args=""
    if [ "${QUICK_MODE}" = true ]; then
        args="--quick"
    fi
    
    if [ -n "${SPECIFIC_DATASET}" ]; then
        args="${args} --dataset ${SPECIFIC_DATASET}"
    fi
    
    if [ -n "${SPECIFIC_ENIID}" ]; then
        args="${args} --eniid ${SPECIFIC_ENIID}"
    fi
    
    log_info "Running: python experiments/run_main_experiments.py ${args}"
    
    if python3 experiments/run_main_experiments.py ${args}; then
        log_success "Main experiments completed successfully."
    else
        log_error "Main experiments failed."
        return 1
    fi
}

run_ablation_study() {
    log_section "Running Ablation Study (Table 3)"
    
    cd "${PROJECT_DIR}"
    
    local args=""
    if [ "${QUICK_MODE}" = true ]; then
        args="--quick"
    fi
    
    if [ -n "${SPECIFIC_DATASET}" ]; then
        args="${args} --dataset ${SPECIFIC_DATASET}"
    fi
    
    if [ -n "${SPECIFIC_ENIID}" ]; then
        args="${args} --eniid ${SPECIFIC_ENIID}"
    fi
    
    log_info "Running: python experiments/run_ablation_study.py ${args}"
    
    if python3 experiments/run_ablation_study.py ${args}; then
        log_success "Ablation study completed successfully."
    else
        log_error "Ablation study failed."
        return 1
    fi
}

run_hyperparameter_analysis() {
    log_section "Running Hyperparameter Analysis (Figure 4)"
    
    cd "${PROJECT_DIR}"
    
    local args=""
    if [ "${QUICK_MODE}" = true ]; then
        args="--quick"
    fi
    
    if [ -n "${SPECIFIC_DATASET}" ]; then
        args="${args} --dataset ${SPECIFIC_DATASET}"
    fi
    
    log_info "Running: python experiments/run_hyperparameter_analysis.py ${args}"
    
    if python3 experiments/run_hyperparameter_analysis.py ${args}; then
        log_success "Hyperparameter analysis completed successfully."
    else
        log_error "Hyperparameter analysis failed."
        return 1
    fi
}

run_mobility_experiments() {
    log_section "Running Mobility Experiments"
    
    cd "${PROJECT_DIR}"
    
    # Run with static scenario (ps=1.0)
    log_info "Running static mobility scenario (ps=1.0)..."
    python3 experiments/run_main_experiments.py --mobility 1.0
    
    # Run with high-mobility scenario (ps=0.5)
    log_info "Running high-mobility scenario (ps=0.5)..."
    python3 experiments/run_main_experiments.py --mobility 0.5
    
    log_success "Mobility experiments completed successfully."
}

generate_summary_report() {
    log_section "Generating Summary Report"
    
    cd "${PROJECT_DIR}"
    
    python3 << 'EOF'
import os
import json
from datetime import datetime

results_dir = "experiments/results"
tables_dir = os.path.join(results_dir, "tables")
figures_dir = os.path.join(results_dir, "figures")

# Generate summary report
report = {
    "timestamp": datetime.now().isoformat(),
    "results_location": results_dir,
    "tables": [],
    "figures": [],
    "status": "complete"
}

# List generated tables
if os.path.exists(tables_dir):
    for f in os.listdir(tables_dir):
        if f.endswith('.csv') or f.endswith('.json'):
            report["tables"].append(f)

# List generated figures
if os.path.exists(figures_dir):
    for f in os.listdir(figures_dir):
        if f.endswith('.png') or f.endswith('.pdf'):
            report["figures"].append(f)

# Save summary
summary_file = os.path.join(results_dir, "experiment_summary.json")
with open(summary_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Summary report saved to: {summary_file}")
print(f"Tables generated: {len(report['tables'])}")
print(f"Figures generated: {len(report['figures'])}")
EOF
    
    log_success "Summary report generated."
}

print_usage() {
    echo "CSAHFL Complete Experiment Runner"
    echo ""
    echo "Usage: bash scripts/run_all_experiments.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick           Run with reduced rounds for testing"
    echo "  --dataset NAME    Specific dataset (fashion_mnist, cifar10, svhn)"
    echo "  --eniid SETTING   Specific ENIID setting (EIID, ENIID50%, ENIID30%)"
    echo "  --main-only       Run only main experiments (Table 1, 2, Figure 3)"
    echo "  --ablation-only   Run only ablation study (Table 3)"
    echo "  --hyperparam-only Run only hyperparameter analysis (Figure 4)"
    echo "  --no-mobility     Skip mobility experiments"
    echo "  --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run_all_experiments.sh"
    echo "  bash scripts/run_all_experiments.sh --quick"
    echo "  bash scripts/run_all_experiments.sh --dataset fashion_mnist --eniid ENIID50%"
    echo "  bash scripts/run_all_experiments.sh --main-only"
    echo ""
}

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --dataset)
            SPECIFIC_DATASET="$2"
            shift 2
            ;;
        --eniid)
            SPECIFIC_ENIID="$2"
            shift 2
            ;;
        --main-only)
            RUN_ABLATION=false
            RUN_HYPERPARAM=false
            RUN_MOBILITY=false
            shift
            ;;
        --ablation-only)
            RUN_MAIN=false
            RUN_HYPERPARAM=false
            RUN_MOBILITY=false
            shift
            ;;
        --hyperparam-only)
            RUN_MAIN=false
            RUN_ABLATION=false
            RUN_MOBILITY=false
            shift
            ;;
        --no-mobility)
            RUN_MOBILITY=false
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_section "CSAHFL Complete Experiment Runner"
    
    echo "Configuration:"
    echo "  Quick Mode: ${QUICK_MODE}"
    echo "  Dataset: ${SPECIFIC_DATASET:-all}"
    echo "  ENIID: ${SPECIFIC_ENIID:-all}"
    echo "  Run Main Experiments: ${RUN_MAIN}"
    echo "  Run Ablation Study: ${RUN_ABLATION}"
    echo "  Run Hyperparameter Analysis: ${RUN_HYPERPARAM}"
    echo "  Run Mobility Experiments: ${RUN_MOBILITY}"
    echo ""
    
    # Pre-flight checks
    check_dependencies
    check_datasets
    setup_environment
    
    # Track overall success
    OVERALL_SUCCESS=true
    
    # Run experiments based on flags
    if [ "${RUN_MAIN}" = true ]; then
        run_main_experiments || OVERALL_SUCCESS=false
    fi
    
    if [ "${RUN_ABLATION}" = true ]; then
        run_ablation_study || OVERALL_SUCCESS=false
    fi
    
    if [ "${RUN_HYPERPARAM}" = true ]; then
        run_hyperparameter_analysis || OVERALL_SUCCESS=false
    fi
    
    if [ "${RUN_MOBILITY}" = true ]; then
        run_mobility_experiments || OVERALL_SUCCESS=false
    fi
    
    # Generate summary report
    generate_summary_report
    
    # Final status
    log_section "Experiment Runner Complete"
    
    if [ "${OVERALL_SUCCESS}" = true ]; then
        log_success "All experiments completed successfully!"
        log_info "Results saved to: ${PROJECT_DIR}/experiments/results/"
        log_info "Logs saved to: ${LOG_FILE}"
        exit 0
    else
        log_error "Some experiments failed. Check logs for details."
        log_info "Partial results saved to: ${PROJECT_DIR}/experiments/results/"
        log_info "Logs saved to: ${LOG_FILE}"
        exit 1
    fi
}

# Run main function
main
