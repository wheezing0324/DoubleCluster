"""
KL Divergence Utilities for CSAHFL

Implements KL divergence calculation for distribution-aware inter-cluster aggregation
(Section 3.5, Eq. 13). Used to measure difference between cluster label distribution
and global target distribution.
"""

import numpy as np
from typing import Union, Optional


def calculate_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate KL divergence D_KL(P || Q) between two probability distributions.
    
    KL divergence measures how one probability distribution P diverges from
    a second, expected probability distribution Q. Used in CSAHFL for
    distribution-aware inter-cluster aggregation (Eq. 13).
    
    Formula: D_KL(P || Q) = Σ P(i) × log(P(i) / Q(i))
    
    Args:
        p: First probability distribution (cluster distribution P_k)
        q: Second probability distribution (global target distribution P_g)
        epsilon: Small constant for numerical stability (avoid log(0))
    
    Returns:
        float: KL divergence value (>= 0, equals 0 when P=Q)
    
    Raises:
        ValueError: If distributions have different shapes or don't sum to ~1
    
    Example:
        >>> p = np.array([0.5, 0.3, 0.2])  # Cluster distribution
        >>> q = np.array([0.33, 0.33, 0.34])  # Uniform target
        >>> kl_div = calculate_kl_divergence(p, q)
        >>> print(f"KL divergence: {kl_div:.4f}")
    """
    # Convert to numpy arrays if needed
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Validate inputs
    if p.shape != q.shape:
        raise ValueError(f"Distributions must have same shape: {p.shape} vs {q.shape}")
    
    if len(p.shape) != 1:
        raise ValueError(f"Distributions must be 1D arrays, got shape {p.shape}")
    
    # Normalize distributions to ensure they sum to 1
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("Distributions must have positive sum")
    
    p_normalized = p / p_sum
    q_normalized = q / q_sum
    
    # Add epsilon for numerical stability (avoid log(0) and division by 0)
    p_safe = p_normalized + epsilon
    q_safe = q_normalized + epsilon
    
    # Re-normalize after adding epsilon
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)
    
    # Calculate KL divergence: Σ P(i) × log(P(i) / Q(i))
    # Only compute for non-zero P values (standard convention)
    kl_div = np.sum(p_normalized * np.log(p_safe / q_safe))
    
    # Ensure non-negative (KL divergence is always >= 0)
    kl_div = max(0.0, kl_div)
    
    return float(kl_div)


def normalize_distribution(
    distribution: np.ndarray,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Normalize a distribution to sum to 1.0 (valid probability distribution).
    
    Args:
        distribution: Input distribution (e.g., label counts)
        epsilon: Small constant to avoid division by zero
    
    Returns:
        np.ndarray: Normalized probability distribution
    
    Example:
        >>> counts = np.array([50, 30, 20])  # Label counts
        >>> prob_dist = normalize_distribution(counts)
        >>> print(prob_dist)  # [0.5, 0.3, 0.2]
        >>> print(np.sum(prob_dist))  # 1.0
    """
    dist = np.asarray(distribution, dtype=np.float64)
    
    # Add epsilon to avoid division by zero
    dist_sum = np.sum(dist) + epsilon
    
    return dist / dist_sum


def create_uniform_distribution(
    num_classes: int,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Create a uniform probability distribution over classes.
    
    Used as the global target distribution P_g in Eq. 13.
    
    Args:
        num_classes: Number of classes in the distribution
        epsilon: Small constant for numerical stability
    
    Returns:
        np.ndarray: Uniform distribution [1/n, 1/n, ..., 1/n]
    
    Example:
        >>> uniform = create_uniform_distribution(10)
        >>> print(uniform)  # [0.1, 0.1, ..., 0.1]
    """
    uniform = np.ones(num_classes, dtype=np.float64) / num_classes
    return uniform


def estimate_label_distribution(
    labels: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Estimate probability distribution from label counts.
    
    Converts raw label assignments into a probability distribution
    for KL divergence calculation.
    
    Args:
        labels: Array of label indices (0 to num_classes-1)
        num_classes: Total number of possible classes
    
    Returns:
        np.ndarray: Probability distribution over classes
    
    Example:
        >>> labels = np.array([0, 0, 1, 1, 1, 2])
        >>> dist = estimate_label_distribution(labels, num_classes=3)
        >>> print(dist)  # [0.33, 0.5, 0.17]
    """
    labels = np.asarray(labels)
    
    # Count occurrences of each class
    label_counts = np.zeros(num_classes, dtype=np.float64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        if 0 <= label < num_classes:
            label_counts[label] = count
    
    # Normalize to probability distribution
    return normalize_distribution(label_counts)


def calculate_js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Jensen-Shannon divergence (symmetric alternative to KL).
    
    JS divergence is symmetric and bounded [0, 1], making it useful
    for some clustering applications.
    
    Formula: JS(P || Q) = 0.5 × D_KL(P || M) + 0.5 × D_KL(Q || M)
    where M = 0.5 × (P + Q)
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small constant for numerical stability
    
    Returns:
        float: JS divergence value in range [0, 1]
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Calculate midpoint distribution
    m = 0.5 * (p + q)
    
    # Calculate JS divergence
    js_div = 0.5 * calculate_kl_divergence(p, m, epsilon) + \
             0.5 * calculate_kl_divergence(q, m, epsilon)
    
    # JS divergence is bounded [0, log(2)], normalize to [0, 1]
    js_div = js_div / np.log(2)
    
    return float(js_div)


def calculate_distribution_weight(
    cluster_distribution: np.ndarray,
    global_distribution: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate distribution weight for inter-cluster aggregation (Eq. 13).
    
    Formula: ω_k^dis = 1 / (1 + D_KL(P_k || P_g))
    
    Higher weight for clusters with distributions closer to global target.
    
    Args:
        cluster_distribution: Cluster label distribution P_k
        global_distribution: Global target distribution P_g
        epsilon: Small constant for numerical stability
    
    Returns:
        float: Distribution weight in range (0, 1]
    
    Example:
        >>> cluster_dist = np.array([0.5, 0.3, 0.2])
        >>> global_dist = np.array([0.33, 0.33, 0.34])
        >>> weight = calculate_distribution_weight(cluster_dist, global_dist)
        >>> print(f"Distribution weight: {weight:.4f}")
    """
    kl_div = calculate_kl_divergence(
        cluster_distribution,
        global_distribution,
        epsilon
    )
    
    # Eq. 13: ω_k^dis = 1 / (1 + D_KL(P_k || P_g))
    weight = 1.0 / (1.0 + kl_div)
    
    return weight


# Convenience function for direct import
def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Alias for calculate_kl_divergence for convenience."""
    return calculate_kl_divergence(p, q, epsilon)
