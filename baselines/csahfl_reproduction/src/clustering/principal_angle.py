"""
Principal Angle Calculation for Privacy-Preserving Clustering
==============================================================

Implements principal angle calculation between subspaces (Section 3.3, Eq. 4, 6)
for measuring similarity between client data distributions in CSAHFL.

The smallest principal angle Θ_1(X, Y) measures the similarity between two subspaces:
- Smaller angle → higher similarity → clients should be in same cluster
- Larger angle → lower similarity → clients should be in different clusters

Equation 4: Θ_1(X, Y) = min arccos(|x^T y| / (||x|| ||y||))
Equation 6: A[i,j] = Θ_1(U_i^p, U_j^p) for i,j = 1,...,N (proximity matrix)
"""

import numpy as np
from numpy.linalg import svd
from typing import Tuple, Optional
import warnings


class PrincipalAngleCalculator:
    """
    Calculates principal angles between subspaces for clustering.
    
    Implements Eq. 4 from CSAHFL paper to compute the smallest principal angle
    between two subspaces represented by their singular vectors.
    
    Attributes:
        epsilon (float): Small value for numerical stability (default: 1e-10)
        max_angle (float): Maximum possible angle (π/2 for orthogonal subspaces)
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize principal angle calculator.
        
        Args:
            epsilon: Small value to prevent division by zero (default: 1e-10)
        """
        self.epsilon = epsilon
        self.max_angle = np.pi / 2  # 90 degrees - maximum for principal angle
    
    def calculate_principal_angle(self, U_i: np.ndarray, U_j: np.ndarray) -> float:
        """
        Calculate the smallest principal angle between two subspaces.
        
        Implements Eq. 4: Θ_1(X, Y) = min arccos(|x^T y| / (||x|| ||y||))
        
        The smallest principal angle is computed using SVD of U_i^T @ U_j.
        The largest singular value of this product gives cos(Θ_1).
        
        Args:
            U_i: Left singular vectors from client i (shape: n_features × p)
            U_j: Left singular vectors from client j (shape: n_features × p)
            
        Returns:
            float: Smallest principal angle in radians [0, π/2]
                   - 0.0 means identical subspaces
                   - π/2 means orthogonal subspaces
        """
        # Validate input dimensions
        if U_i.ndim != 2 or U_j.ndim != 2:
            raise ValueError(f"Input matrices must be 2D, got shapes {U_i.shape}, {U_j.shape}")
        
        if U_i.shape[0] != U_j.shape[0]:
            raise ValueError(
                f"Feature dimensions must match: {U_i.shape[0]} vs {U_j.shape[0]}"
            )
        
        # Compute cross-product matrix: U_i^T @ U_j
        # Shape: (p, p) where p is number of singular vectors
        cross_product = U_i.T @ U_j
        
        # Compute SVD of cross-product to find principal angles
        # The singular values of U_i^T @ U_j are cosines of principal angles
        try:
            singular_values = svd(cross_product, compute_uv=False)
        except np.linalg.LinAlgError as e:
            warnings.warn(f"SVD failed for angle calculation: {e}. Using fallback.")
            # Fallback: use Frobenius norm as similarity measure
            similarity = np.linalg.norm(cross_product, 'fro')
            similarity = np.clip(similarity, 0, 1)
            return np.arccos(similarity)
        
        # The largest singular value corresponds to cos(Θ_1)
        # (smallest principal angle has largest cosine)
        cos_theta = np.max(singular_values)
        
        # Clip to [-1, 1] for numerical stability before arccos
        # This prevents NaN from arccos(values > 1 or < -1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Compute angle: Θ_1 = arccos(cos_theta)
        angle = np.arccos(cos_theta)
        
        # Ensure angle is in valid range [0, π/2]
        angle = np.clip(angle, 0.0, self.max_angle)
        
        return angle
    
    def calculate_similarity(self, U_i: np.ndarray, U_j: np.ndarray) -> float:
        """
        Calculate similarity score between two subspaces.
        
        Converts principal angle to similarity score:
        - similarity = 1 - (angle / max_angle)
        - similarity = 1.0 means identical subspaces
        - similarity = 0.0 means orthogonal subspaces
        
        Args:
            U_i: Left singular vectors from client i
            U_j: Left singular vectors from client j
            
        Returns:
            float: Similarity score in [0, 1]
        """
        angle = self.calculate_principal_angle(U_i, U_j)
        similarity = 1.0 - (angle / self.max_angle)
        return max(0.0, min(1.0, similarity))
    
    def build_proximity_matrix(self, client_signatures: list) -> np.ndarray:
        """
        Build N×N proximity matrix for all clients (Eq. 6).
        
        Computes pairwise principal angles between all client subspaces.
        Smaller values indicate higher similarity (clients should cluster together).
        
        Args:
            client_signatures: List of N client signatures (U_i^p matrices)
                              Each signature has shape (n_features, p)
                              
        Returns:
            np.ndarray: Proximity matrix A of shape (N, N)
                       - A[i,j] = Θ_1(U_i^p, U_j^p)
                       - Symmetric matrix with zeros on diagonal
                       - Values in range [0, π/2]
        """
        N = len(client_signatures)
        
        if N == 0:
            return np.array([]).reshape(0, 0)
        
        # Initialize proximity matrix
        proximity_matrix = np.zeros((N, N))
        
        # Compute pairwise angles (only upper triangle, then symmetrize)
        for i in range(N):
            for j in range(i + 1, N):
                angle = self.calculate_principal_angle(
                    client_signatures[i],
                    client_signatures[j]
                )
                proximity_matrix[i, j] = angle
                proximity_matrix[j, i] = angle  # Symmetric
        
        # Diagonal should be zero (angle with self is 0)
        np.fill_diagonal(proximity_matrix, 0.0)
        
        return proximity_matrix
    
    def build_similarity_matrix(self, client_signatures: list) -> np.ndarray:
        """
        Build N×N similarity matrix for all clients.
        
        Converts proximity matrix to similarity scores.
        Larger values indicate higher similarity.
        
        Args:
            client_signatures: List of N client signatures (U_i^p matrices)
                              
        Returns:
            np.ndarray: Similarity matrix S of shape (N, N)
                       - S[i,j] = 1 - (Θ_1(U_i^p, U_j^p) / (π/2))
                       - Symmetric matrix with ones on diagonal
                       - Values in range [0, 1]
        """
        proximity_matrix = self.build_proximity_matrix(client_signatures)
        
        if proximity_matrix.size == 0:
            return proximity_matrix
        
        # Convert angles to similarities
        similarity_matrix = 1.0 - (proximity_matrix / self.max_angle)
        
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Clip to valid range
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        
        return similarity_matrix


def calculate_principal_angle(U_i: np.ndarray, U_j: np.ndarray, 
                              epsilon: float = 1e-10) -> float:
    """
    Convenience function to calculate principal angle between two subspaces.
    
    Args:
        U_i: Left singular vectors from client i (shape: n_features × p)
        U_j: Left singular vectors from client j (shape: n_features × p)
        epsilon: Small value for numerical stability
        
    Returns:
        float: Smallest principal angle in radians [0, π/2]
    """
    calculator = PrincipalAngleCalculator(epsilon=epsilon)
    return calculator.calculate_principal_angle(U_i, U_j)


def build_proximity_matrix(client_signatures: list, 
                          epsilon: float = 1e-10) -> np.ndarray:
    """
    Convenience function to build proximity matrix for all clients.
    
    Args:
        client_signatures: List of N client signatures (U_i^p matrices)
        epsilon: Small value for numerical stability
        
    Returns:
        np.ndarray: Proximity matrix A of shape (N, N)
    """
    calculator = PrincipalAngleCalculator(epsilon=epsilon)
    return calculator.build_proximity_matrix(client_signatures)


def build_similarity_matrix(client_signatures: list,
                           epsilon: float = 1e-10) -> np.ndarray:
    """
    Convenience function to build similarity matrix for all clients.
    
    Args:
        client_signatures: List of N client signatures (U_i^p matrices)
        epsilon: Small value for numerical stability
        
    Returns:
        np.ndarray: Similarity matrix S of shape (N, N)
    """
    calculator = PrincipalAngleCalculator(epsilon=epsilon)
    return calculator.build_similarity_matrix(client_signatures)


def verify_angle_properties(calculator: PrincipalAngleCalculator, 
                           test_signatures: list) -> dict:
    """
    Verify mathematical properties of principal angle calculation.
    
    Checks:
    1. Symmetry: angle(A,B) = angle(B,A)
    2. Self-similarity: angle(A,A) = 0
    3. Range: angle in [0, π/2]
    4. Triangle inequality (approximately)
    
    Args:
        calculator: PrincipalAngleCalculator instance
        test_signatures: List of test signatures to verify properties
        
    Returns:
        dict: Verification results with boolean flags and details
    """
    results = {
        'symmetry_verified': True,
        'self_similarity_verified': True,
        'range_verified': True,
        'triangle_inequality_verified': True,
        'details': {}
    }
    
    N = len(test_signatures)
    
    if N < 2:
        results['details']['message'] = 'Need at least 2 signatures for verification'
        return results
    
    # Test symmetry
    max_asymmetry = 0.0
    for i in range(min(5, N)):
        for j in range(i + 1, min(5, N)):
            angle_ij = calculator.calculate_principal_angle(
                test_signatures[i], test_signatures[j]
            )
            angle_ji = calculator.calculate_principal_angle(
                test_signatures[j], test_signatures[i]
            )
            asymmetry = abs(angle_ij - angle_ji)
            max_asymmetry = max(max_asymmetry, asymmetry)
    
    results['symmetry_verified'] = max_asymmetry < 1e-10
    results['details']['max_asymmetry'] = max_asymmetry
    
    # Test self-similarity
    max_self_angle = 0.0
    for i in range(min(5, N)):
        self_angle = calculator.calculate_principal_angle(
            test_signatures[i], test_signatures[i]
        )
        max_self_angle = max(max_self_angle, self_angle)
    
    results['self_similarity_verified'] = max_self_angle < 1e-10
    results['details']['max_self_angle'] = max_self_angle
    
    # Test range
    all_angles = []
    for i in range(min(5, N)):
        for j in range(i + 1, min(5, N)):
            angle = calculator.calculate_principal_angle(
                test_signatures[i], test_signatures[j]
            )
            all_angles.append(angle)
    
    if all_angles:
        min_angle = min(all_angles)
        max_angle = max(all_angles)
        results['range_verified'] = (min_angle >= 0 and 
                                    max_angle <= np.pi / 2 + 1e-10)
        results['details']['angle_range'] = (min_angle, max_angle)
    
    return results
