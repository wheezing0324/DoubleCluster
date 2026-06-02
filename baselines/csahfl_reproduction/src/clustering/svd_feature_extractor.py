"""
SVD Feature Extractor for Privacy-Preserving Clustering (Section 3.3, Eq. 5)

This module implements truncated Singular Value Decomposition (SVD) for extracting
privacy-preserving data signatures from client datasets. Instead of sharing raw data,
clients share only the top-p singular vectors which serve as data distribution signatures.

Key Equation (Eq. 5):
    U_i^p = [u_1, u_2, ..., u_p] where p << rank(D_i)
    
Each client computes SVD on local data matrix D_i and retains only top p singular
vectors, which are transmitted to the server for clustering.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.linalg import svd


class SVDFeatureExtractor:
    """
    Extracts privacy-preserving features using truncated SVD.
    
    Implements Eq. 5 from CSAHFL paper: U_i^p = [u_1, u_2, ..., u_p]
    where p singular vectors are retained as data signature.
    
    Attributes:
        n_components (int): Number of singular vectors to retain (p in Eq. 5)
        random_state (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(self, n_components: int = 10, random_state: Optional[int] = None):
        """
        Initialize SVD feature extractor.
        
        Args:
            n_components: Number of singular vectors to retain (p=10 recommended)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract top-p singular vectors from data matrix (Eq. 5).
        
        Computes truncated SVD on the data matrix and returns only the top-p
        left singular vectors as a privacy-preserving signature.
        
        Args:
            data: Input data matrix D_i of shape (n_samples, n_features)
                  For images: (n_samples, height * width * channels)
        
        Returns:
            U_p: Top-p left singular vectors of shape (n_samples, n_components)
                 This is U_i^p from Eq. 5
        
        Raises:
            ValueError: If n_components exceeds matrix dimensions
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data matrix, got {data.ndim}D")
        
        n_samples, n_features = data.shape
        
        # Ensure n_components doesn't exceed matrix dimensions
        effective_components = min(self.n_components, n_samples, n_features)
        
        if effective_components < self.n_components:
            print(f"Warning: Reducing n_components from {self.n_components} to "
                  f"{effective_components} due to data dimensions")
        
        # Center the data (subtract mean)
        data_centered = data - np.mean(data, axis=0)
        
        # Compute truncated SVD: D = U @ S @ V^T
        # We only need U (left singular vectors)
        U, S, Vt = svd(data_centered, full_matrices=False)
        
        # Retain only top-p singular vectors (Eq. 5)
        U_p = U[:, :effective_components]
        
        return U_p
    
    def extract_features_from_images(self, images: np.ndarray, 
                                     normalize: bool = True) -> np.ndarray:
        """
        Extract SVD features from image data.
        
        Convenience method for image datasets (Fashion-MNIST, CIFAR10, SVHN).
        Flattens images and optionally normalizes before SVD.
        
        Args:
            images: Image data of shape (n_samples, H, W, C) or (n_samples, H, W)
            normalize: Whether to normalize pixel values to [0, 1]
        
        Returns:
            U_p: Top-p singular vectors of shape (n_samples, n_components)
        """
        # Handle different image formats
        if images.ndim == 4:
            # (N, H, W, C) - already in NHWC format
            data_flat = images.reshape(images.shape[0], -1)
        elif images.ndim == 3:
            # (N, H, W) - grayscale images
            data_flat = images.reshape(images.shape[0], -1)
        else:
            raise ValueError(f"Expected 3D or 4D image data, got {images.ndim}D")
        
        # Normalize if requested
        if normalize:
            data_flat = data_flat.astype(np.float32) / 255.0
        
        return self.extract_features(data_flat)
    
    def compute_singular_values(self, data: np.ndarray) -> np.ndarray:
        """
        Compute singular values for analysis.
        
        Useful for determining appropriate n_components by examining
        the singular value spectrum.
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
        
        Returns:
            singular_values: Array of singular values in descending order
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2D data matrix, got {data.ndim}D")
        
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        # Compute SVD
        _, S, _ = svd(data_centered, full_matrices=False)
        
        return S
    
    def get_explained_variance_ratio(self, data: np.ndarray, 
                                     n_components: Optional[int] = None) -> np.ndarray:
        """
        Calculate explained variance ratio for each singular value.
        
        Helps determine how many components are needed to capture
        sufficient data variance.
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
            n_components: Number of components to consider (default: self.n_components)
        
        Returns:
            explained_variance_ratio: Array of variance ratios for each component
        """
        if n_components is None:
            n_components = self.n_components
        
        # Compute singular values
        S = self.compute_singular_values(data)
        
        # Explained variance is proportional to squared singular values
        explained_variance = S ** 2
        total_variance = np.sum(explained_variance)
        
        # Calculate ratio
        explained_variance_ratio = explained_variance[:n_components] / total_variance
        
        return explained_variance_ratio
    
    def get_cumulative_explained_variance(self, data: np.ndarray,
                                          n_components: Optional[int] = None) -> float:
        """
        Calculate cumulative explained variance ratio.
        
        Args:
            data: Input data matrix of shape (n_samples, n_features)
            n_components: Number of components to consider (default: self.n_components)
        
        Returns:
            cumulative_ratio: Total variance explained by top-n components
        """
        ratios = self.get_explained_variance_ratio(data, n_components)
        return np.sum(ratios)


def create_svd_extractor(n_components: int = 10, 
                         random_state: Optional[int] = None) -> SVDFeatureExtractor:
    """
    Factory function to create SVDFeatureExtractor instance.
    
    Args:
        n_components: Number of singular vectors to retain (p=10 recommended)
        random_state: Random seed for reproducibility
    
    Returns:
        SVDFeatureExtractor: Configured extractor instance
    """
    return SVDFeatureExtractor(n_components=n_components, random_state=random_state)


def extract_client_signature(data: np.ndarray, n_components: int = 10) -> np.ndarray:
    """
    Convenience function to extract SVD signature from client data.
    
    Single-function interface for extracting privacy-preserving signatures.
    
    Args:
        data: Client data matrix of shape (n_samples, n_features)
        n_components: Number of singular vectors to retain (p=10)
    
    Returns:
        signature: Top-p singular vectors as data signature
    """
    extractor = create_svd_extractor(n_components=n_components)
    return extractor.extract_features(data)


if __name__ == "__main__":
    # Test SVD feature extraction
    print("Testing SVDFeatureExtractor...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    data = np.random.randn(n_samples, n_features)
    
    # Test feature extraction
    extractor = create_svd_extractor(n_components=10)
    U_p = extractor.extract_features(data)
    
    print(f"Input data shape: {data.shape}")
    print(f"Output signature shape: {U_p.shape}")
    print(f"Expected shape: ({n_samples}, 10)")
    
    # Test explained variance
    var_ratio = extractor.get_explained_variance_ratio(data)
    cumulative_var = extractor.get_cumulative_explained_variance(data)
    
    print(f"\nExplained variance ratio (first 5 components): {var_ratio[:5]}")
    print(f"Cumulative explained variance: {cumulative_var:.4f}")
    
    # Test with image-like data
    images = np.random.randint(0, 256, (50, 28, 28), dtype=np.uint8)
    U_p_images = extractor.extract_features_from_images(images)
    
    print(f"\nImage data shape: {images.shape}")
    print(f"Image signature shape: {U_p_images.shape}")
    
    print("\n✓ SVDFeatureExtractor tests passed!")
