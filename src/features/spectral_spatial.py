import numpy as np
from typing import Tuple

def create_patches(
    X: np.ndarray, 
    y: np.ndarray, 
    window_size: int = 11, 
    remove_zero_labels: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts spatial-spectral patches from the hyperspectral cube.
    
    Args:
        X (np.ndarray): Hyperspectral image cube (H, W, Bands) (usually after PCA).
        y (np.ndarray): Ground truth label map (H, W).
        window_size (int): Size of the spatial window (e.g., 11 for 11x11 patch). Must be odd.
        remove_zero_labels (bool): If True, skips background pixels (where y == 0).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X_patches: Numpy array of shape (Num_Samples, window_size, window_size, Bands)
            - y_labels: Numpy array of shape (Num_Samples,)
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number (e.g., 11, 15, 25).")
        
    margin = window_size // 2
    
    # Pad the image symmetrically so we can extract patches for border pixels
    # Pad H and W, but not Bands (0, 0)
    padded_X = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='symmetric')
    
    patches = []
    labels = []
    
    H, W = y.shape
    
    for r in range(H):
        for c in range(W):
            # 0 usually denotes unlabeled background in HSI datasets
            if remove_zero_labels and y[r, c] == 0:
                continue
                
            # Extract the patch around the pixel
            patch = padded_X[r : r + window_size, c : c + window_size, :]
            patches.append(patch)
            
            # Note: Convert labels from 1-indexed to 0-indexed for training
            labels.append(y[r, c] - 1)
            
    X_patches = np.array(patches, dtype=np.float32)
    y_labels = np.array(labels, dtype=np.int64)
    
    return X_patches, y_labels
