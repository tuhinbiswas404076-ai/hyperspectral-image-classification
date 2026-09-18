import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

def apply_pca(X: np.ndarray, num_components: int = 30) -> Tuple[np.ndarray, PCA]:
    """
    Applies Principal Component Analysis (PCA) to hyperspectral data for dimensionality reduction.
    Also whitens the data (zero mean, unit variance).
    
    Args:
        X (np.ndarray): Hyperspectral image cube of shape (H, W, Bands).
        num_components (int): Number of principal components to keep.
        
    Returns:
        Tuple[np.ndarray, PCA]: The reduced image cube of shape (H, W, num_components) and the fitted PCA object.
    """
    H, W, B = X.shape
    new_X = np.reshape(X, (-1, B))
    
    # Apply PCA with whitening to normalize
    pca = PCA(n_components=num_components, whiten=True)
    new_X = pca.fit_transform(new_X)
    
    new_X = np.reshape(new_X, (H, W, num_components))
    return new_X, pca

def split_data(
    X_patches: np.ndarray, 
    y_labels: np.ndarray, 
    test_ratio: float = 0.2, 
    val_ratio: float = 0.1, 
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Splits the extracted patches and labels into training, validation, and test sets.
    
    Args:
        X_patches (np.ndarray): Array of patches.
        y_labels (np.ndarray): Array of corresponding labels.
        test_ratio (float): Proportion of the dataset to include in the test split.
        val_ratio (float): Proportion of the dataset to include in the validation split (taken from train set).
        random_seed (int): Seed for reproducible shuffling.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
    """
    # First, split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_patches, y_labels, test_size=test_ratio, random_state=random_seed, stratify=y_labels
    )
    
    # Calculate relative validation ratio from the remaining train+val set
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_ratio, random_state=random_seed, stratify=y_train_val
    )
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }
