import torch
from torch.utils.data import Dataset
import numpy as np

class HyperspectralDataset(Dataset):
    """
    Custom PyTorch Dataset for Hyperspectral Image Patches.
    """
    def __init__(self, X_patches: np.ndarray, y_labels: np.ndarray):
        """
        Args:
            X_patches (np.ndarray): Array of patches of shape (N, H, W, C).
            y_labels (np.ndarray): Array of labels of shape (N,).
        """
        self.X_patches = X_patches
        self.y_labels = y_labels

    def __len__(self) -> int:
        return len(self.X_patches)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (patch_tensor, label_tensor)
        """
        # Get patch and label
        patch = self.X_patches[idx]
        label = self.y_labels[idx]
        
        # PyTorch expects image format to be (Channels, H, W) for Conv2D
        # Current patch is (H, W, Channels)
        patch = np.transpose(patch, (2, 0, 1))
        
        # Convert to tensors
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch_tensor, label_tensor
