import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Hyperspectral2DCNN(nn.Module):
    """
    A simple 2D Convolutional Neural Network for Hyperspectral Image Classification.
    Accepts spatial-spectral patches of shape (B, C, H, W).
    """
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        hidden_dims: List[int] = [64, 128, 256],
        patch_size: int = 11
    ):
        super(Hyperspectral2DCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Build convolutional layers
        layers = []
        current_channels = in_channels
        
        for h_dim in hidden_dims:
            layers.append(nn.Conv2d(current_channels, h_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            current_channels = h_dim
            
        # Optional Max Pooling to reduce spatial dimensions if patch size is large enough
        if patch_size >= 11:
            layers.append(nn.MaxPool2d(kernel_size=2))
            spatial_dim = patch_size // 2
        else:
            spatial_dim = patch_size
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate flattened dimension
        flattened_size = current_channels * spatial_dim * spatial_dim
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width)
            
        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes)
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
