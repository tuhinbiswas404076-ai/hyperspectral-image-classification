import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List

def plot_confusion_matrix(cm: np.ndarray, num_classes: int, save_path: str = None):
    """
    Plots and optionally saves a confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix array.
        num_classes (int): Number of classes.
        save_path (str, optional): File path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(1, num_classes + 1), 
                yticklabels=range(1, num_classes + 1))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_history(history: Dict[str, list], save_path: str = None):
    """
    Plots training and validation loss and accuracy curves.
    
    Args:
        history (Dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path (str, optional): File path to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax1.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_classification_map(
    gt_map: np.ndarray, 
    pred_map: np.ndarray = None, 
    num_classes: int = 16,
    save_path: str = None
):
    """
    Plots the ground truth and optionally the predicted classification map side by side.
    
    Args:
        gt_map (np.ndarray): Ground truth 2D array.
        pred_map (np.ndarray, optional): Predicted labels 2D array.
        num_classes (int): Number of classes (determines colormap limits).
        save_path (str, optional): File path to save the plot.
    """
    import matplotlib as mpl
    cmap = plt.colormaps['tab20'].resampled(num_classes + 1)
    
    if pred_map is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.imshow(gt_map, cmap=cmap, vmin=0, vmax=num_classes)
        ax1.set_title('Ground Truth Labels')
        ax1.axis('off')
        
        im2 = ax2.imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes)
        ax2.set_title('Predicted Labels')
        ax2.axis('off')
        
        plt.colorbar(im1, ax=[ax1, ax2], fraction=0.04)
        
    else:
        plt.figure(figsize=(7, 6))
        im = plt.imshow(gt_map, cmap=cmap, vmin=0, vmax=num_classes)
        plt.title('Ground Truth Labels')
        plt.axis('off')
        plt.colorbar(im, fraction=0.04)
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
