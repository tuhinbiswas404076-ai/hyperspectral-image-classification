import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
import os
from typing import Dict, Tuple

def evaluate_model(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "results/metrics/evaluation.json"
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluates the model on the test dataset and calculates comprehensive metrics.
    
    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to evaluate on.
        save_path (str): File path to save the metrics as JSON.
        
    Returns:
        Tuple[Dict, np.ndarray]: A dictionary of metrics and the confusion matrix.
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    overall_accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Average Accuracy is the mean of the per-class recalls
    average_accuracy = np.mean(recall)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "overall_accuracy": float(overall_accuracy),
        "average_accuracy": float(average_accuracy),
        "kappa": float(kappa),
        "per_class": {
            str(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            } for i in range(len(precision))
        }
    }
    
    # Save metrics to JSON
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    print(f"Evaluation complete. OA: {overall_accuracy:.4f} | AA: {average_accuracy:.4f} | Kappa: {kappa:.4f}")
    return metrics, cm
