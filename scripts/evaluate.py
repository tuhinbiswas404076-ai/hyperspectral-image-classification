import yaml
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

# Ensure src/ is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import HyperspectralDataset
from src.models import Hyperspectral2DCNN
from src.evaluation import evaluate_model
from src.visualization import plot_confusion_matrix, plot_classification_map
from src.features import create_patches

def reconstruct_classification_map(model, full_X, gt_map, patch_size, device):
    """
    Reconstructs the full classification map using the trained model.
    """
    print("Reconstructing full classification map...")
    # Extract patches for ALL pixels (even unlabeled ones) to build a full map
    # We pass remove_zero_labels=False
    X_all_patches, _ = create_patches(full_X, gt_map, window_size=patch_size, remove_zero_labels=False)
    
    # Create dataset without labels
    # We can fake the labels just to use our Dataset class
    fake_labels = np.zeros(len(X_all_patches))
    full_dataset = HyperspectralDataset(X_all_patches, fake_labels)
    full_loader = DataLoader(full_dataset, batch_size=256, shuffle=False)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs, _ in full_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Add 1 back to predictions so they align with ground truth (1-indexed)
            # 0 will be kept for background
            all_preds.extend((preds + 1).cpu().numpy())
            
    # Reshape back to image dimensions
    pred_map = np.array(all_preds).reshape(gt_map.shape)
    
    # Mask out background pixels (optional, depends if you want to predict background)
    pred_map[gt_map == 0] = 0
    
    return pred_map

def main():
    # Load Configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("=== Evaluating CNN Model ===")
    
    # 1. Load Processed Data
    processed_file = os.path.join(config['paths']['processed_data'], "dataset_splits.npz")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data not found at {processed_file}")
        
    data = np.load(processed_file)
    X_test, y_test = data['X_test'], data['y_test']
    
    # 2. Create Test DataLoader
    test_dataset = HyperspectralDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # 3. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    model = Hyperspectral2DCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['training']['num_classes'],
        hidden_dims=config['model']['hidden_dims'],
        patch_size=config['preprocessing']['patch_size']
    )
    
    model_path = config['paths']['model_save']
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 4. Evaluate Metrics
    metrics_save_path = os.path.join(config['paths']['results_metrics'], "evaluation.json")
    metrics, cm = evaluate_model(model, test_loader, device, save_path=metrics_save_path)
    
    # 5. Plot Confusion Matrix
    cm_fig_path = os.path.join(config['paths']['results_figures'], "confusion_matrix.png")
    plot_confusion_matrix(cm, num_classes=config['training']['num_classes'], save_path=cm_fig_path)
    print(f"Confusion matrix saved to {cm_fig_path}")
    
    # 6. Reconstruct and Plot Classification Map
    # Extract full spatial map arrays saved during preprocessing
    full_X = data['full_X']
    gt_map = data['gt_map']
    
    pred_map = reconstruct_classification_map(
        model, full_X, gt_map, 
        patch_size=config['preprocessing']['patch_size'], 
        device=device
    )
    
    map_fig_path = os.path.join(config['paths']['results_figures'], "classification_map.png")
    plot_classification_map(gt_map, pred_map, num_classes=config['training']['num_classes'], save_path=map_fig_path)
    print(f"Classification map saved to {map_fig_path}")

if __name__ == "__main__":
    main()
