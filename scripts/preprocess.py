import yaml
import os
import numpy as np
import sys

# Ensure src/ is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_hyperspectral_data, load_ground_truth, apply_pca, split_data
from src.features import create_patches

def main():
    # Load Configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("=== Hyperspectral Image Preprocessing ===")
    
    # 1. Load Data
    print(f"Loading {config['dataset']['name']} dataset...")
    X = load_hyperspectral_data(config['dataset']['data_path'], config['dataset']['data_key'])
    y = load_ground_truth(config['dataset']['gt_path'], config['dataset']['gt_key'])
    
    print(f"Original Data Shape: {X.shape}")
    print(f"Ground Truth Shape: {y.shape}")
    
    # 2. Preprocessing (PCA)
    if config['preprocessing']['apply_pca']:
        num_components = config['preprocessing']['num_pca_components']
        print(f"Applying PCA to reduce bands to {num_components}...")
        X, pca_model = apply_pca(X, num_components=num_components)
        print(f"Data Shape after PCA: {X.shape}")
        
    # 3. Patch Extraction
    patch_size = config['preprocessing']['patch_size']
    print(f"Extracting {patch_size}x{patch_size} patches...")
    X_patches, y_labels = create_patches(X, y, window_size=patch_size)
    print(f"Extracted {len(X_patches)} labeled patches.")
    
    # 4. Split Dataset
    print("Splitting dataset into Train/Val/Test...")
    splits = split_data(
        X_patches, y_labels, 
        test_ratio=config['preprocessing']['test_ratio'],
        val_ratio=config['preprocessing']['val_ratio'],
        random_seed=config['preprocessing']['random_seed']
    )
    
    print(f"Train samples: {len(splits['X_train'])}")
    print(f"Val samples: {len(splits['X_val'])}")
    print(f"Test samples: {len(splits['X_test'])}")
    
    # 5. Save Processed Data
    processed_dir = config['paths']['processed_data']
    os.makedirs(processed_dir, exist_ok=True)
    
    save_path = os.path.join(processed_dir, "dataset_splits.npz")
    np.savez_compressed(
        save_path, 
        X_train=splits['X_train'], y_train=splits['y_train'],
        X_val=splits['X_val'], y_val=splits['y_val'],
        X_test=splits['X_test'], y_test=splits['y_test'],
        # Save original gt_map for full map reconstruction later
        gt_map=y,
        # Save X to reconstruct full map
        full_X=X
    )
    print(f"Processed dataset saved to {save_path}")

if __name__ == "__main__":
    main()
