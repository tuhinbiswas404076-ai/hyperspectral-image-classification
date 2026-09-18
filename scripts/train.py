import yaml
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# Ensure src/ is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import HyperspectralDataset
from src.models import Hyperspectral2DCNN, train_model
from src.visualization import plot_training_history

def main():
    # Load Configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("=== Training CNN Model ===")
    
    # 1. Load Processed Data
    processed_file = os.path.join(config['paths']['processed_data'], "dataset_splits.npz")
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data not found at {processed_file}. Run preprocess.py first.")
        
    print("Loading preprocessed dataset splits...")
    data = np.load(processed_file)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # 2. Create PyTorch Datasets & DataLoaders
    batch_size = config['training']['batch_size']
    train_dataset = HyperspectralDataset(X_train, y_train)
    val_dataset = HyperspectralDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize Model, Loss, and Optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Hyperspectral2DCNN(
        in_channels=config['model']['in_channels'],
        num_classes=config['training']['num_classes'],
        hidden_dims=config['model']['hidden_dims'],
        patch_size=config['preprocessing']['patch_size']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 4. Train Model
    print("Starting training loop...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['early_stopping_patience'],
        save_path=config['paths']['model_save'],
        device=device
    )
    
    # 5. Plot and Save History
    fig_path = os.path.join(config['paths']['results_figures'], "training_history.png")
    plot_training_history(history, save_path=fig_path)
    print(f"Training history plot saved to {fig_path}")

if __name__ == "__main__":
    main()
