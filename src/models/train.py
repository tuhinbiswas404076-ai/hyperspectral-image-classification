import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import copy
import os

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 100,
    patience: int = 10,
    save_path: str = "models/best_model.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, list]:
    """
    Trains the PyTorch model with validation and early stopping.
    
    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping early.
        save_path (str): File path to save the best model weights.
        device (str): Device to train on ('cuda' or 'cpu').
        
    Returns:
        Dict[str, list]: Training history containing train_loss, val_loss, train_acc, val_acc.
    """
    model.to(device)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        # ---------- TRAINING PHASE ----------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        
        # ---------- VALIDATION PHASE ----------
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects / len(val_loader.dataset)
        
        # Store history
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        # Early Stopping and Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return history
