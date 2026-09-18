import os
import scipy.io as sio
import numpy as np

def load_hyperspectral_data(file_path: str, data_key: str) -> np.ndarray:
    """
    Loads hyperspectral data from a .mat file.
    
    Args:
        file_path (str): Path to the .mat file containing the dataset.
        data_key (str): The key in the .mat dictionary corresponding to the HSI cube.
        
    Returns:
        np.ndarray: The hyperspectral image cube as a numpy array.
        
    Raises:
        FileNotFoundError: If the .mat file does not exist.
        KeyError: If the data_key is not found in the .mat file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hyperspectral data file not found at: {file_path}")
        
    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load .mat file '{file_path}'. Error: {e}")
        
    if data_key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith('_')]
        raise KeyError(f"Key '{data_key}' not found in {file_path}. Available keys: {available_keys}")
        
    hsi_cube = mat_data[data_key]
    
    if not isinstance(hsi_cube, np.ndarray):
        raise ValueError(f"Expected numpy array for key '{data_key}', got {type(hsi_cube)}")
        
    # Ensure HSI cube is 3D (H, W, Bands)
    if hsi_cube.ndim != 3:
        raise ValueError(f"Expected 3D hyperspectral cube, but got shape {hsi_cube.shape}")
        
    return hsi_cube.astype(np.float32)

def load_ground_truth(file_path: str, gt_key: str) -> np.ndarray:
    """
    Loads ground truth labels from a .mat file.
    
    Args:
        file_path (str): Path to the .mat file containing the ground truth.
        gt_key (str): The key in the .mat dictionary corresponding to the labels.
        
    Returns:
        np.ndarray: The ground truth labels as a 2D numpy array.
        
    Raises:
        FileNotFoundError: If the .mat file does not exist.
        KeyError: If the gt_key is not found in the .mat file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")
        
    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load .mat file '{file_path}'. Error: {e}")
        
    if gt_key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith('_')]
        raise KeyError(f"Key '{gt_key}' not found in {file_path}. Available keys: {available_keys}")
        
    gt_map = mat_data[gt_key]
    
    if not isinstance(gt_map, np.ndarray):
        raise ValueError(f"Expected numpy array for key '{gt_key}', got {type(gt_map)}")
        
    # Ensure ground truth is 2D (H, W)
    if gt_map.ndim != 2:
        raise ValueError(f"Expected 2D ground truth map, but got shape {gt_map.shape}")
        
    return gt_map.astype(np.int32)
