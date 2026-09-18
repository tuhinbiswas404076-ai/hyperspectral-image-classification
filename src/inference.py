import os
import json
import joblib
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

class HSIInferenceEngine:
    """
    Production-ready Hyperspectral Image Inference Engine.
    Handles data loading, PCA reduction, band-wise min-max scaling,
    batched 3D+2D CNN prediction, spatial map reconstruction, and GT evaluation.
    """
    def __init__(self, model_dir="model"):
        self.model_dir = model_dir
        self.model = None
        self.pca = None
        self.preprocessing = None
        self.metadata = None
        self.class_names = None
        self.training_history = None
        self.is_loaded = False
        
        self.load_artifacts()

    def load_artifacts(self):
        """Loads model weights/keras model, PCA transformer, min-max bounds, and metadata."""
        meta_path = os.path.join(self.model_dir, "metadata.json")
        pca_path = os.path.join(self.model_dir, "pca.pkl")
        prep_path = os.path.join(self.model_dir, "preprocessing.pkl")
        classes_path = os.path.join(self.model_dir, "class_names.json")
        hist_path = os.path.join(self.model_dir, "training_history.json")
        model_keras_path = os.path.join(self.model_dir, "model.keras")

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                self.class_names = json.load(f)

        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                self.training_history = json.load(f)

        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)

        if os.path.exists(prep_path):
            self.preprocessing = joblib.load(prep_path)

        if os.path.exists(model_keras_path):
            self.model = tf.keras.models.load_model(model_keras_path)
            self.is_loaded = True
        else:
            patch = self.metadata.get("patch_size", 15) if self.metadata else 15
            bands = self.metadata.get("pca_components", 30) if self.metadata else 30
            n_classes = self.metadata.get("num_classes", 9) if self.metadata else 9
            
            self.model = self.build_model(patch, bands, n_classes)
            weights_path = os.path.join(self.model_dir, "best_model.weights.h5")
            if not os.path.exists(weights_path):
                weights_path = "best_model.weights.h5"
            
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                self.is_loaded = True

    @staticmethod
    def build_model(patch=15, bands=30, num_classes=9):
        """Reconstructs the exact 3D CNN + 2D CNN architecture from the notebook."""
        inp = tf.keras.layers.Input(shape=(patch, patch, bands, 1))

        # 3D Convolutions
        x = tf.keras.layers.Conv3D(32, (3, 3, 7), padding='same', activation='relu')(inp)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv3D(64, (3, 3, 5), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Reshape to flatten spatial/spectral dimensions for 2D convolution
        sh = x.shape
        x = tf.keras.layers.Reshape((sh[1], sh[2], sh[3] * sh[4]))(x)

        # 2D Spatial Convolution
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Classifier Head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def parse_mat_file(self, mat_file_path_or_dict):
        """
        Parses a .mat file or dictionary to extract hyperspectral cube and ground truth map.
        """
        if isinstance(mat_file_path_or_dict, str):
            try:
                data = sio.loadmat(mat_file_path_or_dict)
            except Exception as e:
                raise ValueError(f"Failed to read .mat file: {e}")
        else:
            data = mat_file_path_or_dict

        hsi_cube = None
        gt_map = None

        cube_keys = ['WHU_Hi_LongKou', 'WHU_Hi_HanChuan', 'WHU_Hi_HongHu', 'data', 'image', 'img', 'hsi', 'cube', 'X', 'HSI']
        label_keys = ['WHU_Hi_LongKou_gt', 'WHU_Hi_HanChuan_gt', 'WHU_Hi_HongHu_gt', 'gt', 'groundtruth', 'ground_truth', 'labels', 'label', 'GT', 'map']

        # Check explicit cube keys
        for key in cube_keys:
            if data and key in data and hasattr(data[key], 'shape'):
                arr = data[key]
                if arr.ndim == 3:
                    hsi_cube = arr.astype(np.float32)
                    break

        # Check explicit groundtruth keys
        for key in label_keys:
            if data and key in data and hasattr(data[key], 'shape'):
                arr = data[key]
                if arr.ndim == 2:
                    gt_map = arr.astype(np.int32)
                    break

        # Fallback search for largest 3D array
        if hsi_cube is None:
            candidates = [(k, v) for k, v in data.items()
                          if not k.startswith('_') and hasattr(v, 'ndim') and v.ndim == 3]
            if candidates:
                _, v = max(candidates, key=lambda x: x[1].size)
                hsi_cube = v.astype(np.float32)

        # Fallback search for largest 2D array
        if gt_map is None:
            candidates = [(k, v) for k, v in data.items()
                          if not k.startswith('_') and hasattr(v, 'ndim') and v.ndim == 2]
            if candidates:
                _, v = max(candidates, key=lambda x: x[1].size)
                gt_map = v.astype(np.int32)

        if hsi_cube is None:
            raise ValueError("No valid 3D hyperspectral image cube found in the uploaded .mat file.")

        # Ensure shape (H, W, B)
        if hsi_cube.shape[0] < hsi_cube.shape[2]:
            hsi_cube = np.transpose(hsi_cube, (1, 2, 0))

        return hsi_cube, gt_map

    def predict(self, hsi_cube, gt_map=None, batch_size=256):
        """
        Runs complete inference pipeline:
        PCA -> Normalization -> Reflect Padding -> Batched 3D+2D CNN -> Classification Map
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model artifacts are not loaded.")

        H, W, B_orig = hsi_cube.shape
        patch_size = self.metadata.get("patch_size", 15) if self.metadata else 15
        pad = patch_size // 2

        # 1. PCA Reduction
        flat_cube = hsi_cube.reshape(-1, B_orig)
        if self.pca is not None:
            flat_pca = self.pca.transform(flat_cube)
        else:
            from sklearn.decomposition import PCA
            n_comp = min(30, B_orig)
            pca_tmp = PCA(n_components=n_comp, random_state=42)
            flat_pca = pca_tmp.fit_transform(flat_cube)

        pca_cube = flat_pca.reshape(H, W, -1)
        B_reduced = pca_cube.shape[2]

        # 2. Per-Band Min-Max Normalization
        if self.preprocessing is not None and "band_min" in self.preprocessing and "band_max" in self.preprocessing:
            b_min = self.preprocessing["band_min"]
            b_max = self.preprocessing["band_max"]
            if b_min.shape[-1] == B_reduced:
                cube_norm = (pca_cube - b_min) / (b_max - b_min + 1e-8)
            else:
                b_min_curr = pca_cube.min(axis=(0, 1), keepdims=True)
                b_max_curr = pca_cube.max(axis=(0, 1), keepdims=True)
                cube_norm = (pca_cube - b_min_curr) / (b_max_curr - b_min_curr + 1e-8)
        else:
            b_min_curr = pca_cube.min(axis=(0, 1), keepdims=True)
            b_max_curr = pca_cube.max(axis=(0, 1), keepdims=True)
            cube_norm = (pca_cube - b_min_curr) / (b_max_curr - b_min_curr + 1e-8)

        # 3. Reflect Padding
        cube_padded = np.pad(cube_norm, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

        # 4. Extract all patches efficiently
        patches_list = []
        for r in range(H):
            for c in range(W):
                patch = cube_padded[r:r + patch_size, c:c + patch_size, :]
                patches_list.append(patch)

        X_all = np.array(patches_list, dtype=np.float32)[..., np.newaxis]
        
        # Fast batched prediction in a single Keras call
        probs_all = self.model.predict(X_all, batch_size=batch_size, verbose=0)
        
        top_classes = np.argmax(probs_all, axis=1) + 1  # 1-based class IDs
        top_confs = np.max(probs_all, axis=1)

        pred_map = top_classes.reshape(H, W)
        conf_map = top_confs.reshape(H, W)
        prob_cube = probs_all.reshape(H, W, -1)

        # 5. Calculate Class Distribution Statistics
        total_pixels = H * W
        unique_classes, counts = np.unique(pred_map, return_counts=True)
        class_names_list = self.class_names or [f"Class {i}" for i in range(1, self.model.output_shape[-1] + 1)]
        
        distribution = []
        for cls_id, count in zip(unique_classes, counts):
            cls_name = class_names_list[cls_id - 1] if 1 <= cls_id <= len(class_names_list) else f"Class {cls_id}"
            pct = float((count / total_pixels) * 100.0)
            distribution.append({
                "class_id": int(cls_id),
                "class_name": cls_name,
                "pixel_count": int(count),
                "percentage": round(pct, 2)
            })

        # 6. Evaluate GT Metrics if Available
        gt_metrics = None
        if gt_map is not None:
            valid_mask = (gt_map > 0)
            if np.any(valid_mask):
                y_true = gt_map[valid_mask].astype(int) - 1
                y_pred = pred_map[valid_mask].astype(int) - 1
                
                num_c = self.model.output_shape[-1]
                valid_indices = (y_true < num_c) & (y_pred < num_c)
                y_true = y_true[valid_indices]
                y_pred = y_pred[valid_indices]

                if len(y_true) > 0:
                    acc = float(np.mean(y_true == y_pred) * 100.0)
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_c))).tolist()
                    target_names = [class_names_list[i] if i < len(class_names_list) else f"Class {i+1}" for i in range(num_c)]
                    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
                    
                    gt_metrics = {
                        "accuracy": round(acc, 2),
                        "confusion_matrix": cm,
                        "classification_report": report,
                        "num_evaluated_pixels": len(y_true)
                    }

        return {
            "pred_map": pred_map,
            "conf_map": conf_map,
            "prob_cube": prob_cube,
            "distribution": distribution,
            "gt_metrics": gt_metrics,
            "image_shape": (H, W, B_orig),
            "pca_components": B_reduced,
            "mean_confidence": round(float(np.mean(conf_map)) * 100.0, 2)
        }
