# 🌈 Hyperspectral Image Classification

**AI-powered spectral-spatial land-cover classification using a 3D CNN + 2D CNN**

[![Gradio](https://img.shields.io/badge/Gradio-Live%20Demo-orange?style=for-the-badge&logo=gradio)](https://huggingface.co/spaces/tuhinbiswas404076-ai/hyperspectral-image-classification)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 🚀 Live Demo

> **Replace `tuhinbiswas404076-ai` below with your actual Hugging Face username after deployment.**

```
https://huggingface.co/spaces/tuhinbiswas404076-ai/hyperspectral-image-classification
```

---

## 📌 Project Overview

This project implements an end-to-end **hyperspectral image classification system** that takes raw `.mat` hyperspectral imagery, applies PCA dimensionality reduction and per-band normalization, extracts spatial patches, and classifies each pixel using a deep **3D CNN + 2D CNN** architecture.

The trained model is deployed as a **public Gradio web application** on Hugging Face Spaces — no local Python, TensorFlow, or Jupyter installation is required to use it.

### User Workflow
```
👤 User opens public link
      ↓
📤 Uploads a .mat hyperspectral file
      ↓
🚀 Clicks "Run Classification"
      ↓
🛰️ False-Color Satellite Image
      ↓
🗺️ Predicted Classification Map
      ↓
📊 Class Distribution Charts
      ↓
🎯 Per-Pixel Confidence & Spectral Analysis
      ↓
📌 Confusion Matrix (when ground truth exists)
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **3D + 2D CNN** | Spectral-spatial deep learning combining 3D convolutions (joint spectral-spatial features) and 2D convolutions (refined spatial boundaries) |
| 📉 **PCA Reduction** | Compresses hundreds of spectral bands to 30 decorrelated components |
| 🔄 **Per-Band Normalization** | Min-max scaling preserving inter-band relationships |
| 🛰️ **False-Color Visualization** | Automatic RGB composite from representative spectral bands |
| 🗺️ **Spatial Classification Map** | Complete pixel-level land-cover prediction with colorful legend |
| 📊 **Class Distribution** | Interactive charts showing predicted pixel counts and percentages |
| 🎯 **Confidence Analysis** | Per-pixel prediction confidence from softmax probabilities |
| 🔬 **Pixel Spectroscopy** | Click any (Row, Col) to view its spectral signature and class probabilities |
| 📌 **Ground-Truth Evaluation** | Confusion matrix, precision, recall, F1-score when GT labels exist |
| 📈 **Training History** | Training vs. validation accuracy and loss curves |
| ⚡ **Batched Inference** | Memory-efficient batched prediction suitable for free HF Spaces |

---

## 🧠 Model Architecture

```
Input: (15, 15, 30, 1) — 15×15 Spatial Patch × 30 PCA Bands × 1 Channel

  ┌──────────────────────────────────────────────┐
  │  Conv3D(32, 3×3×7, same, relu) + BatchNorm  │
  ├──────────────────────────────────────────────┤
  │  Conv3D(64, 3×3×5, same, relu) + BatchNorm  │
  ├──────────────────────────────────────────────┤
  │  Reshape(15, 15, 64×30)                      │
  ├──────────────────────────────────────────────┤
  │  Conv2D(128, 3×3, same, relu) + BatchNorm   │
  ├──────────────────────────────────────────────┤
  │  GlobalAveragePooling2D                      │
  ├──────────────────────────────────────────────┤
  │  Dense(256, relu) + Dropout(0.4)             │
  ├──────────────────────────────────────────────┤
  │  Dense(128, relu) + Dropout(0.3)             │
  ├──────────────────────────────────────────────┤
  │  Dense(9, softmax)                           │
  └──────────────────────────────────────────────┘

Output: 9 land-cover class probabilities
```

---

## 🔬 Processing Pipeline

```
Hyperspectral Image (.mat)
        ↓
  PCA Reduction (→ 30 components)
        ↓
  Per-Band Min-Max Normalization (0–1)
        ↓
  Reflect Padding (pad=7)
        ↓
  15×15 Patch Extraction
        ↓
  3D CNN Feature Extraction
        ↓
  2D CNN Spatial Refinement
        ↓
  Global Average Pooling + Dense Head
        ↓
  Softmax Classification (9 classes)
```

---

## 📂 Repository Structure

```
hyperspectral-image-classification/
│
├── app.py                          # Gradio web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
├── best_model.weights.h5           # Original trained weights
│
├── model/
│   ├── model.keras                 # Exported Keras model
│   ├── pca.pkl                     # Fitted PCA transformer
│   ├── preprocessing.pkl           # Per-band min/max bounds
│   ├── metadata.json               # Model configuration
│   ├── class_names.json            # Land-cover class labels
│   └── training_history.json       # Training/validation metrics
│
├── assets/
│   └── sample/
│       └── sample_hsi.mat          # Sample data for demo
│
├── src/
│   ├── __init__.py
│   └── inference.py                # HSIInferenceEngine class
│
├── scripts/
│   └── export_artifacts.py         # Artifact export pipeline
│
└── notebooks/
    └── Hyper_SpectralImage.ipynb    # Original training notebook
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/tuhinbiswas404076-ai/hyperspectral-image-classification.git
cd hyperspectral-image-classification

# Create virtual environment (Python 3.11 recommended)
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
python app.py
```

The application launches at `http://localhost:7860`.

---

## 📊 Dataset

This model was trained on the **WHU-Hi Hyperspectral Dataset** (Wuhan University).

| Property | Value |
|---|---|
| **Dataset** | WHU-Hi (HanChuan / LongKou / HongHu) |
| **Spectral Bands** | 270 (reduced to 30 via PCA) |
| **Classes** | 9 land-cover types |
| **Class Names** | Corn, Cotton, Sesame, Broad-leaf Soybean, Narrow-leaf Soybean, Rice, Water, Roads & Buildings, Mixed Weed |
| **Source** | [Kaggle: WHU Hyperspectral Dataset](https://www.kaggle.com/datasets/rupeshkumaryadav/whu-hyperspectral-dataset) |

---

## 🌐 Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Upload these files: `app.py`, `requirements.txt`, `src/`, `model/`, `assets/`
4. The Space will automatically install dependencies and launch

Or use the Hugging Face CLI:

```bash
pip install huggingface_hub
huggingface-cli login

# Create and push to Space
huggingface-cli repo create hyperspectral-image-classification --type space --space-sdk gradio
git remote add hf https://huggingface.co/spaces/tuhinbiswas404076-ai/hyperspectral-image-classification
git push hf main
```

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{hyperspectral-classification,
  title={Hyperspectral Image Classification using 3D CNN + 2D CNN},
  year={2026},
  url={https://github.com/tuhinbiswas404076-ai/hyperspectral-image-classification}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
