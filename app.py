import os
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from src.inference import HSIInferenceEngine

# Initialize inference engine
engine = HSIInferenceEngine(model_dir="model")

# ─── Visualization helpers ────────────────────────────────────────────────

def get_rgb_image(hsi_cube):
    """Generates a false-color RGB image using 3 representative spectral bands."""
    H, W, B = hsi_cube.shape
    r = hsi_cube[:, :, int(B * 0.75)]
    g = hsi_cube[:, :, int(B * 0.45)]
    b = hsi_cube[:, :, int(B * 0.15)]
    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    return np.stack([norm(r), norm(g), norm(b)], axis=2)


def plot_maps(hsi_cube, pred_map, gt_map=None, class_names=None):
    """Creates side-by-side False-Color RGB, GT Map (if any), and Predicted Map."""
    rgb = get_rgb_image(hsi_cube)
    num_cols = 3 if gt_map is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5.5), facecolor='#111827')

    axes[0].imshow(rgb)
    axes[0].set_title("Satellite Image (False-Color)", fontsize=13, fontweight='bold', color='#38bdf8')
    axes[0].axis('off')

    num_classes = engine.model.output_shape[-1] if engine.model else 9
    cmap = plt.cm.get_cmap('tab20', num_classes + 1)

    col_idx = 1
    if gt_map is not None:
        axes[col_idx].imshow(gt_map, cmap=cmap, vmin=0, vmax=num_classes)
        axes[col_idx].set_title("Ground Truth Labels", fontsize=13, fontweight='bold', color='#4ade80')
        axes[col_idx].axis('off')
        col_idx += 1

    im_pred = axes[col_idx].imshow(pred_map, cmap=cmap, vmin=0, vmax=num_classes)
    axes[col_idx].set_title("Predicted Classification Map", fontsize=13, fontweight='bold', color='#f472b6')
    axes[col_idx].axis('off')

    cbar = fig.colorbar(im_pred, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors='white')
    if class_names:
        ticks = np.arange(0.5, num_classes + 0.5)
        cbar.set_ticks(ticks[:len(class_names)])
        cbar.set_ticklabels(class_names[:len(ticks)])

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names):
    """Generates a confusion matrix heatmap."""
    n_classes = len(cm)
    fig, ax = plt.subplots(figsize=(max(7, n_classes * 0.6), max(6, n_classes * 0.5)), facecolor='#111827')
    ax.set_facecolor('#1f2937')

    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, cmap='Blues')
    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.ax.tick_params(colors='white')

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names[:n_classes], rotation=45, ha='right', fontsize=9, color='white')
    ax.set_yticklabels(class_names[:n_classes], fontsize=9, color='white')
    ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold', color='#38bdf8')
    ax.set_ylabel('True Class', fontsize=11, fontweight='bold', color='#38bdf8')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold', color='white')

    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_arr[i, j]
            ax.text(j, i, f"{val}", ha='center', va='center',
                    color='white' if val > thresh else '#9ca3af',
                    fontsize=max(7, 10 - n_classes // 3))
    plt.tight_layout()
    return fig


def plot_training_history():
    """Plots training accuracy & loss curves from saved history."""
    hist = engine.training_history
    if not hist or 'accuracy' not in hist:
        return None

    epochs = range(1, len(hist['accuracy']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#111827')
    ax1.set_facecolor('#1f2937')
    ax2.set_facecolor('#1f2937')

    ax1.plot(epochs, [a * 100 if a <= 1 else a for a in hist['accuracy']],
             label='Train Accuracy', color='#38bdf8', linewidth=2)
    ax1.plot(epochs, [a * 100 if a <= 1 else a for a in hist['val_accuracy']],
             label='Val Accuracy', color='#f472b6', linewidth=2)
    best_ep = hist.get('best_epoch', np.argmax(hist['val_accuracy']) + 1)
    ax1.axvline(best_ep, color='#4ade80', linestyle='--', label=f'Best Epoch ({best_ep})')
    ax1.set_title('Accuracy over Epochs', fontsize=12, fontweight='bold', color='white')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Accuracy (%)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#111827', edgecolor='none', labelcolor='white')
    ax1.grid(alpha=0.2)

    ax2.plot(epochs, hist['loss'], label='Train Loss', color='#38bdf8', linewidth=2)
    ax2.plot(epochs, hist['val_loss'], label='Val Loss', color='#f472b6', linewidth=2)
    ax2.axvline(best_ep, color='#4ade80', linestyle='--', label=f'Best Epoch ({best_ep})')
    ax2.set_title('Loss over Epochs', fontsize=12, fontweight='bold', color='white')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Loss', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#111827', edgecolor='none', labelcolor='white')
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    return fig


# ─── Session state ─────────────────────────────────────────────────────────

state = {
    "hsi_cube": None,
    "gt_map": None,
    "pred_results": None,
    "file_name": ""
}


# ─── Event handlers ────────────────────────────────────────────────────────

def load_file(file_obj, sample_flag=False):
    """Handles upload of .mat file or sample data loading."""
    try:
        if sample_flag or file_obj is None:
            mat_path = "assets/sample/sample_hsi.mat"
            if not os.path.exists(mat_path):
                return ("### Upload Status\nSample data not found. Upload a .mat file.",
                        None, gr.update(), gr.update(), "Sample data not found.")
            file_name = "sample_hsi.mat"
        else:
            mat_path = file_obj.name
            file_name = os.path.basename(file_obj.name)

        cube, gt = engine.parse_mat_file(mat_path)
        state["hsi_cube"] = cube
        state["gt_map"] = gt
        state["pred_results"] = None
        state["file_name"] = file_name

        H, W, B = cube.shape
        gt_status = (f"Ground Truth Available ({np.unique(gt[gt > 0]).size} classes)"
                     if gt is not None else "Ground Truth Not Provided")

        info_md = f"""### File Summary: `{file_name}`
* **Spatial Dimensions**: {H} rows x {W} columns ({H * W:,} pixels)
* **Spectral Bands**: {B}
* **PCA Target Components**: {engine.metadata.get('pca_components', 30) if engine.metadata else 30}
* **Patch Size**: {engine.metadata.get('patch_size', 15) if engine.metadata else 15} x {engine.metadata.get('patch_size', 15) if engine.metadata else 15}
* **Ground-Truth Status**: {gt_status}"""

        rgb = get_rgb_image(cube)
        fig_rgb, ax = plt.subplots(figsize=(5.5, 4.5), facecolor='#111827')
        ax.imshow(rgb)
        ax.set_title(f"{file_name} (False-Color RGB)", fontsize=11, fontweight='bold', color='#38bdf8')
        ax.axis('off')
        plt.tight_layout()

        return (
            info_md,
            fig_rgb,
            gr.update(maximum=H - 1, value=H // 2),
            gr.update(maximum=W - 1, value=W // 2),
            "Ready! Click 'Run Classification' below."
        )
    except Exception as e:
        return (
            f"Error loading file: {str(e)}",
            None,
            gr.update(),
            gr.update(),
            f"Error: {str(e)}"
        )


def run_classification():
    """Runs batched inference engine on loaded HSI cube."""
    cube = state.get("hsi_cube")
    gt = state.get("gt_map")

    if cube is None:
        return ("Please upload a .mat file or click 'Try Sample Data' first.",
                None, None, "", None, "", "No data loaded")

    try:
        results = engine.predict(cube, gt_map=gt, batch_size=256)
        state["pred_results"] = results

        pred_map = results["pred_map"]
        class_names = engine.class_names or [f"Class {i}" for i in range(1, engine.model.output_shape[-1] + 1)]

        # 1. Main Maps Plot
        fig_maps = plot_maps(cube, pred_map, gt_map=gt, class_names=class_names)

        # 2. Distribution Chart (Plotly)
        dist_data = results["distribution"]
        df_dist = {
            "Class Name": [d["class_name"] for d in dist_data],
            "Pixel Count": [d["pixel_count"] for d in dist_data],
            "Percentage": [d["percentage"] for d in dist_data]
        }
        fig_dist = px.bar(
            df_dist, x="Class Name", y="Pixel Count", text="Percentage",
            color="Class Name", title="Predicted Pixel Count & Coverage (%)",
            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_dist.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dist.update_layout(showlegend=False, font=dict(color="white"))

        # 3. Overall Stats Markdown
        stats_md = f"""### Prediction Highlights
* **Total Image Pixels**: {cube.shape[0] * cube.shape[1]:,}
* **Mean Model Confidence**: **{results['mean_confidence']}%**
* **Identified Land-Cover Classes**: {len(dist_data)}"""

        # 4. Confusion Matrix & GT Metrics
        fig_cm = None
        gt_md = ""
        if results["gt_metrics"]:
            gt_m = results["gt_metrics"]
            fig_cm = plot_confusion_matrix(gt_m["confusion_matrix"], class_names)
            gt_md = f"""### Ground Truth Performance
* **Classification Accuracy**: **{gt_m['accuracy']}%**
* **Evaluated Pixels**: {gt_m['num_evaluated_pixels']:,}"""
        else:
            gt_md = "> Ground-truth labels were not provided in this `.mat` file, so accuracy metrics and confusion matrix cannot be calculated."

        return (
            "Classification completed successfully!",
            fig_maps,
            fig_dist,
            stats_md,
            fig_cm,
            gt_md,
            "Done"
        )
    except Exception as e:
        return (
            f"Error during classification: {str(e)}",
            None, None, "", None, "", f"Error: {str(e)}"
        )


def inspect_pixel(row, col):
    """Plots spectral signature and class probability for selected (Row, Col)."""
    cube = state.get("hsi_cube")
    results = state.get("pred_results")

    if cube is None:
        return None, "Load a dataset first.", None

    H, W, B = cube.shape
    row = max(0, min(int(row), H - 1))
    col = max(0, min(int(col), W - 1))
    spectrum = cube[row, col, :]

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(
        x=list(range(1, B + 1)), y=spectrum.tolist(),
        mode='lines+markers', name=f'Pixel ({row}, {col})',
        line=dict(color='#38bdf8', width=2.5),
        marker=dict(size=3, color='#f472b6')
    ))
    fig_spec.update_layout(
        title=f"Spectral Signature at Pixel Row={row}, Col={col}",
        xaxis_title="Spectral Band Index", yaxis_title="Band Intensity",
        template="plotly_dark", font=dict(color="white")
    )

    if results is not None:
        pred_class_id = int(results["pred_map"][row, col])
        probs = results["prob_cube"][row, col]
        class_names = engine.class_names or [f"Class {i}" for i in range(1, len(probs) + 1)]
        pred_class_name = class_names[pred_class_id - 1] if 1 <= pred_class_id <= len(class_names) else f"Class {pred_class_id}"
        conf = float(probs[pred_class_id - 1]) * 100.0

        details_md = f"""### Pixel ({row}, {col}) Analysis
* **Predicted Class**: **{pred_class_name}** (Class ID: {pred_class_id})
* **Confidence**: **{conf:.2f}%**"""

        df_prob = {"Class Name": class_names[:len(probs)], "Probability (%)": [float(p) * 100.0 for p in probs]}
        fig_prob = px.bar(
            df_prob, x="Class Name", y="Probability (%)", color="Class Name",
            title=f"Probabilities for Pixel ({row}, {col})",
            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_prob.update_layout(showlegend=False, font=dict(color="white"))
        return fig_spec, details_md, fig_prob
    else:
        return fig_spec, f"### Pixel ({row}, {col})\n*Run classification to view predicted class and probabilities.*", None


def show_training_history():
    """Lazy wrapper for training history plot."""
    return plot_training_history()


# ─── Build Gradio UI ───────────────────────────────────────────────────────

custom_css = """
.gradio-container { max-width: 1280px !important; margin: 0 auto; }
h1 { color: #38bdf8 !important; text-align: center; font-weight: 800; font-size: 2.2rem; }
.subtitle { text-align: center; color: #9ca3af; margin-bottom: 1.5rem; font-size: 1.1rem; }
.btn-primary { background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; }
.btn-sample { background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; }
"""

with gr.Blocks(title="Hyperspectral Image Classification", css=custom_css) as demo:
    gr.Markdown("# Hyperspectral Image Classification")
    gr.Markdown("<p class='subtitle'>AI-powered spectral-spatial land-cover classification using a 3D CNN + 2D CNN</p>")

    with gr.Tabs():
        # ── TAB 1: HOME & OVERVIEW ──
        with gr.Tab("Home & Overview"):
            gr.Markdown("""
### Project Overview
This application delivers end-to-end **spectral-spatial hyperspectral image classification** using a deep learning pipeline combining **3D Convolutions** (extracting joint spatial-spectral features across continuous wavelength bands) and **2D Convolutions** (refining high-level land-cover boundaries).

#### Processing & Inference Workflow
```text
User Uploads .mat File
        |
PCA Dimensionality Reduction (30 Components)
        |
Per-Band Min-Max Scaling (0 to 1)
        |
Spatial Patch Extraction (15x15 Patches + Reflect Padding)
        |
3D Conv (32, 64 Filters) + 2D Conv (128 Filters)
        |
Global Average Pooling + Dense Classifier Head
        |
Reconstructed Spatial Classification Map & Visualizations
```

#### Key Features
* **Automated `.mat` Parsing**: Intelligently auto-detects 3D hyperspectral cubes and 2D ground truth maps.
* **Zero Online Retraining**: Online inference utilizes pre-fitted PCA and scaling parameters for instantaneous batched predictions.
* **False-Color & Spatial Maps**: Side-by-side high-resolution visualization comparing False-Color RGB satellite composite with land-cover predictions.
* **Pixel-Level Spectroscopy**: Click or enter (Row, Col) coordinates to inspect exact spectral signatures and class probabilities.
* **Ground-Truth Validation**: Calculates Confusion Matrix, Precision, Recall, and F1-scores whenever ground-truth labels exist.
            """)

        # ── TAB 2: UPLOAD & CLASSIFY ──
        with gr.Tab("Upload & Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Hyperspectral .mat File", file_types=[".mat"])
                    btn_sample = gr.Button("Try Sample Data (WHU-Hi)", elem_classes=["btn-sample"])
                    info_output = gr.Markdown("### Upload Status\nUpload a `.mat` file or click 'Try Sample Data' to begin.")
                    btn_predict = gr.Button("Run Classification", elem_classes=["btn-primary"])
                    status_output = gr.Textbox(label="Status Log", interactive=False)

                with gr.Column(scale=1):
                    preview_plot = gr.Plot(label="Satellite False-Color Composite")

            gr.Markdown("---")
            gr.Markdown("### Visual Results")
            maps_plot = gr.Plot(label="False-Color Image vs Predicted Classification Map")

            with gr.Row():
                with gr.Column():
                    dist_plot = gr.Plot(label="Class Distribution Chart")
                with gr.Column():
                    stats_summary = gr.Markdown()

        # ── TAB 3: PIXEL ANALYSIS ──
        with gr.Tab("Pixel Analysis"):
            gr.Markdown("Inspect the exact spectral intensity curve and model probability distribution for any specific pixel location.")
            with gr.Row():
                slider_row = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Pixel Row Index")
                slider_col = gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Pixel Column Index")
                btn_inspect = gr.Button("Inspect Pixel Spectrum")

            with gr.Row():
                with gr.Column(scale=1):
                    pixel_info = gr.Markdown()
                    prob_plot = gr.Plot(label="Class Probabilities")
                with gr.Column(scale=1):
                    spectrum_plot = gr.Plot(label="Spectral Signature Curve")

        # ── TAB 4: GROUND-TRUTH PERFORMANCE ──
        with gr.Tab("Performance Evaluation"):
            gr.Markdown("Evaluates accuracy and confusion matrix against ground-truth labels if provided.")
            gt_summary = gr.Markdown()
            cm_plot = gr.Plot(label="Confusion Matrix")

        # ── TAB 5: MODEL ARCHITECTURE & TRAINING ──
        with gr.Tab("Model Architecture & History"):
            gr.Markdown("""
### Deep Learning Model Architecture
* **Input Layer**: `(15, 15, 30, 1)` -- 15x15 Spatial Patch, 30 PCA Spectral Bands
* **Layer 1**: `Conv3D(32, kernel=(3,3,7), padding='same', relu)` + `BatchNormalization`
* **Layer 2**: `Conv3D(64, kernel=(3,3,5), padding='same', relu)` + `BatchNormalization`
* **Layer 3**: `Reshape(15, 15, 64*30)` -- Flattens 3D feature maps for 2D spatial conv
* **Layer 4**: `Conv2D(128, kernel=(3,3), padding='same', relu)` + `BatchNormalization`
* **Layer 5**: `GlobalAveragePooling2D()` -- Compresses spatial dimensions into 1D feature vector
* **Layer 6**: `Dense(256, relu)` + `Dropout(0.4)`
* **Layer 7**: `Dense(128, relu)` + `Dropout(0.3)`
* **Output Layer**: `Dense(num_classes, softmax)`
            """)

            btn_show_history = gr.Button("Load Training History Plots")
            history_plot = gr.Plot(label="Training & Validation Accuracy/Loss Curves")

    # ── Wire Event Handlers ──
    file_input.change(fn=lambda f: load_file(f, sample_flag=False),
                      inputs=[file_input],
                      outputs=[info_output, preview_plot, slider_row, slider_col, status_output])
    btn_sample.click(fn=lambda: load_file(None, sample_flag=True),
                     outputs=[info_output, preview_plot, slider_row, slider_col, status_output])
    btn_predict.click(fn=run_classification,
                      outputs=[status_output, maps_plot, dist_plot, stats_summary, cm_plot, gt_summary, status_output])
    btn_inspect.click(fn=inspect_pixel, inputs=[slider_row, slider_col],
                      outputs=[spectrum_plot, pixel_info, prob_plot])
    slider_row.change(fn=inspect_pixel, inputs=[slider_row, slider_col],
                      outputs=[spectrum_plot, pixel_info, prob_plot])
    slider_col.change(fn=inspect_pixel, inputs=[slider_row, slider_col],
                      outputs=[spectrum_plot, pixel_info, prob_plot])
    btn_show_history.click(fn=show_training_history, outputs=[history_plot])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
