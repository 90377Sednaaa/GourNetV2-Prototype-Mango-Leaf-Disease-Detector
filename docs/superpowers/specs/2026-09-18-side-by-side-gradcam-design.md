# Side-by-Side Grad-CAM Feature Design Specification

- **Date**: 2026-09-18
- **Topic**: Side-by-Side Grad-CAM Visual Explainability for GourNet & GourNet v2
- **Placement**: Directly inside Tab 1 (`🔬 Single Image Comparison`)
- **Key Focus**: Symmetrical, height-aligned dual model visual attribution with interactive controls

---

## 1. Executive Summary & Objectives

The goal is to provide visual explainability (Grad-CAM — Gradient-weighted Class Activation Mapping) directly comparing the attention maps of **GourNet (Baseline)** vs **GourNet v2 (Enhanced)** side-by-side.

This feature allows plant pathologists, growers, and researchers to verify whether the models are focusing on actual leaf lesions (e.g. Anthracnose necrotic patches, powdery mildew white fungal growth, gall midge blisters) or background artifacts.

---

## 2. Technical Architecture & Computation (`models/gradcam.py`)

### Layer Targeting
- **GourNet (Baseline)**: Last convolutional layer is `"Convolution-4"`.
- **GourNet v2**: Last convolutional layer is `"Block4_Conv"`.
- **Fallback**: Dynamic inspection backwards through `model.layers` for the final `Conv2D` layer.

### Computation Pipeline
1. **Model Splitting**:
   ```python
   grad_model = tf.keras.Model(
       inputs=model.inputs,
       outputs=[model.get_layer(last_conv_name).output, model.output]
   )
   ```
2. **Gradient Extraction**:
   Using `tf.GradientTape()` on preprocessed tensor `batch`:
   - Feature map activations $A \in \mathbb{R}^{1 \times H \times W \times C}$
   - Predictions $Y \in \mathbb{R}^{1 \times K}$
   - Loss $y^c = Y[:, c]$ where $c$ is the target class index
   - Gradients: $\text{grads} = \frac{\partial y^c}{\partial A}$
3. **Channel Importance Weights**:
   - $\alpha_k^c = \frac{1}{H \cdot W} \sum_{i,j} \text{grads}_{i,j,k}$
4. **Weighted Activation Map & ReLU**:
   - $\text{cam} = \text{ReLU}\left(\sum_k \alpha_k^c A_k\right)$
   - Normalized to $[0.0, 1.0]$.
5. **Upsampling & Blending**:
   - Resized to input shape $(224, 224)$.
   - Colormapped via Matplotlib (`jet`, `viridis`, `magma`).
   - Alpha-blended onto original leaf image:
     $$\text{Blended} = \alpha \cdot \text{Colormap} + (1 - \alpha) \cdot \text{Original}$$

---

## 3. User Interface & Controls (`ui/components.py` & `app.py`)

### Placement in Tab 1
Placed directly beneath the Comparative Metrics Ribbon and above the Per-Class Probabilities chart.

### Interactive Control Toolbar
- **Master Toggle**: `st.toggle("🔥 Enable Side-by-Side Grad-CAM Explainability", value=False)`
  - When **OFF**: No extra computations are executed.
  - When **ON**: Renders the toolbar and the side-by-side heatmaps.
- **Control Bar**:
  - `Target Class`: Dropdown defaulting to `"Top Predicted Class (Per Model)"` or selecting any specific disease class to inspect what features contribute to other diagnoses.
  - `Overlay Opacity`: Slider from $0\%$ (pure leaf) to $100\%$ (pure heatmap), defaulting to $50\%$.
  - `Colormap`: Dropdown offering `jet` (standard research heatmap), `viridis` (colorblind friendly), and `magma`.

### Symmetrical Height-Aligned Cards
Two columns `col_cam_a, col_cam_b = st.columns(2)` with `st.container(border=True)`:
- Left card: **GourNet (Baseline)** Grad-CAM overlay.
- Right card: **GourNet v2** Grad-CAM overlay.
- Both cards share identical DOM structures, identical image aspect ratios $(224 \times 224)$, and CSS flexbox height stretching to guarantee 100% height alignment.

---

## 4. Verification & Testing

- Unit tests in `tests/test_gradcam.py`:
  - Verify layer resolution for both baseline and v2.
  - Verify gradient computation and heatmap shape $(224, 224)$ with values in $[0, 1]$.
  - Verify heatmap overlay blending produces valid RGB PIL Image.
- End-to-end integration test verifying Grad-CAM execution with real weights.
