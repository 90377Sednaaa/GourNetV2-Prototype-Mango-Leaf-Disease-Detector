# Side-by-Side Grad-CAM Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Side-by-Side Grad-CAM visual explainability feature directly in Tab 1 (`🔬 Single Image Comparison`), with interactive controls (on/off toggle, opacity slider, colormap, target class selector) and height-aligned cards.

**Architecture:** Encapsulate Grad-CAM gradient calculation and color overlay generation in `models/gradcam.py`. Provide a clean UI component in `ui/components.py` (`render_gradcam_section`) that renders directly in `app.py` under the comparative metrics ribbon.

**Tech Stack:** Python 3.12, TensorFlow 2.21.0, NumPy, Matplotlib (colormap), Pillow, Streamlit.

## Global Constraints
- Strictly maintain existing model architectures in `models/gournet.py` and `models/gournet_v2.py`.
- Zero new external dependencies (uses standard TensorFlow GradientTape, Matplotlib, and Pillow).
- On-demand calculation only (runs when toggle is ON, preserving sub-30ms normal inference).
- Symmetrical layout with guaranteed equal card heights via CSS flexbox.

---

### Task 1: Grad-CAM Computation Engine (`models/gradcam.py`)

**Files:**
- Create: `models/gradcam.py`
- Create: `tests/test_gradcam.py`

**Interfaces:**
- Produces in `models/gradcam.py`:
  - `find_last_conv_layer(model: tf.keras.Model) -> str`: Finds `"Convolution-4"`, `"Block4_Conv"`, or the last `Conv2D` layer in the model.
  - `generate_gradcam_heatmap(model: tf.keras.Model, batch: np.ndarray, last_conv_layer_name: str | None = None, pred_index: int | None = None) -> np.ndarray`: Returns normalized $(224, 224)$ 2D float array in $[0.0, 1.0]$.
  - `overlay_gradcam(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, colormap_name: str = "jet") -> Image.Image`: Blends the heatmap over `img` and returns an RGB `PIL.Image`.

- [ ] **Step 1: Write the failing test `tests/test_gradcam.py`**

```python
import unittest
import numpy as np
from PIL import Image
from models import build_gournet, build_gournet_v2
from models.gradcam import find_last_conv_layer, generate_gradcam_heatmap, overlay_gradcam

class TestGradCAM(unittest.TestCase):
    def setUp(self):
        self.m_base = build_gournet(8, (224, 224, 3))
        self.m_v2 = build_gournet_v2(8, (224, 224, 3))

    def test_find_last_conv_layer(self):
        self.assertEqual(find_last_conv_layer(self.m_base), "Convolution-4")
        self.assertEqual(find_last_conv_layer(self.m_v2), "Block4_Conv")

    def test_generate_gradcam_heatmap(self):
        batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
        heatmap = generate_gradcam_heatmap(self.m_base, batch, pred_index=0)
        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(0.0 <= np.min(heatmap) <= np.max(heatmap) <= 1.0)

    def test_overlay_gradcam(self):
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
        blended = overlay_gradcam(img, heatmap, alpha=0.5, colormap_name="jet")
        self.assertEqual(blended.size, (224, 224))
        self.assertEqual(blended.mode, "RGB")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `python tests/test_gradcam.py`
Expected: FAIL (ModuleNotFoundError: No module named 'models.gradcam')

- [ ] **Step 3: Implement `models/gradcam.py`**
Implement `find_last_conv_layer`, `generate_gradcam_heatmap` via `tf.GradientTape`, and `overlay_gradcam` with `matplotlib.colormaps`.

- [ ] **Step 4: Run test to verify it passes**
Run: `python tests/test_gradcam.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add models/gradcam.py tests/test_gradcam.py
git commit -m "feat: implement grad-cam computation engine and unit tests"
```

---

### Task 2: Side-by-Side Grad-CAM UI Component (`ui/components.py`)

**Files:**
- Modify: `ui/components.py`
- Modify: `ui/__init__.py`
- Modify: `tests/test_components.py`

**Interfaces:**
- Produces: `render_gradcam_section(img, batch, models, results, class_names, model_keys)`
  - Renders master toggle `st.toggle("🔥 Enable Side-by-Side Grad-CAM Explainability", value=False)`.
  - When ON, renders control toolbar (Target Class dropdown, Opacity slider, Colormap selectbox).
  - Generates heatmaps for both models.
  - Renders side-by-side equal-height cards for Baseline and v2 with overlay images, target class pills, and peak intensity metrics.

- [ ] **Step 1: Write test for `render_gradcam_section` in `tests/test_components.py`**
Add headless invocation test verifying no exceptions are thrown.

- [ ] **Step 2: Implement `render_gradcam_section` in `ui/components.py` and export in `ui/__init__.py`**

- [ ] **Step 3: Run test to verify it passes**
Run: `python tests/test_components.py`
Expected: PASS

- [ ] **Step 4: Commit**
```bash
git add ui/components.py ui/__init__.py tests/test_components.py
git commit -m "feat: implement side-by-side grad-cam UI component and controls"
```

---

### Task 3: Tab 1 Integration & End-to-End Verification

**Files:**
- Modify: `app.py`
- Modify: `tests/test_end_to_end.py`

**Interfaces:**
- Connects `render_gradcam_section` into Tab 1 of `app.py` right below `render_comparative_ribbon`.
- Adds end-to-end Grad-CAM test in `tests/test_end_to_end.py` verifying real weight execution on both models.

- [ ] **Step 1: Add E2E test in `tests/test_end_to_end.py`**
Verify that Grad-CAM produces valid overlays using loaded Standard and 12k model checkpoints.

- [ ] **Step 2: Connect Grad-CAM component into `app.py`**
Call `render_gradcam_section(img, batch, models, results, CLASS_NAMES, MODEL_KEYS)` in Tab 1.

- [ ] **Step 3: Run full test suite**
Run: `python -m unittest discover -s tests -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**
```bash
git add app.py tests/test_end_to_end.py
git commit -m "feat: integrate side-by-side grad-cam into single image comparison tab"
```
