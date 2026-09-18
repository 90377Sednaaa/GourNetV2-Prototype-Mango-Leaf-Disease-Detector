# Modernize Streamlit UI/UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the GourNet Mango Leaf Disease Detector Streamlit app into a modern, responsive research & benchmarking dashboard with a 3-tab layout, botanical CSS design tokens, sample image support, and exportable batch benchmarking.

**Architecture:** Decompose UI styling and reusable UI components into a dedicated `ui/` module (`ui/styles.py` and `ui/components.py`), while `app.py` serves as the high-level tab router and pipeline controller. Preserve the existing low-latency direct tensor inference logic and caching.

**Tech Stack:** Python 3.12, Streamlit 1.64.0, TensorFlow-CPU 2.21.0, NumPy, Pandas, Pillow.

## Global Constraints
- Maintain existing model architectures in `models/` and weight paths in `weights/` without breaking changes.
- Zero new heavy dependencies (no extra pip packages required; fully compatible with `streamlit==1.64.0`).
- Seamless support for user-supplied images in `samples/` with silent graceful handling if empty.
- Fully compatible with Streamlit's native light and dark modes.

---

### Task 1: Scaffolding `samples/` Directory and User Guide

**Files:**
- Create: `samples/README.md`
- Create: `tests/test_samples_dir.py`

**Interfaces:**
- Produces: `samples/` folder with `README.md` explaining format and placement of sample test images.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_samples_dir.py
import pathlib

def test_samples_directory_exists():
    root = pathlib.Path(__file__).parent.parent
    samples_dir = root / "samples"
    assert samples_dir.is_dir()
    readme = samples_dir / "README.md"
    assert readme.is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_samples_dir.py` or run directly with python.
Expected: FAIL (directory does not exist yet).

- [ ] **Step 3: Create `samples/` and `samples/README.md`**

```markdown
# Sample Mango Leaf Images

Place your sample mango leaf photos (`.jpg`, `.jpeg`, `.png`) in this folder.
When images are added here, the GourNet Streamlit application will automatically populate a **Preset Samples** dropdown selector in the *Single Image Comparison* tab, allowing 1-click evaluation without re-uploading files.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_samples_dir.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add samples/README.md tests/test_samples_dir.py
git commit -m "feat: scaffold samples directory and test instructions"
```

---

### Task 2: Botanical CSS Design System (`ui/styles.py`)

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/styles.py`
- Create: `tests/test_styles.py`

**Interfaces:**
- Produces: `apply_theme_styles()` in `ui/styles.py` which calls `st.markdown(..., unsafe_allow_html=True)` injecting the modern botanical stylesheet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_styles.py
from ui.styles import get_botanical_css

def test_get_botanical_css_tokens():
    css = get_botanical_css()
    assert "#0F5132" in css or "var(--primary-color" in css or ".gournet-card" in css
    assert ".metric-badge" in css
    assert ".status-pill" in css
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_styles.py`
Expected: FAIL (ModuleNotFoundError: No module named 'ui')

- [ ] **Step 3: Implement `ui/__init__.py` and `ui/styles.py`**

Create `ui/__init__.py` and `ui/styles.py` with custom botanical CSS:
- Card containers (`.gournet-card`) with subtle elevation, rounded corners (12px), and clean borders.
- Model status pills (`.status-pill`) with emerald and slate accents.
- Consensus and divergence callout ribbons (`.consensus-banner`, `.divergence-banner`).
- Refined typography and tab highlight styling.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_styles.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/__init__.py ui/styles.py tests/test_styles.py
git commit -m "feat: implement botanical CSS styling tokens and test"
```

---

### Task 3: Modular UI Components (`ui/components.py`)

**Files:**
- Create: `ui/components.py`
- Create: `tests/test_components.py`

**Interfaces:**
- Produces:
  - `render_hero_header()`
  - `render_model_status_ribbon(static_infos, trained_flags, variant, model_registry)`
  - `get_sample_images(samples_dir: pathlib.Path) -> list[pathlib.Path]`
  - `render_consensus_banner(pred_a: str, pred_b: str, conf_a: float, conf_b: float, short_a: str, short_b: str)`
  - `render_comparative_ribbon(a_ms: float, b_ms: float, a_conf: float, b_conf: float, a_margin: float, b_margin: float, short_a: str, short_b: str)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_components.py
import pathlib
from ui.components import get_sample_images, format_consensus_data

def test_get_sample_images_empty_dir(tmp_path):
    assert get_sample_images(tmp_path) == []

def test_get_sample_images_with_files(tmp_path):
    (tmp_path / "leaf1.jpg").write_bytes(b"dummy")
    (tmp_path / "leaf2.png").write_bytes(b"dummy")
    (tmp_path / "notes.txt").write_bytes(b"dummy")
    imgs = get_sample_images(tmp_path)
    assert len(imgs) == 2
    assert all(f.suffix in [".jpg", ".png"] for f in imgs)

def test_format_consensus_data():
    data = format_consensus_data("Healthy", "Healthy", 0.95, 0.98, "Baseline", "v2")
    assert data["agreed"] is True
    assert "Healthy" in data["title"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_components.py`
Expected: FAIL with ModuleNotFoundError or ImportError.

- [ ] **Step 3: Implement `ui/components.py`**

Implement `get_sample_images`, `format_consensus_data`, `render_hero_header`, `render_model_status_ribbon`, `render_consensus_banner`, and `render_comparative_ribbon`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_components.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/components.py tests/test_components.py
git commit -m "feat: add reusable UI components and sample image scanner"
```

---

### Task 4: Streamlit App Refactor & Tabbed Navigation (`app.py`)

**Files:**
- Modify: `app.py`
- Create: `tests/test_app_structure.py`

**Interfaces:**
- Consumes: `ui.styles`, `ui.components`, `models`, `weights`.
- Implements:
  - Modern hero header & model status ribbon.
  - Sidebar weights toggle (`Standard` vs `12k`) and top-k slider.
  - Tab 1: `🔬 Single Image Comparison` (preset selector, consensus banner, side-by-side model cards, comparative metrics ribbon, horizontal probability comparison bar chart, delta table, raw distribution expander).
  - Tab 2: `📊 Batch Benchmark Suite` (multi-file uploader, data editor for true labels, benchmark execution with progress bar, KPI metrics cards, agreement rate, accuracy, result table, disagreements inspection, and CSV download).
  - Tab 3: `📐 Model Architecture & Specs` (side-by-side comparison table of parameters, size, normalizations, pooling, dropout, plus dynamic layer expanders).

- [ ] **Step 1: Write integration test verifying imports and helper logic**

```python
# tests/test_app_structure.py
import app

def test_model_keys_and_registry():
    assert "GourNet (baseline)" in app.MODEL_REGISTRY
    assert "GourNet v2 (enhanced)" in app.MODEL_REGISTRY
    assert len(app.CLASS_NAMES) == 8

def test_prediction_stats():
    import numpy as np
    dummy_probs = np.array([0.1, 0.7, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02])
    stats = app.prediction_stats(dummy_probs)
    assert stats["top_idx"] == 1
    assert stats["order"][0] == 1
    assert stats["margin"] > 0.5
```

- [ ] **Step 2: Run test to verify current state**

Run: `python tests/test_app_structure.py`
Expected: PASS

- [ ] **Step 3: Update `app.py` to modernize UI/UX**

Refactor `app.py` to:
- Inject CSS from `ui.styles`.
- Call `render_hero_header` and `render_model_status_ribbon`.
- Organize the interface into the 3 tabs (`st.tabs`).
- Integrate sample image picker into Tab 1 alongside file uploader.
- Add CSV download button to Tab 2 batch benchmark.
- Build the rich architecture comparison table and dynamic layer inspector into Tab 3.

- [ ] **Step 4: Run integration test and syntax validation**

Run: `python tests/test_app_structure.py`
Run: `python -c "import app; print('app imported cleanly')"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_structure.py
git commit -m "feat: modernize app.py with 3-tab layout and research dashboard UX"
```

---

### Task 5: End-to-End Verification and Validation

**Files:**
- Create: `tests/test_end_to_end.py`

**Interfaces:**
- Performs headless end-to-end inference and benchmark test to verify no regressions in accuracy, latency measurement, probability delta, or batch calculations.

- [ ] **Step 1: Write end-to-end test**

```python
# tests/test_end_to_end.py
import numpy as np
from PIL import Image
import app

def test_end_to_end_inference():
    # Create synthetic test image
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    batch = app.preprocess_image(img)
    
    # Load models
    m_base, ok_base = app.load_model(app.MODEL_KEYS[0], "Standard")
    m_v2, ok_v2 = app.load_model(app.MODEL_KEYS[1], "Standard")
    assert ok_base and ok_v2
    
    # Predict
    p_base, dt_base = app.timed_predict(m_base, batch)
    p_v2, dt_v2 = app.timed_predict(m_v2, batch)
    
    assert len(p_base) == 8
    assert len(p_v2) == 8
    assert dt_base > 0
    assert dt_v2 > 0
```

- [ ] **Step 2: Run end-to-end test**

Run: `python tests/test_end_to_end.py`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run:
```bash
python tests/test_samples_dir.py
python tests/test_styles.py
python tests/test_components.py
python tests/test_app_structure.py
python tests/test_end_to_end.py
```
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: add end-to-end inference and model verification test"
```
