# GourNet Streamlit UI/UX Modernization Design Specification

- **Date**: 2026-09-18
- **Topic**: Modernize UI/UX for GourNet Mango Leaf Disease Detector
- **Target Audience & Focus**: Modern Research & Benchmarking Dashboard
- **Visual Aesthetic**: Modern Botanical / Bio-Tech (Deep Emerald, Soft Mint, Crisp Slate)

---

## 1. Executive Summary & Goals

The GourNet prototype compares two convolutional neural network architectures for mango leaf disease detection:
1. **GourNet (Baseline)**: Faithful implementation of Alam et al., featuring 683,656 parameters.
2. **GourNet v2 (Enhanced)**: GroupNormalization + GlobalAveragePooling2D + Dropout variant with 98,280 parameters (~85% parameter reduction).

This design transforms the current single-stream vertical page into a high-performance **Research & Benchmarking Dashboard** utilizing a 3-tab layout, custom botanical design tokens, polished metric cards, structured probability distributions, and a dedicated batch benchmarking suite with CSV export.

---

## 2. Architecture & File Structure

The project code will be organized into modular components to keep styling and helper functions separated from application routing and inference logic:

```
GourNetV2-Prototype-Mango-Leaf-Disease-Detector/
├── app.py                      # Main Streamlit application entry point & tab router
├── ui/
│   ├── __init__.py             # Exports for UI components and styles
│   ├── styles.py               # Botanical CSS design tokens, card styling, and layout polish
│   └── components.py           # Reusable UI widgets: hero banner, metric cards, model pills, sample picker
├── samples/                    # User sample image storage directory
│   └── README.md               # Instructions for placing sample images
├── models/
│   ├── __init__.py
│   ├── gournet.py               # Baseline architecture
│   └── gournet_v2.py            # Enhanced architecture
├── weights/
│   ├── gournet_model.keras
│   ├── gournet_model_12k.keras
│   ├── gournet_v2_model.keras
│   └── gournet_v2_model_12k.keras
├── requirements.txt            # Unchanged (no extra heavy dependencies required)
├── runtime.txt
└── README.md
```

---

## 3. Visual Design System ("Modern Botanical & Research Tech")

### Color Palette
- **Primary Emerald**: `#0F5132` / `#166534` (Deep foliage accent, active states, consensus badges)
- **Secondary Mint**: `#10B981` / `#34D399` (Accent highlights, progress fills, high confidence)
- **Neutral Dark**: `#0F172A` / `#1E293B` (Typography, high-contrast borders)
- **Neutral Light / Card Surface**: `#FFFFFF` with soft background `#F8FAFC`
- **Warning Amber**: `#D97706` / `#F59E0B` (Model divergence, low confidence)
- **Error Coral**: `#DC2626` / `#EF4444` (Missing weights, corrupt files)

### Visual Tokens & Styling
- **Card Containers**: Subtle elevation (`box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04)`), rounded corners (`12px`), crisp borders (`1px solid #E2E8F0`).
- **Status Pills**: Compact badges for model architecture, parameter efficiency, and weight variant status.
- **Data Tables**: Enhanced with `st.column_config.ProgressColumn` and formatted number columns for visual probability tracking.
- **Theme Adaptability**: Compatible with both Streamlit dark and light themes through CSS variable awareness.

---

## 4. Workspaces & Tab Specifications

### Header & Global Controls
- **Hero Title**: Modern branding header: "GourNet Mango Leaf Disease Detector" with descriptive research subtext.
- **Model Status Ribbon**: Symmetrical overview cards for GourNet Baseline vs. GourNet v2 showing parameter count, compression ratio (~7x), loaded weights indicator, and architecture notes.
- **Sidebar**:
  - Weight variant segmented control: `Standard` vs `12k`.
  - Top-k probability slider (1 to 8, default 3).
  - Quick links / metadata.

### Tab 1: 🔬 Single Image Comparison
- **Input Methods**:
  - Drag-and-drop file uploader (`.jpg`, `.jpeg`, `.png`).
  - Sample preset selector: Scans `samples/` directory for user-provided test images. If images exist, renders a visual selector; if empty, renders a clean, non-intrusive hint.
- **Consensus & Divergence Banner**:
  - Display consensus notification if both models agree on the top disease class.
  - Display divergence notification if models predict different classes, highlighting confidence differences.
- **Dual-Model Inference Cards**:
  - Two equal-width columns.
  - Primary metric: Top predicted disease class with confidence bar.
  - Secondary metrics: Latency (ms), Decision Margin (pp), Shannon Entropy.
  - Top-k class breakdown with visual probability progress bars.
- **Comparative Metrics Ribbon**:
  - Which model is faster (with difference in ms).
  - Which model has higher confidence.
  - Which model has larger decision margin.
- **Probability Distribution & Delta**:
  - Per-class horizontal probability comparison bar chart.
  - Probability Shift table ($v2 - \text{baseline}$) sorted by highest positive/negative delta.
  - Expandable full distribution raw values.

### Tab 2: 📊 Batch Benchmark Suite
- **Multi-File Upload**: Supports simultaneous upload of multiple leaf images.
- **Ground Truth Annotation**:
  - Interactive data editor (`st.data_editor`) for assigning true disease labels.
  - Quick action helper to set labels or auto-detect labels from filenames if applicable.
- **Batch Inference Runner**:
  - Progress bar with processing counter.
  - Sequential inference execution preserving low memory footprint.
- **Aggregate KPI Summary Cards**:
  - Total Images Evaluated.
  - Agreement Rate (% of identical predictions between models).
  - Mean Latency per model.
  - Model Accuracy comparison (when true labels are assigned).
- **Results Explorer & Export**:
  - Full evaluation table with color-coded agreement tags.
  - Disagreements filter table to inspect corner-case images.
  - **One-Click CSV Download** button to export benchmark data.

### Tab 3: 📐 Model Architecture & Specs
- **Side-by-Side Comparison Table**:
  - Total Parameters: 683,656 (Baseline) vs 98,280 (v2).
  - Size on Disk: ~8.3 MB vs ~1.3 MB (~85% reduction).
  - Normalization: None / Standard vs Group Normalization.
  - Pooling: Flatten + Dense(128) vs GlobalAveragePooling2D.
  - Regularization: None vs Dropout(0.3).
- **Layer Breakdown Expanders**:
  - Dynamic layer inspection summarizing layer names, output shapes, and parameter counts fetched directly from the loaded Keras models.

---

## 5. Performance, Error Handling & Graceful Degradation

1. **Tensor Execution**:
   - Maintains low-latency direct tensor call (`model(batch, training=False)`), avoiding overhead.
   - Resource caching via `@st.cache_resource` with warmup zero-tensor pass.
2. **Missing Weights & Corrupt Files**:
   - Incomplete or missing weights gracefully fallback to uninitialized architectures with clear "Untrained" badges.
   - Corrupt files caught with specific try/except blocks informing the user without application crashes.
3. **Empty `samples/` Directory**:
   - Handled silently and cleanly; user is informed how to drop test images without throwing file-not-found errors.

---

## 6. Verification & Test Plan

1. **Static Analysis & Architecture Inspection**:
   - Ensure imports from `ui.styles` and `ui.components` work cleanly.
   - Verify model building and loading functions operate without regression.
2. **Interactive UI Testing**:
   - Verify tab navigation and visual styles render properly.
   - Verify single-image prediction and sample image selector.
   - Verify batch benchmarking, progress tracking, and CSV export.
   - Verify light and dark mode appearance.
