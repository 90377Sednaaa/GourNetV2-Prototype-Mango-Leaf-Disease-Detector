"""Mango leaf disease detector: side-by-side comparison of GourNet and GourNet v2."""

import pathlib
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = pathlib.Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from models import build_gournet, build_gournet_v2
from ui.styles import apply_theme_styles
from ui.components import (
    get_sample_images,
    format_consensus_data,
    render_hero_header,
    render_model_status_ribbon,
    render_consensus_banner,
    render_comparative_ribbon,
    render_model_card,
)

WEIGHTS_DIR = APP_DIR / "weights"
SAMPLES_DIR = APP_DIR / "samples"

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

MODEL_REGISTRY = {
    "GourNet (baseline)": {
        "short": "GourNet",
        "builder": build_gournet,
        "weights": {
            "Standard": WEIGHTS_DIR / "gournet_model.keras",
            "12k": WEIGHTS_DIR / "gournet_model_12k.keras",
        },
        "notes": "4-conv-block CNN.",
    },
    "GourNet v2 (enhanced)": {
        "short": "GourNet v2",
        "builder": build_gournet_v2,
        "weights": {
            "Standard": WEIGHTS_DIR / "gournet_v2_model.keras",
            "12k": WEIGHTS_DIR / "gournet_v2_model_12k.keras",
        },
        "notes": "GroupNorm + GAP variant.",
    },
}
MODEL_KEYS = list(MODEL_REGISTRY.keys())
WEIGHT_OPTIONS = ["Standard", "12k"]


@st.cache_resource(show_spinner=True)
def load_model(model_key: str, variant: str):
    """Load trained model for the selected weights variant, else build untrained architecture."""
    spec = MODEL_REGISTRY[model_key]
    weights_path = spec["weights"][variant]
    if weights_path.exists():
        try:
            m = tf.keras.models.load_model(weights_path)
            ok = True
        except Exception as e:
            st.warning(f"Could not load {weights_path.name} ({e}).")
            m = spec["builder"](num_classes=len(CLASS_NAMES), input_shape=IMG_SIZE + (3,))
            ok = False
    else:
        m = spec["builder"](num_classes=len(CLASS_NAMES), input_shape=IMG_SIZE + (3,))
        ok = False

    # Warmup graph execution so cold-start initialization does not skew latency metrics
    try:
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype="float32")
        _ = m(dummy, training=False)
    except Exception:
        pass

    return m, ok


def model_static_info(model, weights_path: pathlib.Path):
    try:
        params = int(model.count_params())
    except Exception:
        params = -1
    size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else 0.0
    return params, size_mb


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def timed_predict(model, batch: np.ndarray):
    t0 = time.perf_counter()
    raw = model(batch, training=False)
    probs = raw.numpy()[0].astype(float)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return probs, dt_ms


def prediction_stats(probs: np.ndarray):
    order = np.argsort(probs)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    conf1, conf2 = float(probs[top1]), float(probs[top2])
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log(clipped)))
    return {
        "top_idx": top1,
        "conf": conf1,
        "margin": conf1 - conf2,
        "entropy": entropy,
        "order": order,
    }


def main():
    st.set_page_config(
        page_title="GourNet Mango Leaf Disease Detector",
        page_icon="🌿",
        layout="wide",
    )

    # Apply botanical & research CSS design tokens
    apply_theme_styles()

    # Sidebar controls & documentation
    with st.sidebar:
        st.markdown("### 🌿 GourNet Controls")
        variant = st.segmented_control("Weights Variant", WEIGHT_OPTIONS, default="Standard")
        if variant is None:
            variant = "Standard"
        top_k = st.slider("Top-k classes per model", 1, 8, 3)

        st.divider()
        st.markdown("### ℹ️ About Dataset")
        st.caption(
            "Designed for detection of 8 mango leaf health conditions:\n\n"
            + "\n".join(f"• {cls}" for cls in CLASS_NAMES)
        )
        st.markdown("### 🔬 Model Profiles")
        st.caption(
            "• **GourNet (Baseline)**: Alam et al. (683,656 params, 8.3 MB)\n\n"
            "• **GourNet v2 (Enhanced)**: GroupNorm + GAP + Dropout (98,280 params, 1.3 MB, -85.6%)"
        )

    # Hero branding header
    render_hero_header()

    # Load models
    models, trained_flags, static_infos = {}, {}, {}
    for key in MODEL_KEYS:
        m, ok = load_model(key, variant)
        models[key] = m
        trained_flags[key] = ok
        static_infos[key] = model_static_info(m, MODEL_REGISTRY[key]["weights"][variant])

    # Model status overview ribbon
    render_model_status_ribbon(static_infos, trained_flags, variant, MODEL_REGISTRY)

    missing = [k for k in MODEL_KEYS if not trained_flags[k]]
    if missing:
        st.warning(f"{variant} weights missing for: {', '.join(missing)}. Predictions are using untrained architecture.")
    elif variant == "12k":
        st.info("ℹ️ Note: GourNet (baseline) uses Standard weights for the 12k option; GourNet v2 uses distinct 12-seed trained weights.")

    a_key, b_key = MODEL_KEYS[0], MODEL_KEYS[1]

    # Modern 3-Tab Dashboard Navigation
    tab_single, tab_batch, tab_specs = st.tabs([
        "🔬 Single Image Comparison",
        "📊 Batch Benchmark Suite",
        "📐 Model Architecture & Specs",
    ])

    # =========================================================================
    # TAB 1: Single Image Comparison
    # =========================================================================
    with tab_single:
        samples = get_sample_images(SAMPLES_DIR)
        img = None
        img_source_desc = None

        if samples:
            source_mode = st.radio(
                "Image Source Mode",
                ["Upload Image", "Select Sample Preset"],
                horizontal=True,
            )
            if source_mode == "Upload Image":
                uploaded = st.file_uploader(
                    "Upload a mango leaf photo",
                    type=["jpg", "jpeg", "png"],
                    key="single_image_uploader",
                )
                if uploaded is not None:
                    try:
                        img = Image.open(uploaded)
                        img_source_desc = f"Uploaded: {uploaded.name}"
                    except Exception as e:
                        st.error(f"Error loading uploaded image: {e}")
            else:
                selected_sample = st.selectbox(
                    "Choose a preset sample image:",
                    samples,
                    format_func=lambda p: p.name,
                )
                if selected_sample:
                    try:
                        img = Image.open(selected_sample)
                        img_source_desc = f"Sample preset: {selected_sample.name}"
                    except Exception as e:
                        st.error(f"Error loading sample image: {e}")
        else:
            uploaded = st.file_uploader(
                "Upload a mango leaf photo",
                type=["jpg", "jpeg", "png"],
                key="single_image_uploader",
            )
            if uploaded is not None:
                try:
                    img = Image.open(uploaded)
                    img_source_desc = f"Uploaded: {uploaded.name}"
                except Exception as e:
                    st.error(f"Error loading uploaded image: {e}")

            with st.expander("ℹ️ How to use sample presets", expanded=False):
                st.markdown(
                    "Place `.jpg`, `.jpeg`, or `.png` images into the `samples/` directory "
                    "to quickly select and test presets from this interface without re-uploading. "
                    "See `samples/README.md` for guidelines."
                )

        if img is None:
            st.info("Upload or select a mango leaf photo above to begin comparative inference.")
        else:
            col_preview, col_meta = st.columns([1, 2])
            with col_preview:
                st.image(img, caption=img_source_desc or "Target Leaf Image", width="stretch")
            with col_meta:
                with st.container(border=True):
                    st.markdown("**Image Metadata**")
                    st.write(f"• **Dimensions:** {img.size[0]} × {img.size[1]} px")
                    st.write(f"• **Color Mode:** {img.mode}")
                    st.write(f"• **Format:** {img.format or 'Image File'}")
                    st.write(f"• **Inference Input Shape:** {IMG_SIZE[0]} × {IMG_SIZE[1]} × 3")

            batch = preprocess_image(img)
            results = {}
            with st.spinner("Running inference across both models..."):
                for key in MODEL_KEYS:
                    probs, dt_ms = timed_predict(models[key], batch)
                    results[key] = {"probs": probs, "ms": dt_ms, "stats": prediction_stats(probs)}

            a, b = results[a_key], results[b_key]
            a_stats, b_stats = a["stats"], b["stats"]

            # Consensus / Divergence banner
            consensus_data = format_consensus_data(
                pred_a=CLASS_NAMES[a_stats["top_idx"]],
                pred_b=CLASS_NAMES[b_stats["top_idx"]],
                conf_a=a_stats["conf"],
                conf_b=b_stats["conf"],
                short_a=MODEL_REGISTRY[a_key]["short"],
                short_b=MODEL_REGISTRY[b_key]["short"],
            )
            render_consensus_banner(consensus_data)

            # Symmetrical side-by-side model prediction cards
            col_a, col_b = st.columns(2)
            with col_a:
                render_model_card(
                    short_name=MODEL_REGISTRY[a_key]["short"],
                    full_name=a_key,
                    res=a,
                    top_k=top_k,
                    class_names=CLASS_NAMES,
                )
            with col_b:
                render_model_card(
                    short_name=MODEL_REGISTRY[b_key]["short"],
                    full_name=b_key,
                    res=b,
                    top_k=top_k,
                    class_names=CLASS_NAMES,
                )

            # Comparative metrics ribbon
            render_comparative_ribbon(
                a_ms=a["ms"],
                b_ms=b["ms"],
                a_conf=a_stats["conf"],
                b_conf=b_stats["conf"],
                a_margin=a_stats["margin"],
                b_margin=b_stats["margin"],
                short_a=MODEL_REGISTRY[a_key]["short"],
                short_b=MODEL_REGISTRY[b_key]["short"],
            )

            # Per-class probability horizontal bar chart
            st.subheader("Per-class probabilities")
            comp_df = pd.DataFrame(
                {
                    MODEL_REGISTRY[a_key]["short"]: a["probs"],
                    MODEL_REGISTRY[b_key]["short"]: b["probs"],
                },
                index=CLASS_NAMES,
            )
            st.bar_chart(comp_df, horizontal=True, stack=False, x_label="Probability", y_label="Disease class")

            # Probability delta (v2 − baseline)
            st.subheader("Probability delta (v2 − baseline)")
            delta = b["probs"] - a["probs"]
            delta_df = pd.DataFrame(
                {
                    "Class": CLASS_NAMES,
                    MODEL_REGISTRY[a_key]["short"]: np.round(a["probs"], 4),
                    MODEL_REGISTRY[b_key]["short"]: np.round(b["probs"], 4),
                    "Delta": np.round(delta, 4),
                }
            ).sort_values("Delta", ascending=False)
            st.dataframe(
                delta_df,
                hide_index=True,
                width="stretch",
                column_config={
                    MODEL_REGISTRY[a_key]["short"]: st.column_config.NumberColumn(format="%.4f"),
                    MODEL_REGISTRY[b_key]["short"]: st.column_config.NumberColumn(format="%.4f"),
                    "Delta": st.column_config.NumberColumn(format="+%.4f"),
                },
            )

            # Raw probabilities expander
            with st.expander("Raw probabilities", expanded=False):
                for key in MODEL_KEYS:
                    st.markdown(f"**{key}**")
                    for i in results[key]["stats"]["order"]:
                        st.write(f"{CLASS_NAMES[int(i)]}: {float(results[key]['probs'][int(i)]) * 100:.2f}%")

    # =========================================================================
    # TAB 2: Batch Benchmark Suite
    # =========================================================================
    with tab_batch:
        st.subheader("Batch Benchmark Suite")
        st.caption("Upload multiple images to evaluate agreement rate, model accuracy, and inference latency.")

        multi = st.file_uploader(
            "Upload leaf photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader",
        )

        if multi:
            label_rows = pd.DataFrame([{"File": f.name, "True label": "Unknown"} for f in multi])
            edited = st.data_editor(
                label_rows,
                key="batch_labels",
                hide_index=True,
                width="stretch",
                column_config={
                    "File": st.column_config.TextColumn("File Name", disabled=True),
                    "True label": st.column_config.SelectboxColumn(
                        "True label",
                        options=["Unknown"] + CLASS_NAMES,
                        required=True,
                    ),
                },
            )

            col_run, col_clear = st.columns([2, 5])
            run_clicked = col_run.button("Run batch benchmark", icon=":material/play_arrow:")
            if col_clear.button("Clear benchmark results", disabled="batch_results" not in st.session_state):
                st.session_state.pop("batch_results", None)
                st.rerun()

            if run_clicked:
                records = []
                bar = st.progress(0, text="Running batch inference...")
                for i, f in enumerate(multi):
                    try:
                        f.seek(0)
                        bimg = Image.open(f)
                        bbatch = preprocess_image(bimg)
                    except Exception as e:
                        st.warning(f"Skipping {f.name}: {e}")
                        continue
                    row = {"File": f.name}
                    for key in MODEL_KEYS:
                        short = MODEL_REGISTRY[key]["short"]
                        probs, dt_ms = timed_predict(models[key], bbatch)
                        s = prediction_stats(probs)
                        row[f"{short} pred"] = CLASS_NAMES[s["top_idx"]]
                        row[f"{short} conf"] = round(s["conf"], 4)
                        row[f"{short} ms"] = round(dt_ms, 1)
                        row[f"{short} margin"] = round(s["margin"], 4)
                        row[f"{short} entropy"] = round(s["entropy"], 4)
                    records.append(row)
                    bar.progress((i + 1) / len(multi), text=f"Processed {i + 1}/{len(multi)}")
                bar.empty()

                if records:
                    res_df = pd.DataFrame(records)
                    truth = {r["File"]: r["True label"] for r in edited.to_dict("records")}
                    res_df["True label"] = res_df["File"].map(truth)
                    res_df["Agree"] = res_df[f"{MODEL_REGISTRY[a_key]['short']} pred"] == res_df[
                        f"{MODEL_REGISTRY[b_key]['short']} pred"
                    ]
                    st.session_state["batch_results"] = res_df
                else:
                    st.error("No images could be processed.")

            if "batch_results" in st.session_state:
                res_df = st.session_state["batch_results"]
                truth = {r["File"]: r["True label"] for r in edited.to_dict("records")}
                res_df["True label"] = res_df["File"].map(truth)

                scored = res_df[res_df["True label"] != "Unknown"]
                with st.container(horizontal=True):
                    st.metric("Total Images", str(len(res_df)), border=True)
                    st.metric("Agreement Rate", f"{res_df['Agree'].mean() * 100:.1f}%", border=True)
                    for key in MODEL_KEYS:
                        short = MODEL_REGISTRY[key]["short"]
                        st.metric(f"{short} Mean Latency", f"{res_df[f'{short} ms'].mean():.1f} ms", border=True)

                if len(scored):
                    with st.container(horizontal=True):
                        for key in MODEL_KEYS:
                            short = MODEL_REGISTRY[key]["short"]
                            acc = (scored[f"{short} pred"] == scored["True label"]).mean() * 100
                            st.metric(f"{short} Accuracy (n={len(scored)})", f"{acc:.1f}%", border=True)
                else:
                    st.info("Assign true labels above to calculate accuracy metrics.")

                st.markdown("**Evaluation Table**")
                st.dataframe(res_df, hide_index=True, width="stretch")

                disagreements = res_df[~res_df["Agree"]]
                if len(disagreements):
                    st.subheader("Model Disagreements Inspection")
                    st.caption("Cases where GourNet and GourNet v2 predicted different classes.")
                    st.dataframe(disagreements, hide_index=True, width="stretch")
                else:
                    st.success("🎉 Perfect agreement across all evaluated images! Both models produced identical predictions.")

                # One-click CSV download
                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Benchmark CSV",
                    data=csv_bytes,
                    file_name="gournet_batch_benchmark.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
        else:
            st.info("Upload multiple mango leaf photos above to launch batch benchmarking.")

    # =========================================================================
    # TAB 3: Model Architecture & Specs
    # =========================================================================
    with tab_specs:
        st.subheader("Model Architecture & Technical Specifications")
        st.caption("Detailed architectural comparison between baseline GourNet (Alam et al.) and GourNet v2 (enhanced).")

        params_a, size_a = static_infos[a_key]
        params_b, size_b = static_infos[b_key]
        params_a_str = f"{params_a:,}" if params_a >= 0 else "683,656"
        params_b_str = f"{params_b:,} (-85.6%)" if params_b >= 0 else "98,280 (-85.6%)"
        size_a_str = f"{size_a:.1f} MB" if size_a > 0 else "~8.3 MB"
        size_b_str = f"{size_b:.1f} MB (~7x smaller)" if size_b > 0 else "~1.3 MB (~7x smaller)"

        specs_data = [
            {"Specification": "Parameters", "GourNet (Baseline)": params_a_str, "GourNet v2 (Enhanced)": params_b_str},
            {"Specification": "Size on Disk", "GourNet (Baseline)": size_a_str, "GourNet v2 (Enhanced)": size_b_str},
            {"Specification": "Normalization", "GourNet (Baseline)": "None", "GourNet v2 (Enhanced)": "Group Normalization (groups=8/16/32)"},
            {"Specification": "Classifier / Pooling", "GourNet (Baseline)": "Flatten + Dense(128)", "GourNet v2 (Enhanced)": "GlobalAveragePooling2D"},
            {"Specification": "Regularization", "GourNet (Baseline)": "None", "GourNet v2 (Enhanced)": "Dropout (0.3)"},
            {"Specification": "Input Resolution", "GourNet (Baseline)": "224 × 224 × 3", "GourNet v2 (Enhanced)": "224 × 224 × 3"},
            {"Specification": "Output Classes", "GourNet (Baseline)": "8 disease classes", "GourNet v2 (Enhanced)": "8 disease classes"},
            {"Specification": "Activation", "GourNet (Baseline)": "ReLU + Softmax", "GourNet v2 (Enhanced)": "ReLU + Softmax"},
        ]
        specs_df = pd.DataFrame(specs_data)
        st.dataframe(specs_df, hide_index=True, width="stretch")

        st.divider()
        st.markdown("### 🔬 Architectural Innovations in GourNet v2")
        c_innov1, c_innov2, c_innov3 = st.columns(3)
        with c_innov1:
            with st.container(border=True):
                st.markdown("**1. Group Normalization**")
                st.caption(
                    "Replaces standard batch-dependent normalization with intra-channel group divisions. "
                    "Ensures stable activations and robust feature representations even during single-image edge inferences."
                )
        with c_innov2:
            with st.container(border=True):
                st.markdown("**2. Global Average Pooling**")
                st.caption(
                    "Replaces the heavy Flatten + Dense(128) layers with spatial average pooling. "
                    "Eliminates ~580,000 parameters and prevents spatial overfitting."
                )
        with c_innov3:
            with st.container(border=True):
                st.markdown("**3. Dropout Regularization**")
                st.caption(
                    "Incorporates 30% dropout rate prior to final softmax classification. "
                    "Prevents co-adaptation of features and provides superior generalization on unseen leaf samples."
                )

        st.subheader("Dynamic Layer Inspection")
        st.caption("Extracted directly from loaded Keras computational graphs.")
        for key in MODEL_KEYS:
            model = models[key]
            with st.expander(f"Layer details: {key}", expanded=False):
                try:
                    layer_rows = []
                    for idx, layer in enumerate(model.layers):
                        shape = getattr(layer, "output_shape", None) or getattr(layer, "batch_shape", None)
                        if shape is None and hasattr(layer, "output"):
                            try:
                                shape = tuple(layer.output.shape)
                            except Exception:
                                shape = str(layer.output.shape)
                        layer_rows.append({
                            "#": idx + 1,
                            "Layer Name": layer.name,
                            "Layer Type": layer.__class__.__name__,
                            "Output Shape": str(shape) if shape is not None else "N/A",
                            "Parameters": f"{layer.count_params():,}",
                        })
                    st.dataframe(pd.DataFrame(layer_rows), hide_index=True, width="stretch")
                except Exception as e:
                    st.warning(f"Could not extract layer details for {key}: {e}")


if __name__ == "__main__" or (hasattr(st, "runtime") and st.runtime.exists()):
    main()
