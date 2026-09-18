"""Reusable modern UI components for GourNet Mango Leaf Disease Detector."""

import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from models.gradcam import generate_gradcam_heatmap, overlay_gradcam


def get_sample_images(samples_dir: pathlib.Path) -> list[pathlib.Path]:
    """Scans samples_dir for valid images (.jpg, .jpeg, .png) and returns them sorted by name."""
    if not isinstance(samples_dir, pathlib.Path):
        samples_dir = pathlib.Path(samples_dir)
    if not samples_dir.exists() or not samples_dir.is_dir():
        return []

    valid_exts = {".jpg", ".jpeg", ".png"}
    images = [
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    return sorted(images, key=lambda p: p.name)


def format_consensus_data(
    pred_a: str,
    pred_b: str,
    conf_a: float,
    conf_b: float,
    short_a: str = "GourNet",
    short_b: str = "GourNet v2",
) -> dict:
    """Formats consensus or divergence details between two model predictions."""
    agreed = pred_a == pred_b
    if agreed:
        title = f"Consensus Reached: Both models predict {pred_a}"
        detail = f"<strong>{short_a}</strong> ({conf_a * 100:.1f}%) vs <strong>{short_b}</strong> ({conf_b * 100:.1f}%)"
        css_class = "consensus-banner"
    else:
        title = "Models Disagree — Further Diagnostic Review Recommended"
        detail = (
            f"<strong>{short_a}</strong>: {pred_a} ({conf_a * 100:.1f}%) vs "
            f"<strong>{short_b}</strong>: {pred_b} ({conf_b * 100:.1f}%)"
        )
        css_class = "divergence-banner"

    return {
        "agreed": agreed,
        "title": title,
        "detail": detail,
        "css_class": css_class,
    }


def render_hero_header():
    """Renders modern botanical branding header with research subtext."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 800; color: var(--text-color, inherit); display: flex; align-items: center; gap: 0.6rem;">
                🌿 <span style="color: #10B981;">GourNet</span> Mango Leaf Disease Detector
            </h1>
            <p class="header-subtext" style="margin-top: 0.35rem;">
                Comparative Botanical Deep Learning Prototype &amp; Benchmarking Dashboard
                <span class="status-pill" style="margin-left: 0.5rem;">Baseline vs v2</span>
                <span class="status-pill">8 Disease Classes</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status_ribbon(
    static_infos: dict,
    trained_flags: dict,
    variant: str,
    model_registry: dict,
):
    """Renders side-by-side model overview cards (Baseline vs v2)."""
    keys = list(model_registry.keys())
    if not keys:
        return

    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        meta = model_registry[key]
        short_name = meta.get("short", key)
        notes = meta.get("notes", "")
        params, size_mb = static_infos.get(key, (-1, 0.0))
        is_trained = trained_flags.get(key, False)

        with col:
            with st.container(border=True):
                header_html = f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="font-weight: 700; font-size: 1.15rem; color: var(--text-color, inherit);">{short_name}</span>
                    <span class="status-pill{' warning' if not is_trained else ''}">
                        {'● Trained' if is_trained else '⚠ Fallback / Missing'}
                    </span>
                </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)
                if notes:
                    st.caption(notes)

                c1, c2, c3 = st.columns(3)
                param_str = f"{params:,}" if params >= 0 else "N/A"
                if "v2" in short_name.lower():
                    c1.metric("Parameters", param_str, delta="~7x smaller (-85.6%)", delta_color="off")
                else:
                    c1.metric("Parameters", param_str, delta="Baseline reference", delta_color="off")

                c2.metric("Size", f"{size_mb:.1f} MB" if size_mb > 0 else "N/A")
                c3.metric("Weights", variant)


def render_consensus_banner(
    consensus_data: dict | str,
    pred_b: str | None = None,
    conf_a: float | None = None,
    conf_b: float | None = None,
    short_a: str = "GourNet",
    short_b: str = "GourNet v2",
):
    """Renders the consensus/divergence banner using appropriate botanical styles."""
    if isinstance(consensus_data, dict):
        data = consensus_data
    else:
        data = format_consensus_data(
            pred_a=str(consensus_data),
            pred_b=str(pred_b or ""),
            conf_a=float(conf_a if conf_a is not None else 0.0),
            conf_b=float(conf_b if conf_b is not None else 0.0),
            short_a=short_a,
            short_b=short_b,
        )

    css_class = data.get("css_class", "consensus-banner")
    title = data.get("title", "")
    detail = data.get("detail", "")
    icon = "✅" if data.get("agreed", False) else "⚠️"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="banner-title">
                <span>{icon}</span> <span>{title}</span>
            </div>
            <div class="banner-detail">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparative_ribbon(
    a_ms: float,
    b_ms: float,
    a_conf: float,
    b_conf: float,
    a_margin: float,
    b_margin: float,
    short_a: str = "GourNet",
    short_b: str = "GourNet v2",
):
    """Renders 3 metric comparison cards: Faster model, Higher confidence, Larger decision margin."""
    faster = short_a if a_ms <= b_ms else short_b
    ms_diff = abs(a_ms - b_ms)

    higher_conf = short_a if a_conf >= b_conf else short_b
    conf_diff = abs(a_conf - b_conf) * 100.0

    larger_margin = short_a if a_margin >= b_margin else short_b
    margin_diff = abs(a_margin - b_margin) * 100.0

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "⚡ Faster Model",
        faster,
        delta=f"{ms_diff:.1f} ms advantage" if ms_diff > 0 else "Identical latency",
        delta_color="off",
        border=True,
    )
    c2.metric(
        "🎯 Higher Confidence",
        higher_conf,
        delta=f"{conf_diff:.1f} pp higher" if conf_diff > 0 else "Identical confidence",
        delta_color="off",
        border=True,
    )
    c3.metric(
        "📏 Larger Decision Margin",
        larger_margin,
        delta=f"{margin_diff:.1f} pp margin" if margin_diff > 0 else "Identical margin",
        delta_color="off",
        border=True,
    )


def render_model_card(
    short_name: str,
    full_name: str,
    res: dict,
    top_k: int,
    class_names: list,
):
    """Renders a single model's prediction card with top class, metrics, and top-k breakdown."""
    if not res or res.get("probs") is None:
        with st.container(border=True):
            st.subheader(short_name)
            if full_name and full_name != short_name:
                st.caption(full_name)
            st.info("No prediction data available.")
        return

    probs = res.get("probs")
    ms = res.get("ms", 0.0)
    stats = res.get("stats")

    if stats is None and probs is not None:
        order = np.argsort(probs)[::-1]
        top1 = int(order[0])
        top2 = int(order[1]) if len(order) > 1 else top1
        conf1 = float(probs[top1])
        conf2 = float(probs[top2])
        clipped = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(clipped * np.log(clipped)))
        stats = {
            "top_idx": top1,
            "conf": conf1,
            "margin": conf1 - conf2,
            "entropy": entropy,
            "order": order,
        }

    top_idx = stats["top_idx"]
    conf = stats["conf"]
    margin = stats["margin"]
    entropy = stats["entropy"]
    order = stats["order"]

    top_class_name = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"

    with st.container(border=True):
        st.subheader(short_name)
        if full_name and full_name != short_name:
            st.caption(full_name)

        st.metric(
            label="Predicted Class",
            value=top_class_name,
            delta=f"{conf * 100.0:.1f}% confidence",
            delta_color="off",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{ms:.1f} ms")
        m2.metric("Margin", f"{margin * 100.0:.1f} pp")
        m3.metric("Entropy", f"{entropy:.2f}")

        st.markdown("**Top Predictions**")
        topk_indices = order[:top_k]
        topk_rows = [
            {
                "Class": class_names[int(i)] if int(i) < len(class_names) else f"Class {i}",
                "Probability": float(probs[int(i)]),
            }
            for i in topk_indices
        ]
        topk_df = pd.DataFrame(topk_rows)
        st.dataframe(
            topk_df,
            hide_index=True,
            column_config={
                "Class": st.column_config.TextColumn("Class"),
                "Probability": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
            },
        )


def _format_model_name(key: str) -> str:
    """Formats model key into short botanical display name."""
    key_lower = key.lower()
    if "v2" in key_lower:
        return "GourNet v2"
    elif "baseline" in key_lower or key_lower == "gournet":
        return "GourNet (Baseline)"
    return key


def _render_gradcam_cards(
    img: Image.Image,
    batch: np.ndarray,
    models: dict,
    results: dict,
    class_names: list,
    model_keys: list,
    target_choice: str,
    opacity: float,
    colormap: str,
):
    """Computes and renders symmetrical side-by-side cards with full-width overlays and focus metrics."""
    cam_data = {}
    with st.spinner("Generating Grad-CAM heatmaps..."):
        for key in model_keys:
            model = models.get(key)
            if model is None:
                cam_data[key] = {"error": f"Model '{key}' is not available."}
                continue

            if target_choice == "Top Predicted Class (Default)":
                top_idx = None
                if results and key in results and isinstance(results[key], dict):
                    res_k = results[key]
                    if "stats" in res_k and "top_idx" in res_k["stats"]:
                        top_idx = int(res_k["stats"]["top_idx"])
                    elif "probs" in res_k and res_k["probs"] is not None:
                        top_idx = int(np.argmax(res_k["probs"]))

                target_idx = top_idx
                if top_idx is not None and class_names and 0 <= top_idx < len(class_names):
                    target_name = class_names[top_idx]
                else:
                    target_name = "Top Predicted"
            else:
                target_idx = class_names.index(target_choice) if (class_names and target_choice in class_names) else None
                target_name = target_choice

            try:
                heatmap = generate_gradcam_heatmap(model, batch, pred_index=target_idx)
                overlay = overlay_gradcam(img, heatmap, alpha=opacity, colormap_name=colormap)
                peak = float(np.max(heatmap))
                mean = float(np.mean(heatmap))
                cam_data[key] = {
                    "target_name": target_name,
                    "overlay": overlay,
                    "peak": peak,
                    "mean": mean,
                }
            except Exception as e:
                cam_data[key] = {
                    "target_name": target_name,
                    "error": str(e),
                }

    # Symmetrical side-by-side cards
    cols = st.columns(len(model_keys)) if model_keys else st.columns(2)
    for col, key in zip(cols, model_keys):
        short_name = _format_model_name(key)
        data = cam_data.get(key, {})
        target_name = data.get("target_name", "N/A")

        with col:
            with st.container(border=True):
                header_html = f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 700; font-size: 1.15rem; color: var(--text-color, inherit);">{short_name}</span>
                    <span class="status-pill">{target_name}</span>
                </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)

                if "error" in data:
                    st.error(f"Grad-CAM error: {data['error']}")
                elif "overlay" in data:
                    display_img = data["overlay"].resize((448, 448), Image.Resampling.LANCZOS)
                    st.image(
                        display_img,
                        caption=f"{short_name} Grad-CAM Overlay",
                        use_container_width=True,
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("Peak Activation Intensity", f"{data['peak']:.2f}")
                    m2.metric("Mean Activation Focus", f"{data['mean']:.2f}")

    st.caption("🔴 **Red / Warm tones**: High activation salient regions • 🔵 **Blue / Cool tones**: Low activation background")


def render_gradcam_section(
    img: Image.Image,
    batch: np.ndarray,
    models: dict,
    results: dict,
    class_names: list,
    model_keys: list,
    target_choice: str = "Top Predicted Class (Default)",
    opacity: float = 0.5,
    colormap: str = "jet",
    key_prefix: str = "gradcam",
    show_controls: bool = True,
):
    """
    Renders side-by-side Grad-CAM explainability comparison.

    When show_controls is True, displays an inline toggle and parameter controls.
    When show_controls is False (e.g. controls moved to sidebar), directly computes
    and displays the symmetrical side-by-side heatmap cards and activation metrics.
    """
    if show_controls:
        with st.container(border=True):
            enable_gradcam = st.toggle(
                "🔥 Enable Side-by-Side Grad-CAM Explainability",
                value=False,
                key=f"{key_prefix}_toggle",
                help="Generate visual attention heatmaps explaining model predictions.",
            )
            if not enable_gradcam:
                return

            if img is None or batch is None or not models:
                st.info("Grad-CAM explainability requires an active input image and loaded models.")
                return

            # 3-column control bar / toolbar
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                target_choice = st.selectbox(
                    "Target Class",
                    options=["Top Predicted Class (Default)"] + list(class_names),
                    key=f"{key_prefix}_target_class",
                )
            with ctrl2:
                opacity = st.slider(
                    "Heatmap Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key=f"{key_prefix}_opacity",
                )
            with ctrl3:
                colormap = st.selectbox(
                    "Colormap",
                    options=["jet", "viridis", "magma"],
                    key=f"{key_prefix}_colormap",
                )

            _render_gradcam_cards(
                img=img,
                batch=batch,
                models=models,
                results=results,
                class_names=class_names,
                model_keys=model_keys,
                target_choice=target_choice,
                opacity=opacity,
                colormap=colormap,
            )
    else:
        if img is None or batch is None or not models:
            st.info("Grad-CAM explainability requires an active input image and loaded models.")
            return

        st.subheader("🔥 Grad-CAM Explainability Comparison")
        st.caption(
            f"Active target: **{target_choice}** • Colormap: `{colormap}` • Opacity: `{int(opacity * 100)}%` (Configured in Sidebar)"
        )

        _render_gradcam_cards(
            img=img,
            batch=batch,
            models=models,
            results=results,
            class_names=class_names,
            model_keys=model_keys,
            target_choice=target_choice,
            opacity=opacity,
            colormap=colormap,
        )
